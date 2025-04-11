import asyncio
import signal
import numpy as np
import matplotlib.pyplot as plt
from grpclib.server import Server
from grpclib.exceptions import GRPCError
from grpclib.const import Status
from stable_baselines3 import DQN
from . import drone_pb2
from . import drone_grpc
from .training_env import ObstacleAvoidance
from .training_tools import (
    find_critical_points,
    state_to_observation_OA,
    get_state_from_env_OA,
    find_X_i,
    M_i,
    M_ext,
    HyRL_agent,
    simulate_obstacleavoidance,
)

# Global variables for models and environment
model = None
agent_0 = None
agent_1 = None
hybrid_agent = None
M_ext0 = None
M_ext1 = None
obstacle_vertices = None
generate_sim_plot = False  # Used to generate simulation plots


def initialize_models():
    global model, agent_0, agent_1, M_ext0, M_ext1, hybrid_agent

    # Load pre-trained models
    model = DQN.load("rl_policy/dqn_models/dqn_obstacleavoidance")
    agent_0 = DQN.load("rl_policy/dqn_models/dqn_obstacleavoidance_0")
    agent_1 = DQN.load("rl_policy/dqn_models/dqn_obstacleavoidance_1")
    print("✅ Succesfully loaded pre-trained models")

    # Compute critical points (simplify if no obstacle)
    resolution = 30
    x_ = np.linspace(0, 3, resolution)
    y_ = np.linspace(-1.5, 1.5, resolution)
    state_difference = np.linalg.norm(np.array([x_[1] - x_[0], y_[1] - y_[0]]))
    initial_points = []
    for idx in range(resolution):
        for idy in range(resolution):
            initial_points.append(np.array([x_[idx], y_[idy]], dtype=np.float32))

    M_star = find_critical_points(
        initial_points,
        state_difference,
        model,
        ObstacleAvoidance,
        min_state_difference=1e-2,
        steps=5,
        threshold=1e-1,
        n_clusters=8,
        custom_state_to_observation=state_to_observation_OA,
        get_state_from_env=get_state_from_env_OA,
        verbose=False,
    )
    M_star = M_star[np.argsort(M_star[:, 0])]
    print("✅ Succesfully computed M_star")

    # Build regions and extension regions
    M_0 = M_i(M_star, index=0)
    M_1 = M_i(M_star, index=1) if len(M_star) > 1 else M_0  # Fallback if only one point
    print("✅ Succesfully built M_0 and M_1")
    X_0 = find_X_i(M_0, model)  # Minimal extension
    X_1 = find_X_i(M_1, model)
    print("✅ Succesfully built X_0 and X_1")
    M_ext0 = M_ext(M_0, X_0)
    M_ext1 = M_ext(M_1, X_1)
    print("✅ Succesfully built M_ext0 and M_ext1")

    # Initialize hybrid agent
    hybrid_agent = HyRL_agent(agent_0, agent_1, M_ext0, M_ext1, q_init=0)
    print("✅ Succesfully initialized hybrid agent")

    # Simulate the hybrid agent compared to the original agent
    if generate_sim_plot:
        print("✅ Starting simulation")
        starting_conditions = [
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.055], dtype=np.float32),
            np.array([0.0, -0.055], dtype=np.float32),
            np.array([0.0, 0.15], dtype=np.float32),
            np.array([0.0, -0.15], dtype=np.float32),
        ]
        for q in range(2):
            for state_init in starting_conditions:
                hybrid_agent = HyRL_agent(agent_0, agent_1, M_ext0, M_ext1, q_init=q)
                simulate_obstacleavoidance(
                    hybrid_agent, model, state_init, figure_number=3 + q
                )
            save_name = "OA_HyRLDQN_Sim_q" + str(q) + ".png"
            plt.savefig(save_name, format="png")
        print("✅ saved png file")


class DroneService(drone_grpc.DroneServiceBase):
    # Calls training_env -> train_agent -> HyRL -> utils
    async def SetEnvironment(self, stream):
        raise GRPCError(  # Corrected to GRPCError
            status=Status.UNIMPLEMENTED,
            message="The SetEnvironment endpoint is deprecated.",
        )
        global obstacle_centroid, obstacle_radius, goal_position, hybrid_agent, M_ext0, M_ext1
        obstacle_centroid = [1.5, 0.0]
        obstacle_radius = 0.75
        goal_position = [3.0, 0.0]
        request: drone_pb2.SetEnvironmentRequest = await stream.recv_message()
        vertices = [(p.x, p.y, p.z) for v in request.vertex for p in v.vertices]
        print(
            f"Received environment with {len(vertices)} obstacles and goal:\n{request.goal}"
        )

        # Edge Case: An empty vertex request will initialize an environment with obstacle set at goal with rad=0
        if not vertices:
            has_obstacle = False
            obstacle_centroid = goal_position  # No obstacle, align with goal
            obstacle_radius = 0.0
            # Update goal (default if not provided)
            if hasattr(request, "goal") and request.goal:
                goal_position = [request.goal.x, request.goal.y]
            else:
                goal_position = [3.0, 0.0]

            # Reinitialize with no obstacle
            initialize_models(
                x_obst=obstacle_centroid[0],
                y_obst=obstacle_centroid[1],
                radius_obst=0.0,
                x_goal=goal_position[0],
                y_goal=goal_position[1],
                has_obstacle=False,
            )

            response = drone_pb2.SetEnvironmentResponse(
                message=f"Environment set without obstacle and goal at {goal_position}"
            )
            await stream.send_message(response)
            return

        # Edge Case: Less than 3 verticies in request returns an error
        if len(vertices) < 3:
            response = drone_pb2.SetEnvironmentResponse(
                message="Error: At least 3 vertices required for obstacle"
            )
            await stream.send_message(response)
            return

        # Compute 2D centroid in XY ignorizing Z and store in obstacle_centroid
        x_sum, y_sum = 0, 0
        for x, y, z in vertices:
            x_sum += x
            y_sum += y
        num_vertices = len(vertices)
        x_obst = x_sum / num_vertices
        y_obst = y_sum / num_vertices
        obstacle_centroid = [x_obst, y_obst]

        # Sort the verticies to find max dist from cetroid and use to aproximate a radial boundary for obstacle
        max_dist = 0
        for x, y, z in vertices:
            dist = np.sqrt((x - x_obst) ** 2 + (y - y_obst) ** 2)
            max_dist = max(max_dist, dist)
        obstacle_radius = max_dist

        # Update goal (default to [3, 0] if not provided; adjust proto if goal is included)
        # Assuming request.goal exists with x, y, z fields; modify if different
        if request.HasField("goal"):
            goal_position = [request.goal.x, request.goal.y]
        else:
            goal_position = [3.0, 0.0]  # Fallback

        # Reinitialize environment with new obstacle/goal
        initialize_models(
            x_obst=obstacle_centroid[0],
            y_obst=obstacle_centroid[1],
            radius_obst=obstacle_radius,
            x_goal=goal_position[0],
            y_goal=goal_position[1],
        )

        response = drone_pb2.SetEnvironmentResponse(
            message=f"Environment set: Obstacle at {obstacle_centroid}, radius {obstacle_radius}, goal at {goal_position}"
        )
        await stream.send_message(response)

    async def GetDirection(self, stream):
        global hybrid_agent, obstacle_centroid, obstacle_radius, goal_position, M_ext0, M_ext1

        request: drone_pb2.DirectionRequest = await stream.recv_message()
        print(f"Received drone state: {request}")

        state = np.array([request.drone_state.x, request.drone_state.y])
        obs = state_to_observation_OA(state)
        action_array, _ = hybrid_agent.predict(obs)
        if isinstance(action_array, np.ndarray):
            action = float(action_array.item())
        else:
            action = float(action_array)

        direction_map = {
            0: drone_pb2.STRAIGHT,  # 1
            1: drone_pb2.LEFT,  # 2
            2: drone_pb2.HARD_LEFT,  # 3
            3: drone_pb2.RIGHT,  # 4
            4: drone_pb2.HARD_RIGHT,  # 5
        }
        direction = direction_map.get(action, drone_pb2.STRAIGHT)
        
        # Send response
        heading = drone_pb2.DiscreteHeading()
        heading.direction = direction
        response = drone_pb2.DirectionResponse(discrete_heading=heading)
        print(f"Response direction: {heading.direction}")
        await stream.send_message(response)


async def main():
    # Initialize models and hybrid agent at startup
    print("Initializing RL models...")
    initialize_models()

    server = Server([DroneService()])
    await server.start("127.0.0.1", 50051)
    print("gRPC server running on 127.0.0.1:50051")

    # Set up signal handling
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def shutdown():
        print("\nShutting Down Server...")
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown)

    # Wait for Ctrl+c shutdown signal
    await stop.wait()
    server.close()
    await server.wait_closed()
    print("Server Stopped.")


if __name__ == "__main__":
    print("Starting gRPC server...")
    asyncio.run(main())
