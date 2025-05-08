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


def __Init__():
    # Load pre-trained models
    model = DQN.load("rl_policy/dqn_models/dqn_obstacleavoidance")
    agent_0 = DQN.load("rl_policy/dqn_models/dqn_obstacleavoidance_0")
    agent_1 = DQN.load("rl_policy/dqn_models/dqn_obstacleavoidance_1")
    print("✅ Successfully loaded pre-trained models")

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
    print("✅ Successfully computed M_star")

    # Build regions and extension regions
    M_0 = M_i(M_star, index=0)
    M_1 = M_i(M_star, index=1) if len(M_star) > 1 else M_0
    print("✅ Successfully built M_0 and M_1")
    X_0 = find_X_i(M_0, model)
    X_1 = find_X_i(M_1, model)
    print("✅ Successfully built X_0 and X_1")
    M_ext0 = M_ext(M_0, X_0)
    M_ext1 = M_ext(M_1, X_1)
    print("✅ Successfully built M_ext0 and M_ext1")

    # Initialize hybrid agent
    hybrid_agent = HyRL_agent(agent_0, agent_1, M_ext0, M_ext1, q_init=0)
    print("✅ Successfully initialized hybrid agent")

    # Simulate the hybrid agent compared to the original agent
    generate_sim_plot=False
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
                hybrid_agent_sim = HyRL_agent(agent_0, agent_1, M_ext0, M_ext1, q_init=q)
                simulate_obstacleavoidance(
                    hybrid_agent_sim, model, state_init, figure_number=3 + q
                )
            save_name = "OA_HyRLDQN_Sim_q" + str(q) + ".png"
            plt.savefig(save_name, format="png")
        print("✅ Saved png file")
    return hybrid_agent

class DroneService(drone_grpc.DroneServiceBase):
    def __init__(self, hybrid_agent):
        self.hybrid_agent = hybrid_agent

    async def GetDirection(self, stream):
        request: drone_pb2.DirectionRequest = await stream.recv_message()
        print(f"Received drone state: {request}")

        state = np.array([request.drone_state.x, request.drone_state.y])
        obs = state_to_observation_OA(state)
        action_array, _ = self.hybrid_agent.predict(obs)
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
    # Initialize the hybrid agent at startup
    print("Initializing RL models...")
    hybrid_agent = __Init__()

    server = Server([DroneService(hybrid_agent)])
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

    # Wait for Ctrl+C shutdown signal
    await stop.wait()
    server.close()
    await server.wait_closed()
    print("Server Stopped.")


if __name__ == "__main__":
    print("Starting gRPC server...")
    asyncio.run(main())