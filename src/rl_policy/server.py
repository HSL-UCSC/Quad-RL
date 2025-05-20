import asyncio
import signal
import numpy as np
import matplotlib.pyplot as plt
from grpclib.server import Server
from stable_baselines3 import DQN

from hyrl_api import obstacle_avoidance_grpc
from hyrl_api import obstacle_avoidance_pb2 as oa_proto
from rl_policy.training_env import ObstacleAvoidance
from rl_policy.training_tools import (
    find_critical_points,
    state_to_observation_OA,
    get_state_from_env_OA,
    find_X_i,
    M_i,
    M_ext,
    HyRL_agent,
    simulate_obstacleavoidance,
)
from dataclasses import dataclass
from pathlib import Path
from importlib.resources import path
import importlib.resources as pkg_resources


@dataclass
class ObstacleAvoidanceModels:
    hybrid: HyRL_agent
    standard: DQN


def initialize_hybrid_models():
    # Load pre-trained models
    # with pkg_resources.path("dqn_models", "dqn_obstacleavoidance") as p:
    #     print(p)
    with path("dqn_models", "dqn_obstacleavoidance") as model_dir:
        model_path = Path(model_dir)
        print(f"Loading main model from {model_path}")
        model = DQN.load(str(model_path))

    with path("dqn_models", "dqn_obstacleavoidance_0") as a0_dir:
        a0_path = Path(a0_dir)
        print(f"Loading agent_0 from {a0_path}")
        agent_0 = DQN.load(str(a0_path))

    with path("dqn_models", "dqn_obstacleavoidance_1") as a1_dir:
        a1_path = Path(a1_dir)
        print(f"Loading agent_1 from {a1_path}")
        agent_1 = DQN.load(str(a1_path))

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
    generate_sim_plot = False
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
                hybrid_agent_sim = HyRL_agent(
                    agent_0, agent_1, M_ext0, M_ext1, q_init=q
                )
                simulate_obstacleavoidance(
                    hybrid_agent_sim, model, state_init, figure_number=3 + q
                )
            save_name = "OA_HyRLDQN_Sim_q" + str(q) + ".png"
            plt.savefig(save_name, format="png")
        print("✅ Saved png file")
    return ObstacleAvoidanceModels(hybrid=hybrid_agent, standard=model)


class DroneService(obstacle_avoidance_grpc.ObstacleAvoidanceServiceBase):

    direction_map = {
        0: oa_proto.HeadingDirection.STRAIGHT,  # 1
        1: oa_proto.HeadingDirection.LEFT,  # 2
        2: oa_proto.HeadingDirection.HARD_LEFT,  # 3
        3: oa_proto.HeadingDirection.RIGHT,  # 4
        4: oa_proto.HeadingDirection.HARD_RIGHT,  # 5
    }

    def __init__(self, models: ObstacleAvoidanceModels):
        self.hybrid_agent = models.hybrid
        self.agent = models.standard

    async def GetDirection(self, stream):
        request: oa_proto.DirectionRequest = await stream.recv_message()
        print(f"Received drone state: {request}")

        state = np.array([request.drone_state.x, request.drone_state.y])
        obs = state_to_observation_OA(state)
        action_array, _ = self.hybrid_agent.predict(obs)
        if isinstance(action_array, np.ndarray):
            action = int(action_array.item())
        else:
            action = int(action_array)

        # Send response
        response = oa_proto.DirectionResponse(
            discrete_heading=oa_proto.DiscreteHeading(
                direction=self.direction_map.get(
                    action, oa_proto.HeadingDirection.STRAIGHT
                )
            )
        )
        print(f"Response direction: {response.discrete_heading}")
        await stream.send_message(response)

    async def GetTrajectory(self, stream):
        request: oa_proto.TrajectoryRequest = await stream.recv_message()
        print(f"Received trajectory request: {request}")

        # todo: implement get trajectory loop here
        # obs = state_to_observation_OA(state)
        # action_array, _ = self.hybrid_agent.predict(obs)
        # if isinstance(action_array, np.ndarray):
        #     action = int(action_array.item())
        # else:
        #     action = int(action_array)
        #
        # Send response

        x_start = request.current_state.x
        y_start = request.current_state.y
        z_start = request.current_state.z
        x_target = request.target_state.x
        y_target = request.target_state.y
        z_target = request.target_state.z
        num_waypoints = request.num_waypoints
        duration_s = request.duration_s

        state = np.array([x_start, y_start], dtype=np.float32)
        t_sampling = 0.05
        steps = int(duration_s / t_sampling)
        env = ObstacleAvoidance(state_init=state, random_init=False, steps=steps)
        done = False
        states = [[state[0], state[1], z_start]]

        target_pos = np.array([x_target, y_target])
        dist_threshold = 0.1

        while not done:
            obs = state_to_observation_OA(state)
            action_array, _ = self.hybrid_agent.predict(obs)

            if isinstance(action_array, np.ndarray):
                action = int(action_array.item())
            else:
                action = int(action_array)

            env.state = state
            _, _, done, _ = env.step(action)
            state = get_state_from_env_OA(env)
            dist_to_target = np.linalg.norm(state - target_pos)
            if dist_to_target < dist_threshold:
                done = True

            if not env.terminate: 
                states.append([state[0], state[1], z_start])
                print(f"Step: State=[{state[0]:.4f}, {state[1]:.4f}], Action={action}")
                
        if dist_to_target < dist_threshold:
            states[-1] = [x_target, y_target, z_target]
        else:
            states.append[x_target, y_target, z_target]

        total_states = len(states)
        if total_states < num_waypoints:
            states.extend([[x_target, y_target, z_target]] * (num_waypoints - total_states))
        elif total_states > num_waypoints:
            indices = np.linspace(0, total_states - 1, num_waypoints, dtype=int)
            states = [states[i] for i in indices]

              
        response = oa_proto.TrajectoryResponse(
            trajectory = [
                oa_proto.DroneState(x=x, y=y, z=z)
                for x, y, z in states
        ])
        print(f"Generated trajectory: {[wp for wp in response.trajectory]}")
        await stream.send_message(response)


async def main():
    # Initialize the hybrid agent at startup
    print("Initializing RL models...")
    hybrid_agent = initialize_hybrid_models()

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
