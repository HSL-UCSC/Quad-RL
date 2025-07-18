import asyncio
import os
import signal
import numpy as np
import matplotlib.pyplot as plt
from grpclib.server import Server
from grpclib.exceptions import GRPCError
from grpclib.const import Status
from dotenv import load_dotenv

from stable_baselines3 import DQN
from typing import List

from scipy.interpolate import splprep, splev


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
    hybrid_q0: HyRL_agent
    hybrid_q1: HyRL_agent
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
        steps=15,
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

    # Initialize hybrid agents
    hybrid_agent_q0 = HyRL_agent(agent_0, agent_1, M_ext0, M_ext1, q_init=0)
    hybrid_agent_q1 = HyRL_agent(agent_0, agent_1, M_ext0, M_ext1, q_init=1)
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
    return ObstacleAvoidanceModels(
        hybrid_q0=hybrid_agent_q0, hybrid_q1=hybrid_agent_q1, standard=model
    )


class DroneService(obstacle_avoidance_grpc.ObstacleAvoidanceServiceBase):
    direction_map = {
        0: oa_proto.HeadingDirection.STRAIGHT,  # 1
        1: oa_proto.HeadingDirection.LEFT,  # 2
        2: oa_proto.HeadingDirection.HARD_LEFT,  # 3
        3: oa_proto.HeadingDirection.RIGHT,  # 4
        4: oa_proto.HeadingDirection.HARD_RIGHT,  # 5
    }

    def __init__(self, models: ObstacleAvoidanceModels):
        self.hybrid_agent_q0 = models.hybrid_q0
        self.hybrid_agent_q1 = models.hybrid_q1
        self.agent = models.standard

    def agent_select(
        self, state: oa_proto.DroneState, model_type: oa_proto.ModelType.ValueType
    ) -> HyRL_agent | DQN:
        if model_type == oa_proto.ModelType.STANDARD:
            agent = self.agent
        else:
            agent = self.hybrid_agent_q0 if state.y > 0 else self.hybrid_agent_q1
        return agent

    async def GetDirection(self, stream):
        request: oa_proto.DirectionRequest = await stream.recv_message()
        print(f"Received drone state: {request}")

        state = np.array([request.state.x, request.state.y])
        obs = state_to_observation_OA(state)

        agent = self.agent_select(request.state, request.model_type)
        action_array, _ = agent.predict(obs)
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
        def smooth_path(states, num_waypoints):
            if len(states) < 3:
                return states
            # Fit a spline to the states
            tck, u = splprep(np.array(states).T, s=2)
            new_u = np.linspace(0, 1, num_waypoints)
            smoothed_states = splev(new_u, tck)
            return np.array(smoothed_states).T.tolist()

        request: oa_proto.TrajectoryRequest = await stream.recv_message()
        print(f"Received trajectory request: {request}")
        if 0 < request.num_waypoints < 3:
            raise GRPCError(
                Status.INVALID_ARGUMENT,
                "num_waypoints must be greater than or equal to 3, or less than 0",
            )

        [x_start, y_start, z_start] = [
            request.state.x,
            request.state.y,
            request.state.z,
        ]
        [x_target, y_target, z_target] = [
            request.target_state.x,
            request.target_state.y,
            request.target_state.z,
        ]
        state = np.array([x_start, y_start], dtype=np.float32)
        states: List[List[float]] = [list(state) + [z_start]]
        duration_s = request.duration_s

        # Agent select
        # The q0 agent will bias to go up and around the obstacle, while q1 will bias to go the other way around
        agent = self.agent_select(request.state, request.model_type)

        t_sampling = request.sampling_time if request.sampling_time else 0.05
        steps = int(duration_s / t_sampling)
        env = ObstacleAvoidance(
            state_init=state, random_init=False, steps=steps, t_sampling=t_sampling
        )

        target_pos = np.array([x_target, y_target])
        dist_threshold = 0.1

        done = False
        dist_to_target = np.inf
        noise_mag = os.getenv("NOISE_MAG", "0.3")
        sign = 1
        while not done:
            noise = noise_mag * sign
            disturbance = np.array([0, noise], dtype=np.float32)
            if request.noise:
                obs = state_to_observation_OA(state + disturbance)
            else:
                obs = state_to_observation_OA(state)
            action_array, _ = agent.predict(obs)

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
            sign *= -1

        total_states = len(states)
        states = smooth_path(states, request.num_waypoints)
        response = oa_proto.TrajectoryResponse(
            trajectory=[oa_proto.DroneState(x=x, y=y, z=z) for x, y, z in states]
        )
        print(f"Total states generated: {total_states}")
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
    load_dotenv()
    print("Starting gRPC server...")
    asyncio.run(main())
