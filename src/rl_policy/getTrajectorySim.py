import asyncio
import numpy as np
import matplotlib.pyplot as plt
import argparse
from grpclib.client import Channel
from hyrl_api import obstacle_avoidance_pb2 as oa_proto
from hyrl_api import obstacle_avoidance_grpc as oa_grpc

HOST = "127.0.0.1"
PORT = 50051
X_MAX = 3.0
Y_MAX = 1.5
OBSTACLE_CENTER = (1.5, 0.0)
OBSTACLE_RADIUS = 0.75

async def get_trajectory(channel, x, y, z, num_waypoints, duration_s):
    stub = oa_grpc.ObstacleAvoidanceServiceStub(channel)
    request = oa_proto.TrajectoryRequest(
        current_state=oa_proto.DroneState(x=x, y=y, z=z),
        target_state=oa_proto.DroneState(x=3.0, y=0.0, z=z),
        num_waypoints=num_waypoints,
        duration_s=duration_s
    )
    response = await stub.GetTrajectory(request)
    # Extract waypoints from the response
    waypoints = []
    for waypoint in response.trajectory:
        x = waypoint.x if waypoint.x != 0 else 0.0
        y = waypoint.y if waypoint.y != 0 else 0.0
        z = waypoint.z if waypoint.z != 0 else 0.5
        waypoints.append([x, y, z])
    return np.array(waypoints)

async def simulate_trajectory(channel, state_init, num_waypoints, duration_s):
    # Get the trajectory directly from GetTrajectory
    states = await get_trajectory(channel, state_init[0], state_init[1], 0.5, num_waypoints, duration_s)
    return states

async def simulate_client(num_waypoints, duration_s):
    starting_conditions = [
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.055], dtype=np.float32),
        np.array([0.0, -0.055], dtype=np.float32),
        np.array([0.0, 0.15], dtype=np.float32),
        np.array([0.0, -0.4], dtype=np.float32),
    ]

    async with Channel(HOST, PORT) as channel:
        all_states = []
        for i, state_init in enumerate(starting_conditions):
            states = await simulate_trajectory(channel, state_init, num_waypoints, duration_s)
            all_states.append(states)
            print(f"\nTrajectory {i+1}:")
            for j, waypoint in enumerate(states):
                print(f"Waypoint {j+1}: x={waypoint[0]:.4f}, y={waypoint[1]:.4f}, z={waypoint[2]:.4f}")

        plot_trajectories(all_states)

def plot_trajectories(all_states):
    fig, ax = plt.subplots(figsize=(12, 8))
    obstacle = plt.Circle(
        OBSTACLE_CENTER, OBSTACLE_RADIUS, color="red", alpha=0.5, label="Obstacle"
    )
    ax.add_patch(obstacle)
    goal = plt.scatter([3.0], [0.0], color="black", marker="*", s=100, label="Goal")

    colors = ["blue", "green", "cyan", "orange", "red"]
    for i, states in enumerate(all_states):
        ax.plot(
            states[:, 0],
            states[:, 1],
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"Trajectory {i+1}",
        )
        ax.plot(
            states[0, 0],
            states[0, 1],
            "o",
            color=colors[i % len(colors)],
            markersize=10,
        )
        ax.plot(
            states[-1, 0],
            states[-1, 1],
            "x",
            color=colors[i % len(colors)],
            markersize=12,
        )

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(-Y_MAX, Y_MAX)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Trajectory Simulation")
    ax.grid(True)
    ax.legend(loc="upper right")

    plt.savefig("trajectory_simulation.png")
    plt.show()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test GetTrajectory endpoint with configurable waypoints and duration.")
    parser.add_argument('--num-waypoints', type=int, default=12, help='Number of waypoints in the trajectory')
    parser.add_argument('--duration-s', type=int, default=12, help='Duration of the trajectory in seconds')
    args = parser.parse_args()

    print(f"Testing GetTrajectory with num_waypoints={args.num_waypoints}, duration_s={args.duration_s}...")
    asyncio.run(simulate_client(args.num_waypoints, args.duration_s))