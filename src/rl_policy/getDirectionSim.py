import asyncio
import numpy as np
import matplotlib.pyplot as plt
from grpclib.client import Channel
from . import drone_pb2
from . import drone_grpc
import time

HOST = "127.0.0.1"
PORT = 50051
X_MAX = 3.0
Y_MAX = 1.5
OBSTACLE_CENTER = (1.5, 0.0)
OBSTACLE_RADIUS = 0.75
STEP_SIZE = 0.05
NOISE_MAG = 0

DIRECTION_CONFIG = {
    drone_pb2.STRAIGHT: {"angle": 0, "color": "blue", "label": "Straight"},
    drone_pb2.LEFT: {"angle": 15, "color": "green", "label": "Left"},
    drone_pb2.HARD_LEFT: {"angle": 30, "color": "cyan", "label": "Hard Left"},
    drone_pb2.RIGHT: {"angle": -15, "color": "orange", "label": "Right"},
    drone_pb2.HARD_RIGHT: {"angle": -30, "color": "red", "label": "Hard Right"},
}


async def get_direction(channel, x, y, z=0.0):
    stub = drone_grpc.DroneServiceStub(channel)
    request = drone_pb2.DirectionRequest(
        drone_state=drone_pb2.DroneState(x=x, y=y, z=z)
    )
    start_time = time.time()
    response = await stub.GetDirection(request)
    return response.discrete_heading.direction, time.time() - start_time


def update_state(state, direction, noise, prev_states, window=5):
    config = DIRECTION_CONFIG[direction]
    angle_rad = np.deg2rad(config["angle"])
    dx = np.cos(angle_rad) * STEP_SIZE
    dy = np.sin(angle_rad) * STEP_SIZE
    proposed_state = state + np.array([dx, dy], dtype=np.float32)
    proposed_state += np.array([0, noise], dtype=np.float32)

    # Stronger smoothing
    if len(prev_states) >= window:
        avg_state = np.mean(prev_states[-window:], axis=0)
        proposed_state = 0.4 * proposed_state + 0.6 * avg_state  # Favor average more

    # Stronger obstacle avoidance
    dist_obst = np.sqrt(
        (proposed_state[0] - OBSTACLE_CENTER[0]) ** 2
        + (proposed_state[1] - OBSTACLE_CENTER[1]) ** 2
    )
    if dist_obst < OBSTACLE_RADIUS + 0.15:  # Larger buffer
        away_vector = proposed_state - np.array(OBSTACLE_CENTER, dtype=np.float32)
        away_vector /= np.linalg.norm(away_vector)
        proposed_state += away_vector * STEP_SIZE * 1.5  # Stronger repulsion

    # Enhanced goal bias after obstacle
    if proposed_state[0] > 2.25:
        goal_vector = np.array([3.0, 0.0], dtype=np.float32) - proposed_state
        goal_vector /= max(np.linalg.norm(goal_vector), 0.01)
        proposed_state += goal_vector * STEP_SIZE * 0.8  # Stronger pull to goal

    proposed_state[0] = np.clip(proposed_state[0], 0, X_MAX)
    proposed_state[1] = np.clip(proposed_state[1], -Y_MAX, Y_MAX)
    return proposed_state


def is_done(state):
    x, y = state
    dist_goal = np.sqrt((x - 3.0) ** 2 + (y - 0.0) ** 2)
    dist_obst = np.sqrt((x - 1.5) ** 2 + (y - 0.0) ** 2) - OBSTACLE_RADIUS
    return dist_goal < 0.1 or x >= X_MAX or dist_obst <= 0


async def simulate_trajectory(channel, state_init):
    state = np.array(state_init, dtype=np.float32)
    states = [state.copy()]
    sign = 1
    response_times = []

    while not is_done(state):
        noise = NOISE_MAG * sign
        direction, response_time = await get_direction(channel, state[0], state[1])
        state = update_state(state, direction, noise, states)
        states.append(state.copy())
        response_times.append(response_time)
        # Calculate distances for logging
        dist_obst = (
            np.sqrt((state[0] - 1.5) ** 2 + (state[1] - 0) ** 2) - OBSTACLE_RADIUS
        )
        dist_goal = np.sqrt((state[0] - 3.0) ** 2 + (state[1] - 0.0) ** 2)
        print(
            f"State: {state}, Direction: {drone_pb2.HeadingDirection.Name(direction)}, "
            f"Dist to Obst: {dist_obst:.3f}, Dist to Goal: {dist_goal:.3f}, Time: {response_time:.4f}s"
        )
        sign *= -1

    return np.array(states), response_times


async def simulate_client():
    starting_conditions = [
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.055], dtype=np.float32),
        np.array([0.0, -0.055], dtype=np.float32),
        np.array([0.0, 0.15], dtype=np.float32),
        np.array([0.0, -0.15], dtype=np.float32),
    ]

    async with Channel(HOST, PORT) as channel:
        all_states = []
        all_response_times = []
        for i, state_init in enumerate(starting_conditions):
            states, response_times = await simulate_trajectory(channel, state_init)
            all_states.append(states)
            all_response_times.extend(response_times)

        avg_response_time = np.mean(all_response_times)
        print(f"\nAverage Response Time: {avg_response_time:.4f} seconds")
        print(f"Max Sampling Rate: {1/avg_response_time:.2f} Hz")

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
    ax.set_title(
        "Trajectory Simulation"
    )
    ax.grid(True)
    ax.legend(loc="upper right")

    plt.savefig("trajectory_simulation.png")
    plt.show()


if __name__ == "__main__":
    asyncio.run(simulate_client())
