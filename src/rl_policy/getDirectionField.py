import asyncio
import numpy as np
import matplotlib.pyplot as plt
from grpclib.client import Channel
from hyrl_api import obstacle_avoidance_pb2 as oa_proto
from hyrl_api import obstacle_avoidance_grpc as oa_grpc
import time

# Server connection details
HOST = "127.0.0.1"
PORT = 50051

# Environment and obstacle parameters
X_MAX = 3.0
Y_MAX = 1.5
OBSTACLE_CENTER = (1.5, 0.0)  # Matches server default
OBSTACLE_RADIUS = 0.75

# Direction angles (degrees) and colors
DIRECTION_CONFIG = {
    oa_proto.HeadingDirection.STRAIGHT: {"angle": 0, "color": "blue", "label": "Straight"},
    oa_proto.HeadingDirection.LEFT: {"angle": 15, "color": "green", "label": "Left"},
    oa_proto.HeadingDirection.HARD_LEFT: {"angle": 30, "color": "cyan", "label": "Hard Left"},
    oa_proto.HeadingDirection.RIGHT: {"angle": -15, "color": "orange", "label": "Right"},
    oa_proto.HeadingDirection.HARD_RIGHT: {"angle": -30, "color": "red", "label": "Hard Right"},
}

async def get_direction(channel, x, y, z=0.0):
    """Query the GetDirection endpoint and return direction with response time."""
    stub = oa_grpc.ObstacleAvoidanceServiceStub(channel)
    request = oa_proto.DirectionRequest(
        drone_state=oa_proto.DroneState(x=x, y=y, z=z),
        model_type=oa_proto.DirectionRequest.ModelType.HYBRID  # Use HYBRID model
    )
    start_time = time.time()
    response = await stub.GetDirection(request)
    end_time = time.time()
    return response.discrete_heading.direction, end_time - start_time

async def simulate_client():
    """Test GetDirection across a grid, log results, and measure response time."""
    resolution = 50
    x_coords = np.linspace(0, X_MAX, resolution)
    y_coords = np.linspace(-Y_MAX, Y_MAX, resolution)
    points = [(x, y) for x in x_coords for y in y_coords]

    async with Channel(HOST, PORT) as channel:
        directions = []
        response_times = []
        for x, y in points:
            direction, response_time = await get_direction(channel, x, y)
            directions.append(direction)
            response_times.append(response_time)
            print(
                f"Point ({x:.2f}, {y:.2f}) -> Direction: {oa_proto.HeadingDirection.Name(direction)}, Time: {response_time:.4f}s"
            )

        # Calculate and print average response time
        avg_response_time = np.mean(response_times)
        print(f"\nAverage Response Time: {avg_response_time:.4f} seconds")
        print(f"Max Sampling Rate: {1/avg_response_time:.2f} Hz")

        # Visualize results
        plot_directions(points, directions)

def plot_directions(points, directions):
    """Plot points with colored direction arrows, obstacle, and legend."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot obstacle
    obstacle = plt.Circle(
        OBSTACLE_CENTER, OBSTACLE_RADIUS, color="red", alpha=0.5, label="Obstacle"
    )
    ax.add_patch(obstacle)

    # Plot points and arrows with colors
    handles = {}
    for (x, y), direction in zip(points, directions):
        config = DIRECTION_CONFIG[direction]
        angle_rad = np.deg2rad(config["angle"])
        dx = np.cos(angle_rad) * 0.05
        dy = np.sin(angle_rad) * 0.05
        arrow = ax.arrow(
            x,
            y,
            dx,
            dy,
            head_width=0.02,
            head_length=0.03,
            fc=config["color"],
            ec=config["color"],
            alpha=0.5,
        )
        if direction not in handles:
            handles[direction] = arrow

    # Set grid and bounds
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(-Y_MAX, Y_MAX)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(
        "Simulation of DQN Hybrid Agent Responses Using RPC Server Based Responses"
    )
    ax.grid(True)

    # Add legend
    legend_handles = [
        plt.Line2D([0], [0], color=config["color"], lw=2, label=config["label"])
        for config in DIRECTION_CONFIG.values()
    ]
    ax.legend(handles=legend_handles + [obstacle], loc="upper right")

    # Save and show plot
    plt.savefig("direction_simulation_colored.png")
    plt.show()

if __name__ == "__main__":
    asyncio.run(simulate_client())