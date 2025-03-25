import asyncio
from grpclib.server import Server
import signal
import numpy as np
from stable_baselines3 import DQN
from . import drone_pb2  # Use generated message types
from . import drone_grpc  # Service base
from HyRL import HyRL_agent, ObstacleAvoidance, M_ext, find_X_i, M_i, find_critical_points, state_to_observation_OA, get_state_from_env_OA

# Global variables for models and environment
model = None
agent_0 = None
agent_1 = None
M_ext0 = None
M_ext1 = None
hybrid_agent = None
obstacle_vertices = None
obstacle_centroid = [1.5, 0.0]  # Default [x_obst, y_obst]
obstacle_radius = 0.75          # Default radius
goal_position = [3.0, 0.0]      # Default [x_goal, y_goal]

def initialize_models(x_obst=1.5, y_obst=0.0, radius_obst=0.75, x_goal=3.0, y_goal=0.0):
    global model, agent_0, agent_1, M_ext0, M_ext1, hybrid_agent
    
    # Load pre-trained models
    model = DQN.load("dqn_obstacleavoidance")
    agent_0 = DQN.load("dqn_obstacleavoidance_0")
    agent_1 = DQN.load("dqn_obstacleavoidance_1")

    # Define environment with current obstacle/goal
    env_class = lambda **kwargs: ObstacleAvoidance(x_obst=x_obst, y_obst=y_obst, 
                                                  radius_obst=radius_obst, 
                                                  x_goal=x_goal, y_goal=y_goal, 
                                                  **kwargs)

    # Compute critical points
    resolution = 30
    x_ = np.linspace(0, 3, resolution)  # Adjust to lab scale if needed (e.g., 0-5)
    y_ = np.linspace(-1.5, 1.5, resolution)
    state_difference = np.linalg.norm(np.array([x_[1] - x_[0], y_[1] - y_[0]]))
    initial_points = [np.array([x_[idx], y_[idy]], dtype=np.float32) 
                     for idx in range(resolution) for idy in range(resolution)]
    
    M_star = find_critical_points(initial_points, state_difference, model, env_class,
                                 min_state_difference=1e-2, steps=5, threshold=1e-1, 
                                 n_clusters=8, custom_state_to_observation=state_to_observation_OA,
                                 get_state_from_env=get_state_from_env_OA, verbose=False)
    M_star = M_star[np.argsort(M_star[:, 0])]

    # Build regions and extension regions
    M_0 = M_i(M_star, index=0)
    M_1 = M_i(M_star, index=1)
    X_0 = find_X_i(M_0, model)
    X_1 = find_X_i(M_1, model)
    M_ext0 = M_ext(M_0, X_0)
    M_ext1 = M_ext(M_1, X_1)

    # Initialize hybrid agent
    hybrid_agent = HyRL_agent(agent_0, agent_1, M_ext0, M_ext1, q_init=0)

class DroneService(drone_grpc.DroneServiceBase):

    # Calls training_env -> train_agent -> HyRL -> utils 
    async def SetEnvironment(self, stream):
        global obstacle_cetroid, obstacle_radius, goal_position, hybrid_agent, M_ext0, M_ext1

        request: drone_pb2.SetEnvironmentRequest = await stream.recv_message()
        vertices = [(v.x, v.y, v.z) for v in request.vertex]  # List of (x, y, z) tuples
        print(f"Received environment with {len(vertices)} vertices")
        
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
            dist = np.sqrt((x - x_obst)**2 + (y - y_obst)**2)
            max_dist = max(max_dist, dist)
        obstacle_radius = max_dist
        
        # Update goal (default to [3, 0] if not provided; adjust proto if goal is included)
        # Assuming request.goal exists with x, y, z fields; modify if different
        if hasattr(request, 'goal') and request.goal:
            goal_position = [request.goal.x, request.goal.y]
        else:
            goal_position = [3.0, 0.0]  # Fallback
        
        # Reinitialize environment with new obstacle/goal
        initialize_models(x_obst=obstacle_centroid[0], y_obst=obstacle_centroid[1], 
                         radius_obst=obstacle_radius, x_goal=goal_position[0], 
                         y_goal=goal_position[1])
        
        response = drone_pb2.SetEnvironmentResponse(
            message=f"Environment set: Obstacle at {obstacle_centroid}, radius {obstacle_radius}, goal at {goal_position}"
        )
        await stream.send_message(response)
            
    async def GetDirection(self, stream):
        global hybrid_agent, obstacle_vertices
        request: drone_pb2.DirectionRequest = await stream.recv_message()
        print(f"Received drone state: {request}")

        # Assume request has x, y fields for drone position (adjust based on actual proto definition)
        state = np.array([request.x, request.y], dtype=np.float32)

        # Configure environment with stored vertices (if any)
        env = ObstacleAvoidance(hybridlearning=True, M_ext=M_ext0 if hybrid_agent.q == 0 else M_ext1)
        if obstacle_vertices:
            env.obstacles = obstacle_vertices  # Assuming ObstacleAvoidance has an obstacles attribute

        # Convert state to observation (as in HyRL.py)
        obs = state_to_observation_OA(state, env)

        # Get action from hybrid agent
        action, _ = hybrid_agent.predict(obs, deterministic=True)

        # Map action to DiscreteHeading (adjust mapping based on your action space)
        direction_map = {
            0: drone_pb2.HeadingDirection.LEFT,
            1: drone_pb2.HeadingDirection.RIGHT,
            2: drone_pb2.HeadingDirection.UP,
            3: drone_pb2.HeadingDirection.DOWN
        }
        direction = direction_map.get(action, drone_pb2.HeadingDirection.LEFT)  # Default to LEFT if unknown

        # Create and send response
        heading = drone_pb2.DiscreteHeading(direction=direction)
        response = drone_pb2.DirectionResponse(discrete_heading=heading)
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

    # Wait for shutdown signal
    await stop.wait()
    server.close()
    await server.wait_closed()
    print("Server Stopped.")

if __name__ == "__main__":
    print("Starting gRPC server...")
    asyncio.run(main())