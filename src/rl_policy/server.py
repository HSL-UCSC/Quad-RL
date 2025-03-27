import asyncio
import signal
import numpy as np
from stable_baselines3 import DQN
from grpclib.server import Server
from . import drone_pb2  # Use generated message types
from . import drone_grpc  # Service base
from HyRL import HyRL_agent, M_ext, find_X_i, M_i, find_critical_points, state_to_observation_OA, get_state_from_env_OA
from training_env import ObstacleAvoidance
from training_tools import find_critical_points, state_to_observation_OA, get_state_from_env_OA, find_X_i, train_hybrid_agent, M_i, M_ext, HyRL_agent, simulate_obstacleavoidance, visualize_M_ext

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
has_obstacle = True             # Flag for obstacle presence

def initialize_models(x_obst=1.5, y_obst=0.0, radius_obst=0.75, x_goal=3.0, y_goal=0.0):
    global model, agent_0, agent_1, M_ext0, M_ext1, hybrid_agent
    
    # Load pre-trained models
    model = DQN.load("dqn_obstacleavoidance")
    agent_0 = DQN.load("dqn_obstacleavoidance_0")
    agent_1 = DQN.load("dqn_obstacleavoidance_1")

    # Define environment with current obstacle/goal to use has_obstacle flag
    env_class = lambda **kwargs: ObstacleAvoidance(x_obst=x_obst, y_obst=y_obst, 
                                                  radius_obst=radius_obst if has_obstacle else 0.0, 
                                                  x_goal=x_goal, y_goal=y_goal, 
                                                  **kwargs)

    # Compute critical points (simplify if no obstacle)
    resolution = 30
    x_ = np.linspace(0, 3, resolution)
    y_ = np.linspace(-1.5, 1.5, resolution)
    state_difference = np.linalg.norm(np.array([x_[1] - x_[0], y_[1] - y_[0]]))
    initial_points = [np.array([x_[idx], y_[idy]], dtype=np.float32) 
                     for idx in range(resolution) for idy in range(resolution)]
    
    if has_obstacle:
        M_star = find_critical_points(initial_points, state_difference, model, env_class,
                                     min_state_difference=1e-2, steps=5, threshold=1e-1, 
                                     n_clusters=8, custom_state_to_observation=state_to_observation_OA,
                                     get_state_from_env=get_state_from_env_OA, verbose=False)
        M_star = M_star[np.argsort(M_star[:, 0])]
    else:
        # No obstacle: minimal or no critical points (e.g., single region)
        M_star = np.array([[x_goal, y_goal]])  # Simplistic, forces straight path

    # Build regions and extension regions
    M_0 = M_i(M_star, index=0)
    M_1 = M_i(M_star, index=1) if len(M_star) > 1 else M_0  # Fallback if only one point
    X_0 = find_X_i(M_0, model) if has_obstacle else [M_star[0]]  # Minimal extension
    X_1 = find_X_i(M_1, model) if has_obstacle else [M_star[0]]
    M_ext0 = M_ext(M_0, X_0)
    M_ext1 = M_ext(M_1, X_1)

    # Initialize hybrid agent
    hybrid_agent = HyRL_agent(agent_0, agent_1, M_ext0, M_ext1, q_init=0)


class DroneService(drone_grpc.DroneServiceBase):
    # Calls training_env -> train_agent -> HyRL -> utils 
    async def SetEnvironment(self, stream):
        global obstacle_centroid, obstacle_radius, goal_position, hybrid_agent, M_ext0, M_ext1

        request: drone_pb2.SetEnvironmentRequest = await stream.recv_message()
        vertices = [(v.x, v.y, v.z) for v in request.vertex]  # List of (x, y, z) tuples
        print(f"Received environment with {len(vertices)} vertices")
        
        # Edge Case: An empty vertex request will initialize an environment with obstacle set at goal with rad=0 
        if not vertices:  
            has_obstacle = False
            obstacle_centroid = goal_position  # No obstacle, align with goal
            obstacle_radius = 0.0
            # Update goal (default if not provided)
            if hasattr(request, 'goal') and request.goal:
                goal_position = [request.goal.x, request.goal.y]
            else:
                goal_position = [3.0, 0.0]
            
            # Reinitialize with no obstacle
            initialize_models(x_obst=obstacle_centroid[0], y_obst=obstacle_centroid[1], 
                             radius_obst=0.0, x_goal=goal_position[0], y_goal=goal_position[1], 
                             has_obstacle=False)
            
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
        global hybrid_agent, obstacle_centroid, obstacle_radius, goal_position, M_ext0, M_ext1
        
        request: drone_pb2.DirectionRequest = await stream.recv_message()
        print(f"Received drone state: {request}")
        
        # Extract 2D position
        state = np.array([request.x, request.y], dtype=np.float32)
        
        # Compute observation
        obs = state_to_observation_OA(state, 
                                    x_obst=obstacle_centroid[0], 
                                    y_obst=obstacle_centroid[1], 
                                    radius_obst=obstacle_radius if has_obstacle else 0.0, 
                                    x_goal=goal_position[0], 
                                    y_goal=goal_position[1])
        
        # Get action from hybrid agent
        action, _ = hybrid_agent.predict(obs, deterministic=True)
        
        # Map action to DiscreteHeading (adjust based on your 5-action space: -1, -0.5, 0, 0.5, 1)
        direction_map = {
            -1: drone_pb2.HeadingDirection.LEFT,    # Full left
            -0.5: drone_pb2.HeadingDirection.LEFT,  # Slight left
            0: drone_pb2.HeadingDirection.UP,       # Straight (assuming UP means no y-change)
            0.5: drone_pb2.HeadingDirection.RIGHT,  # Slight right
            1: drone_pb2.HeadingDirection.RIGHT     # Full right
        }
        direction = direction_map.get(float(action), drone_pb2.HeadingDirection.UP)  # Default to straight
        
        # Send response
        heading = drone_pb2.DiscreteHeading(direction=direction)
        response = drone_pb2.DirectionResponse(discrete_heading=heading)
        await stream.send_message(response)

async def main():
    # Initialize models and hybrid agent at startup
    print("Initializing RL models...")
    # initialize_models()

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