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

def initialize_models():
    global model, agent_0, agent_1, M_ext0, M_ext1, hybrid_agent
    # Load pre-trained models
    model = DQN.load("dqn_obstacleavoidance")
    agent_0 = DQN.load("dqn_obstacleavoidance_0")
    agent_1 = DQN.load("dqn_obstacleavoidance_1")

    # Compute critical points and extended sets (simplified from HyRL.py)
    resolution = 30
    x_ = np.linspace(0, 3, resolution)
    y_ = np.linspace(-1.5, 1.5, resolution)
    state_difference = np.linalg.norm(np.array([x_[1] - x_[0], y_[1] - y_[0]]))
    initial_points = [np.array([x_[idx], y_[idy]], dtype=np.float32) 
                     for idx in range(resolution) for idy in range(resolution)]
    
    M_star = find_critical_points(initial_points, state_difference, model, 
                                  ObstacleAvoidance, min_state_difference=1e-2, 
                                  steps=5, threshold=1e-1, n_clusters=8, 
                                  custom_state_to_observation=state_to_observation_OA,
                                  get_state_from_env=get_state_from_env_OA,
                                  verbose=False)
    M_star = M_star[np.argsort(M_star[:, 0])]

    # Build M_0, M_1 and their extensions
    M_0 = M_i(M_star, index=0)
    M_1 = M_i(M_star, index=1)
    X_0 = find_X_i(M_0, model)
    X_1 = find_X_i(M_1, model)
    M_ext0 = M_ext(M_0, X_0)
    M_ext1 = M_ext(M_1, X_1)

    # Initialize hybrid agent (q_init=0 as default, can be adjusted)
    hybrid_agent = HyRL_agent(agent_0, agent_1, M_ext0, M_ext1, q_init=0)

class DroneService(drone_grpc.DroneServiceBase):
    async def SetEnvironment(self, stream):
        global obstacle_vertices
        request: drone_pb2.SetEnvironmentRequest = await stream.recv_message()
        obstacle_vertices = [(v.x, v.y) for v in request.vertex]  # Store vertices as list of tuples
        print(f"Received environment with {len(request.vertex)} vertices")
        response = drone_pb2.SetEnvironmentResponse(message="Environment set")
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