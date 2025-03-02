import grpc
from concurrent import futures
import drone_service_pb2
import drone_service_pb2_grpc
from my_hybrid_rl_policy import HyRLPolicy  # Import your existing HyRL policy

class DroneServiceServicer(drone_service_pb2_grpc.DroneServiceServicer):
    def __init__(self):
        self.hyrl_policy = HyRLPolicy()  # Load the trained policy

    def GetDirection(self, request, context):
        # Convert incoming request to model input
        drone_state = [request.x, request.y, request.z, 
                       request.velocity, request.heading]

        # Use the existing HyRL policy to get direction
        direction = self.hyrl_policy.predict(drone_state)  # Expects "left" or "right"

        return drone_service_pb2.DirectionResponse(direction=direction)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    drone_service_pb2_grpc.add_DroneServiceServicer_to_server(DroneServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Server running on port 50051...")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
