import grpc
import drone_service_pb2
import drone_service_pb2_grpc

def run():
    channel = grpc.insecure_channel("localhost:50051")
    stub = drone_service_pb2_grpc.DroneServiceStub(channel)
    
    position = drone_service_pb2.DronePosition(
        x=10.0, y=5.0, z=3.0, velocity=20.0, heading=90.0
    )
    
    response = stub.GetDirection(position)
    print(f"Recommended Direction: {response.direction}")

if __name__ == "__main__":
    run()
