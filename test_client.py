import grpc
import drone_pb2
import drone_pb2_grpc

def run():
    channel = grpc.insecure_channel("localhost:50051")
    stub = drone_pb2_grpc.DroneServiceStub(channel)

    # Test SetEnvironment
    response = stub.SetEnvironment(drone_pb2.SetEnvironmentRequest())
    print(f"SetEnvironment Response: {response.message}")

    # Test GetDirection
    drone_state = drone_pb2.DroneState(x=1.0, y=2.0, z=3.0, yaw=0.5)
    response = stub.GetDirection(drone_pb2.DirectionRequest(drone_state=drone_state))
    print(f"GetDirection Response: {response}")

if __name__ == "__main__":
    run()
