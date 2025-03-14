import grpc
import drone_pb2
import drone_pb2_grpc


def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = drone_pb2_grpc.DroneServiceStub(channel)

        # Test SetEnvironment
        vertex = drone_pb2.Vertex(
            vertices=[
                drone_pb2.Point(x=1.0, y=2.0, z=3.0),
                drone_pb2.Point(x=4.0, y=5.0, z=6.0),
            ]
        )
        request = drone_pb2.SetEnvironmentRequest(vertex=[vertex])
        response = stub.SetEnvironment(request)
        print(f"SetEnvironment Response: {response.message}")

        # Test GetDirection
        drone_state = drone_pb2.DroneState(
            x=1.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, roll=0.0, pitch=0.0, yaw=0.0
        )
        direction_request = drone_pb2.DirectionRequest(drone_state=drone_state)
        direction_response = stub.GetDirection(direction_request)
        if direction_response.HasField("discrete_heading"):
            print(
                f"Direction Response: Discrete Heading = {direction_response.discrete_heading.direction}"
            )
        elif direction_response.HasField("continuous_heading"):
            print(
                f"Direction Response: Continuous Heading = {direction_response.continuous_heading.rad}"
            )


if __name__ == "__main__":
    run()
