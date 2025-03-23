import grpc
from concurrent import futures
import drone_pb2
import drone_pb2_grpc


class DroneServiceServicer(drone_pb2_grpc.DroneServiceServicer):
    def SetEnvironment(self, request, context):
        # Placeholder: Echo a success message
        response = drone_pb2.SetEnvironmentResponse(
            message="Environment set successfully"
        )
        return response

    def GetDirection(self, request, context):
        # Placeholder: Decide direction based on drone state
        drone_state = request.drone_state
        if drone_state.x > 0:
            direction = drone_pb2.HeadingDirection.RIGHT
        else:
            direction = drone_pb2.HeadingDirection.STRAIGHT
        response = drone_pb2.DirectionResponse(
            discrete_heading=drone_pb2.DiscreteHeading(direction=direction)
        )
        return response


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    drone_pb2_grpc.add_DroneServiceServicer_to_server(
        DroneServiceServicer(), server
    )
    server.add_insecure_port("[::]:50051")  # Listen on port 50051
    print("Server starting on port 50051...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
