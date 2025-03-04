import grpc
import time
from concurrent import futures

import drone_pb2
import drone_pb2_grpc

class DroneServiceHandler(drone_pb2_grpc.DroneServiceHandler):
    def SetEnvironment(self, request, context):
        print("Received SetEnvironment request")
        return drone_pb2.SetEnvironmentResponse(message="Ping! Environment set.")

    def GetDirection(self, request, context):
        print("Received GetDirection request")
        return drone_pb2.DirectionResponse()  # Empty response just to confirm ping

    def ResetEnvironment(self, request, context):
        print("Received ResetEnvironment request")
        return drone_pb2.ResetEnvironmentResponse(message="Ping! Environment reset.")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    drone_pb2_grpc.add_DroneServiceHandler_to_server(DroneServiceHandler(), server)
    server.add_insecure_port("[::]:50051")
    print("gRPC server is running on port 50051...")
    server.start()

    try:
        while True:
            time.sleep(86400)  # Keeps server running
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve()
