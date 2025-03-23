import asyncio
from grpclib.server import Server
import signal
from . import drone_pb2  # Use generated message types
from . import drone_grpc  # Service base


class DroneService(drone_grpc.DroneServiceBase):
    async def SetEnvironment(self, stream):
        request: drone_pb2.SetEnvironmentRequest = await stream.recv_message()
        print(f"Received environment with {len(request.vertex)} vertices")
        response = drone_pb2.SetEnvironmentResponse(message="Environment set")
        await stream.send_message(response)

    async def GetDirection(self, stream):
        request: drone_pb2.DirectionRequest = await stream.recv_message()
        print(f"Received drone state: {request}")
        heading = drone_pb2.DiscreteHeading(direction=drone_pb2.HeadingDirection.LEFT)
        response = drone_pb2.DirectionResponse(discrete_heading=heading)
        await stream.send_message(response)


async def main():
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
    server.close()  # Stop accepting new connections
    await server.wait_closed()  # Wait for existing connections to finish
    print("Server Stopped.")


if __name__ == "__main__":
    print("Starting gRPC server...")
    asyncio.run(main())
