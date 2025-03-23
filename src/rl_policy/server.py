import asyncio
from grpclib.server import Server
import signal
from typing import List
from dronecontrol import (
    SetEnvironmentRequest,
    SetEnvironmentResponse,
    DirectionRequest,
    DirectionResponse,
    DroneState,
    DiscreteHeading,
    ContinuousHeading,
    HeadingDirection,
    DroneServiceStub,
    Vertex,
)
import drone_grpc


class DroneService(drone_grpc.DroneServiceBase):
    async def SetEnvironment(self, stream):
        request: SetEnvironmentRequest = await stream.recv_message()
        print(f"Received environment with {len(request.vertex)} vertices")

        response = SetEnvironmentResponse(message="Environment set")
        await stream.send_message(response)

    async def GetDirection(self, stream):
        request: DirectionRequest = await stream.recv_message()
        print(f"Received drone state: {request}")

        heading = DiscreteHeading(direction=HeadingDirection.LEFT)  # Example
        response = DirectionResponse(discrete_heading=heading)
        await stream.send_message(response)


async def main():
    server = Server([DroneService()])
    await server.start("127.0.0.1", 50051)
    print("gRPC server running on 127.0.0.1:50051")

    # Set up signal handling
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    #  Handle Server Stop 
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
