import asyncio
from grpclib.server import Server
import sys
sys.path.append('lib')
from dronecontrol.dronecontrol import (
    DroneServiceStub,  # Use the correct base class from the auto-generated code
    SetEnvironmentRequest,
    SetEnvironmentResponse,
    DirectionRequest,
    DirectionResponse,
    DroneState,
    DiscreteHeading,
    ContinuousHeading,
    HeadingDirection,
)


class DroneServiceStub:
    async def set_environment(self, request: SetEnvironmentRequest) -> SetEnvironmentResponse:
        raise NotImplementedError()

    async def get_direction(self, request: DirectionRequest) -> DirectionResponse:
        raise NotImplementedError()


class DroneService(DroneServiceStub):
    async def set_environment(self, request: SetEnvironmentRequest) -> SetEnvironmentResponse:
        print(f"Received SetEnvironmentRequest with vertices: {request.vertex}")
        response = SetEnvironmentResponse(message="Environment Set Successfully!")
        return response

    async def get_direction(self, request: DirectionRequest) -> DirectionResponse:
        print(f"Received DirectionRequest for drone state: {request.drone_state}")
        # Example response
        response = DirectionResponse(
            discrete_heading=DiscreteHeading(direction=HeadingDirection.STRAIGHT)
        )
        return response


async def main():
    # Create the server and add the service to it
    server = Server([DroneService()])

    # Start the server at 127.0.0.1:50051
    await server.start("127.0.0.1", 50051)
    print("gRPC server running on 127.0.0.1:50051")

    # Keep the server running
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
