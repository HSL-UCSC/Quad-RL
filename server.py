import asyncio
from grpclib.server import Server
from dronecontrol import (
    SetEnvironmentRequest,
    SetEnvironmentResponse,
    DirectionRequest,
    DirectionResponse,
    DroneState,
    DiscreteHeading,
    ContinuousHeading,
    HeadingDirection,
)


class DroneService:
    async def set_environment(self, stream):
        request = await stream.recv_message()
        print(f"Received SetEnvironmentRequest with vertices: {request.vertex}")
        response = SetEnvironmentResponse(message="Environment Set Successfully!")
        await stream.send_message(response)

    async def get_direction(self, stream):
        request = await stream.recv_message()
        print(f"Received DirectionRequest for drone state: {request.drone_state}")
        # Example response
        response = DirectionResponse(
            discrete_heading=DiscreteHeading(direction=HeadingDirection.STRAIGHT)
        )
        await stream.send_message(response)


async def main():
    # Create the server
    server = Server()

    # Add handlers for the DroneService
    server.add_generic_rpc_handlers(
        [
            # Set environment
            (
                "/dronecontrol.DroneService/SetEnvironment",
                DroneService().set_environment,
            ),
            # Get direction
            ("/dronecontrol.DroneService/GetDirection", DroneService().get_direction),
        ]
    )

    # Start the server at 127.0.0.1:50051
    await server.start("127.0.0.1", 50051)
    print("gRPC server running on 127.0.0.1:50051")

    # Keep the server running
    await server.wait_closed()


if __name__ == "__main__":
    print("Starting gRPC server...")
    asyncio.run(main())
