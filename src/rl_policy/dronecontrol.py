# Generated by the protocol buffer compiler.  DO NOT EDIT!
# sources: drone.proto
# plugin: python-betterproto
from dataclasses import dataclass
from typing import List, Optional

import betterproto
import grpclib


class HeadingDirection(betterproto.Enum):
    RESERVED = 0
    STRAIGHT = 1
    LEFT = 2
    HARD_LEFT = 3
    RIGHT = 4
    HARD_RIGHT = 5


@dataclass
class DirectionRequest(betterproto.Message):
    """DEFINE DIRECTION REQUEST"""

    drone_state: "DroneState" = betterproto.message_field(1)


@dataclass
class DroneState(betterproto.Message):
    """define drone state"""

    x: float = betterproto.float_field(1)
    y: float = betterproto.float_field(2)
    z: float = betterproto.float_field(3)


@dataclass
class DirectionResponse(betterproto.Message):
    """DEFINE DIRECTION RESPONSE"""

    discrete_heading: "DiscreteHeading" = betterproto.message_field(
        2, group="control_type"
    )
    continuous_heading: "ContinuousHeading" = betterproto.message_field(
        3, group="control_type"
    )


@dataclass
class DiscreteHeading(betterproto.Message):
    direction: "HeadingDirection" = betterproto.enum_field(1)


@dataclass
class ContinuousHeading(betterproto.Message):
    rad: float = betterproto.float_field(1)


@dataclass
class SetEnvironmentRequest(betterproto.Message):
    """DEFINE ENVIRONMENT REQUEST"""

    vertex: List["Vertex"] = betterproto.message_field(1)
    goal: "Point" = betterproto.message_field(2)


@dataclass
class Vertex(betterproto.Message):
    vertices: List["Point"] = betterproto.message_field(1)


@dataclass
class Point(betterproto.Message):
    x: float = betterproto.float_field(1)
    y: float = betterproto.float_field(2)
    z: float = betterproto.float_field(3)


@dataclass
class SetEnvironmentResponse(betterproto.Message):
    """DEFINE ENVIRONMENT RESPONSE"""

    message: str = betterproto.string_field(1)


class DroneServiceStub(betterproto.ServiceStub):
    async def set_environment(
        self, *, vertex: List["Vertex"] = [], goal: Optional["Point"] = None
    ) -> SetEnvironmentResponse:
        request = SetEnvironmentRequest()
        if vertex is not None:
            request.vertex = vertex
        if goal is not None:
            request.goal = goal

        return await self._unary_unary(
            "/dronecontrol.DroneService/SetEnvironment",
            request,
            SetEnvironmentResponse,
        )

    async def get_direction(
        self, *, drone_state: Optional["DroneState"] = None
    ) -> DirectionResponse:
        request = DirectionRequest()
        if drone_state is not None:
            request.drone_state = drone_state

        return await self._unary_unary(
            "/dronecontrol.DroneService/GetDirection",
            request,
            DirectionResponse,
        )
