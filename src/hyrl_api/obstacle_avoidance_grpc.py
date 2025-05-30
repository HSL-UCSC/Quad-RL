# Generated by the Protocol Buffers compiler. DO NOT EDIT!
# source: hyrl_api/obstacle_avoidance.proto
# plugin: grpclib.plugin.main
import abc
import typing

import grpclib.const
import grpclib.client
if typing.TYPE_CHECKING:
    import grpclib.server

import hyrl_api.obstacle_avoidance_pb2


class ObstacleAvoidanceServiceBase(abc.ABC):

    @abc.abstractmethod
    async def GetDirection(self, stream: 'grpclib.server.Stream[hyrl_api.obstacle_avoidance_pb2.DirectionRequest, hyrl_api.obstacle_avoidance_pb2.DirectionResponse]') -> None:
        pass

    @abc.abstractmethod
    async def GetTrajectory(self, stream: 'grpclib.server.Stream[hyrl_api.obstacle_avoidance_pb2.TrajectoryRequest, hyrl_api.obstacle_avoidance_pb2.TrajectoryResponse]') -> None:
        pass

    def __mapping__(self) -> typing.Dict[str, grpclib.const.Handler]:
        return {
            '/hyrl.ObstacleAvoidanceService/GetDirection': grpclib.const.Handler(
                self.GetDirection,
                grpclib.const.Cardinality.UNARY_UNARY,
                hyrl_api.obstacle_avoidance_pb2.DirectionRequest,
                hyrl_api.obstacle_avoidance_pb2.DirectionResponse,
            ),
            '/hyrl.ObstacleAvoidanceService/GetTrajectory': grpclib.const.Handler(
                self.GetTrajectory,
                grpclib.const.Cardinality.UNARY_UNARY,
                hyrl_api.obstacle_avoidance_pb2.TrajectoryRequest,
                hyrl_api.obstacle_avoidance_pb2.TrajectoryResponse,
            ),
        }


class ObstacleAvoidanceServiceStub:

    def __init__(self, channel: grpclib.client.Channel) -> None:
        self.GetDirection = grpclib.client.UnaryUnaryMethod(
            channel,
            '/hyrl.ObstacleAvoidanceService/GetDirection',
            hyrl_api.obstacle_avoidance_pb2.DirectionRequest,
            hyrl_api.obstacle_avoidance_pb2.DirectionResponse,
        )
        self.GetTrajectory = grpclib.client.UnaryUnaryMethod(
            channel,
            '/hyrl.ObstacleAvoidanceService/GetTrajectory',
            hyrl_api.obstacle_avoidance_pb2.TrajectoryRequest,
            hyrl_api.obstacle_avoidance_pb2.TrajectoryResponse,
        )
