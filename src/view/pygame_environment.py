from dataclasses import dataclass

import pygame
from planar import Point, Vec2
from pygame.surface import Surface

from model.environment.environment import Environment
from model.simulation import CarState
from model.track.track import SegmentId, Track
from view import colors


@dataclass
class EnvironmentContext:
    surface: Surface
    frame_time: float
    offset: Vec2
    scale: float
    point_of_interest: Point = Point(0, 0)


@dataclass
class EnvironmentState:
    best_car_segment: SegmentId
    cars_count: int = 0


@dataclass
class PyGameEnvironment(Environment[EnvironmentContext, EnvironmentState]):
    def __init__(self, track: Track):
        super().__init__(track)
        pygame.font.init()
        self.font = pygame.font.SysFont("Verdana", 24)

    def _initialize(self) -> EnvironmentState:
        return EnvironmentState(0)

    def _finalize_iteration(
        self, state: EnvironmentState, context: EnvironmentContext
    ) -> EnvironmentState:
        txt_surface = self.font.render(f"Cars: {state.cars_count}", False, colors.WHITE)
        context.surface.blit(txt_surface, dest=(0, 0))
        return EnvironmentState(0)

    def _process_car_step(
        self,
        state: EnvironmentState,
        context: EnvironmentContext,
        group_id: str,
        car_state: CarState,
    ) -> None:
        if car_state.active:
            color = colors.LIME
            if car_state.car.active_segment > state.best_car_segment:
                context.point_of_interest = car_state.car.rect.shape.centroid
                state.best_car_segment = car_state.car.active_segment
        else:
            color = colors.RED

        state.cars_count += 1
        pygame.draw.polygon(
            context.surface,
            color,
            [(a - context.offset) * context.scale for a in car_state.car.rect.shape],
        )
