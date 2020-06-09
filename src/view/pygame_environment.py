from dataclasses import dataclass
from typing import Mapping, Iterable, Generator, List

import pygame
from planar import Point, Vec2
from pygame.surface import Surface

from model.environment.environment import Environment
from model.neural_network.neural_network import NeuralNetwork
from model.simulation import Simulation, SimState, CarState, FIXED_DELTA_TIME
from model.track.track import Track
from view import colors


@dataclass
class EnvironmentContext:
    surface: Surface
    frame_time: float
    offset: Vec2
    scale: float
    point_of_interest: Point = Point(0, 0)


@dataclass
class PyGameEnvironment(Environment[EnvironmentContext]):
    track: Track

    def generate_adaptations(
        self, networks_groups: Mapping[str, List[NeuralNetwork]]
    ) -> Generator[None, EnvironmentContext, Mapping[str, Iterable[float]]]:
        simulation = Simulation(self.track, networks_groups)
        frame_time = 0.0
        any_active = True

        while any_active:
            best_car_segment = 0
            context: EnvironmentContext = (yield)
            frame_time += FIXED_DELTA_TIME
            frame_time, cars = simulation.update(frame_time)

            any_active = False
            # TODO: differentiate cars on group_id
            for _group_id, car_group in cars.items():
                for car_state in car_group:
                    if car_state.active:
                        color = colors.LIME
                        any_active = True
                        if car_state.car.active_segment > best_car_segment:
                            context.point_of_interest = (
                                car_state.car.rect.shape.centroid
                            )
                            best_car_segment = car_state.car.active_segment
                    else:
                        color = colors.RED

                    pygame.draw.polygon(
                        context.surface,
                        color,
                        [
                            (a - context.offset) * context.scale
                            for a in car_state.car.rect.shape
                        ],
                    )

        return self._compute(cars)

    def _compute(self, cars: SimState) -> Mapping[str, Iterable[float]]:
        return {name: self._group_adaptation(group) for name, group in cars.items()}

    def _group_adaptation(self, car_group: Iterable[CarState]) -> Iterable[float]:
        for car_state in car_group:
            yield self._car_adaptation(car_state)

    @staticmethod
    def _car_adaptation(car_state: CarState) -> float:
        return car_state.car.active_segment ** 2 / car_state.car.ticks
