from dataclasses import dataclass
from typing import Mapping, Iterable, Generator, List

import pygame
from pygame.surface import Surface

from model.environment.environment import Environment
from model.geom.track import Track
from model.neural_network.neural_network import NeuralNetwork
from model.simulation import Simulation, SimState, CarState
from view import colors


@dataclass
class EnvironmentContext:
    surface: Surface
    frame_time: float


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
            context: EnvironmentContext = (yield)
            frame_time += context.frame_time
            frame_time, cars = simulation.update(frame_time)

            any_active = False
            # TODO: differentiate cars on group_id
            for _group_id, car_group in cars.items():
                for car_state in car_group:
                    if car_state.active:
                        color = colors.LIGHTBLUE
                        any_active = True
                    else:
                        color = colors.RED

                    # TODO: drawing in the right place
                    pygame.draw.polygon(
                        context.surface, color, car_state.car.rect.shape, 10
                    )

        return self._compute(cars)

    def _compute(self, cars: SimState) -> Mapping[str, Iterable[float]]:
        return {name: self._group_adaptation(group) for name, group in cars.items()}

    def _group_adaptation(self, car_group: Iterable[CarState]) -> Iterable[float]:
        for car_state in car_group:
            yield self._car_adaptation(car_state)

    def _car_adaptation(self, car_state: CarState) -> float:
        # TODO: something more sophisticated
        return car_state.car.active_segment
