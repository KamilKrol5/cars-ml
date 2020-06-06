from dataclasses import dataclass
from typing import Mapping, Iterable, Generator, List, Dict, Tuple

import pygame
from pygame.surface import Surface

from model.car import Car
from model.environment.environment import Environment
from model.geom.track import Track
from model.neural_network.neural_network import NeuralNetwork
from model.simulation import Simulation, SimState
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
            for group_id, car_group in cars.items():
                for car, active in car_group:
                    if active:
                        color = colors.LIGHTBLUE
                        any_active = True
                    else:
                        color = colors.RED
                    pygame.draw.polygon(context.surface, color, car.rect.shape, 10)

        return self._compute(
            {name: (car for car, _ in group) for name, group in cars.items()}
        )

    def _compute(
        self, cars: Mapping[str, Iterable[Car]]
    ) -> Mapping[str, Iterable[float]]:
        return {name: self._group_adaptation(group) for name, group in cars.items()}

    def _group_adaptation(self, car_group: Iterable[Car]) -> Iterable[float]:
        for car in car_group:
            yield self._car_adaptation(car)

    def _car_adaptation(self, car: Car) -> float:
        # TODO: something more sophisticated
        return car.active_segment
