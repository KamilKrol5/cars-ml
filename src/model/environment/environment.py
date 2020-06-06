from typing import Iterable, Mapping, Generator

import pygame
from pygame.surface import Surface

import utils
from model.Simulation import Simulation
from model.neural_network.neural_network import NeuralNetwork


class Environment:
    simulation: Simulation

    def generate_adaptations(
        self, networks_groups: Mapping[str, Iterable[NeuralNetwork]]
    ) -> Generator[None, Surface, Mapping[str, Iterable[float]]]:
        while cars := self.simulation.tick(0.1):
            surface = yield

            for car in cars:
                pygame.draw.circle(
                    surface, self.car_color, (car.position_x, car.position_y), 10
                )
        return cars

    def generate_adaptations2(
        self, networks_groups: Mapping[str, Iterable[NeuralNetwork]]
    ) -> Generator[None, Surface, None]:
        yield from self.generate_adaptations(networks_groups)

    def compute_adaptations(
        self, networks_groups: Mapping[str, Iterable[NeuralNetwork]],
    ) -> Mapping[str, Iterable[float]]:
        return utils.generator_value(self.generate_adaptations(networks_groups))
