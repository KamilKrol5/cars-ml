from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Generator, Generic, TypeVar

import pygame
from pygame.surface import Surface

import utils
from model.simulation import Simulation
from model.neural_network.neural_network import NeuralNetwork


T_contra = TypeVar("T_contra", contravariant=True)


class Environment(ABC, Generic[T_contra]):
    @abstractmethod
    def generate_adaptations(
        self, networks_groups: Mapping[str, Iterable[NeuralNetwork]]
    ) -> Generator[None, T_contra, Mapping[str, Iterable[float]]]:
        raise NotImplementedError

    def run(
        self, networks_groups: Mapping[str, Iterable[NeuralNetwork]]
    ) -> Generator[None, T_contra, None]:
        yield from self.generate_adaptations(networks_groups)

    @staticmethod
    def compute_adaptations(
        env: "Environment[None]",
        networks_groups: Mapping[str, Iterable[NeuralNetwork]],
    ) -> Mapping[str, Iterable[float]]:
        return utils.generator_value(env.generate_adaptations(networks_groups))


class PyGameEnvironment(Environment[Surface]):
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
