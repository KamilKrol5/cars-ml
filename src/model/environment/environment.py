from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Generator, Generic, TypeVar, Dict, List

import utils
from model.neural_network.neural_network import NeuralNetwork

T_contra = TypeVar("T_contra", contravariant=True)


class Environment(ABC, Generic[T_contra]):
    @abstractmethod
    def generate_adaptations(
        self, networks_groups: Mapping[str, List[NeuralNetwork]]
    ) -> Generator[None, T_contra, Mapping[str, Iterable[float]]]:
        raise NotImplementedError

    def run(
        self, networks_groups: Mapping[str, List[NeuralNetwork]]
    ) -> Generator[None, T_contra, None]:
        yield from self.generate_adaptations(networks_groups)

    @staticmethod
    def compute_adaptations(
        env: "Environment[None]", networks_groups: Dict[str, List[NeuralNetwork]],
    ) -> Mapping[str, Iterable[float]]:
        return utils.generator_value(env.generate_adaptations(networks_groups))
