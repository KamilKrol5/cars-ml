from typing import Iterable, Mapping

from model.neural_network.neural_network import NeuralNetwork


class Environment:
    def compute_adaptations(
        self, networks_groups: Mapping[str, Iterable[NeuralNetwork]],
    ) -> Mapping[str, Iterable[float]]:
        raise NotImplementedError
