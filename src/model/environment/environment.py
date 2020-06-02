from typing import List, Iterable

from model.neural_network.neural_network import NeuralNetwork


class Environment:
    def compute_adaptations(self, networks: Iterable[NeuralNetwork]) -> List[float]:
        raise NotImplementedError
