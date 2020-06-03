from typing import List, Iterable

from model.neural_network.neural_network import NeuralNetwork


class Environment:
    def compute_adaptations(
        self,
        children_networks: Iterable[NeuralNetwork],
        parents_networks: Iterable[NeuralNetwork],
    ) -> List[float]:
        # we might want to mark children and parents in various ways
        # remember to return only children adaptations
        raise NotImplementedError
