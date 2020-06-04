from typing import List, Iterable, Dict

from model.neural_network.neural_network import NeuralNetwork


class Environment:
    def compute_adaptations(
        self, networks_groups: Dict[str, Iterable[NeuralNetwork]],
    ) -> Dict[str, List[float]]:
        # we might want to mark children and parents in various ways
        # remember to return only children adaptations
        raise NotImplementedError
