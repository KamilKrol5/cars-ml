from model.neural_network.neural_network import NeuralNetwork
from typing import Optional


class Individual:
    def __init__(self, neural_network: NeuralNetwork):
        self._neural_network: NeuralNetwork = neural_network
        self.adaptation: Optional[int] = None

    @property
    def neural_network(self) -> NeuralNetwork:
        return self._neural_network
