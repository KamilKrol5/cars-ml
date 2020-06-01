from typing import Optional

import numpy as np

from model.neural_network.neural_network import NeuralNetwork, Layer


class Individual:
    def __init__(self, neural_network: NeuralNetwork):
        self._neural_network: NeuralNetwork = neural_network
        self.adaptation: Optional[int] = None

    @property
    def neural_network(self) -> NeuralNetwork:
        return self._neural_network

    def get_random_layer(self) -> Layer:
        random_layer: Layer = np.random.choice(self.neural_network.hidden_layers)
        return random_layer
