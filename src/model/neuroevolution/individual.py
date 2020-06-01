from typing import Optional, Tuple

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

    def get_random_layer_index(self) -> int:
        return np.random.randint(low=0, high=len(self.neural_network.hidden_layers))

    @staticmethod
    def get_random_weight_index(neural_network_layer: Layer) -> Tuple:
        return tuple(np.random.randint(neural_network_layer.weights.shape))

    @staticmethod
    def get_random_bias_index(neural_network_layer: Layer) -> int:
        return np.random.randint(neural_network_layer.biases.shape[0])
