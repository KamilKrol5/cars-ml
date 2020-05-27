from dataclasses import dataclass
from typing import List

import numpy as np

from model.neural_network.activation_function_utils import (
    activation_functions_utils,
    ActivationFunctionUtils,
)

np.set_printoptions(suppress=True)


@dataclass
class NeuralNetworkHiddenLayerInfo:
    activation_function_name: str
    neurons_count: int


class NeuralNetworkHiddenLayer:
    def __init__(self, basic_info: NeuralNetworkHiddenLayerInfo, weights: np.ndarray, biases: np.ndarray):
        self.info = basic_info
        self.utils: ActivationFunctionUtils = activation_functions_utils[
            basic_info.activation_function_name
        ]
        self.weights: np.ndarray = weights
        self.biases = biases
        self.values: np.ndarray = np.zeros(shape=(self.info.neurons_count,))


class NeuralNetwork:
    """
    Class representing a neural network (multilayer perceptron).
    """

    def __init__(
            self,
            hidden_layers_info: List[NeuralNetworkHiddenLayerInfo],
            output_neurons_count
    ):
        """
        Args:
            hidden_layers_info (List[NeuralNetworkHiddenLayerInfo): A list of objects
                with info needed for creation of hidden layers. There must be at least two layers,
                since single layer network is not supported.
        """
        if len(hidden_layers_info) < 2:
            raise NotImplementedError(
                "This implementation of neural network does not support single layer networks"
            )
        self.output_neurons_count = output_neurons_count

        self.hidden_layers_info = hidden_layers_info
        self.input_layer_neuron_count = self.hidden_layers_info[0].neurons_count
        self.hidden_layers = self._create_hidden_layers(hidden_layers_info)

        self.eta = 0.5

    def _feed_forward(
            self,
            training_data_sets: np.ndarray,
    ) -> np.ndarray:
        """
        Feeds the neural network with the data provided.

        Args:
            training_data_sets (numpy.ndarray): Data which will be used for feeding the network.
        """
        input_layer = self.hidden_layers[0]
        input_layer.values = training_data_sets
        for layer, next_layer in zip(self.hidden_layers[0:-1], self.hidden_layers[1:]):
            next_layer.values = layer.utils.function(
                np.dot(layer.values, layer.weights.T) + layer.biases
            )
        last_hidden_layer = self.hidden_layers[-1]
        return last_hidden_layer.utils.function(
            np.dot(last_hidden_layer.values, last_hidden_layer.weights.T) + last_hidden_layer.biases
        )

    def predict(self, input_data_set: np.ndarray) -> np.ndarray:
        output = self._feed_forward(input_data_set)
        return output

    @staticmethod
    def _create_hidden_layers(hidden_layers_info: List[NeuralNetworkHiddenLayerInfo]) -> List[NeuralNetworkHiddenLayer]:
        hidden_layers = []
        for layer_info, next_layer_info in zip(
                hidden_layers_info[0:-1], hidden_layers_info[1:]
        ):
            weights = np.random.rand(
                next_layer_info.neurons_count, layer_info.neurons_count,
            )
            biases = np.random.rand(layer_info.neurons_count)
            hidden_layers.append(NeuralNetworkHiddenLayer(layer_info, weights, biases))

        last_hidden_layer_info = hidden_layers_info[-1]
        last_hidden_layer_info_weights = np.random.rand(
            1, last_hidden_layer_info.neurons_count
        )
        last_hidden_layer_biases = np.random.rand(last_hidden_layer_info.neurons_count)
        hidden_layers.append(
            NeuralNetworkHiddenLayer(
                last_hidden_layer_info, last_hidden_layer_info_weights, last_hidden_layer_biases
            )
        )
        return hidden_layers
