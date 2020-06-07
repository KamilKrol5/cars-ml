from dataclasses import dataclass
from typing import List

import numpy as np

from model.neural_network.activation_functions import (
    activation_functions_utils,
    Activation,
)

np.set_printoptions(suppress=True)


@dataclass
class LayerInfo:
    neurons_count: int
    activation_function_name: str


class Layer:
    def __init__(
        self, basic_info: LayerInfo, weights: np.ndarray, biases: np.ndarray,
    ):
        self.info = basic_info
        self.activation: Activation = activation_functions_utils[
            basic_info.activation_function_name
        ]
        self.weights: np.ndarray = weights
        self.biases = biases


class NeuralNetwork:
    """
    Class representing a neural network (multilayer perceptron).
    """

    def __init__(
        self, hidden_layers_info: List[LayerInfo], output_neurons_count: int,
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
        self.hidden_layers = self._create_hidden_layers(
            hidden_layers_info, output_neurons_count
        )

    def _feed_forward(self, training_data_set: np.ndarray,) -> np.ndarray:
        """
        Feeds the neural network with the data provided.

        Args:
            training_data_set (numpy.ndarray): Data which will be used for feeding the network.
                The dimension of this array should be (number of data samples, single data sample length).
                Single data sample length must be equal to input_layer_neuron_count of the neural network.
                Example: Let's assume that [[a], [b], [c]] is training_data_set given as argument.
                Its shape is (3, 1), so there are 3 data samples, and each of the samples has length 1.
        """
        if training_data_set.shape[1] != self.input_layer_neuron_count:
            raise ValueError("Dimension mismatch")

        values = training_data_set
        for layer in self.hidden_layers:
            values = layer.activation.value(
                np.dot(values, layer.weights.T) + layer.biases.reshape(1, -1)
            )

        return values

    def predict(self, input_data_set: np.ndarray) -> np.ndarray:
        output = self._feed_forward(input_data_set)
        return output

    @staticmethod
    def _create_hidden_layers(
        hidden_layers_info: List[LayerInfo], output_neurons_count: int,
    ) -> List[Layer]:
        hidden_layers = []
        for layer_info, next_layer_info in zip(
            hidden_layers_info[:-1], hidden_layers_info[1:]
        ):
            weights = np.random.rand(
                next_layer_info.neurons_count, layer_info.neurons_count,
            )
            biases = np.random.rand(next_layer_info.neurons_count)
            hidden_layers.append(Layer(layer_info, weights, biases))

        last_hidden_layer_info = hidden_layers_info[-1]
        last_hidden_layer_info_weights = np.random.rand(
            output_neurons_count, last_hidden_layer_info.neurons_count
        )
        last_hidden_layer_biases = np.random.rand(output_neurons_count)
        hidden_layers.append(
            Layer(
                last_hidden_layer_info,
                last_hidden_layer_info_weights,
                last_hidden_layer_biases,
            )
        )
        return hidden_layers
