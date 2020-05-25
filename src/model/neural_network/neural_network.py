from typing import List

import numpy as np

from model.neural_network.activation_function_utils import (
    activation_functions_utils,
    ActivationFunctionUtils,
)

np.set_printoptions(suppress=True)


class NeuralNetworkHiddenLayerInfo:
    def __init__(self, activation_function_name: str, neurons_count: int):
        self.activation_function_name: str = activation_function_name
        self.neurons_count: int = neurons_count


class NeuralNetworkHiddenLayer:
    def __init__(self, basic_info: NeuralNetworkHiddenLayerInfo, weights: np.ndarray):
        self.info = basic_info
        self.utils: ActivationFunctionUtils = activation_functions_utils[
            basic_info.activation_function_name
        ]
        self.weights: np.ndarray = weights
        self.values: np.ndarray = np.zeros(shape=(self.info.neurons_count,))
        self.weights_change: np.ndarray = np.zeros(shape=(self.info.neurons_count,))

    def update_weights(self) -> None:
        self.weights += self.weights_change
        self.weights_change.fill(0)


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
        self.output = np.zeros((output_neurons_count, 1))

        self.hidden_layers_info = hidden_layers_info
        self.input_layer_neuron_count = self.hidden_layers_info[0].neurons_count
        self.hidden_layers = self._create_hidden_layers()

        self.eta = 0.5

    def _feed_forward(
            self,
            training_data_sets: np.ndarray,
    ) -> None:
        """
        Feeds the neural network with the data provided.

        Args:
            training_data_sets (numpy.ndarray): Data which will be used for feeding the network.
        """
        input_layer = self.hidden_layers[0]
        input_layer.values = training_data_sets
        for layer, next_layer in zip(self.hidden_layers[0:-1], self.hidden_layers[1:]):
            next_layer.values = layer.utils.function(
                np.dot(layer.values, layer.weights.T)
            )
        last_hidden_layer = self.hidden_layers[-1]
        self.output = last_hidden_layer.utils.function(
            np.dot(last_hidden_layer.values, last_hidden_layer.weights.T)
        )

    def _create_hidden_layers(self) -> List[NeuralNetworkHiddenLayer]:
        hidden_layers = []
        for layer_info, next_layer_info in zip(
                self.hidden_layers_info[0:-1], self.hidden_layers_info[1:]
        ):
            weights = np.random.rand(
                next_layer_info.neurons_count, layer_info.neurons_count,
            )
            hidden_layers.append(NeuralNetworkHiddenLayer(layer_info, weights))

        last_hidden_layer_info = self.hidden_layers_info[-1]
        last_hidden_layer_info_weights = np.random.rand(
            self.output.shape[1], last_hidden_layer_info.neurons_count
        )
        hidden_layers.append(
            NeuralNetworkHiddenLayer(
                last_hidden_layer_info, last_hidden_layer_info_weights
            )
        )
        return hidden_layers

    def predict(self, input_data_set: np.ndarray) -> np.ndarray:
        self._feed_forward(input_data_set)
        return self.output
