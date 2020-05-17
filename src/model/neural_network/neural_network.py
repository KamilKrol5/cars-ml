from typing import List, Optional

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

    Args:
        hidden_layers_info (List[NeuralNetworkHiddenLayerInfo): A list of objects
            with info needed for creation of hidden layers. There must be at least two layers,
            since single layer network is not supported. The first layer's neurons count
            must be consistent with training data if provided.
        training_data_sets (numpy.ndarray): Data which will be used for training. If not provided,
            there is up to user to set up weights and biases to make any predictions 'sensible'
        labels (numpy.ndarray): same as for training_data_sets. It's dimensions must be consistent with
            training data sets and first layer's neuron count.
    """

    def __init__(
            self,
            hidden_layers_info: List[NeuralNetworkHiddenLayerInfo],
            training_data_sets: Optional[np.ndarray],
            labels: Optional[np.ndarray],
    ):
        if len(hidden_layers_info) < 2:
            raise NotImplementedError(
                "This implementation of neural network does not support single layer networks"
            )
        if (labels is None) != (training_data_sets is None):
            raise ValueError("'training_data_sets' and 'labels' must be both None or provided as an argument")

        # if training data is provided, learn method must be invoked before the network can do predictions
        self.ready_for_prediction = True
        if labels is not None:
            self.ready_for_prediction = False

        self.hidden_layers_info = hidden_layers_info
        self.input_layer_neuron_count = self.hidden_layers_info[0].neurons_count
        self.training_data_sets = training_data_sets
        self.labels = labels
        self.hidden_layers = self._create_hidden_layers()

        self.output = np.zeros(self.labels.shape)
        self.eta = 0.5

    def _feed_forward(self, with_training_data: bool = True) -> None:
        input_layer = self.hidden_layers[0]
        if with_training_data:
            input_layer.values = self.training_data_sets
        for layer, next_layer in zip(self.hidden_layers[0:-1], self.hidden_layers[1:]):
            next_layer.values = layer.utils.function(
                np.dot(layer.values, layer.weights.T)
            )
        last_hidden_layer = self.hidden_layers[-1]
        self.output = last_hidden_layer.utils.function(
            np.dot(last_hidden_layer.values, last_hidden_layer.weights.T)
        )

    def _back_propagation(self) -> None:
        last_hidden_layer = self.hidden_layers[-1]
        error = (self.labels - self.output) * last_hidden_layer.utils.derivative(
            self.output
        )
        last_hidden_layer.weights_change = self.eta * np.dot(
            error.T, last_hidden_layer.values
        )

        for layer, next_layer in zip(
                reversed(self.hidden_layers[0:-1]), reversed(self.hidden_layers[1:])
        ):
            error = layer.utils.derivative(next_layer.values) * np.dot(
                error, next_layer.weights
            )
            layer.weights_change = self.eta * np.dot(error.T, layer.values)

        for layer in self.hidden_layers:
            layer.update_weights()

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
            self.labels.shape[1], last_hidden_layer_info.neurons_count
        )
        hidden_layers.append(
            NeuralNetworkHiddenLayer(
                last_hidden_layer_info, last_hidden_layer_info_weights
            )
        )
        return hidden_layers

    def learn(self, iterations: int) -> None:
        """
        Method which trains the network using gradient descent and backward propagation methods.
        If training data is not provided, each call of this method will result in an error.
        """

        if self.training_data_sets is None:
            raise RuntimeError("The network cannot learn when learning data is not provided.")

        self.hidden_layers[0].values = self.training_data_sets
        for _ in range(iterations):
            self._feed_forward()
            self._back_propagation()
        self.ready_for_prediction = True

    def predict(self, input_data_set: np.ndarray) -> np.ndarray:
        if not self.ready_for_prediction:
            raise RuntimeError(
                "Neural network must learn before it can make any predictions"
            )
        input_layer = self.hidden_layers[0]
        input_layer.values = input_data_set
        self._feed_forward(with_training_data=False)
        return self.output
