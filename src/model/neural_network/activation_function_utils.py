from collections import Callable
from dataclasses import dataclass
import numpy as np

from model.neural_network.activation_functions import (
    sigmoid_derivative_from_value,
    relu,
    relu_derivative_from_value,
    tanh_derivative_from_value,
    sigmoid,
    tanh,
)


@dataclass
class ActivationFunction:
    function: Callable[[np.ndarray], np.ndarray]
    derivative_from_value: Callable[[np.ndarray], np.ndarray]


activation_functions_utils = {
    "sigmoid": ActivationFunction(function=sigmoid, derivative_from_value=sigmoid_derivative_from_value),
    "relu": ActivationFunction(function=relu, derivative_from_value=relu_derivative_from_value),
    "tanh": ActivationFunction(function=tanh, derivative_from_value=tanh_derivative_from_value),
}
