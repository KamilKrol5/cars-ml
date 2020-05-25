from collections import namedtuple

from model.neural_network.activation_functions import (
    sigmoid_derivative_from_value,
    relu,
    relu_derivative_from_value,
    tanh_derivative_from_value,
    sigmoid,
    tanh,
)

ActivationFunctionUtils = namedtuple(
    "ActivationFunctionUtils", ["function", "derivative"]
)

activation_functions_utils = {
    "sigmoid": ActivationFunctionUtils(function=sigmoid, derivative=sigmoid_derivative_from_value()),
    "relu": ActivationFunctionUtils(function=relu, derivative=relu_derivative_from_value),
    "tanh": ActivationFunctionUtils(function=tanh, derivative=tanh_derivative_from_value),
}
