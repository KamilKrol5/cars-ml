from collections import namedtuple


from model.neural_network.activation_functions import (
    sigmoid_star,
    relu,
    relu_derivative,
    tanh_derivative,
    sigmoid,
    tanh,
)

ActivationFunctionUtils = namedtuple(
    "_ActivationFunctionUtils", ["function", "derivative"]
)

activation_functions_utils = {
    "sigmoid": ActivationFunctionUtils(function=sigmoid, derivative=sigmoid_star),
    "relu": ActivationFunctionUtils(function=relu, derivative=relu_derivative),
    "tanh": ActivationFunctionUtils(function=tanh, derivative=tanh_derivative),
}
