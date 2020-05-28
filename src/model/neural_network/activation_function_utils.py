from typing import Dict

from model.neural_network.activation_functions import Activation, Sigmoid, Relu, Tanh

activation_functions_utils: Dict[str, Activation] = {
    "sigmoid": Sigmoid(),
    "relu": Relu(),
    "tanh": Tanh(),
}
