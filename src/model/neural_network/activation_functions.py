from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class Activation(ABC):
    @staticmethod
    @abstractmethod
    def value(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def derivative_from_value(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Sigmoid(Activation):
    @staticmethod
    def value(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(np.negative(x)))

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        fx = Sigmoid.value(x)
        return Sigmoid.derivative_from_value(fx)

    @staticmethod
    def derivative_from_value(x: np.ndarray) -> np.ndarray:
        return x * (1.0 - x)

    def __repr__(self) -> str:
        return "sigmoid"


class Relu(Activation):
    @staticmethod
    def value(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)

    @staticmethod
    def derivative_from_value(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)

    def __repr__(self) -> str:
        return "ReLU"


class Tanh(Activation):
    @staticmethod
    def value(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        fx = Tanh.value(x)
        return Tanh.derivative_from_value(fx)

    @staticmethod
    def derivative_from_value(x: np.ndarray) -> np.ndarray:
        return 1.0 - np.square(x)

    def __repr__(self) -> str:
        return "tanh"


activation_functions_utils: Dict[str, Activation] = {
    "sigmoid": Sigmoid(),
    "relu": Relu(),
    "tanh": Tanh(),
}
