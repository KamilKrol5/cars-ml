import numpy as np

tanh = np.tanh


def tanh_derivative_from_value(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative_from_value(x: np.ndarray) -> np.ndarray:
    return x * (1.0 - x)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_derivative_from_value(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)
