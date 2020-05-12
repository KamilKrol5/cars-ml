import random
from typing import Tuple

import numpy as np


class NeuralNetwork:
    def __init__(self) -> None:
        pass

    def compute(self, data: np.ndarray) -> Tuple[int, int]:
        return random.choice([-1, 0, 1]), random.choice([-1, 0, 1])
