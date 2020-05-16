import numpy as np


class Sensor:
    def __init__(self) -> None:
        pass

    def detect(self, position_x: float, position_y: float, turn: float) -> np.ndarray:
        return np.random.randint(-10, 10 + 1, 8)
