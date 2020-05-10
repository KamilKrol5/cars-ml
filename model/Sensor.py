import random

import numpy as np


class Sensor:
    def __init__(self):
        self.__car = None

    def bind(self, car) -> None:
        self.__car = car

    def detect(self) -> np.ndarray:
        return np.array([random.randrange(-10, 11) for i in range(8)])
