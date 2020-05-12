from typing import Optional

import numpy as np

from model.car import Car


class Sensor:
    def __init__(self) -> None:
        self.__car: Optional[Car] = None

    def bind(self, car: Car) -> None:
        self.__car = car

    def detect(self) -> np.ndarray:
        return np.random.randint(-10, 10 + 1, 8)
