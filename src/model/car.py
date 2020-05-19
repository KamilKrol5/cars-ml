from math import sin, cos, pi
from typing import Tuple

import numpy as np

from model.neural_network_old import NeuralNetwork
from model.geom.sensor import Sensor


class Car:
    _ACCELERATION_RATE: float = 10
    _BRAKING_RATE: float = 10  # braking means "accelerate backwards" also
    _MAX_FORWARD_VELOCITY: float = 200
    _MAX_BACKWARD_VELOCITY: float = 50
    _ROTATION_RATE: float = 1
    _SIZE: float = 10

    def __init__(
        self,
        neural_network: NeuralNetwork,
        sensor: Sensor,
        position: Tuple[float, float, float],
    ):
        self._position_x: float = position[0]
        self._position_y: float = position[1]
        self._velocity: float = 0
        self._turn: float = position[2]

        self._neural_network: NeuralNetwork = neural_network
        self._sensor: Sensor = sensor
        self._crashed: bool = False

    @property
    def position_x(self) -> float:
        return self._position_x

    @property
    def position_y(self) -> float:
        return self._position_y

    @property
    def velocity(self) -> float:
        return self._velocity

    @property
    def turn(self) -> float:
        return self._turn

    @property
    def crashed(self) -> bool:
        return self._crashed

    def _accelerate(self, acceleration: float) -> None:
        if acceleration > 0:
            self._velocity += Car._ACCELERATION_RATE
            if self._velocity > Car._MAX_FORWARD_VELOCITY:
                self._velocity = Car._MAX_FORWARD_VELOCITY
        elif acceleration < 0:
            self._velocity -= Car._BRAKING_RATE
            if self._velocity < -Car._MAX_BACKWARD_VELOCITY:
                self._velocity = -Car._MAX_BACKWARD_VELOCITY

    def _rotate(self, rotation: float) -> None:
        if (
            0.05 * -Car._MAX_BACKWARD_VELOCITY
            < self._velocity
            < 0.05 * Car._MAX_FORWARD_VELOCITY
        ):
            return  # vehicle is not able to rotate when goes very slow to avoid rotation "like tank"
        if rotation > 0:
            self._turn += Car._ROTATION_RATE
        elif rotation < 0:
            self._turn -= Car._ROTATION_RATE

    def _go(self) -> None:
        self._position_x += sin(self._turn / 180 * pi) * self._velocity
        self._position_y += cos(self._turn / 180 * pi) * self._velocity

    def _check_collision_and_set_crashed(self) -> None:
        #  set self._crashed using self._sensor or basing on last detection
        self._crashed = False

    def tick(self) -> None:
        if self._crashed:
            return
        detection: np.ndarray = self._sensor.detect(
            self._position_x, self._position_y, self._turn
        )
        acceleration, rotation = self._neural_network.compute(detection)
        self._accelerate(acceleration)
        self._rotate(rotation)
        self._go()
        self._check_collision_and_set_crashed()
        if self._crashed:
            self._velocity = 0

    def go_brrrr(self) -> None:
        print("brrr!")
