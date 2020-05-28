from math import sin, cos, pi
from typing import Tuple

import numpy as np

from model.neural_network_old import NeuralNetwork
from model.sensor import Sensor


class Car:
    __ACCELERATION_RATE: float = 10
    __BRAKING_RATE: float = 10  # braking means "accelerate backwards" also
    __MAX_FORWARD_VELOCITY: float = 200
    __MAX_BACKWARD_VELOCITY: float = 50
    __ROTATION_RATE: float = 1
    __SIZE: float = 10

    def __init__(
        self,
        neural_network: NeuralNetwork,
        sensor: Sensor,
        position: Tuple[float, float, float],
    ):
        self.__position_x: float = position[0]
        self.__position_y: float = position[1]
        self.__velocity: float = 0
        self.__turn: float = position[2]

        self.__neural_network: NeuralNetwork = neural_network
        self.__sensor: Sensor = sensor
        self.__crashed: bool = False

    @property
    def position_x(self) -> float:
        return self.__position_x

    @property
    def position_y(self) -> float:
        return self.__position_y

    @property
    def velocity(self) -> float:
        return self.__velocity

    @property
    def turn(self) -> float:
        return self.__turn

    @property
    def crashed(self) -> bool:
        return self.__crashed

    def __accelerate(self, acceleration: float) -> None:
        if acceleration > 0:
            self.__velocity += Car.__ACCELERATION_RATE
            if self.__velocity > Car.__MAX_FORWARD_VELOCITY:
                self.__velocity = Car.__MAX_FORWARD_VELOCITY
        elif acceleration < 0:
            self.__velocity -= Car.__BRAKING_RATE
            if self.__velocity < -Car.__MAX_BACKWARD_VELOCITY:
                self.__velocity = -Car.__MAX_BACKWARD_VELOCITY

    def __rotate(self, rotation: float) -> None:
        if (
            0.05 * -Car.__MAX_BACKWARD_VELOCITY
            < self.__velocity
            < 0.05 * Car.__MAX_FORWARD_VELOCITY
        ):
            return  # vehicle is not able to rotate when goes very slow to avoid rotation "like tank"
        if rotation > 0:
            self.__turn += Car.__ROTATION_RATE
        elif rotation < 0:
            self.__turn -= Car.__ROTATION_RATE

    def __go(self) -> None:
        self.__position_x += sin(self.__turn / 180 * pi) * self.__velocity
        self.__position_y += cos(self.__turn / 180 * pi) * self.__velocity

    def __check_collision_and_set_crashed(self) -> None:
        #  set self.__crashed using self.__sensor or basing on last detection
        self.__crashed = False

    def tick(self) -> None:
        if self.__crashed:
            return
        detection: np.ndarray = self.__sensor.detect(
            self.__position_x, self.__position_y, self.__turn
        )
        acceleration, rotation = self.__neural_network.compute(detection)
        self.__accelerate(acceleration)
        self.__rotate(rotation)
        self.__go()
        self.__check_collision_and_set_crashed()
        if self.__crashed:
            self.__velocity = 0

    def go_brrrr(self) -> None:
        print("brrr!")
