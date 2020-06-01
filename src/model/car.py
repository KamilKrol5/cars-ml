from typing import Tuple, List, ClassVar

from model.geom.track import Track
from model.neural_network_old import NeuralNetwork
from model.geom.directed_rect import DirectedRectangle
from model.geom.sensor import Sensor

import numpy as np


class Car:
    TRACTION: ClassVar[float] = 1.0
    _ACCELERATION_RATE: float = 10
    _BRAKING_RATE: float = 10  # braking means "accelerate backwards" also
    _MAX_FORWARD_VELOCITY: float = 200
    _MAX_BACKWARD_VELOCITY: float = 50
    _ROTATION_RATE: float = 1
    _SIZE: float = 10

    __slots__ = ("rect", "sensors", "neural_network", "speed")

    def __init__(
        self,
        size: Tuple[float, float],
        sensors: List[Sensor],
        neural_network: NeuralNetwork,
    ) -> None:
        """Creates a new car at point (0,0) headed in the X axis direction."""
        self.rect = DirectedRectangle.new_origin_x(*size)
        self.sensors = sensors
        self.neural_network = neural_network
        self.speed = 0.0

    def sense_surroundings(self, track: Track) -> List[float]:
        # TODO: 0 -> active_sector
        return [track.sense_closest(sensor, 0) for sensor in self.sensors]

    def check_collision(self, track: Track) -> bool:
        # TODO
        return False

    def move(self, turning_rate: float, delta_time: float) -> None:
        transform = self.rect.turn_curve_transform(
            self.speed, Car.TRACTION, turning_rate, delta_time
        )
        for sensor in self.sensors:
            sensor.transform(transform)
        self.rect.transform(transform)

    def update_speed(self, acceleration: float, delta_time: float) -> None:
        self.speed += acceleration * delta_time

    def tick(self, track: Track, delta_time: float) -> None:
        distances = self.sense_surroundings(track)
        turning_rate, acceleration = self.neural_network.compute(np.array(distances))
        self.move(turning_rate, delta_time)
        if self.check_collision(track):
            # TODO: communicate collision outside
            return
        self.update_speed(acceleration, delta_time)

    def go_brrrr(self) -> None:
        print("brrr!")
