from typing import Tuple, List, ClassVar

import numpy as np

from model.geom.directed_rect import DirectedRectangle
from model.geom.sensor import Sensor
from model.geom.track import Track, SegmentId
from model.neural_network_old import NeuralNetwork


class Collision(Exception):
    pass


class Car:
    TRACTION: ClassVar[float] = 1.0
    _ACCELERATION_RATE: float = 10
    _BRAKING_RATE: float = 10  # braking means "accelerate backwards" also
    _MAX_FORWARD_VELOCITY: float = 200
    _MAX_BACKWARD_VELOCITY: float = 50
    _ROTATION_RATE: float = 1
    _SIZE: float = 10

    __slots__ = ("rect", "sensors", "neural_network", "speed", "active_segment")

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
        self.active_segment: SegmentId = 0

    def _sense_surroundings(self, track: Track) -> List[float]:
        return [
            track.sense_closest(sensor, self.active_segment) for sensor in self.sensors
        ]

    def _check_collision(self, track: Track) -> bool:
        return track.intersects(self.rect.shape, self.active_segment)

    def _move(self, turning_rate: float, delta_time: float, track: Track) -> None:
        transform = self.rect.turn_curve_transform(
            self.speed, Car.TRACTION, turning_rate, delta_time
        )
        for sensor in self.sensors:
            sensor.transform(transform)
        self.rect.transform(transform)

        self.active_segment = track.update_active(self.active_segment, self.rect.center)

    def _update_speed(self, acceleration: float, delta_time: float) -> None:
        # TODO: limit speed
        self.speed += acceleration * delta_time

    def tick(self, track: Track, delta_time: float) -> None:
        distances = self._sense_surroundings(track)
        turning_rate, acceleration = self.neural_network.compute(np.array(distances))
        self._move(turning_rate, delta_time, track)
        if self._check_collision(track):
            raise Collision
        self._update_speed(acceleration, delta_time)

    def go_brrrr(self) -> None:
        print("brrr!")
