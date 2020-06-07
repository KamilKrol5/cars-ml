from typing import Tuple, List

import numpy as np
from planar.transform import Affine

from model.geom.directed_rect import DirectedRectangle
from model.geom.sensor import Sensor
from model.geom.track import Track, SegmentId
from model.neural_network.neural_network import NeuralNetwork


class Collision(Exception):
    pass


class Car:
    _TRACTION: float = 1.0
    _ACCELERATION_RATE: float = 10.0
    _BRAKING_RATE: float = 5.0
    _MAX_FORWARD_SPEED: float = 200.0
    _MAX_BACKWARD_SPEED: float = 50.0

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

    @classmethod
    def with_standard_sensors(
        cls, size: Tuple[float, float], neural_network: NeuralNetwork
    ) -> "Car":
        self = cls(size, [], neural_network)

        rays = self.rect.surrounding_rays()
        self.sensors = [Sensor(ray) for ray in rays]

        return self

    def _sense_surroundings(self, track: Track) -> List[float]:
        return track.sense_closest(self.sensors, self.active_segment)

    def _check_collision(self, track: Track) -> bool:
        return track.intersects(self.rect.shape, self.active_segment)

    def transform(self, trans: Affine) -> None:
        for sensor in self.sensors:
            sensor.transform(trans)
        self.rect.transform(trans)

    def _move(self, turning_rate: float, delta_time: float, track: Track) -> None:
        transform = self.rect.turn_curve_transform(
            self.speed, Car._TRACTION, turning_rate, delta_time
        )

        self.transform(transform)

        self.active_segment = track.update_active(self.active_segment, self.rect.center)

    def _update_speed(self, acceleration: float, delta_time: float) -> None:
        speed_dir: float = np.sign(self.speed)
        acceleration_dir: float = np.sign(acceleration)
        speed_limits = {
            1.0: (0.0, Car._MAX_FORWARD_SPEED),
            -1.0: (-Car._MAX_BACKWARD_SPEED, 0.0),
        }
        speed_limit = speed_limits.get(
            speed_dir, speed_limits.get(acceleration_dir, (0.0, 0.0))
        )

        if acceleration_dir == speed_dir or speed_dir == 0.0:
            # car accelerates with the same rate forward and backward
            rate = Car._ACCELERATION_RATE
        else:
            rate = Car._BRAKING_RATE

        scaled_acceleration = rate * acceleration
        new_speed = self.speed + scaled_acceleration * delta_time
        new_speed_clipped = np.clip(new_speed, *speed_limit)
        self.speed = new_speed_clipped
        if new_speed_clipped != new_speed and new_speed_clipped == 0.0:
            remaining_speed = new_speed - new_speed_clipped
            remaining_time = remaining_speed / scaled_acceleration
            self._update_speed(acceleration, remaining_time)

    def tick(self, track: Track, delta_time: float) -> None:
        distances = self._sense_surroundings(track)
        turning_rate, acceleration = self.neural_network.predict(
            np.expand_dims(np.array(distances), 0)
        )[0]
        self._move(turning_rate, delta_time, track)
        if self._check_collision(track):
            raise Collision
        self._update_speed(acceleration, delta_time)
