from dataclasses import dataclass

from planar import Vec2, Point, EPSILON
from planar.line import Ray
from planar.polygon import Polygon
from planar.transform import Affine

import numpy as np


_ANGLE_CALCULATION_HELPER: float = 90 / np.square(np.pi)


@dataclass
class DirectedRectangle:
    __slots__ = ("heading", "shape")
    heading: Ray
    shape: Polygon

    @classmethod
    def new_origin_x(cls, width: float, length: float) -> "DirectedRectangle":
        """Creates a new rect, centered at origin, headed in direction of the X axis."""
        center = Point(0, 0)
        top_right = center + Vec2(length / 2, width / 2)
        shape = Polygon.from_points(
            [
                top_right,
                top_right - Vec2(length, 0),
                top_right - Vec2(length, width),
                top_right - Vec2(0, width),
            ]
        )

        return cls(Ray(center, Vec2(1, 0)), shape)

    @property
    def center(self) -> Point:
        return self.heading.anchor

    @property
    def direction(self) -> Vec2:
        return self.heading.direction

    @property
    def right_direction(self) -> Vec2:
        return self.heading.normal

    @property
    def left_direction(self) -> Vec2:
        return -self.heading.normal

    def transform(self, trans: Affine) -> None:
        self.shape *= trans
        self.heading *= trans

    def turn_curve_transform(
        self, speed: float, traction: float, turning_rate: float, delta_time: float
    ) -> Affine:
        """Produces a transform representing circular motion.

        Args:
            speed (float): current speed, positive if forward, negative if backward
            traction (float): the factor of how well can this object grip the surface it is travelling on
            turning_rate (float): rate at which this object should turn;
                positive to go right, negative to go left
            delta_time (float): time that passed in this "instant"
        """
        if np.abs(speed) < EPSILON:
            return Affine.identity()

        if np.abs(turning_rate) < EPSILON:
            return Affine.translation(self.direction * speed * delta_time)

        # increased traction means smaller radius
        # so does increased turning rate, but only to a threshold of 1
        # radius will be positive while turning right and negative while turning left
        turning_rate = np.clip(turning_rate, -1.0, 1.0)
        radius = speed * speed / (traction * turning_rate)

        # sign depends on which way we're turning and whether forward or backward
        # if pivot is on the right, positive angle (clockwise) represents
        # moving backward and negative angle moving forward
        # if pivot is on the left, angle's sign is reversed
        angle = -1.0 * _ANGLE_CALCULATION_HELPER * speed * delta_time / radius

        transform = Affine.rotation(angle, self.right_direction * radius)

        return transform
