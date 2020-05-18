from dataclasses import dataclass
from typing import Optional

import planar
from planar.line import Ray, LineSegment
from planar.transform import Affine

import numpy as np

from model.geom.wall import Wall


@dataclass
class Sensor:
    ray: Ray

    def check_distance(self, wall: Wall) -> Optional[float]:
        """Return distance to a given wall or None if the ray doesn't intersect it."""
        ls: LineSegment = wall.line_segment
        p, r = self.ray.anchor, self.ray.direction
        q, s = ls.anchor, ls.direction

        q_p = q - p
        rxs = r.cross(s)

        if abs(rxs) < planar.EPSILON:
            # parallel
            return None

        wall_along = q_p.cross(r) / rxs
        if wall_along < 0 or wall_along > ls.length:
            # wall doesn't intersect the line
            return None

        along: float = q_p.cross(s) / rxs
        if along >= 0:
            return along
        else:
            # behind the ray
            return None

    def transform(self, trans: Affine) -> None:
        trans.itransform(self.ray)

    # TODO: use check_distance
    def detect(self, position_x: float, position_y: float, turn: float) -> np.ndarray:
        return np.random.randint(-10, 10 + 1, 8)
