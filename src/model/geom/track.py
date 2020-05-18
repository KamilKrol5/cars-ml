from dataclasses import dataclass
from typing import List, Tuple, Iterable

from planar import Vec2
from planar.line import LineSegment

from model.geom.segment import TrackSegment
from model.geom.wall import Wall
from utils import pairwise


@dataclass
class Track:
    segments: List[TrackSegment]

    @classmethod
    def from_points(cls, points: Iterable[Tuple[Vec2, Vec2]]) -> "Track":
        segments: List[TrackSegment] = []
        for (s1, s2), (e1, e2) in pairwise(points):
            segments.append(
                TrackSegment(
                    Wall(LineSegment.from_points((s1, e1))),
                    Wall(LineSegment.from_points((s2, e2))),
                )
            )

        return cls(segments)
