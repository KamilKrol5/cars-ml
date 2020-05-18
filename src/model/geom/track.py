from dataclasses import dataclass
from typing import List, Tuple, Iterable

from planar import Vec2
from planar.line import LineSegment

import utils
from model.geom.segment import TrackSegment
from model.geom.sensor import Sensor
from model.geom.wall import Wall
from utils import pairwise


SegmentId = int


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

    def segment_walls(self, segment_id: SegmentId) -> List[Wall]:
        current_segment = self.segments[segment_id]
        walls = [current_segment.left_wall, current_segment.right_wall]
        if segment_id == 0:
            walls.append(current_segment.back_wall)
        if segment_id == -1 or segment_id == len(self.segments) - 1:
            walls.append(current_segment.front_wall)

        return walls

    def sense_closest(self, sensor: Sensor, active_segment: SegmentId) -> float:
        """Return the distance to the closest wall of the track sensed with the given sensor."""
        # TODO: edge cases when a later segment has the closest wall
        # TODO: optimize for multiple sensors
        for segment_id in utils.spread_int(active_segment, 0, len(self.segments)):
            for wall in self.segment_walls(segment_id):
                if (d := sensor.check_distance(wall)) is not None:
                    return d
        raise RuntimeError("couldn't find any walls in any of the segments")
