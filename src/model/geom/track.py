from dataclasses import dataclass
from functools import cached_property
from typing import List, Tuple, Iterable, Optional, cast

from planar import Point, Vec2
from planar.line import LineSegment
from planar.polygon import Polygon

import utils
from model.geom.segment import TrackSegment
from model.geom.sensor import Sensor
from model.geom.wall import Wall
from utils import pairwise


SegmentId = int


@dataclass
class Track:
    __slots__ = ("segments", "__dict__")
    segments: List[TrackSegment]

    @classmethod
    def from_points(cls, points: Iterable[Tuple[Point, Point]]) -> "Track":
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

    def _spread_segment_ids(self, segment_id: SegmentId) -> Iterable[SegmentId]:
        return utils.spread_int(segment_id, 0, len(self.segments))

    def sense_closest(
        self, sensors: List[Sensor], active_segment: SegmentId
    ) -> List[float]:
        """Returns the distances to the closest walls of the track sensed with the given sensors."""
        # TODO?: edge cases when a later segment has
        #  the closest wall (not sure if that ever happens)

        sensors_to_check: List[Tuple[int, Sensor]] = list(enumerate(sensors))
        distances: List[Optional[float]] = [None] * len(sensors_to_check)
        for segment_id in self._spread_segment_ids(active_segment):
            for wall in self.segment_walls(segment_id):
                for i in reversed(range(len(sensors_to_check))):
                    dist_index, sensor = sensors_to_check[i]
                    if (d := sensor.check_distance(wall)) is not None:
                        distances[dist_index] = d
                        # while iterating backward removing elements won't affect the order
                        del sensors_to_check[i]
            if not sensors_to_check:
                break

        if sensors_to_check:
            raise RuntimeError(
                "one of the sensors couldn't find any walls in any of the segments"
            )

        return cast(List[float], distances)

    def intersects(self, shape: Polygon, active_segment: SegmentId) -> bool:
        """Checks whether a given shape intersects any of the track's walls."""
        found_limit = [False, False]
        for i, segment_id in enumerate(self._spread_segment_ids(active_segment)):
            if found_limit[i % 2]:
                continue
            for wall in self.segment_walls(segment_id):
                if wall.intersects(shape):
                    return True
            current_bbx = self.segments[segment_id].bounding_box
            if not any(current_bbx.contains_point(p) for p in shape):
                found_limit[i % 2] = True
                if all(found_limit):
                    return False
        return False

    def update_active(self, active_segment: SegmentId, center: Point) -> SegmentId:
        """
        Returns segment currently containing the given point
        assuming it was moved by a small amount and is still inside the track.
        """
        for segment_id in self._spread_segment_ids(active_segment):
            if self.segments[segment_id].is_point_inside(center):
                return segment_id
        raise RuntimeError("none of the track's segments contain the given point")

    @cached_property
    def boundaries(self) -> Tuple[Vec2, Vec2]:
        """:return: Tuple[Vec2, Vec2] - upper left and lower right corners of rectangle enclosing track
        """
        first_point = self.segments[0].right_wall.line_segment.points[0]
        min_x = first_point.x
        min_y = first_point.y
        max_x = first_point.x
        max_y = first_point.y
        for segment in self.segments:
            for point in (
                segment.left_wall.line_segment.points
                + segment.right_wall.line_segment.points
            ):
                min_x = min(min_x, point.x)
                min_y = min(min_y, point.y)
                max_x = max(max_x, point.x)
                max_y = max(max_y, point.y)
        return Vec2(min_x, min_y), Vec2(max_x, max_y)
