from dataclasses import dataclass
from functools import cached_property
from typing import cast

from planar import Polygon, Vec2
from planar.box import BoundingBox
from planar.line import LineSegment

from model.track.wall import Wall


@dataclass
class TrackSegment:
    """Represents a segment of a track with 2 walls

    Walls should be oriented in the same direction
    """

    # __dict__ needed for cached properties
    __slots__ = (
        "left_wall",
        "right_wall",
        "__dict__",
    )

    left_wall: Wall
    right_wall: Wall

    @cached_property
    def region(self) -> Polygon:
        return Polygon(
            [
                *self.left_wall.line_segment.points,
                *reversed(self.right_wall.line_segment.points),
            ]
        )

    @cached_property
    def bounding_box(self) -> BoundingBox:
        return self.region.bounding_box

    @cached_property
    def back_wall(self) -> Wall:
        return Wall(
            LineSegment.from_points(
                [self.left_wall.line_segment.start, self.right_wall.line_segment.start]
            )
        )

    @cached_property
    def front_wall(self) -> Wall:
        return Wall(
            LineSegment.from_points(
                [self.left_wall.line_segment.end, self.right_wall.line_segment.end]
            )
        )

    def is_point_inside(self, center: Vec2) -> bool:
        """Checks whether the given point is inside this segment."""
        region = self.region
        return cast(bool, region.contains_point(center))

    def is_fully_inside(self, shape: Polygon) -> bool:
        """Checks whether the given polygon is fully contained in this segment."""
        region = self.region
        return all(region.contains_point(p) for p in shape)
