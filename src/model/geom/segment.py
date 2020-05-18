from dataclasses import dataclass
from functools import cached_property
from typing import cast

from planar import Polygon, Vec2

from model.geom.wall import Wall


@dataclass
class TrackSegment:
    """Represents a segment of a track with 2 walls

    Walls should be oriented in the same direction
    """

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

    def is_point_inside(self, center: Vec2) -> bool:
        """Check whether the given point is inside this segment."""
        region = self.region
        return cast(bool, region.contains_point(center))

    def is_fully_inside(self, shape: Polygon) -> bool:
        """Check whether the given polygon is fully contained in this segment."""
        region = self.region
        return all(region.contains_point(p) for p in shape)
