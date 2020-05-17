from dataclasses import dataclass
from functools import cached_property
from typing import cast

from planar import Polygon

from model.geom.wall import Wall


@dataclass
class Segment:
    """Represents a segment of a track with 2 walls

    Walls should be oriented in the same direction
    """

    left_wall: Wall
    right_wall: Wall

    @cached_property
    def region(self) -> Polygon:
        return Polygon(
            [*self.left_wall.segment.points, *reversed(self.right_wall.segment.points)]
        )

    def is_center_inside(self, shape: Polygon) -> bool:
        """Check whether the center of the given polygon is inside this segment."""
        region = self.region
        return cast(bool, region.contains_point(shape.centroid))

    def is_fully_inside(self, shape: Polygon) -> bool:
        """Check whether the given polygon is fully contained in this segment."""
        region = self.region
        return all(region.contains_point(p) for p in shape)
