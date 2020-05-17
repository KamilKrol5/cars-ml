from dataclasses import dataclass
from itertools import cycle, islice

from planar.line import LineSegment, Line
from planar.polygon import Polygon

from utils import pairwise


@dataclass
class Wall:
    segment: LineSegment

    def intersects(self, shape: Polygon) -> bool:
        """Check whether this wall intersects a Polygon."""
        segment_line = self.segment.line
        vertex1, vertex2 = self.segment.points
        relative_sides = ((p, segment_line.point_left(p)) for p in shape)
        for (p1, s1), (p2, s2) in islice(pairwise(cycle(relative_sides)), len(shape)):
            if s1 is not s2:
                # p1 and p2 are on opposite sides of the line that contains this wall
                edge = Line.from_points([p1, p2])
                if edge.point_left(vertex1) is not edge.point_left(vertex2):
                    return True
        return False
