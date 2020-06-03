from __future__ import annotations

from typing import List, Optional, Type, Tuple

import numpy as np
import pygame
from pygame.event import EventType
from pygame.surface import Surface

from model import Simulation
from model.geom.track import Track
from view import Colors
from view.Action import Action
from view.View import View


class TrackView(View):
    simulation: Simulation
    simulation_class: Type[Simulation]
    dataset = None
    board: Surface
    coord_start: Tuple[float, float]
    scale = 1.0
    margin = 0
    background_color = Colors.GRAY
    foreground_color = Colors.BLACK

    def __init__(self, simulation: Simulation):
        super().__init__()

    @classmethod
    def from_dataset(cls, simulation_class: Type[Simulation],
                     dataset) -> TrackView:
        tv = cls(None)
        tv.simulation_class = simulation_class
        tv.dataset = dataset
        return tv

    def draw(self,
             destination: Surface,
             events: List[EventType]) -> Optional[Action]:
        pass

    def activate(self) -> None:
        super().activate()
        if self.simulation:
            self.simulation = self.simulation_class(**self.dataset)

        track: Track = self.simulation.track
        board_size = [(self.scale * (b - a)) + (2 * self.margin) for a, b in
                      track.boundaries]
        board_start = [self.scale * a + self.margin for a, _ in
                       track.boundaries]

        board_surf = Surface(board_size)
        # board_rect = Rect(board_start, board_size)
        board_surf.fill(self.background_color)  # maybe texture one day

        for segment in track.segments:
            points = np.array(
                segment.left_wall.line_segment.points + segment.right_wall.line_segment.points)
            points *= self.scale
            points += self.margin
            points += np.array([board_start] * len(points))
            pygame.draw.polygon(board_surf, self.foreground_color,
                                list(points))

        self.board = board_surf
        self.coord_start = tuple(board_start)
