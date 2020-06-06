from __future__ import annotations

from typing import List, Optional, Type, Dict, Any, Iterable, Tuple

import pygame
from planar import Vec2, Point
from pygame.event import EventType
from pygame.rect import Rect
from pygame.surface import Surface

from model.geom.track import Track
from model.neuroevolution.neuroevolution import Neuroevolution
from model.simulation import Simulation
from view import colors
from view.action import Action, ActionType
from view.pygame_environment import EnvironmentContext
from view.view import View


class TrackView(View):
    simulation: Optional[Simulation]
    simulation_class: Type[Simulation]
    dataset: Dict[str, Any]
    board: Surface
    coord_start: Vec2
    scale = 1.0
    point_of_interest = Vec2(0, 0)
    # margin = 0
    background_color = colors.GRAY
    foreground_color = colors.BLACK
    car_color = colors.RED

    def __init__(self, track_data: List[Tuple[Point, Point]]) -> None:
        super().__init__()
        self.track_data = track_data

    neuroevolution = Neuroevolution(None)
    evo = neuroevolution.generate_evolution(None, False)

    def draw(self, surface: Surface, events: List[EventType]) -> Optional[Action]:
        if (x := self._process_events(events)) is not None:
            return x

        # TODO draw cars/optimisation
        # Displaying cars will be re-added when the final version of simulation interface will be available
        # surface stitching may be optimised

        # to be refactored
        board = self.board.copy()
        new_size = (
            max(board.get_width(), surface.get_width()),
            max(board.get_height(), surface.get_height()),
        )
        anchor = Vec2(*new_size)
        anchor -= board.get_size()
        anchor /= 2
        new_board = Surface(new_size)
        new_board.fill(self.background_color)
        new_board.blit(board, Rect(anchor, self.board.get_size()))
        board = new_board

        try:
            self.evo.send(EnvironmentContext(surface, 1.0 / 60.0))
        except StopIteration:
            self.evo = self.neuroevolution.generate_evolution(None, False)

        view_rect: Rect = surface.get_rect()
        limit_x = surface.get_width() // 2
        limit_y = surface.get_height() // 2
        center_x = min(
            max(limit_x, self.point_of_interest.x), board.get_width() - limit_x
        )
        center_y = min(
            max(limit_y, self.point_of_interest.y), board.get_height() - limit_y
        )
        view_rect.center = (int(center_x), int(center_y))

        surface.blit(board.subsurface(view_rect), (0, 0))

        return None

    def activate(self) -> None:
        super().activate()
        if not self.simulation:
            self.simulation = self.simulation_class(**self.dataset)
        self._prepare_board()
        pygame.key.set_repeat(350, 50)

    def _prepare_board(self) -> None:
        assert self.simulation is not None
        track: Track = self.simulation.track
        board_size = (
            track.bounding_box.width * self.scale,
            track.bounding_box.height * self.scale,
        )
        self.coord_start = track.bounding_box.min_point
        # board_size += 2 * Vec2(self.margin, self.margin)
        # board_start = track.boundaries[0] * self.scale
        board_surf = Surface(board_size)
        # board_rect = Rect(board_start, board_size)
        board_surf.fill(colors.BIZARRE_MASKING_PURPLE)
        board_surf.set_colorkey(colors.BIZARRE_MASKING_PURPLE, pygame.RLEACCEL)

        for segment in track.segments:
            points: Iterable[Vec2] = segment.region
            points = [(point - self.coord_start) * self.scale for point in points]
            pygame.draw.polygon(board_surf, self.foreground_color, points)

        self.board = board_surf

    def _process_events(self, events: List[EventType]) -> Optional[Action]:
        change = 20
        for event in events:
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    return Action(ActionType.POP_VIEW, 0)
                elif event.key == pygame.K_KP_PLUS:
                    self.scale *= 1.6
                    self.point_of_interest *= 1.6
                    self._prepare_board()
                elif event.key == pygame.K_KP_MINUS:
                    self.scale *= 0.625
                    self.point_of_interest *= 0.625
                    self._prepare_board()
                print(self.point_of_interest, self.scale, self.board.get_size())

            # TODO change to previous view
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    self.point_of_interest = self.point_of_interest + (0, change)
                elif event.key == pygame.K_UP:
                    self.point_of_interest = self.point_of_interest + (0, -change)
                elif event.key == pygame.K_LEFT:
                    self.point_of_interest = self.point_of_interest + (-change, 0)
                elif event.key == pygame.K_RIGHT:
                    self.point_of_interest = self.point_of_interest + (change, 0)
                elif event.key == pygame.K_RIGHT:
                    self.point_of_interest = self.point_of_interest + (change, 0)
                print(self.point_of_interest, self.scale, self.board.get_size())

        return None
