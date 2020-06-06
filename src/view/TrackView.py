from __future__ import annotations

from typing import List, Optional, Type, Dict, Any, Iterable

import pygame
from planar import Vec2
from pygame.event import EventType
from pygame.rect import Rect
from pygame.surface import Surface

from model.Simulation import Simulation
from model.geom.track import Track
from view import Colors
from view.Action import Action, ActionType
from view.View import View


class TrackView(View):
    simulation: Simulation
    simulation_class: Type[Simulation]
    dataset: Dict[str, Any]
    board: Surface
    coord_start: Vec2
    scale = 1.0
    poi = Vec2(0, 0)
    # margin = 0
    background_color = Colors.GRAY
    foreground_color = Colors.BLACK
    car_color = Colors.RED

    def __init__(self, simulation: Optional[Simulation]):
        super().__init__()
        # to make non optional, as it really is, builder would be required
        self.simulation = simulation

    @classmethod
    def from_dataset(
        cls, simulation_class: Type[Simulation], dataset: Dict[str, Any]
    ) -> TrackView:
        tv = cls(None)
        tv.simulation_class = simulation_class
        tv.dataset = dataset
        return tv

    def draw(self, destination: Surface, events: List[EventType]) -> Optional[Action]:
        if x := self._process_events(events):
            return x
        # TODO draw cars/optimisation
        board = self.board.copy()
        for car in self.simulation.get_positions():
            pygame.draw.circle(
                board, self.car_color, (car.position_x, car.position_y), 10
            )

        # may be optimised
        if (
            destination.get_width() > board.get_width()
            or destination.get_height() > board.get_height()
        ):
            board = self.board
            new_size = (
                max(board.get_width(), destination.get_width()),
                max(board.get_height(), destination.get_height()),
            )
            anchor = Vec2(*new_size)
            anchor -= board.get_size()
            anchor /= 2
            new_board = Surface(new_size)
            new_board.fill((0, 250, 0))
            new_board.blit(board, Rect(anchor, self.board.get_size()))
            board = new_board

        view_rect: Rect = destination.get_rect()
        limit_x = destination.get_width() // 2
        limit_y = destination.get_height() // 2
        center_x = min(max(limit_x, self.poi.x), board.get_width() - limit_x)
        center_y = min(max(limit_y, self.poi.y), board.get_height() - limit_y)
        view_rect.center = (int(center_x), int(center_y))

        destination.blit(board.subsurface(view_rect), (0, 0))

        return None

    def activate(self) -> None:
        super().activate()
        if not self.simulation:
            self.simulation = self.simulation_class(**self.dataset)
        self._prepare_board()
        pygame.key.set_repeat(350, 50)

    def _prepare_board(self) -> None:
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
        board_surf.fill(Colors.BIZARRE_MASKING_PURPLE)
        board_surf.set_colorkey(Colors.BIZARRE_MASKING_PURPLE, pygame.RLEACCEL)

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
                    return Action(ActionType.CHANGE_VIEW, 0)
                elif event.key == pygame.K_KP_PLUS:
                    self.scale *= 1.6
                    self.poi *= 1.6
                    self._prepare_board()
                elif event.key == pygame.K_KP_MINUS:
                    self.scale *= 0.625
                    self.poi *= 0.625
                    self._prepare_board()
                print(self.poi, self.scale, self.board.get_size())

            # TODO change to previous view
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    self.poi = self.poi + (0, change)
                elif event.key == pygame.K_UP:
                    self.poi = self.poi + (0, -change)
                elif event.key == pygame.K_LEFT:
                    self.poi = self.poi + (-change, 0)
                elif event.key == pygame.K_RIGHT:
                    self.poi = self.poi + (change, 0)
                elif event.key == pygame.K_RIGHT:
                    self.poi = self.poi + (change, 0)
                print(self.poi, self.scale, self.board.get_size())

        return None
