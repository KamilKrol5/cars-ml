from __future__ import annotations

from typing import List, Optional, Iterable, Generator

import pygame
from planar import Vec2
from pygame.event import EventType
from pygame.rect import Rect
from pygame.surface import Surface

from model.car.directed_rect import SURROUNDING_RAYS_COUNT
from model.track.track import Track
from model.neural_network.neural_network import LayerInfo
from model.neuroevolution.neuroevolution import Neuroevolution
from view import colors
from view.action import Action, ActionType
from view.pygame_environment import PyGameEnvironment, EnvironmentContext
from view.view import View


class TrackView(View):
    generator: Generator[None, EnvironmentContext, None]
    track: Track
    environment: PyGameEnvironment

    neuroevolution: Neuroevolution

    _paused = False
    scale = 1.0
    coord_start: Vec2
    board: Surface
    last_frame: Surface = Surface((1, 1))
    background_color = colors.GRAY
    foreground_color = colors.BLACK
    car_color = colors.RED

    def __init__(self, track: Track):
        super().__init__()
        self.track = track
        layers_infos: List[LayerInfo] = [
            LayerInfo(SURROUNDING_RAYS_COUNT + 1, "tanh"),
            LayerInfo(4, "tanh"),
        ]
        self.environment = PyGameEnvironment(track)
        self.neuroevolution = Neuroevolution.init_with_neural_network_info(
            layers_infos, 2
        )

    def draw(
        self, destination: Surface, events: List[EventType], delta_time: float
    ) -> Optional[Action]:
        if (x := self._process_events(events)) is not None:
            return x

        if self._paused:
            width, height = destination.get_size()
            frame_size = (int(width * 0.9), int(height * 0.9))
            frame_start = (int(width * 0.05), int(height * 0.05))
            last_frame = pygame.transform.scale(self.last_frame, frame_size)
            # TODO: Paused label
            destination.fill(colors.BLACK)
            destination.blit(last_frame, frame_start)
            return None

        board = self.board.copy()
        context = EnvironmentContext(board, delta_time, self.coord_start, self.scale)
        try:
            self.generator.send(context)
        except StopIteration:
            self.generator = self._get_generator()

        new_size = (
            max(board.get_width(), destination.get_width()),
            max(board.get_height(), destination.get_height()),
        )
        anchor = Vec2(*new_size)
        anchor -= board.get_size()
        anchor /= 2
        new_board = Surface(new_size)
        new_board.fill(self.background_color)
        new_board.blit(board, Rect(anchor, self.board.get_size()))
        board = new_board

        view_rect: Rect = destination.get_rect()
        limit_x = destination.get_width() // 2
        limit_y = destination.get_height() // 2
        center_x = min(
            max(
                limit_x, (context.point_of_interest.x - self.coord_start.x) * self.scale
            ),
            board.get_width() - limit_x,
        )
        center_y = min(
            max(
                limit_y, (context.point_of_interest.y - self.coord_start.y) * self.scale
            ),
            board.get_height() - limit_y,
        )
        view_rect.center = (int(center_x), int(center_y))

        self.last_frame = board.subsurface(view_rect)
        destination.blit(self.last_frame, (0, 0))
        return None

    def activate(self) -> None:
        super().activate()
        self._prepare_board()
        self.generator = self._get_generator()

    def _prepare_board(self) -> None:
        board_size = (
            self.track.bounding_box.width * self.scale,
            self.track.bounding_box.height * self.scale,
        )
        self.coord_start = self.track.bounding_box.min_point
        board_surf = Surface(board_size)
        board_surf.fill(colors.BIZARRE_MASKING_PURPLE)
        board_surf.set_colorkey(colors.BIZARRE_MASKING_PURPLE, pygame.RLEACCEL)

        for segment in self.track.segments:
            points: Iterable[Vec2] = segment.region
            points = [(point - self.coord_start) * self.scale for point in points]
            pygame.draw.polygon(board_surf, self.foreground_color, points)
            pygame.draw.polygon(board_surf, colors.LIGHTGRAY, points, 2)
            segment_center = segment.region.centroid
            if segment_center is not None:
                segment_center = tuple(
                    map(int, (segment_center - self.coord_start) * self.scale)
                )
                pygame.draw.circle(board_surf, colors.WHITE, segment_center, 1)

        self.board = board_surf

    def _process_events(self, events: List[EventType]) -> Optional[Action]:
        for event in events:
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    return Action(ActionType.CHANGE_VIEW, 0)
                elif event.key == pygame.K_KP_PLUS and self.scale < 5:
                    self.scale *= 1.6
                    self._prepare_board()
                elif event.key == pygame.K_KP_MINUS:
                    self.scale *= 0.625
                    self._prepare_board()
                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    self._paused = not self._paused
        # TODO change to previous view
        return None

    def _get_generator(self) -> Generator[None, EnvironmentContext, None]:
        generator = self.neuroevolution.generate_evolution(self.environment, True)
        next(generator)
        return generator
