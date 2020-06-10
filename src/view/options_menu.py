from typing import List, Optional, Dict, Tuple, Any
from collections import OrderedDict

import pygame

from pygame.event import EventType
from pygame.surface import Surface
from pygame.rect import Rect

from view.action import Action, ActionType
from view.view import View
from view import colors


class OptionsMenu(View):
    ARROW_COLOR = colors.ORANGE
    FONT_COLOR = colors.WHITE
    LOGO_IMAGE = pygame.image.load("resources/graphics/logo.png")
    DIVIDER = 0.4

    def __init__(self, tracks: Dict[str, Action], tracks_thumbnails: List[Any]) -> None:
        super().__init__()
        pygame.font.init()
        self.font = pygame.font.SysFont("Verdana", 30)
        self.main_font = pygame.font.SysFont("comicsansms", 60)
        self.selected_item = 0
        self._options = OrderedDict(tracks)
        self._background: Optional[Tuple[Surface, Tuple[int, int]]] = None
        self.background_image: Optional[Surface] = None
        self.selected_action: Optional[Action] = None
        self.thumbnails = tracks_thumbnails

    def draw(self, destination: Surface, events: List[EventType], delta_time: float) -> Optional[Action]:
        size = destination.get_size()
        self._update_geometry(size)

        if self._process_events(events):
            return self.selected_action

        if self._background:
            destination.blit(*self._background)
        if self._logo:
            destination.blit(*self._logo)

        mini_track = self._button_rect
        thumbnail_image = pygame.transform.scale(self.thumbnails[self.selected_item], (mini_track.w, mini_track.h))
        thumbnail = (thumbnail_image, mini_track)
        destination.blit(*thumbnail)
        pygame.draw.rect(destination, colors.WHITE, mini_track, 4)

        main_label = self.main_font.render("Select track", True, self.FONT_COLOR)
        destination.blit(main_label, (size[0] // 2 - 185, size[1] // 10))

        track_label = self.font.render(
            list(self._options.keys())[self.selected_item], True, self.FONT_COLOR
        )
        destination.blit(
            track_label,
            (
                mini_track.centerx - (track_label.get_height() // 2) - 28,
                mini_track.centery - (mini_track.size[1] // 2) - 68,
            ),
        )

        left_arrow_points = (
            (self.offset_x - size[0] // 20, self.offset_y + mini_track.h // 2),
            (self.offset_x - 10, self.offset_y + mini_track.h // 2 - size[0] // 40),
            (self.offset_x - 10, self.offset_y + mini_track.h // 2 + size[0] // 40),
        )
        right_arrow_points = (
            (3 * self.offset_x + size[0] // 20, self.offset_y + mini_track.h // 2),
            (3 * self.offset_x + 10, self.offset_y + mini_track.h // 2 - size[0] // 40),
            (3 * self.offset_x + 10, self.offset_y + mini_track.h // 2 + size[0] // 40),
        )

        if self.selected_item != 0:
            pygame.draw.polygon(destination, self.ARROW_COLOR, left_arrow_points)
        if self.selected_item != len(self._options) - 1:
            pygame.draw.polygon(destination, self.ARROW_COLOR, right_arrow_points)
        return None

    def _update_geometry(self, size: Tuple[int, int]) -> None:
        self.offset_y = int(self.DIVIDER * size[1])
        self.offset_x = size[0] // 4

        self.button_dims = size[0] // 2, size[1] // 2

        background_shape = Rect((0, 0), size)
        background_image = pygame.transform.scale(self.background_image, size)
        self._background = (background_image, background_shape)
        self._button_rect = Rect((self.offset_x, self.offset_y), self.button_dims)

        logo_image = pygame.transform.scale(
            self.LOGO_IMAGE, (size[1] // 3, size[1] // 12)
        )
        logo_shape = Rect((10, 10), (size[0] // 8, size[1] // 8))
        self._logo = (logo_image, logo_shape)

    def _process_events(self, events: List[EventType]) -> bool:
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.selected_item != 0:
                    self.selected_item = self.selected_item - 1
                elif (
                        event.key == pygame.K_RIGHT
                        and self.selected_item != len(self._options) - 1
                ):
                    self.selected_item = self.selected_item + 1
                elif event.key == pygame.K_a and self.selected_item != 0:
                    self.selected_item = self.selected_item - 1
                elif (
                        event.key == pygame.K_d
                        and self.selected_item != len(self._options) - 1
                ):
                    self.selected_item = self.selected_item + 1
                elif event.key == pygame.K_RETURN:
                    self.selected_action = list(self._options.values())[
                        self.selected_item
                    ]
                    return True
                elif event.key == pygame.K_ESCAPE:
                    self.selected_action = Action(ActionType.CHANGE_VIEW, 0)
                    return True
        return False
