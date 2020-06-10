from collections import OrderedDict
from typing import Dict, Tuple, Optional, List

import pygame
from pygame.event import EventType
from pygame.rect import Rect
from pygame.surface import Surface

from view import colors
from view.action import Action
from view.view import View


class Menu(View):
    FONT_COLOR = colors.ORANGE
    SELECTED_FONT_COLOR = colors.WHITE
    BUTTON_COLOR = colors.WHITE
    SELECTED_BUTTON_COLOR = colors.ORANGE

    def __init__(self, menu_options: Dict[str, Action]):
        super().__init__()
        pygame.font.init()
        self.font = pygame.font.SysFont("Verdana", 24)
        self.selected_item = 0
        self._options = OrderedDict(menu_options)
        self.font_color = colors.WHITE
        self.button_highlight = colors.GRAY
        self.button_color = colors.LIGHTGRAY
        self._background: Optional[Tuple[Surface, Tuple[int, int]]] = None
        self._logo: Optional[Tuple[Surface, Tuple[int, int]]] = None
        self.background_image: Surface = Surface((1, 1))
        self.logo_image: Optional[Surface] = None
        self.button_dims = (360, 80)
        self.divider = 0.4

    def draw(
        self, destination: Surface, events: List[EventType], delta_time: float
    ) -> Optional[Action]:
        # TODO add activation check for consistency
        size = destination.get_size()
        self._update_geometry(size)

        if self._process_events(events):
            return list(self._options.values())[self.selected_item]

        if self._background:
            destination.blit(*self._background)
        if self._logo:
            destination.blit(*self._logo)

        for pos, button in enumerate(self._options):
            ofset_y = int(pos * 1.5 * self.button_dims[1])
            shifted_button_rect = self._button_rect.move(0, ofset_y)
            color = (
                self.SELECTED_BUTTON_COLOR
                if pos == self.selected_item
                else self.BUTTON_COLOR
            )

            if pos == self.selected_item:
                shifted_button_rect = shifted_button_rect.inflate(20, 20)
            else:
                pygame.draw.rect(destination, colors.ORANGE, shifted_button_rect, 4)

            pygame.draw.rect(destination, color, shifted_button_rect)

            font_color = (
                self.SELECTED_FONT_COLOR
                if pos == self.selected_item
                else self.FONT_COLOR
            )

            label = self.font.render(button, True, font_color)
            destination.blit(
                label,
                (
                    shifted_button_rect.centerx - (label.get_width() // 2),
                    shifted_button_rect.centery - 16,
                ),
            )
        return None

    def _update_geometry(self, size: Tuple[int, int]) -> None:
        ofset_y = int(self.divider * size[1])
        ofset_x = (size[0] - self.button_dims[0]) // 2

        background_shape = Rect((0, 0), size)
        background_image = pygame.transform.scale(self.background_image, size)
        self._background = (background_image, background_shape)
        self._button_rect = Rect((ofset_x, ofset_y), self.button_dims)

        if not self.logo_image:
            return

        logo_dims = self.logo_image.get_size()
        logo_shape = Rect(
            ((size[0] - logo_dims[0]) // 2, (ofset_y - logo_dims[1]) // 2), logo_dims
        )
        self._logo = (self.logo_image, logo_shape)

    def _process_events(self, events: List[EventType]) -> bool:
        for event in events:
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    self.selected_item = (self.selected_item - 1) % len(self._options)
                elif event.key == pygame.K_DOWN:
                    self.selected_item = (self.selected_item + 1) % len(self._options)
                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    return True
        return False