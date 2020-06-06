from typing import Tuple, Optional, Union, Dict

import pygame
from pygame.surface import Surface

from view.action import ActionType
from view.view import View


class Window:
    loading_screen: Union[View, Surface, None] = None

    def __init__(
        self,
        name: str,
        size: Tuple[int, int],
        fullscreen: bool = False,
        resizable: bool = False,
        min_size: Optional[Tuple[int, int]] = None,
    ):
        pygame.init()
        pygame.display.set_caption(name)
        self._mode = pygame.HWSURFACE | pygame.DOUBLEBUF
        if fullscreen:
            self._mode |= pygame.FULLSCREEN
        if resizable:
            self._mode |= pygame.RESIZABLE
        self._screen: pygame.Surface = pygame.display.set_mode(size, self._mode)
        self._closing = False
        self._view_manager = self.ViewManager()
        self._min_size = min_size

    def run(self) -> None:

        while not self._closing:
            event_passthrough = []
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._closing = True
                elif event.type == pygame.KEYUP and event.key == pygame.K_F11:
                    print(self._screen.get_size())
                    self._mode ^= pygame.FULLSCREEN
                    self._screen = pygame.display.set_mode((0, 0), self._mode)
                elif event.type == pygame.VIDEORESIZE:
                    size = list(event.size)
                    if self._min_size and size[0] < self._min_size[0]:
                        size[0] = self._min_size[0]
                    if self._min_size and size[1] < self._min_size[1]:
                        size[1] = self._min_size[1]
                    self._screen = pygame.display.set_mode(size, self._mode)
                    print(self._screen.get_size())
                else:
                    event_passthrough.append(event)

            if (
                x := self._view_manager.active_view.draw(
                    self._screen, event_passthrough
                )
            ) is not None:
                if x.type == ActionType.SYS_EXIT:
                    self._closing = True
                elif x.type == ActionType.CHANGE_VIEW:
                    assert x.extra is not None
                    print(x)
                    self._view_manager.change_view(x.extra)
                else:
                    raise NotImplementedError

            pygame.display.update()
        pygame.quit()

    class ViewManager:
        def __init__(self) -> None:
            self._active: Optional[Union[int, str]] = None
            self._views: Dict[Union[int, str], View] = {}

        def remove(self, id: Union[int, str]) -> View:
            return self._views.pop(id)

        def add(self, view: View, id: Union[int, str], active: bool) -> None:
            self._views[id] = view
            if active:
                self.change_view(id)

        def change_view(self, id: Union[int, str]) -> None:
            if self._active is not None:
                self._views[self._active].deactivate()
            self._active = id
            self._views[id].activate()

        @property
        def active_view(self) -> View:
            if self._active is None:
                raise AssertionError("No activated views found")
            return self._views[self._active]

    def add_view(self, view: View, id: Union[int, str], active: bool = False) -> None:
        self._view_manager.add(view, id, active)

    def remove_view(self, id: Union[int, str]) -> View:
        return self._view_manager.remove(id)
