from typing import Tuple, Any, Optional, Union, Dict

import pygame
from pygame.surface import Surface

from view import Colors
from view.Action import ActionType
from view.View import View


class Window:
    loading_screen: Union[View, Surface, None] = None

    def __init__(
            self,
            name: str,
            size: Tuple[int, int],
            mode: Any = None,
            fullscreen: bool = False,
            resizable: bool = False,
    ):
        pygame.init()
        pygame.display.set_caption(name)
        self._screen: pygame.Surface = pygame.display.set_mode(
            size)  # TODO modes
        self._closing = False
        self._view_manager = self.ViewManager()

    def run(self) -> None:

        while not self._closing:
            event_passthrough = []
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.K_ESCAPE:
                    self._closing = True
                else:
                    event_passthrough.append(event)

            self._screen.fill(Colors.BLACK)

            if x := self._view_manager.active_view.draw(self._screen,
                                                        event_passthrough):
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

    def add_view(self, view: View, id: Union[int, str],
                 active: bool = False) -> None:
        self._view_manager.add(view, id, active)

    def remove_view(self, id: Union[int, str]) -> View:
        return self._view_manager.remove(id)
