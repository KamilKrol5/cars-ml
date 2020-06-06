from typing import Tuple, Optional, Union, Callable, List, NoReturn, Mapping

import pygame
from pygame.surface import Surface

from view.action import ActionType, Action
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

    def run(self, initial_view: View) -> None:
        self._view_manager.push(initial_view)

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
                action := self._view_manager.active_view.draw(
                    self._screen, event_passthrough
                )
            ) is not None:
                self._execute_action(action)

            pygame.display.update()
        pygame.quit()

    def _execute_action(self, action: Action) -> None:
        actions: Mapping[ActionType, Callable[[Action], None]]
        actions = {
            ActionType.SYS_EXIT: self._exit,
            ActionType.PUSH_VIEW: self._push_view,
            ActionType.POP_VIEW: self._pop_view,
        }
        actions.get(action.type, self.__invalid_action)(action)

    def _push_view(self, action: Action) -> None:
        assert action.extra is not None, "empty action data"
        print(action)
        view_constructor, *args = action.extra
        new_view = view_constructor(*args)
        self._view_manager.push(new_view)

    def _pop_view(self, _: Action) -> None:
        self._view_manager.pop()

    def _exit(self, _: Action) -> None:
        self._closing = True

    def __invalid_action(self, _: Action) -> NoReturn:
        raise NotImplementedError

    class ViewManager:
        def __init__(self) -> None:
            self._views: List[View] = []

        def push(self, view: View) -> None:
            if (active := self._active_view) is not None:
                active.deactivate()
            self._views.append(view)
            view.activate()

        def pop(self) -> View:
            assert self._views, "no views found"
            view = self._views.pop()
            view.deactivate()
            return view

        @property
        def _active_view(self) -> Optional[View]:
            if self._views:
                return self._views[-1]
            else:
                return None

        @property
        def active_view(self) -> View:
            view = self._active_view
            assert view is not None, "no views found"
            return view
