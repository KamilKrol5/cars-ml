from abc import ABC, abstractmethod
from typing import List, Optional

from pygame.event import EventType
from pygame.surface import Surface

from view.Action import Action


class View(ABC):
    def __init__(self) -> None:
        self._active = False

    def activate(self) -> None:
        self._active = True

    @abstractmethod
    def draw(self, destination: Surface, events: List[EventType]) -> Optional[Action]:
        pass

    def deactivate(self) -> None:
        self._active = False

    @property
    def active(self) -> bool:
        return self._active
