from typing import List, Optional, Any, Dict

from pygame.event import EventType
from pygame.surface import Surface

from view.Action import Action
from view.View import View


class OptionsMenu(View):
    def __init__(
        self, available: Any, output_dataset: Dict[str, Any], next: Optional[int] = None
    ) -> None:
        super().__init__()

    def draw(self, destination: Surface, events: List[EventType]) -> Optional[Action]:
        pass
