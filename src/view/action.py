from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional


class ActionType(Enum):
    CONTINUE = auto()  # Unused
    SYS_EXIT = auto()
    PUSH_VIEW = auto()
    POP_VIEW = auto()


@dataclass
class Action:
    type: ActionType
    extra: Optional[Any] = None
