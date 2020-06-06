from dataclasses import dataclass
from enum import Enum, auto
from typing import Union


class ActionType(Enum):
    CONTINUE = auto()  # Unused
    SYS_EXIT = auto()
    CHANGE_VIEW = auto()
    BACK_VIEW = auto()


@dataclass
class Action:
    type: ActionType
    extra: Union[str, int, None] = None
