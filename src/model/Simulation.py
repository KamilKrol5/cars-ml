from dataclasses import dataclass
from typing import Any

from model.geom.track import Track


@dataclass
class Simulation():
    track: Track
    ai: Any