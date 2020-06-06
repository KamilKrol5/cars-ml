from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List

from model.car import Car
from model.geom.track import Track


@dataclass
class Simulation:
    track: Track
    cars: List[Car]

    def get_positions(self) -> List[Car]:
        return []

    def tick(self, delta_time: float) -> List[Car]:
        for car in self.cars:
            car.tick()
