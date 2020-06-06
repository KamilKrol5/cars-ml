from __future__ import annotations
from typing import List, Dict, Mapping, Tuple

from model.car import Car
from model.geom.track import Track
from model.neural_network.neural_network import NeuralNetwork


FIXED_DELTA_TIME = 1.0 / 120.0

SimState = Mapping[str, List[Tuple[Car, bool]]]


class Simulation:
    track: Track
    cars: SimState

    def __init__(self, track: Track, cars: Mapping[str, List[NeuralNetwork]]):
        self.track = track
        # TODO: car initialization
        car = Car((10.0, 20.0), sensors=[], neural_network=None)
        self.cars = {
            name: [(car, False) for _ in group] for name, group in cars.items()
        }

    def update(self, delta_time: float) -> Tuple[float, SimState]:
        state = None
        while delta_time > FIXED_DELTA_TIME:
            state = self.fixed_update(FIXED_DELTA_TIME)
            delta_time -= FIXED_DELTA_TIME

        # TODO: deal with short frames
        assert state is not None, "frame was too short"
        return delta_time, state

    def fixed_update(self, delta_time: float) -> SimState:
        state = self.cars
        for car_group in state.values():
            for car, active in car_group:
                if active:
                    car.tick(self.track, delta_time)
        return state
