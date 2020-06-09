from dataclasses import dataclass
from typing import List, Mapping, Tuple

from model.car.car import Car, Collision
from model.neural_network.neural_network import NeuralNetwork
from model.track.track import Track

from planar.transform import Affine

FIXED_DELTA_TIME = 1.0 / 10.0


@dataclass
class CarState:
    car: Car
    active: bool
    ticks: int


SimState = Mapping[str, List[CarState]]


class Simulation:
    track: Track
    cars: SimState

    def __init__(self, track: Track, cars: Mapping[str, List[NeuralNetwork]]):
        self.track = track
        self.cars = {
            name: [self._make_car(nn, track) for nn in group]
            for name, group in cars.items()
        }

    @staticmethod
    def _make_car(nn: NeuralNetwork, track: Track) -> CarState:
        car = Car.with_standard_sensors((10.0, 20.0), nn)
        car.transform(Affine.translation(track.segments[2].region.centroid))
        return CarState(car, True, 1)

    def update(self, delta_time: float) -> Tuple[float, SimState]:
        """
        Updates simulation state with however many fixed steps fit in the given time.

        Returns:
            Tuple with remaining time and current state of the simulation.
        """
        state = self.cars
        adjusted = 1 * FIXED_DELTA_TIME
        while delta_time >= adjusted:
            state = self.fixed_update(FIXED_DELTA_TIME)
            delta_time -= adjusted
        # print(self.cars[0])
        return delta_time, state

    def fixed_update(self, delta_time: float) -> SimState:
        """
        Updates simulations state.

        delta_time should be the same for each call to get deterministic results.
        """
        state = self.cars
        for car_group in state.values():
            for car_state in car_group:
                if car_state.active:
                    try:
                        car_state.car.tick(self.track, delta_time)
                        car_state.ticks += 1
                    except Collision:
                        car_state.active = False
        return state
