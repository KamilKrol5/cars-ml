from typing import List, Mapping, Tuple

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
        self.cars = {
            name: [self._make_car(nn) for nn in group] for name, group in cars.items()
        }

    @staticmethod
    def _make_car(nn: NeuralNetwork) -> Tuple[Car, bool]:
        car = Car.with_standard_sensors((10.0, 20.0), nn)
        return car, False

    def update(self, delta_time: float) -> Tuple[float, SimState]:
        """
        Updates simulation state with however many fixed steps fit in the given time.

        Returns:
            Tuple with remaining time and current state of the simulation.
        """
        state = self.cars
        while delta_time > FIXED_DELTA_TIME:
            state = self.fixed_update(FIXED_DELTA_TIME)
            delta_time -= FIXED_DELTA_TIME

        return delta_time, state

    def fixed_update(self, delta_time: float) -> SimState:
        """
        Updates simulations state.

        delta_time should be the same for each call to get deterministic results.
        """
        state = self.cars
        for car_group in state.values():
            for car, active in car_group:
                if active:
                    car.tick(self.track, delta_time)
        return state
