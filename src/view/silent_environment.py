from dataclasses import dataclass
from typing import Mapping, Iterable, Generator, List

from model.environment.environment import Environment
from model.neural_network.neural_network import NeuralNetwork
from model.simulation import Simulation, SimState, CarState, FIXED_DELTA_TIME
from model.track.track import Track


@dataclass
class SilentEnvironment(Environment[None]):
    track: Track

    def generate_adaptations(
        self, networks_groups: Mapping[str, List[NeuralNetwork]]
    ) -> Generator[None, None, Mapping[str, Iterable[float]]]:
        simulation = Simulation(self.track, networks_groups)
        frame_time = 0.0
        any_active = True

        while any_active:
            (yield)
            frame_time += FIXED_DELTA_TIME
            frame_time, cars = simulation.update(frame_time)

            any_active = False
            for _group_id, car_group in cars.items():
                for car_state in car_group:
                    if car_state.active:
                        any_active = True

        return self._compute(cars)

    def _compute(self, cars: SimState) -> Mapping[str, Iterable[float]]:
        return {name: self._group_adaptation(group) for name, group in cars.items()}

    def _group_adaptation(self, car_group: Iterable[CarState]) -> Iterable[float]:
        for car_state in car_group:
            yield self._car_adaptation(car_state)

    @staticmethod
    def _car_adaptation(car_state: CarState) -> float:
        return car_state.car.active_segment ** 2 / car_state.car.ticks
