from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Generator, Generic, TypeVar, List, Any

import utils
from model.neural_network.neural_network import NeuralNetwork
from model.simulation import SimState, Simulation, FIXED_DELTA_TIME, CarState
from model.track.track import Track

T_CONTEXT = TypeVar("T_CONTEXT", contravariant=True)
T_STATE = TypeVar("T_STATE")


class Environment(ABC, Generic[T_CONTEXT, T_STATE]):
    def __init__(self, track: Track):
        self._track = track

    def __run_simulation(
        self, networks_groups: Mapping[str, List[NeuralNetwork]]
    ) -> Generator[None, T_CONTEXT, SimState]:
        simulation = Simulation(self._track, networks_groups)
        frame_time = 0.0
        any_active = True

        state: T_STATE = self._initialize()

        while any_active:
            context: T_CONTEXT = (yield)
            frame_time += FIXED_DELTA_TIME
            frame_time, cars = simulation.update(frame_time)

            any_active = False
            # TODO: differentiate cars on group_id
            for group_id, car_group in cars.items():
                for car_state in car_group:
                    if car_state.active:
                        any_active = True

                    self._process_car_step(state, context, group_id, car_state)
            state = self._finalize_iteration(state, context)
        return cars

    def generate_adaptations(
        self, networks_groups: Mapping[str, List[NeuralNetwork]]
    ) -> Generator[None, T_CONTEXT, Mapping[str, Iterable[float]]]:
        cars = yield from self.__run_simulation(networks_groups)

        return self._finalize(cars)

    def run(
        self, networks_groups: Mapping[str, List[NeuralNetwork]]
    ) -> Generator[None, T_CONTEXT, None]:
        yield from self.__run_simulation(networks_groups)

    @staticmethod
    def compute_adaptations(
        env: "Environment[None, Any]",
        networks_groups: Mapping[str, List[NeuralNetwork]],
    ) -> Mapping[str, Iterable[float]]:
        return utils.generator_value(env.generate_adaptations(networks_groups))

    @abstractmethod
    def _process_car_step(
        self, state: T_STATE, context: T_CONTEXT, group_id: str, car: CarState
    ) -> None:
        pass

    @abstractmethod
    def _initialize(self) -> T_STATE:
        raise NotImplementedError

    @abstractmethod
    def _finalize_iteration(self, state: T_STATE, context: T_CONTEXT) -> T_STATE:
        return state

    def _finalize(self, cars: SimState) -> Mapping[str, Iterable[float]]:
        return {name: self._group_adaptation(group) for name, group in cars.items()}

    def _group_adaptation(self, car_group: Iterable[CarState]) -> Iterable[float]:
        for car_state in car_group:
            yield self._car_adaptation(car_state)

    def _car_adaptation(self, car_state: CarState) -> float:
        return car_state.car.active_segment ** 2 / car_state.active_ticks
