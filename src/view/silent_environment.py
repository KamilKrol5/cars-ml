from dataclasses import dataclass

from model.environment.environment import Environment
from model.simulation import CarState
from model.track.track import Track


@dataclass
class SilentEnvironment(Environment[None, None]):
    def __init__(self, track: Track):
        super().__init__(track)

    _initialize = type(None)

    def _finalize_iteration(self, state: None, context: None) -> None:
        return state

    def _process_car_step(
        self, state: None, context: None, group_id: str, car: CarState
    ) -> None:
        pass
