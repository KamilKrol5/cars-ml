from typing import List, Generator

from model.neural_network.neural_network import NeuralNetwork
from model.neuroevolution.neuroevolution import Neuroevolution
from model.track.track import Track
from view.pygame_environment import EnvironmentContext, PyGameEnvironment
from view.track_view import TrackView


class PlayTrackView(TrackView):
    _paused = False

    def __init__(self, track: Track, neural_networks: List[NeuralNetwork]):
        super().__init__(track)

        self.environment = PyGameEnvironment(track)
        self.neuroevolution = Neuroevolution(neural_networks)

    def _get_generator(self) -> Generator[None, EnvironmentContext, None]:
        generator = self.neuroevolution.generate_evolution(self.environment, True)
        next(generator)
        return generator
