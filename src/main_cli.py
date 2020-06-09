import json
from itertools import count
from typing import List

from model.track.track import Track
from model.neural_network.neural_network import LayerInfo
from model.neural_network.neural_network_store import NeuralNetworkStore
from model.neuroevolution.neuroevolution import Neuroevolution
from view.silent_environment import SilentEnvironment

input_neurons = 7

layers_infos: List[LayerInfo] = [
    LayerInfo(input_neurons, "tanh"),
    LayerInfo(8, "tanh"),
    LayerInfo(12, "tanh"),
    LayerInfo(18, "tanh"),
    LayerInfo(9, "tanh"),
]


def main() -> None:
    try:
        with open("resources/tracks.json") as file:
            tracks = json.load(file)["tracks"]
    except (IOError, IndexError):
        raise IOError("Unable to load tracks from file")

    print("Initialization ...")
    track = Track.from_points(tracks[1]["points"])
    env = SilentEnvironment(track)
    neuroevolution = Neuroevolution.init_with_neural_network_info(layers_infos, 2)

    for i in count(0):
        neuroevolution.evolve(env, False)
        if i % 100 == 0:
            NeuralNetworkStore.store(
                [i.neural_network for i in neuroevolution.individuals], "data"
            )


if __name__ == "__main__":
    main()
