import json
import time
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
    print(
        f"Initialization finished. Running evolution, individuals count: {len(neuroevolution.individuals)}"
    )
    step = 100
    time_start = time.time()
    for i in count(0):
        neuroevolution.evolve(env, False)
        if i % step == 0:
            time_for_step_iterations = time.time() - time_start
            print("Saving networks to file...")
            NeuralNetworkStore.store(
                [i.neural_network for i in neuroevolution.individuals], f"data{i}"
            )
            print(
                f"Time for {max(0, i-step)} to {i} iterations: {time_for_step_iterations}."
            )
            print(
                f"Iteration: {i}; Results: {[i.adaptation for i in neuroevolution.individuals]}"
            )
            time_start = time.time()


if __name__ == "__main__":
    main()
