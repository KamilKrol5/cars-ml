from dataclasses import dataclass
from typing import Tuple, ClassVar, List, Callable, cast

import numpy as np

from model.neural_network.neural_network import NeuralNetwork, Layer


@dataclass
class ChildIndividual:
    neural_network: NeuralNetwork

    available_mutations: ClassVar[List[Callable[["ChildIndividual"], None]]]

    def get_random_layer(self) -> Layer:
        random_layer: Layer = np.random.choice(self.neural_network.hidden_layers)
        return random_layer

    def get_random_layer_index(self) -> int:
        return cast(
            int, np.random.randint(low=0, high=len(self.neural_network.hidden_layers))
        )

    @staticmethod
    def get_random_weight_index(neural_network_layer: Layer) -> Tuple[int, ...]:
        return tuple(np.random.randint(neural_network_layer.weights.shape))

    @staticmethod
    def get_random_bias_index(neural_network_layer: Layer) -> int:
        return cast(int, np.random.randint(neural_network_layer.biases.shape[0]))

    @staticmethod
    def random_weight_mutation(individual: "ChildIndividual") -> None:
        """
        Changes a randomly chosen weight in the neural network to random value from range [-1, 1)

        Args:
            individual (ChildIndividual): Individual (neural network) to be modified.
        """
        random_layer = individual.get_random_layer()
        random_index = individual.get_random_weight_index(random_layer)
        # TODO discuss new weight value's range, update method documentation
        random_layer.weights[random_index] = np.random.uniform(-3, 3)

    @staticmethod
    def random_bias_mutation(individual: "ChildIndividual") -> None:
        """
        Changes a randomly chosen bias in the neural network to random value from range [-1, 1)

        Args:
            individual (ChildIndividual): Individual (neural network) to be modified.
        """
        random_layer = individual.get_random_layer()
        random_index = individual.get_random_bias_index(random_layer)
        # TODO discuss new bias value's range, update method documentation
        random_layer.biases[random_index] = np.random.uniform(-3, 3)

    @staticmethod
    def change_weight_sign_mutation(individual: "ChildIndividual") -> None:
        """
        Changes a sign of randomly chosen weight in the neural network.

        Args:
            individual (ChildIndividual): Individual (neural network) to be modified.
        """
        random_layer = individual.get_random_layer()
        random_index = individual.get_random_weight_index(random_layer)
        random_layer.weights[random_index] = -random_layer.weights[random_index]

    @staticmethod
    def multiply_neuron_weights_mutation(individual: "ChildIndividual") -> None:
        """
        Multiplies all weights for single randomly chosen neuron,
        from randomly chosen layer, by random numbers from the range [0.5, 1.5].
        There is a random multiplier chosen for every single weight.

        Args:
            individual (ChildIndividual): Individual (neural network) to be modified.
        """
        random_layer = individual.get_random_layer()
        random_neuron_index = np.random.randint(0, random_layer.weights.shape[0])
        multipliers = np.random.uniform(
            low=0.1, high=10, size=(random_layer.weights.shape[1])
        )
        random_layer.weights[random_neuron_index] *= multipliers

    @staticmethod
    def random_neuron_weights_mutation(individual: "ChildIndividual") -> None:
        """
        Changes all weights for single randomly chosen neuron from randomly chosen layer.

        Args:
            individual (ChildIndividual): Individual (neural network) to be modified.
        """
        random_layer = individual.get_random_layer()
        random_neuron_index = np.random.randint(0, random_layer.weights.shape[0])
        # TODO decide the range of new random weights
        random_layer.weights[random_neuron_index] = (
            np.random.random((random_layer.weights.shape[1])) * 6 - 3
        )


ChildIndividual.available_mutations = [
    ChildIndividual.random_weight_mutation,
    ChildIndividual.random_bias_mutation,
    ChildIndividual.change_weight_sign_mutation,
    ChildIndividual.multiply_neuron_weights_mutation,
    ChildIndividual.random_neuron_weights_mutation,
]


@dataclass
class AdultIndividual:
    neural_network: NeuralNetwork
    adaptation: float
