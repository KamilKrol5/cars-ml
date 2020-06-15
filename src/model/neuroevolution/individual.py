from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, ClassVar, List, Callable, cast

import numpy as np

from model.neural_network.neural_network import NeuralNetwork, Layer
from utils import swap_same_index, swap_numpy_same_index


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
        random_layer.weights[random_neuron_index] = np.random.uniform(
            -3, 3, random_layer.weights.shape[1]
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

    available_reproductions: ClassVar[
        List[
            Callable[
                ["AdultIndividual", "AdultIndividual"],
                Tuple[ChildIndividual, ChildIndividual],
            ]
        ]
    ]

    @staticmethod
    def weight_swap_reproduction(
        father1: "AdultIndividual",
        father2: "AdultIndividual",
        weights_to_be_swapped: int = 1,
    ) -> Tuple[ChildIndividual, ChildIndividual]:
        """
        Return two new individuals (children). They are created by
        swapping one or more parents's weights.
        Args:
            father1 (AdultIndividual): First parent
            father2 (AdultIndividual): Second parent
            weights_to_be_swapped (int): Number of weights to be swapped
        Returns:
            Tuple[ChildIndividual, ChildIndividual]: Tuple consisting two new individuals.
        """
        child_1 = ChildIndividual(deepcopy(father1.neural_network))
        child_2 = ChildIndividual(deepcopy(father2.neural_network))
        for _ in range(weights_to_be_swapped):
            random_layer_index = child_1.get_random_layer_index()
            child_1_rand_layer = child_1.neural_network.hidden_layers[
                random_layer_index
            ]
            child_2_rand_layer = child_2.neural_network.hidden_layers[
                random_layer_index
            ]
            weight_index = ChildIndividual.get_random_weight_index(child_1_rand_layer)
            swap_numpy_same_index(
                child_1_rand_layer.weights, child_2_rand_layer.weights, weight_index
            )
        return child_1, child_2

    @staticmethod
    def neuron_swap_reproduction(
        mother1: "AdultIndividual", mother2: "AdultIndividual", neurons_to_swap: int = 1
    ) -> Tuple[ChildIndividual, ChildIndividual]:
        """
        Return two new individuals (children). They are created by
        swapping one or more randomly chosen neurons.
        Args:
            mother1 (AdultIndividual): First parent
            mother2 (AdultIndividual): Second parent
            neurons_to_swap (int): Number of neurons to be swapped
        Returns:
            Tuple[ChildIndividual, ChildIndividual]: Tuple consisting two new individuals.
        """
        child_1 = ChildIndividual(deepcopy(mother1.neural_network))
        child_2 = ChildIndividual(deepcopy(mother2.neural_network))
        for _ in range(neurons_to_swap):
            random_layer_index = child_1.get_random_layer_index()
            child_1_rand_layer_w = child_1.neural_network.hidden_layers[
                random_layer_index
            ].weights
            child_2_rand_layer_w = child_2.neural_network.hidden_layers[
                random_layer_index
            ].weights
            neuron_index = np.random.randint(0, child_2_rand_layer_w.shape[0])
            swap_numpy_same_index(
                child_1_rand_layer_w, child_2_rand_layer_w, neuron_index
            )
        return child_1, child_2

    @staticmethod
    def layer_swap_reproduction(
        mother: "AdultIndividual", father: "AdultIndividual", layers_to_swap: int = 1
    ) -> Tuple[ChildIndividual, ChildIndividual]:
        """
        Return two new individuals (children). They are created by
        swapping one or more randomly chosen layers.
        All data from one layer (weights and biases) is swapped
        with all data from another layer.
        mother (Individual): First parent
        father (Individual): Second parent
        layers_to_swap (int): Number of layers to be swapped
        Returns:
            Tuple[ChildIndividual, ChildIndividual]: Tuple consisting two new individuals.
        """
        child_1 = ChildIndividual(deepcopy(mother.neural_network))
        child_2 = ChildIndividual(deepcopy(father.neural_network))
        for _ in range(layers_to_swap):
            layer_index = child_1.get_random_layer_index()
            child_1_rand_layers = child_1.neural_network.hidden_layers
            child_2_rand_layers = child_2.neural_network.hidden_layers
            swap_same_index(child_1_rand_layers, child_2_rand_layers, layer_index)
        return child_1, child_2


AdultIndividual.available_reproductions = [
    AdultIndividual.weight_swap_reproduction,
    AdultIndividual.neuron_swap_reproduction,
    AdultIndividual.layer_swap_reproduction,
]
