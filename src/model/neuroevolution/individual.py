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
    def get_random_weight_indexes(
        neural_network_layer: Layer, indexes_count: int
    ) -> Tuple[Tuple[int, ...], ...]:
        return tuple(
            np.random.randint(neural_network_layer.weights.shape, size=indexes_count)
        )

    @staticmethod
    def get_random_bias_index(neural_network_layer: Layer) -> int:
        return cast(int, np.random.randint(neural_network_layer.biases.shape[0]))

    @staticmethod
    def get_random_bias_indexes(
        neural_network_layer: Layer, indexes_count: int
    ) -> Tuple[int, ...]:
        return cast(
            Tuple[int, ...],
            np.random.randint(neural_network_layer.biases.shape[0], size=indexes_count),
        )

    @staticmethod
    def random_weight_mutation(
        individual: "ChildIndividual", weights_to_mutate: int = 1
    ) -> None:
        """
        Changes a randomly chosen weight in the neural network.

        Args:
            individual (ChildIndividual): Individual (neural network) to be modified.
            weights_to_mutate (int): Number of random choices of weight to be mutated.
        """
        random_layer = individual.get_random_layer()
        random_indexes = individual.get_random_weight_indexes(
            random_layer, weights_to_mutate
        )
        # TODO discuss new weight value's range, update method documentation
        random_layer.weights.put(
            random_indexes, np.random.uniform(-1, 1, size=weights_to_mutate)
        )

    @staticmethod
    def random_bias_mutation(
        individual: "ChildIndividual", biases_to_mutate: int = 1
    ) -> None:
        """
        Changes a randomly chosen bias in the neural network.

        Args:
            individual (ChildIndividual): Individual (neural network) to be modified.
            biases_to_mutate (int): Number of random choices of bias to be mutated.
        """
        random_layer = individual.get_random_layer()
        random_indexes = individual.get_random_bias_indexes(
            random_layer, biases_to_mutate
        )
        # TODO discuss new bias value's range, update method documentation
        random_layer.biases.put(
            random_indexes, np.random.uniform(-1, 1, size=biases_to_mutate)
        )

    @staticmethod
    def biases_shuffle_mutation(individual: "ChildIndividual") -> None:
        """
        Shuffles all biases of randomly chosen layer.

        Args:
            individual (ChildIndividual): Individual (neural network) to be modified.
        """
        random_layer = individual.get_random_layer()
        np.random.shuffle(random_layer.biases)

    @staticmethod
    def change_weight_sign_mutation(individual: "ChildIndividual") -> None:
        """
        Changes the sign of a randomly chosen weight in the neural network.

        Args:
            individual (ChildIndividual): Individual (neural network) to be modified.
        """
        random_layer = individual.get_random_layer()
        random_index = individual.get_random_weight_index(random_layer)
        random_layer.weights[random_index] = -random_layer.weights[random_index]

    @staticmethod
    def multiply_neuron_weights_mutation(individual: "ChildIndividual") -> None:
        """
        Multiplies all weights for a single randomly chosen neuron.

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
        Changes all weights for a single randomly chosen neuron.

        Args:
            individual (ChildIndividual): Individual (neural network) to be modified.
        """
        random_layer = individual.get_random_layer()
        random_neuron_index = np.random.randint(0, random_layer.weights.shape[0])
        # TODO decide the range of new random weights
        random_layer.weights[random_neuron_index] = np.random.uniform(
            -3, 3, random_layer.weights.shape[1]
        )

    @staticmethod
    def neuron_weights_shuffle_mutation(individual: "ChildIndividual") -> None:
        """
        Shuffles all weights for single randomly chosen neuron from randomly chosen layer.

        Args:
            individual (ChildIndividual): Individual (neural network) to be modified.
        """
        random_layer = individual.get_random_layer()
        random_neuron_index = np.random.randint(0, random_layer.weights.shape[0])
        np.random.shuffle(random_layer.weights[random_neuron_index])

    @staticmethod
    def multiply_shuffle_neuron_weights_mutation(individual: "ChildIndividual") -> None:
        """
        Multiplies all weights for single randomly chosen neuron,
        from randomly chosen layer, by random numbers from the range [0.5, 1.5].
        Then shuffles these weights.
        There is a random multiplier chosen for every single weight.

        Args:
            individual (ChildIndividual): Individual (neural network) to be modified.
        """
        ChildIndividual.multiply_neuron_weights_mutation(individual)
        ChildIndividual.neuron_weights_shuffle_mutation(individual)


ChildIndividual.available_mutations = [
    ChildIndividual.random_weight_mutation,
    ChildIndividual.random_bias_mutation,
    ChildIndividual.biases_shuffle_mutation,
    ChildIndividual.change_weight_sign_mutation,
    ChildIndividual.multiply_neuron_weights_mutation,
    ChildIndividual.random_neuron_weights_mutation,
    ChildIndividual.neuron_weights_shuffle_mutation,
    ChildIndividual.multiply_shuffle_neuron_weights_mutation,
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
        Performs reproduction by swapping weights of parents.

        Weights to be swapped are chosen randomly.

        Args:
            father1 (AdultIndividual): First parent.
            father2 (AdultIndividual): Second parent.
            weights_to_be_swapped (int): Number of weights to be swapped.

        Returns:
            Tuple[ChildIndividual, ChildIndividual]: Two new individuals.
        """
        child_1 = ChildIndividual(deepcopy(father1.neural_network))
        child_2 = ChildIndividual(deepcopy(father2.neural_network))

        for _ in range(weights_to_be_swapped):
            (
                child_1_rand_layer,
                child_2_rand_layer,
            ) = AdultIndividual._get_random_layers_from_children(child_1, child_2)
            weight_index = ChildIndividual.get_random_weight_index(child_1_rand_layer)
            swap_numpy_same_index(
                child_1_rand_layer.weights, child_2_rand_layer.weights, weight_index
            )

        return child_1, child_2

    @staticmethod
    def bias_swap_reproduction(
        father1: "AdultIndividual",
        father2: "AdultIndividual",
        biases_to_be_swapped: int = 1,
    ) -> Tuple[ChildIndividual, ChildIndividual]:
        """
        Return two new individuals (children). They are created by
        swapping one or more biases in one randomly chosen layer from parents.

        Args:
            father1 (AdultIndividual): First parent
            father2 (AdultIndividual): Second parent
            biases_to_be_swapped (int): Number of biases in single layer to be swapped

        Returns:
            Tuple[ChildIndividual, ChildIndividual]: Tuple consisting two new individuals.
        """
        child_1 = ChildIndividual(deepcopy(father1.neural_network))
        child_2 = ChildIndividual(deepcopy(father2.neural_network))
        for _ in range(biases_to_be_swapped):
            (
                child_1_rand_layer,
                child_2_rand_layer,
            ) = AdultIndividual._get_random_layers_from_children(child_1, child_2)
            bias_index = ChildIndividual.get_random_bias_index(child_1_rand_layer)
            swap_numpy_same_index(
                child_1_rand_layer.biases, child_2_rand_layer.biases, bias_index
            )
        return child_1, child_2

    @staticmethod
    def neuron_swap_reproduction(
        mother1: "AdultIndividual", mother2: "AdultIndividual", neurons_to_swap: int = 1
    ) -> Tuple[ChildIndividual, ChildIndividual]:
        """
        Performs reproduction by swapping neurons of parents.

        Neurons to be swapped are chosen randomly.

        Args:
            mother1 (AdultIndividual): First parent.
            mother2 (AdultIndividual): Second parent.
            neurons_to_swap (int): Number of neurons to be swapped.

        Returns:
            Tuple[ChildIndividual, ChildIndividual]: Two new individuals.
        """
        child_1 = ChildIndividual(deepcopy(mother1.neural_network))
        child_2 = ChildIndividual(deepcopy(mother2.neural_network))

        for _ in range(neurons_to_swap):
            ch1_layer, ch2_layer = AdultIndividual._get_random_layers_from_children(
                child_1, child_2
            )
            neuron_index = np.random.randint(0, ch2_layer.weights.shape[0])
            swap_numpy_same_index(ch1_layer.weights, ch2_layer.weights, neuron_index)
        return child_1, child_2

    @staticmethod
    def layer_swap_reproduction(
        mother: "AdultIndividual", father: "AdultIndividual", layers_to_swap: int = 1
    ) -> Tuple[ChildIndividual, ChildIndividual]:
        """
        Performs reproduction by swapping layers of parents.

        Layers to be swapped are chosen randomly.
        Entire parents' layers (weights and biases) are swapped.

        Args:
            mother (Individual): First parent.
            father (Individual): Second parent.
            layers_to_swap (int): Number of layers to be swapped.

        Returns:
            Tuple[ChildIndividual, ChildIndividual]: Two new individuals.
        """
        child_1 = ChildIndividual(deepcopy(mother.neural_network))
        child_2 = ChildIndividual(deepcopy(father.neural_network))

        for _ in range(layers_to_swap):
            layer_index = child_1.get_random_layer_index()
            child_1_rand_layers = child_1.neural_network.hidden_layers
            child_2_rand_layers = child_2.neural_network.hidden_layers
            swap_same_index(child_1_rand_layers, child_2_rand_layers, layer_index)

        return child_1, child_2

    @staticmethod
    def _get_random_layers_from_children(
        *children: ChildIndividual,
    ) -> Tuple[Layer, ...]:
        """
        Choose a random layer index and returns
        tuple containing provided children's layers at this index.

        Args:
            *children (ChildIndividual): children to get layers from

        Returns:
            Tuple[Layer, ...]: tuple consisting of provided children's layers
        """
        if children:
            layers: List[Layer] = []
            random_layer_index = children[0].get_random_layer_index()
            for child in children:
                child_rand_layer = child.neural_network.hidden_layers[
                    random_layer_index
                ]
                layers.append(child_rand_layer)
            return tuple(layers)
        else:
            return ()


AdultIndividual.available_reproductions = [
    AdultIndividual.weight_swap_reproduction,
    AdultIndividual.bias_swap_reproduction,
    AdultIndividual.neuron_swap_reproduction,
    AdultIndividual.layer_swap_reproduction,
]
