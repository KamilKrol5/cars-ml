from copy import deepcopy
from typing import Tuple, Dict, Callable, List

import numpy as np

from model.neural_network.neural_network import NeuralNetwork, LayerInfo
from model.neuroevolution.individual import Individual


class Neuroevolution:
    """
    Genetic algorithm.
    """

    _INDIVIDUALS: int = 300
    """
    Amount of individuals per generation, minimum 3.
    """

    _GOLDEN_TICKETS: int = 20
    """
    Amount of individuals allowed to reproduce for sure.
    """

    @staticmethod
    def _reproduction_probability(
        individual: Individual, bound_adaptation: int
    ) -> float:
        """
        Probability of reproduction for every given individual
        without golden ticket according to bound adaptation.
        """
        if individual.adaptation is None:
            # this should never happen
            raise Exception(
                "Cannot compute probability of" "individual with undefined adaptation"
            )
        # thanks to mypy
        assert isinstance(individual.adaptation, int)
        return (individual.adaptation / bound_adaptation) ** 3

    _MUTATION_CHANCE: float = 0.15
    """
    Mutation chance for every child.
    """

    _REPRODUCTION_RATE: Tuple[int, int, int] = (5, 3, 1)
    """
    Proportional chance for reproduction by respectively:
    1. swapping single weight,
    2. swapping single neuron,
    3. swapping entire layer.
    """

    _MUTATION_RATE: Tuple[int, int, int, int, int] = (10, 10, 5, 5, 1)
    """
    Proportional chance for mutation by respectively (apply only to children with mutation):
    1. generate random value for single weight,
    2. generate random value for single bias,
    3. change the sign of single weight,
    4. multiply all weights of single neuron by numbers in range 0.5 to 1.5,
    5  random value for all weights of single neuron.
    """

    _REPRODUCTION_PROBABILITIES = np.array(_REPRODUCTION_RATE) / sum(_REPRODUCTION_RATE)
    _MUTATION_PROBABILITIES = np.array(_MUTATION_RATE) / sum(_MUTATION_RATE)

    def __init__(
        self,
        individuals: List[Individual],
        layers_infos: List[LayerInfo],
        output_neurons: int,
    ) -> None:
        self.individuals: List[Individual] = individuals
        self._layers_infos = layers_infos
        self._output_neurons = output_neurons
        self._parents: List[Individual] = []
        self._reproduce: Dict[int, Callable] = {
            0: self._weight_swap_reproduction,
            1: self._neuron_swap_reproduction,
            2: self._layer_swap_reproduction,
        }
        self._mutate: Dict[int, Callable] = {
            0: self._random_weight_mutation,
            1: self._random_bias_mutation,
            2: self._change_weight_sign_mutation,
            3: self._multiply_neuron_weights_mutation,
            4: self._random_neuron_weights_mutation,
        }

    @classmethod
    def init_with_neural_network_info(
        cls, layers_infos: List[LayerInfo], output_neurons: int
    ) -> "Neuroevolution":
        return cls(
            [
                Individual(NeuralNetwork(layers_infos, output_neurons))
                for _ in range(cls._INDIVIDUALS)
            ],
            layers_infos,
            output_neurons,
        )

    def _ranking(self) -> None:
        self.individuals.sort(
            key=lambda individual: individual.adaptation, reverse=True
        )

    def _selection(self) -> List[Individual]:
        bound_adaptation = self.individuals[self._GOLDEN_TICKETS].adaptation
        if bound_adaptation is None:
            # this should never happen
            raise Exception("Bound adaptation cannot be undefined")
        parents = self.individuals[: self._GOLDEN_TICKETS]
        # i would prefer to commit this line
        # for individual in self.individuals[self._GOLDEN_TICKETS : self._INDIVIDUALS]:
        # but pre-commit sucks :<
        for i in range(self._GOLDEN_TICKETS, self._INDIVIDUALS):
            if np.random.rand() < self._reproduction_probability(
                self.individuals[i], bound_adaptation
            ):
                parents.append(self.individuals[i])
        return parents

    @staticmethod
    def _weight_swap_reproduction(
        father1: Individual, father2: Individual, weights_to_be_swapped: int = 1
    ) -> Tuple[Individual, Individual]:
        """
        Return two new individuals (children). They are created by
        swapping one or more parents's weights.

        :param father1: First parent
        :param father2: Second parent
        :return: Tuple consisting two new individuals.
        """
        child_1 = deepcopy(father1)
        child_2 = deepcopy(father2)
        for _ in range(weights_to_be_swapped):
            random_layer_index = child_1.get_random_layer_index()
            child_1_rand_layer = child_1.neural_network.hidden_layers[
                random_layer_index
            ]
            child_2_rand_layer = child_2.neural_network.hidden_layers[
                random_layer_index
            ]
            weight_index = Individual.get_random_weight_index(child_1_rand_layer)
            (
                child_1_rand_layer.weights[weight_index],
                child_2_rand_layer.weights[weight_index],
            ) = (
                child_2_rand_layer.weights[weight_index],
                child_1_rand_layer.weights[weight_index],
            )
        return child_1, child_2

    @staticmethod
    def _neuron_swap_reproduction(
        mother1: Individual, mother2: Individual, neurons_to_swap: int = 1
    ) -> Tuple[Individual, Individual]:
        """
        Return two new individuals (children). They are created by
        swapping one or more randomly chosen neurons.

        :param mother1: First parent
        :param mother2: Second parent
        :return: Tuple consisting two new individuals.
        """
        child_1 = deepcopy(mother1)
        child_2 = deepcopy(mother2)
        for _ in range(neurons_to_swap):
            random_layer_index = child_1.get_random_layer_index()
            child_1_rand_layer_w = child_1.neural_network.hidden_layers[
                random_layer_index
            ].weights
            child_2_rand_layer_w = child_2.neural_network.hidden_layers[
                random_layer_index
            ].weights
            neuron_index = np.random.randint(0, child_2_rand_layer_w.shape[0])
            child_1_rand_layer_w[neuron_index], child_2_rand_layer_w[neuron_index] = (
                child_2_rand_layer_w[neuron_index],
                child_1_rand_layer_w[neuron_index].copy(),
            )
        return child_1, child_2

    @staticmethod
    def _layer_swap_reproduction(
        mother: Individual, father: Individual, layers_to_swap: int = 1
    ) -> Tuple[Individual, Individual]:
        """
        Return two new individuals (children). They are created by
        swapping one or more randomly chosen layers.
        All data from one layer (weights and biases) is swapped
        with all data from another layer.

        :param mother: First parent
        :param father: Second parent
        :return: Tuple consisting two new individuals.
        """
        child_1 = deepcopy(mother)
        child_2 = deepcopy(father)
        for _ in range(layers_to_swap):
            layer_index = child_1.get_random_layer_index()
            child_1_rand_layers = child_1.neural_network.hidden_layers
            child_2_rand_layers = child_2.neural_network.hidden_layers
            child_2_rand_layers[layer_index], child_1_rand_layers[layer_index] = (
                child_1_rand_layers[layer_index],
                child_2_rand_layers[layer_index],
            )
        return child_1, child_2

    def _reproduction(self, parents: List[Individual]) -> List[Individual]:
        children_remain = self._INDIVIDUALS - len(parents)
        children = []
        while children_remain > 0:
            mother, father = np.random.choice(parents, size=2)
            while mother == father:
                father = np.random.choice(parents)
            daughter, son = self._reproduce[
                np.random.choice(
                    list(self._reproduce.keys()), p=self._REPRODUCTION_PROBABILITIES
                )
            ](mother, father)
            children.append(Individual(daughter))
            children_remain -= 1
            if children_remain > 0:
                children.append(Individual(son))
                children_remain -= 1
        return children

    @staticmethod
    def _random_weight_mutation(individual: Individual) -> None:
        """
        Changes a randomly chosen weight in the neural network to random value from range [-1, 1)

        :param individual: Individual (neural network) to be modified.
        :return: None
        """
        random_layer = individual.get_random_layer()
        random_index = individual.get_random_weight_index(random_layer)
        # TODO discuss new weight value's range, update method documentation
        random_layer.weights[random_index] = np.random.uniform(-1, 1)

    @staticmethod
    def _random_bias_mutation(individual: Individual) -> None:
        """
        Changes a randomly chosen bias in the neural network to random value from range [-1, 1)

        :param individual: Individual (neural network) to be modified.
        :return: None
        """
        random_layer = individual.get_random_layer()
        random_index = individual.get_random_bias_index(random_layer)
        # TODO discuss new bias value's range, update method documentation
        random_layer.biases[random_index] = np.random.uniform(-1, 1)
        pass

    @staticmethod
    def _change_weight_sign_mutation(individual: Individual) -> None:
        """
        Changes a sign of randomly chosen weight in the neural network.

        :param individual: Individual (neural network) to be modified.
        :return: None
        """
        random_layer = individual.get_random_layer()
        random_index = individual.get_random_weight_index(random_layer)
        random_layer.weights[random_index] = -random_layer.weights[random_index]

    @staticmethod
    def _multiply_neuron_weights_mutation(individual: Individual) -> None:
        """
        Multiplies all weights for single randomly chosen neuron,
        from randomly chosen layer, by random numbers from the range [0.5, 1.5].
        There is a random multiplier chosen for every single weight.
        :param individual: Individual to be changed.
        :return: None
        """
        random_layer = individual.get_random_layer()
        random_neuron_index = np.random.randint(0, random_layer.weights.shape[0])
        multipliers = np.random.uniform(
            low=0.5, high=1.5, size=(random_layer.weights.shape[1])
        )
        random_layer.weights[random_neuron_index] *= multipliers

    @staticmethod
    def _random_neuron_weights_mutation(individual: Individual) -> None:
        """
        Changes all weights for single randomly chosen neuron from randomly chosen layer.
        :param individual: Individual to be changed.
        :return: None
        """
        random_layer = individual.get_random_layer()
        random_neuron_index = np.random.randint(0, random_layer.weights.shape[0])
        # TODO decide the range of new random weights
        random_layer.weights[random_neuron_index] = np.random.random(
            (random_layer.weights.shape[1])
        )

    def _mutation(self, individuals: List[Individual]) -> List[Individual]:
        for individual in individuals:
            if np.random.rand() < self._MUTATION_CHANCE:
                self._mutate[
                    np.random.choice(
                        list(self._mutate.keys()), p=self._MUTATION_PROBABILITIES
                    )
                ](individual)
        return individuals

    def evolve(self) -> None:
        self._ranking()
        parents = self._selection()
        children = self._reproduction(parents)
        self.individuals = parents + self._mutation(children)
