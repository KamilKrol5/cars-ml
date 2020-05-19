from typing import List, Tuple, Dict, Callable

import numpy as np

from model.individual import Individual
from model.neural_network import NeuralNetwork


class Neuroevolution:
    """
    genetic algorithm
    """

    _INDIVIDUALS: int = 300
    """
    amount of individuals per generation, minimum 3
    """

    _INDIVIDUALS_FOR_REPRODUCTION: int = 5
    """
    amount of individuals for reproduction per generation, valid values in range from 2 to _INDIVIDUALS - 1
    """

    _MUTATION_CHANCE: float = 0.15
    """
    mutation chance for every child
    """

    _REPRODUCTION_RATE: Tuple[int, int, int] = (5, 3, 1)
    """
    proportional chance for reproduction by respectively:
    1. swapping single weight,
    2. swapping single neuron,
    3. swapping entire layer
    """

    _MUTATION_RATE: Tuple[int, int, int, int, int] = (10, 10, 5, 5, 1)
    """
    proportional chance for mutation by respectively (apply only to children with mutation):
    1. generate random value for single weight,
    2. generate random value for single bias,
    3. change the sign of single weight,
    4. multiply all weights of single neuron by numbers in range 0.5 to 1.5,
    5  random value for all weights of single neuron
    """

    _REPRODUCTION_PROBABILITIES = np.array(_REPRODUCTION_RATE) / sum(_REPRODUCTION_RATE)
    _MUTATION_PROBABILITIES = np.array(_MUTATION_RATE) / sum(_MUTATION_RATE)

    def __init__(self, individuals: List[Individual]) -> None:
        self.individuals: List[Individual] = individuals
        self._individuals_for_reproduction: List[Individual] = []
        self._reproduce: Dict[int, Callable] = {
            0: lambda x, y: Neuroevolution._single_weight_reproduction(x, y),
            1: lambda x, y: Neuroevolution._single_neuron_reproduction(x, y),
            2: lambda x, y: Neuroevolution._entire_layer_reproduction(x, y),
        }
        self._mutate: Dict[int, Callable] = {
            0: lambda x: Neuroevolution._random_weight_mutation(x),
            1: lambda x: Neuroevolution._random_bias_mutation(x),
            2: lambda x: Neuroevolution._change_weight_sign_mutation(x),
            3: lambda x: Neuroevolution._multiply_neuron_weights_mutation(x),
            4: lambda x: Neuroevolution._random_neuron_weights_mutation(x),
        }

    @classmethod
    def init_with_prepared_neural_networks(
        cls, neural_networks: List[NeuralNetwork]
    ) -> "Neuroevolution":
        return Neuroevolution(
            [Individual(neural_network) for neural_network in neural_networks]
        )

    @classmethod
    def init_without_prepared_neural_networks(cls) -> "Neuroevolution":
        return Neuroevolution(
            [Individual(NeuralNetwork()) for i in range(Neuroevolution._INDIVIDUALS)]
        )

    def _ranking(self) -> None:
        self.individuals.sort(
            key=lambda individual: individual.adaptation, reverse=True
        )

    def _selection(self) -> None:
        self._individuals_for_reproduction = self.individuals[
            : Neuroevolution._INDIVIDUALS_FOR_REPRODUCTION
        ]
        self.individuals = self._individuals_for_reproduction[:]

    @staticmethod
    def _single_weight_reproduction(
        mother: NeuralNetwork, father: NeuralNetwork
    ) -> Tuple[NeuralNetwork, NeuralNetwork]:
        # TODO
        return NeuralNetwork(), NeuralNetwork()

    @staticmethod
    def _single_neuron_reproduction(
        mother: NeuralNetwork, father: NeuralNetwork
    ) -> Tuple[NeuralNetwork, NeuralNetwork]:
        # TODO
        return NeuralNetwork(), NeuralNetwork()

    @staticmethod
    def _entire_layer_reproduction(
        mother: NeuralNetwork, father: NeuralNetwork
    ) -> Tuple[NeuralNetwork, NeuralNetwork]:
        # TODO
        return NeuralNetwork(), NeuralNetwork()

    def _reproduction(self) -> None:
        children_remain = (
            Neuroevolution._INDIVIDUALS - Neuroevolution._INDIVIDUALS_FOR_REPRODUCTION
        )
        while children_remain > 0:
            mother, father = np.random.choice(
                self._individuals_for_reproduction, size=2
            )
            while mother == father:
                father = np.random.choice(self._individuals_for_reproduction)
            child1, child2 = self._reproduce[
                np.random.choice(
                    [0, 1, 2], p=Neuroevolution._REPRODUCTION_PROBABILITIES
                )
            ](mother, father)
            self.individuals.append(Individual(child1))
            children_remain -= 1
            if children_remain > 0:
                self.individuals.append(Individual(child2))
                children_remain -= 1

    @staticmethod
    def _random_weight_mutation(child: NeuralNetwork) -> None:
        # TODO
        pass

    @staticmethod
    def _random_bias_mutation(child: NeuralNetwork) -> None:
        # TODO
        pass

    @staticmethod
    def _change_weight_sign_mutation(child: NeuralNetwork) -> None:
        # TODO
        pass

    @staticmethod
    def _multiply_neuron_weights_mutation(child: NeuralNetwork) -> None:
        # TODO
        pass

    @staticmethod
    def _random_neuron_weights_mutation(child: NeuralNetwork) -> None:
        # TODO
        pass

    def _mutation(self) -> None:
        pre_commit_sucks = Neuroevolution._INDIVIDUALS_FOR_REPRODUCTION
        for individual in self.individuals[pre_commit_sucks:]:
            if np.random.choice(
                [True, False],
                p=[
                    Neuroevolution._MUTATION_CHANCE,
                    1 - Neuroevolution._MUTATION_CHANCE,
                ],
            ):
                self._mutate[
                    np.random.choice(
                        [0, 1, 2, 3, 4], p=Neuroevolution._MUTATION_PROBABILITIES
                    )
                ](individual)

    def evolve(self) -> None:
        self._ranking()
        self._selection()
        self._reproduction()
        self._mutation()
