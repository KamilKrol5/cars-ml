from typing import List, Tuple, Dict, Callable

import numpy as np

from model.individual import Individual
from model.neural_network import NeuralNetwork


class Neuroevolution:
    """
    amount of individuals per generation, minimum 3
    """

    __INDIVIDUALS: int = 300

    """
    amount of individuals for reproduction per generation, valid values in range from 2 to __INDIVIDUALS - 1
    """
    __INDIVIDUALS_FOR_REPRODUCTION: int = 5

    """
    mutation chance for every child
    """
    __MUTATION_CHANCE: float = 0.15

    """
    proportional chance for reproduction by respectively:
    1. swapping single weight,
    2. swapping single neuron,
    3. swapping entire layer
    """
    __REPRODUCTION_RATE: List[int] = [5, 3, 1]

    """
    proportional chance for mutation by respectively (applied only for child with mutation):
    1. generate random value for single weight,
    2. generate random value for single bias,
    3. change the sign of single weight,
    4. multiply all weights of single neuron by numbers in range 0.5 to 1.5,
    5  random value for all weights of single neuron
    """
    __MUTATION_RATE: List[int] = [10, 10, 5, 5, 1]

    __REPRODUCTION_PROBABILITIES = np.array(__REPRODUCTION_RATE) / sum(
        __REPRODUCTION_RATE
    )
    __MUTATION__PROBABILITIES = np.array(__MUTATION_RATE) / sum(__MUTATION_RATE)

    def __init__(self, individuals: List[Individual]) -> None:
        self.individuals: List[Individual] = individuals
        self.__individuals_for_reproduction: List[Individual] = []
        self.__reproduce: Dict[int, Callable] = {
            0: lambda x, y: Neuroevolution.__single_weight_reproduction(x, y),
            1: lambda x, y: Neuroevolution.__single_neuron_reproduction(x, y),
            2: lambda x, y: Neuroevolution.__entire_layer_reproduction(x, y),
        }
        self.__mutate: Dict[int, Callable] = {
            0: lambda x: Neuroevolution.__random_weight_mutation(x),
            1: lambda x: Neuroevolution.__random_bias_mutation(x),
            2: lambda x: Neuroevolution.__change_weight_sign_mutation(x),
            3: lambda x: Neuroevolution.__multiply_neuron_weights_mutation(x),
            4: lambda x: Neuroevolution.__random_neuron_weights_mutation(x),
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
            [Individual(NeuralNetwork()) for i in range(Neuroevolution.__INDIVIDUALS)]
        )

    def __ranking(self) -> None:
        self.individuals.sort(
            key=lambda individual: individual.adaptation, reverse=True
        )

    def __selection(self) -> None:
        self.__individuals_for_reproduction = self.individuals[
            : Neuroevolution.__INDIVIDUALS_FOR_REPRODUCTION
        ]
        self.individuals = self.__individuals_for_reproduction[:]

    @staticmethod
    def __single_weight_reproduction(
        mother: NeuralNetwork, father: NeuralNetwork
    ) -> Tuple[NeuralNetwork, NeuralNetwork]:
        # TODO
        return NeuralNetwork(), NeuralNetwork()

    @staticmethod
    def __single_neuron_reproduction(
        mother: NeuralNetwork, father: NeuralNetwork
    ) -> Tuple[NeuralNetwork, NeuralNetwork]:
        # TODO
        return NeuralNetwork(), NeuralNetwork()

    @staticmethod
    def __entire_layer_reproduction(
        mother: NeuralNetwork, father: NeuralNetwork
    ) -> Tuple[NeuralNetwork, NeuralNetwork]:
        # TODO
        return NeuralNetwork(), NeuralNetwork()

    def __reproduction(self) -> None:
        children_remain = (
            Neuroevolution.__INDIVIDUALS - Neuroevolution.__INDIVIDUALS_FOR_REPRODUCTION
        )
        while children_remain > 0:
            mother, father = np.random.choice(
                self.__individuals_for_reproduction, size=2
            )
            while mother == father:
                father = np.random.choice(self.__individuals_for_reproduction)
            reproduce = self.__reproduce.get(
                np.random.choice(
                    [0, 1, 2], p=Neuroevolution.__REPRODUCTION_PROBABILITIES
                )
            )
            assert callable(reproduce)
            child1, child2 = reproduce(mother, father)
            self.individuals.append(Individual(child1))
            children_remain -= 1
            if children_remain > 0:
                self.individuals.append(Individual(child2))
                children_remain -= 1

    @staticmethod
    def __random_weight_mutation(child: NeuralNetwork) -> None:
        # TODO
        pass

    @staticmethod
    def __random_bias_mutation(child: NeuralNetwork) -> None:
        # TODO
        pass

    @staticmethod
    def __change_weight_sign_mutation(child: NeuralNetwork) -> None:
        # TODO
        pass

    @staticmethod
    def __multiply_neuron_weights_mutation(child: NeuralNetwork) -> None:
        # TODO
        pass

    @staticmethod
    def __random_neuron_weights_mutation(child: NeuralNetwork) -> None:
        # TODO
        pass

    def __mutation(self) -> None:
        pre_commit_sucks = Neuroevolution.__INDIVIDUALS_FOR_REPRODUCTION
        for individual in self.individuals[pre_commit_sucks:]:
            if np.random.choice(
                [True, False],
                p=[
                    Neuroevolution.__MUTATION_CHANCE,
                    1 - Neuroevolution.__MUTATION_CHANCE,
                ],
            ):
                mutate = self.__mutate.get(
                    np.random.choice(
                        [0, 1, 2, 3, 4], p=Neuroevolution.__MUTATION__PROBABILITIES
                    )
                )
                assert callable(mutate)
                mutate(individual)

    def evolve(self) -> None:
        self.__ranking()
        self.__selection()
        self.__reproduction()
        self.__mutation()
