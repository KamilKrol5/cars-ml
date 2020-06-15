from copy import deepcopy
from typing import Tuple, List, Generator, TypeVar, Any

import numpy as np

import utils
from model.environment.environment import Environment
from model.neural_network.neural_network import NeuralNetwork, LayerInfo
from model.neuroevolution.individual import AdultIndividual, ChildIndividual

T = TypeVar("T")


class Neuroevolution:
    """
    Genetic algorithm.
    """

    _iteration_counter = 1
    _INDIVIDUALS: int = 70
    """
    Amount of individuals per generation, minimum 3.
    """

    _MUTATION_RATE: Tuple[int, int, int, int, int] = (10, 3, 0, 5, 1)
    """
    Proportional chance for mutation by respectively (apply only to children with mutation):
    1. generate random value for single weight, <-- makes diversity
    2. generate random value for single bias, <-- makes diversity
    3. change the sign of single weight, <-- very poor, useless
    4. multiply all weights of single neuron by numbers in range 0.1 to 10,
    ^^ maintains individuals value, enhance good individuals
    5  random value for all weights of single neuron.
    ^^ makes strong diversity, useful in case of stagnancy
    """

    _MUTATION_PROBABILITIES = np.array(_MUTATION_RATE) / sum(_MUTATION_RATE)

    def __init__(self, networks: List[NeuralNetwork]) -> None:
        self._new_generation: List[ChildIndividual] = [
            ChildIndividual(nn) for nn in networks[:-1]
        ]
        #  one parent is needed in first iteration
        self._parent: AdultIndividual = AdultIndividual(networks[-1], -1)

    @classmethod
    def init_with_neural_network_info(
        cls, layers_infos: List[LayerInfo], output_neurons: int,
    ) -> "Neuroevolution":
        return cls(
            [
                NeuralNetwork(layers_infos, output_neurons)
                for _ in range(cls._INDIVIDUALS)
            ],
        )

    @staticmethod
    def _selection(individuals: List[AdultIndividual]) -> AdultIndividual:
        return max(individuals, key=lambda individual: individual.adaptation)

    def _mutation(self, parent: AdultIndividual) -> List[ChildIndividual]:
        children: List[ChildIndividual] = [
            ChildIndividual(deepcopy(parent.neural_network))
            for _ in range(self._INDIVIDUALS - 1)
        ]
        for child in children:
            np.random.choice(
                ChildIndividual.available_mutations, p=self._MUTATION_PROBABILITIES
            )(child)
        return children

    def generate_evolution(
        self, environment: Environment[T, Any], with_parents: bool
    ) -> Generator[None, T, None]:
        """
        Evaluates individuals from current generation and
        produce new generation out of the best.

        Args:
            environment (Environment): environment
            with_parents (bool): Indicates whether parents of current generation
                should take part in race.
        """
        print(f"Iteration: {self._iteration_counter}")
        self._iteration_counter += 1
        network_groups = {
            "children": [child.neural_network for child in self._new_generation]
        }
        if with_parents:
            network_groups["parents"] = [self._parent.neural_network]

        adaptations = yield from environment.generate_adaptations(network_groups)
        new_individuals = [
            AdultIndividual(child.neural_network, adaptation)
            for child, adaptation in zip(self._new_generation, adaptations["children"])
        ]
        self._parent = self._selection(new_individuals + [self._parent])
        print(f"Best adaptation: {self._parent.adaptation}")
        self._new_generation = self._mutation(self._parent)

    def evolve(self, environment: Environment[None, Any], with_parents: bool) -> None:
        return utils.generator_value(self.generate_evolution(environment, with_parents))
