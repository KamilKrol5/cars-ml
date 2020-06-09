import sys
from typing import Tuple, List, Generator, TypeVar

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

    _INDIVIDUALS: int = 10
    """
    Amount of individuals per generation, minimum 3.
    """

    _GOLDEN_TICKETS: int = 2
    """
    Amount of individuals allowed to reproduce for sure.
    """

    _MAX_PARENTS: int = _INDIVIDUALS // 2
    """
    Maximal parents count.
    """

    @staticmethod
    def _reproduction_probability(
        individual: AdultIndividual, bound_adaptation: float
    ) -> float:
        """
        Probability of reproduction for every given individual
        without golden ticket according to bound adaptation.
        """
        return (individual.adaptation / bound_adaptation) ** 3

    _MUTATION_CHANCE: float = 0.30
    """
    Mutation chance for every child.
    """

    _REPRODUCTION_RATE: Tuple[int, int, int] = (1, 1, 1)
    """
    Proportional chance for reproduction by respectively:
    1. swapping single weight,
    2. swapping single neuron,
    3. swapping entire layer.
    """

    _MUTATION_RATE: Tuple[int, int, int, int, int] = (7, 7, 7, 7, 2)
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

    def __init__(self, networks: List[NeuralNetwork]) -> None:
        self._new_generation: List[ChildIndividual] = [
            ChildIndividual(nn) for nn in networks
        ]
        self.individuals: List[AdultIndividual] = []
        self._parents: List[AdultIndividual] = []

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

    def _sort_individuals_and_kill_unnecessary(self) -> None:
        self.individuals.sort(
            key=lambda individual: individual.adaptation, reverse=True
        )
        self.individuals = self.individuals[: self._INDIVIDUALS]

    def _selection(self) -> List[AdultIndividual]:
        bound_adaptation = self.individuals[0].adaptation
        parents = self.individuals[: self._GOLDEN_TICKETS]
        for individual in self.individuals[self._GOLDEN_TICKETS :]:
            if np.random.rand() < self._reproduction_probability(
                individual, bound_adaptation
            ):
                parents.append(individual)
                if len(parents) == self._MAX_PARENTS:
                    break

        print(
            f"Individuals to reproduce: {len(parents)} out of {self._INDIVIDUALS}",
            file=sys.stderr,
        )
        return parents

    def _reproduction(self, parents: List[AdultIndividual]) -> List[ChildIndividual]:
        children_to_make = self._INDIVIDUALS - len(parents)
        children: List[ChildIndividual] = []
        while len(children) < children_to_make:
            mother, father = np.random.choice(parents, size=2)
            while mother == father:
                father = np.random.choice(parents)
            daughter, son = np.random.choice(
                AdultIndividual.available_reproductions,
                p=self._REPRODUCTION_PROBABILITIES,
            )(mother, father)
            children.append(daughter)
            if len(children) < children_to_make:
                children.append(son)
        return children

    def _mutation(self, individuals: List[ChildIndividual]) -> List[ChildIndividual]:
        for individual in individuals:
            if np.random.rand() < self._MUTATION_CHANCE:
                np.random.choice(
                    ChildIndividual.available_mutations, p=self._MUTATION_PROBABILITIES
                )(individual)
        return individuals

    def generate_evolution(
        self, environment: Environment[T], with_parents: bool
    ) -> Generator[None, T, None]:
        """
        Evaluates individuals from current generation and
        produce new generation out of the best.

        Args:
            environment (Environment): environment
            with_parents (bool): Indicates whether parents of current generation
                should take part in race.
        """
        network_groups = {
            "children": [child.neural_network for child in self._new_generation]
        }
        if with_parents:
            network_groups["parents"] = [
                parent.neural_network for parent in self._parents
            ]

        print(f"New generation size: {len(self._new_generation)}")
        adaptations = yield from environment.generate_adaptations(network_groups)

        new_individuals = [
            AdultIndividual(child.neural_network, adaptation)
            for child, adaptation in zip(self._new_generation, adaptations["children"])
        ]
        self.individuals.extend(new_individuals)
        self._sort_individuals_and_kill_unnecessary()
        self._parents = self._selection()
        children: List[ChildIndividual] = self._reproduction(self._parents)
        self._new_generation = self._mutation(children)

    def evolve(self, environment: Environment[None], with_parents: bool) -> None:
        return utils.generator_value(self.generate_evolution(environment, with_parents))
