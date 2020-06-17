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
    """
    Iteration number.
    """

    _INDIVIDUALS: int = 70
    """
    Amount of individuals per generation.
    """

    _GOLDEN_TICKETS: int = 1
    """
    Amount of individuals allowed to reproduce for sure.
    """

    _MAX_PARENTS: int = 1
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

    _MUTATION_CHANCE: float = 1.0
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
                if len(parents) >= self._MAX_PARENTS:
                    break

        return parents

    def _reproduction(self, parents: List[AdultIndividual]) -> List[ChildIndividual]:
        children_to_make = self._INDIVIDUALS - len(parents)
        children: List[ChildIndividual] = []

        while len(children) < children_to_make:
            mother, father = np.random.choice(parents, size=2)
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
        self, environment: Environment[T, Any], with_parents: bool
    ) -> Generator[None, T, None]:
        """
        Executes one step of evolution.

        Evaluates individuals from current generation and
        produces new generation out of the best ones.

        Args:
            environment (Environment): environment.
            with_parents (bool): Indicates whether parents of current generation
                should take part in the evaluation process.
        """
        print(f"Iteration: {self._iteration_counter}")
        self._iteration_counter += 1

        network_groups = {
            "children": [child.neural_network for child in self._new_generation]
        }
        if with_parents:
            network_groups["parents"] = [
                parent.neural_network for parent in self._parents
            ]

        adaptations = yield from environment.generate_adaptations(network_groups)

        new_individuals = [
            AdultIndividual(child.neural_network, adaptation)
            for child, adaptation in zip(self._new_generation, adaptations["children"])
        ]
        self.individuals.extend(new_individuals)
        self._sort_individuals_and_kill_unnecessary()

        self._parents = self._selection()
        print(f"Best adaptation: {self._parents[0].adaptation}")

        children: List[ChildIndividual] = self._reproduction(self._parents)
        self._new_generation = self._mutation(children)

    def evolve(self, environment: Environment[None, Any], with_parents: bool) -> None:
        return utils.generator_value(self.generate_evolution(environment, with_parents))
