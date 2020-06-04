from typing import Tuple, List

import numpy as np

from model.environment.environment import Environment
from model.neural_network.neural_network import NeuralNetwork, LayerInfo
from model.neuroevolution.individual import AdultIndividual, ChildIndividual


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
        individual: AdultIndividual, bound_adaptation: float
    ) -> float:
        """
        Probability of reproduction for every given individual
        without golden ticket according to bound adaptation.
        """
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
        self, networks: List[ChildIndividual], environment: Environment,
    ) -> None:
        # TODO move environment somewhere else
        self._environment: Environment = environment
        self._new_generation: List[ChildIndividual] = networks
        self.individuals: List[AdultIndividual] = []
        self._parents: List[AdultIndividual] = []

    @classmethod
    def init_with_neural_network_info(
        cls,
        layers_infos: List[LayerInfo],
        output_neurons: int,
        environment: Environment,
    ) -> "Neuroevolution":
        return cls(
            [
                ChildIndividual(NeuralNetwork(layers_infos, output_neurons))
                for _ in range(cls._INDIVIDUALS)
            ],
            environment,
        )

    def _sort_individuals_and_kill_unnecessary(self) -> None:
        self.individuals.sort(
            key=lambda individual: individual.adaptation, reverse=True
        )
        self.individuals = self.individuals[: self._INDIVIDUALS]

    def _selection(self) -> List[AdultIndividual]:
        bound_adaptation = self.individuals[self._GOLDEN_TICKETS].adaptation
        parents = self.individuals[: self._GOLDEN_TICKETS]
        for individual in self.individuals[self._GOLDEN_TICKETS : self._INDIVIDUALS]:
            if np.random.rand() < self._reproduction_probability(
                individual, bound_adaptation
            ):
                parents.append(individual)
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

    def evolve(self, with_parents: bool) -> None:
        """
        Evaluates individuals from current generation and
        produce new generation out of the best.

        Args:
            with_parents (bool): Indicates whether parents of current generation
                should take part in race.
        """
        if with_parents:
            adaptations = self._environment.compute_adaptations(
                {
                    "children": (
                        child.neural_network for child in self._new_generation
                    ),
                    "parents": (parent.neural_network for parent in self._parents),
                }
            )
        else:
            adaptations = self._environment.compute_adaptations(
                {"children": (child.neural_network for child in self._new_generation)}
            )

        new_individuals = [
            AdultIndividual(child.neural_network, adaptation)
            for child, adaptation in zip(self._new_generation, adaptations["children"])
        ]
        self.individuals.extend(new_individuals)
        self._sort_individuals_and_kill_unnecessary()
        self._parents = self._selection()
        children: List[ChildIndividual] = self._reproduction(self._parents)
        self._new_generation = self._mutation(children)
