from typing import Tuple, List

import numpy as np

from model.environment.environment import Environment
from model.neural_network.neural_network import NeuralNetwork, LayerInfo
from model.neuroevolution.individual import Individual, ChildIndividual


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
        individual: Individual, bound_adaptation: float
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
        self.individuals: List[Individual] = []
        self._parents: List[Individual] = []

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

    def _reproduction(self, parents: List[Individual]) -> List[ChildIndividual]:
        children_remain = self._INDIVIDUALS - len(parents)
        children: List[ChildIndividual] = []
        while children_remain > 0:
            mother, father = np.random.choice(parents, size=2)
            while mother == father:
                father = np.random.choice(parents)
            daughter, son = np.random.choice(
                Individual.available_reproductions, p=self._REPRODUCTION_PROBABILITIES
            )(mother, father)
            children.append(daughter)
            children_remain -= 1
            if children_remain > 0:
                children.append(son)
                children_remain -= 1
        return children

    def _mutation(self, individuals: List[ChildIndividual]) -> List[ChildIndividual]:
        for individual in individuals:
            if np.random.rand() < self._MUTATION_CHANCE:
                np.random.choice(
                    ChildIndividual.available_mutations, p=self._MUTATION_PROBABILITIES
                )(individual)
        return individuals

    def evolve(self, with_parents: bool) -> None:
        adaptations = self._environment.compute_adaptations(
            (child.neural_network for child in self._new_generation),
            (parent.neural_network for parent in self.individuals)
            if with_parents
            else [],
        )
        new_individuals = [
            Individual(child.neural_network, adaptation)
            for child, adaptation in zip(self._new_generation, adaptations)
        ]
        self.individuals.extend(new_individuals)
        self._sort_individuals_and_kill_unnecessary()
        self._parents = self._selection()
        children: List[ChildIndividual] = self._reproduction(self._parents)
        self._new_generation = self._mutation(children)
