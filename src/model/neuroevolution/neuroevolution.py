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

    counter = 1
    _INDIVIDUALS: int = 500
    """
    Amount of individuals per generation, minimum 3.
    """

    _GOLDEN_TICKETS: int = 2
    """
    Amount of individuals allowed to reproduce for sure.
    """

    _MAX_PARENTS: int = 100
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
        x = (individual.adaptation / bound_adaptation) ** 5
        print(f"({individual.adaptation} / {bound_adaptation}) ** 4= {x}")
        return x

    _REPRODUCTION_RATE: Tuple[int, int, int] = (50, 10, 1)
    """
    Proportional chance for reproduction by respectively:
    1. swapping single weight,  7/10
    2. swapping single neuron,  6/10
    3. swapping entire layer.  1/10
    """

    _MUTATION_CHANCE: float = 0.70
    """
    Mutation chance for every child.
    """

    _MUTATION_RATE: Tuple[int, int, int, int, int] = (50, 50, 10, 100, 1)
    """
    Proportional chance for mutation by respectively (apply only to children with mutation):
    1. generate random value for single weight, 3/10
    2. generate random value for single bias,  2/10
    3. change the sign of single weight,  3/10
    4. multiply all weights of single neuron by numbers in range 0.5 to 1.5,  8/10
    5  random value for all weights of single neuron.  1/10
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
                print("yes")
                parents.append(individual)
                if len(parents) == self._MAX_PARENTS:
                    break
            else:
                print("no")

        print(f"Individuals to reproduce: {len(parents)} out of {self._INDIVIDUALS}")
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

    def _parent_mutation(
        self, individuals: List[AdultIndividual]
    ) -> List[ChildIndividual]:

        _mutated_parents = [ChildIndividual(i.neural_network) for i in individuals]
        for individual in _mutated_parents:
            np.random.choice(
                ChildIndividual.available_mutations, p=self._MUTATION_PROBABILITIES
            )(individual)
        return _mutated_parents

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
        print(f"------------ {self.counter} ------------")
        self.counter += 1
        network_groups = {
            "children": [child.neural_network for child in self._new_generation]
        }
        # if with_parents:
        #     network_groups["parents"] = [
        #         parent.neural_network for parent in self._parents
        #     ]

        adaptations = yield from environment.generate_adaptations(network_groups)
        new_individuals = [
            AdultIndividual(child.neural_network, adaptation)
            for child, adaptation in zip(self._new_generation, adaptations["children"])
        ]
        self.individuals.extend(new_individuals)
        self._sort_individuals_and_kill_unnecessary()
        self._parents = self._selection()
        for i in self._parents:
            print(i.neural_network)
        children: List[ChildIndividual] = self._reproduction(self._parents)
        self._new_generation = self._mutation(children) + self._parent_mutation(
            self._parents
        )

    def evolve(self, environment: Environment[None, Any], with_parents: bool) -> None:
        return utils.generator_value(self.generate_evolution(environment, with_parents))
