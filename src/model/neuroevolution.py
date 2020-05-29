from typing import Tuple, Dict, Callable, List

import numpy as np

from model.individual import Individual
from model.neural_network.neural_network import NeuralNetwork, LayerInfo
from model.neural_network_old import NeuralNetwork as NeuralNetworkOld


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
            0: self._single_weight_reproduction,
            1: self._single_neuron_reproduction,
            2: self._entire_layer_reproduction,
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
                for i in range(cls._INDIVIDUALS)
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
    def _single_weight_reproduction(
        mother: NeuralNetworkOld, father: NeuralNetworkOld
    ) -> Tuple[NeuralNetworkOld, NeuralNetworkOld]:
        # TODO
        return NeuralNetworkOld(), NeuralNetworkOld()

    @staticmethod
    def _single_neuron_reproduction(
        mother: NeuralNetworkOld, father: NeuralNetworkOld
    ) -> Tuple[NeuralNetworkOld, NeuralNetworkOld]:
        # TODO
        return NeuralNetworkOld(), NeuralNetworkOld()

    @staticmethod
    def _entire_layer_reproduction(
        mother: NeuralNetworkOld, father: NeuralNetworkOld
    ) -> Tuple[NeuralNetworkOld, NeuralNetworkOld]:
        # TODO
        return NeuralNetworkOld(), NeuralNetworkOld()

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
    def _random_weight_mutation(individual: NeuralNetworkOld) -> None:
        # TODO
        pass

    @staticmethod
    def _random_bias_mutation(individual: NeuralNetworkOld) -> None:
        # TODO
        pass

    @staticmethod
    def _change_weight_sign_mutation(individual: NeuralNetworkOld) -> None:
        # TODO
        pass

    @staticmethod
    def _multiply_neuron_weights_mutation(individual: NeuralNetworkOld) -> None:
        # TODO
        pass

    @staticmethod
    def _random_neuron_weights_mutation(individual: NeuralNetworkOld) -> None:
        # TODO
        pass

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
