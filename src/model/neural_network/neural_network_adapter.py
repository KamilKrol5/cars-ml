from dataclasses import dataclass
from typing import List

from model.neural_network.neural_network import NeuralNetwork
from model.car.car_instructions import CarInstruction
import numpy as np


@dataclass
class NeuralNetworkAdapter:
    """
    Convert data formats between Car and NeuralNetwork.
    """

    _PREPROCESSING_RATE = 1 / 100
    """
    Every input number will be multiplied by this before enter to neural network.
    """

    def __init__(self, neural_network: NeuralNetwork):
        if neural_network.output_neurons_count != 2:
            raise ValueError("Neural network need 2 output neurons to be compatible.")
        self.neural_network: NeuralNetwork = neural_network

    def get_instructions(self, distances: List[float], speed: float) -> CarInstruction:
        """
        Get instructions for car based on neural network output.
        Args:
            distances (List[float]): Distances counted by sensors.
            speed (float): Current car's speed.
        Return:
            Instructions that define car's behavior.
        """
        if self.neural_network.input_layer_neuron_count - 1 != len(distances):
            raise ValueError(
                "Neural network incompatible with sensors, amount of input neurons "
                "must be equals to numbers of sensors plus one for speed.\n"
                f"Neural network input neurons: {self.neural_network.input_layer_neuron_count}.\n"
                f"Sensors count: {len(distances)}."
            )
        valid_input = np.expand_dims(
            np.array(distances + [speed]) * NeuralNetworkAdapter._PREPROCESSING_RATE, 0
        )
        output = self.neural_network.predict(valid_input)
        return CarInstruction(*output[0])
