import pickle
from os import makedirs
from os.path import exists
from typing import List

from model.neural_network.neural_network import NeuralNetwork


class NeuralNetworkStore:
    """
    Store and load neural networks from disc.
    """

    _DEFAULT_DIRECTORY = "store"

    @staticmethod
    def store(
        neural_networks: List[NeuralNetwork],
        file_name: str,
        directory_name: str = _DEFAULT_DIRECTORY,
    ) -> None:
        """
        Stores neural networks in a binary file with .nn extension.

        Args:
            neural_networks (List[NeuralNetwork]): Neural networks to store in file.
            file_name (str): Name of the file to store neural networks in,
                will be created if it doesn't exist.
            directory_name (str): Name of the directory where the file is,
                will be created if it doesn't exist.
        """
        if not exists(directory_name):
            makedirs(directory_name)
        with open(f"{directory_name}/{file_name}.nn", "wb") as store:
            for neural_network in neural_networks:
                pickle.dump(neural_network, store, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(
        file_name: str,
        directory_name: str = _DEFAULT_DIRECTORY,
        amount_to_load: int = -1,
    ) -> List[NeuralNetwork]:
        """
        Loads neural networks from a binary file with .nn extension.

        Args:
            file_name (str): Name of the file to load neural networks from.
            directory_name (str): Name of directory where file to load is.
            amount_to_load (int): Maximal amount of neural networks to load,
                if there is not enough neural networks in file,
                everything will be loaded without any exception,
                same for negative value.

        Raises:
            FileNotFoundError: If the file or directory don't exist.
            UnpicklingError: If an inappropriate file is given.
        """
        neural_networks = []
        with open(f"{directory_name}/{file_name}.nn", "rb") as store:
            counter = 0
            while counter != amount_to_load:
                try:
                    neural_networks.append(pickle.load(store))
                except EOFError:
                    break
                counter += 1
        return neural_networks
