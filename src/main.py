from time import sleep

from model.car import Car
from model.neural_network import NeuralNetwork
from model.sensor import Sensor


def main() -> None:
    neural_network = NeuralNetwork()
    sensor = Sensor()
    car = Car(neural_network, sensor, (0, 0, 45))
    while True:
        sleep(1)
        car.tick()
        car.go_brrrr()


if __name__ == "__main__":
    main()