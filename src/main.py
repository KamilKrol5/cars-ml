from time import sleep

from planar.line import Ray

from model.car import Car
from model.neural_network_old import NeuralNetwork
from model.geom.sensor import Sensor


def main() -> None:
    neural_network = NeuralNetwork()
    sensor = Sensor(Ray((0, 0), (0, 1)))  # TODO: move construction elsewhere
    car = Car(neural_network, sensor, (0, 0, 45))
    while True:
        sleep(1)
        car.tick()
        car.go_brrrr()


if __name__ == "__main__":
    main()
