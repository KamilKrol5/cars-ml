from time import sleep

from model.Car import Car
from model.NeuralNetwork import NeuralNetwork
from model.Sensor import Sensor

if __name__ == '__main__':
    neural_network = NeuralNetwork()
    sensor = Sensor()
    car = Car(neural_network, sensor, (0, 0, 45))
    while True:
        sleep(1)
        car.tick()
        car.go_brrrr()
