import json

import pygame
from planar import Point

from planar.line import Ray

from model.Simulation import Simulation
from model.car import Car
from model.geom.sensor import Sensor
from model.geom.track import Track
from model.neural_network_old import NeuralNetwork
from view.action import Action, ActionType
from view.menu import Menu
from view.track_view import TrackView
from view.window import Window

WINDOW_NAME = "CarsML"
WINDOW_SIZE = (1280, 800)


def main_no_ui() -> None:
    from time import sleep

    neural_network = NeuralNetwork()
    sensor = Sensor(Ray((0, 0), (0, 1)))  # TODO: move construction elsewhere

    car = Car((1, 2), [sensor], neural_network)

    track = Track.from_points(
        [
            (Point(-10.0, -10.0), Point(-10.0, 10.0)),
            (Point(10.0, -10.0), Point(10.0, 10.0)),
            (Point(20.0, -10.0), Point(20.0, 10.0)),
        ]
    )
    sleep_time = 0.1
    while True:
        sleep(sleep_time)
        car.tick(track, sleep_time)
        car.go_brrrr()

def main() -> None:
    tracks = {}
    try:
        with open("resources/tracks.json") as file:
            tracks = json.load(file)["tracks"]
    except (IOError, IndexError):
        raise IOError("Unable to load tracks from file")

    window = Window(WINDOW_NAME, WINDOW_SIZE, resizable=True, min_size=WINDOW_SIZE)

    menu_options = {
        "Train": Action(ActionType.CHANGE_VIEW, 1),
        "Play": Action(ActionType.CHANGE_VIEW, 2),
        "Exit": Action(ActionType.SYS_EXIT),
    }
    menu = Menu(menu_options)
    menu.background_image = pygame.image.load("resources/graphics/menu-background.png")
    menu.logo_image = pygame.image.load("resources/graphics/logo.png")

    track = Track.from_points(tracks[0]["points"])
    sim = Simulation(track)
    tv = TrackView(sim)

    track1 = Track.from_points(tracks[1]["points"])
    sim1 = Simulation(track1)
    tv1 = TrackView(sim1)

    window.add_view(menu, 0, True)
    window.add_view(tv, 1)
    window.add_view(tv1, 2)

    window.run()


if __name__ == "__main__":
    main()
