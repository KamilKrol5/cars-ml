import json

import pygame
from planar import Point
from planar.line import Ray

from model.car.car import Car
from model.car.sensor import Sensor
from model.neural_network.neural_network import NeuralNetwork, LayerInfo
from model.neural_network.neural_network_store import NeuralNetworkStore
from model.track.track import Track
from view.action import Action, ActionType
from view.menu import Menu
from view.track_view import TrackView
from view.window import Window
from view.options_menu import OptionsMenu

WINDOW_NAME = "CarsML"
WINDOW_SIZE = (1280, 800)
WINDOW_MIN_SIZE = (800, 600)


def main_no_ui() -> None:
    from time import sleep

    neural_networks = []
    for _ in range(100):
        neural_networks.append(
            NeuralNetwork(
                [
                    LayerInfo(8, "tanh"),
                    LayerInfo(15, "sigmoid"),
                    LayerInfo(10, "sigmoid"),
                ],
                2,
            )
        )
    NeuralNetworkStore.store(neural_networks, "data")
    neural_networks.clear()
    neural_networks = NeuralNetworkStore.load("data")
    print(len(neural_networks))

    neural_network = neural_networks[0]

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
    try:
        with open("resources/tracks.json") as file:
            tracks = json.load(file)["tracks"]
    except (IOError, IndexError):
        raise IOError("Unable to load tracks from file")

    window = Window(WINDOW_NAME, WINDOW_SIZE, resizable=True, min_size=WINDOW_MIN_SIZE)

    menu_options = {
        "Train": Action(ActionType.CHANGE_VIEW, 1),
        "Exit": Action(ActionType.SYS_EXIT),
    }
    menu = Menu(menu_options)
    menu.background_image = pygame.image.load("resources/graphics/menu-background.png")
    menu.logo_image = pygame.image.load("resources/graphics/logo.png")

    tv1 = TrackView(Track.from_points(tracks[1]["points"]))
    tv2 = TrackView(Track.from_points(tracks[3]["points"]))
    tv3 = TrackView(Track.from_points(tracks[4]["points"]))

    track_options = {
        "Track1": Action(ActionType.CHANGE_VIEW, 2),
        "Track2": Action(ActionType.CHANGE_VIEW, 3),
        "Track3": Action(ActionType.CHANGE_VIEW, 4),
    }
    track_images = [
        pygame.image.load("resources/graphics/track1.jpg"),
        pygame.image.load("resources/graphics/track2.jpg"),
        pygame.image.load("resources/graphics/track3.jpg")
    ]
    options_menu = OptionsMenu(tracks=track_options, tracks_thumbnails=track_images)
    options_menu.background_image = pygame.image.load(
        "resources/graphics/menu-background.png"
    )

    window.add_view(menu, 0, True)
    window.add_view(tv1, 2)
    window.add_view(tv2, 3)
    window.add_view(tv3, 4)
    window.add_view(options_menu, 1)

    window.run()


if __name__ == "__main__":
    main()
