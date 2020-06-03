import pygame
from planar import Point, Vec2

from planar.line import Ray

from model.Simulation import Simulation
from model.car import Car
from model.geom.sensor import Sensor
from model.geom.track import Track
from model.neural_network_old import NeuralNetwork
from view.Action import Action, ActionType
from view.Menu import Menu
from view.TrackView import TrackView
from view.Window import Window

WINDOW_NAME = "CarsML"
WINDOW_SIZE = (800, 800)


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
    window = Window(WINDOW_NAME, WINDOW_SIZE)

    menu_options = {
        "Train": Action(ActionType.CHANGE_VIEW, 1),
        "Play": Action(ActionType.CHANGE_VIEW, 2),
        "Exit": Action(ActionType.SYS_EXIT),
    }
    menu = Menu(menu_options)
    menu.background_image = pygame.image.load(
        "../resources/graphics/menu-background.png"
    )
    menu.logo_image = pygame.image.load("../resources/graphics/logo.png")

    xy = [
        ((2, 5), (0, 5)),
        ((4, 2), (4, 0)),
        ((6, 5), (8, 5)),
        ((4, 8), (4, 10)),
        ((2, 5), (0, 5)),
    ]
    xa = (
        ((34.206844, 146.0878), (-13.79613, -0.37798)),
        ((41.766368, 118.87351), (32.69494, 107.72321)),
        ((86.745534, 97.13988), (-7.9375, -13.98512)),
        ((103.37649, 67.468749), (89.202381, 65.578867)),
        ((113.58184, 42.900298), (2.07887, 13.040177)),
        ((175.19196, 42.522322), (-8.69345, 13.607142)),
        ((169.75855, 59.436756), (195.17745, 58.633555)),
        ((174.625, 75.21726), (166.87649, 62.744048)),
        ((132.85863, 95.249999), (126.24405, 82.398808)),
        ((118.87351, 110.74702), (-13.79613, -7.9375)),
        ((85.233631, 120.38542), (-8.504465, -9.82739)),
        ((69.547617, 126.62202), (51.782739, 126.05506)),
        ((70.492558, 132.85863), (-14.552083, 9.82738)),
        ((88.25744, 131.15774), (-2.645834, 12.47321)),
        ((112.44792, 130.77976), (0.56696, 12.47321)),
        ((125.48809, 122.65327), (11.71727, 7.9375)),
        ((145.14286, 103.56547), (10.20535, 11.1503)),
        ((164.60863, 93.549106), (7.18155, 12.284224)),
        ((181.42857, 83.721725), (8.31548, 11.528274)),
        ((197.87053, 73.894343), (9.07143, 11.150298)),
        ((216.39137, 58.586308), (225.4628, 71.24851)),
        ((238.88095, 46.302083), (4.53571, 14.363094)),
        ((272.89881, 52.53869), (260.80357, 68.224701)),
        ((286.50594, 96.00595), (271.5759, 99.407738)),
        ((285.75, 140.41815), (-17.38691, -1.70089)),
        ((268.1741, 165.36458), (-7.74851, -15.11905)),
        ((239.25893, 166.49851), (1.70089, -11.52827)),
        ((205.80803, 166.12053), (4.15774, -10.39434)),
        ((183.12946, 155.72619), (11.90625, -4.7247)),
        ((194.09077, 131.15774), (6.80357, 11.52827)),
        ((225.08482, 130.40178), (6.80357, 13.41816)),
        ((243.79464, 121.33036), (15.49702, -0.56697)),
        ((242.47172, 102.80952), (255.70089, 87.690475)),
        ((231.6994, 99.596725), (227.73065, 85.611606)),
        ((212.23363, 113.95982), (-6.80357, -11.90625)),
        ((197.11458, 119.44047), (-5.10268, -10.9613)),
        ((181.42857, 129.26786), (-6.61458, -14.17411)),
        ((168.38839, 145.70982), (156.67113, 130.59077)),
        ((155.34821, 165.55357), (-7.18155, -21.73363)),
        ((129.26786, 166.87649), (1.70089, -13.04018)),
        ((114.3378, 165.93155), (4.34672, -11.52828)),
        ((91.848214, 166.30952), (4.913691, -11.52827)),
        ((57.830357, 166.12053), (59.34226, 153.08036)),
        ((26.269345, 162.90774), (38.553572, 151.19047)),
    )

    multiplier = 100  # fixme for tests, duplicates TrackView.scale
    points = [(Vec2(*a) * multiplier, Vec2(*b) * multiplier) for b, a in xy]
    track = Track.from_points(points)
    sim = Simulation(track)
    tv = TrackView(sim)

    window.add_view(menu, 0, True)
    window.add_view(tv, 1)

    window.run()


if __name__ == "__main__":
    main()
