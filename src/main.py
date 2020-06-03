import pygame, sys

from model.geom.track import Track
from planar.line import Ray

from model.car import Car
from model.neural_network_old import NeuralNetwork
from model.geom.sensor import Sensor

from view.Action import Action, ActionType
from view.Menu import Menu
from view.Window import Window

WINDOW_NAME = "CarsML"
WINDOW_SIZE = (1280, 854)

x = [[(20, 20), (20, 120)],
     [(20,120), (60,190)],
     [(60,190),(5,260)]]

y = [[(100, 20), (100, 120)],
     [(100,120), (140,190)],
     [(140,190), (85,260)]]


def main() -> None:
    mainClock = pygame.time.Clock()
    pygame.init()
    pygame.display.set_caption('game base')
    screen = pygame.display.set_mode(WINDOW_SIZE, 0, 32)
    font = pygame.font.SysFont(None, 40)

    menu_background = pygame.image.load("../resources/graphics/menu-background.png")
    logo_image = pygame.image.load("../resources/graphics/logo.png")
    track_image = pygame.image.load("../resources/graphics/track.png")


    def draw_text(text, font, color, surface, x, y):
        textobj = font.render(text, 1, color)
        textrect = textobj.get_rect()
        textrect.topleft = (x, y)
        surface.blit(textobj, textrect)

    def main_menu():
        click = False
        while True:

            screen.fill((0, 0, 0))
            screen.blit(menu_background, (0, 0))
            screen.blit(logo_image, (300, 0))

            button_1 = pygame.Rect(50, 100, 200, 50)
            button_2 = pygame.Rect(50, 200, 200, 50)

            pygame.draw.rect(screen, (255, 0, 0), button_1)
            pygame.draw.rect(screen, (255, 0, 0), button_2)

            mx, my = pygame.mouse.get_pos()

            if button_1.collidepoint((mx, my)):
                if click:
                    game()
            if button_2.collidepoint((mx, my)):
                if click:
                    track_generator()

            click = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        click = True

            pygame.display.update()
            mainClock.tick(60)

    def game():
        running = True

        screen.fill((255, 255, 255))
        track = Track.from_points([[((100, 100), (200, 100)), ((100, 100), (100, 200))]])
        red = (255, 0, 0)
        for xi, yi in zip(x, y):
            pygame.draw.polygon(screen, red, (xi[0], yi[0], yi[1], xi[1]),1)
        while running:
            draw_text('game', font, (0, 0, 0), screen, 20, 20)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            pygame.display.update()
            mainClock.tick(60)

    def track_generator():
        click = False
        running = True
        screen.fill((255, 255, 255))
        red = (255, 0, 0)
        x=[]
        y=[]
        x_turn = True
        cur_mx = (20,20)
        cur_my = (40,20)
        screen.blit(track_image, (0,0))
        while running:
            mx, my = pygame.mouse.get_pos()

            if click == True:
                if x_turn:
                    x.append((cur_mx,(mx, my)))
                    pygame.draw.line(screen,(255,0,0),cur_mx,(mx,my),5)
                    cur_mx = (mx, my)
                    x_turn = False

                else:
                    y.append((cur_my,(mx,my)))
                    pygame.draw.line(screen, red, cur_my, (mx, my),5)
                    cur_my = (mx, my)
                    x_turn = True
                    print(cur_my)

            click = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        click = True

            pygame.display.update()
            mainClock.tick(60)
        with open("track.txt", "w") as file:
            file.write(f'x = {x}\n y = {y}')

    main_menu()

    # neural_network = NeuralNetwork()
    # sensor = Sensor(Ray((0, 0), (0, 1)))  # TODO: move construction elsewhere
    # car = Car(neural_network, sensor, (0, 0, 45))
    # car.tick()
    # car.go_brrrr()
    #
    # window = Window(WINDOW_NAME, WINDOW_SIZE)
    #
    # menu_options = {
    #     "Train": Action(ActionType.CHANGE_VIEW, 1),
    #     "Play": Action(ActionType.CHANGE_VIEW, 2),
    #     "Exit": Action(ActionType.SYS_EXIT),
    # }
    # menu = Menu(menu_options)
    # menu.background_image = pygame.image.load(
    #     "../resources/graphics/menu-background.png"
    # )
    # menu.logo_image = pygame.image.load("../resources/graphics/logo.png")
    #
    # window.add_view(menu, 0, True)
    #
    # window.run()


if __name__ == "__main__":
    main()
