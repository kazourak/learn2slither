import pygame
from snake import settings
from snake.states import *
from snake.utils.loader import AssetLoader


class Game:
    def __init__(self):
        pygame.init()
        self.loader = AssetLoader()
        flags = pygame.SCALED | pygame.HWSURFACE | pygame.DOUBLEBUF
        self.screen = pygame.display.set_mode((settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT), flags, vsync=1)
        pygame.display.set_caption("Learn2Slither")
        self.clock  = pygame.time.Clock()
        self.running = True

        self.states = {
            "TITLE": TitleState(self),
            "GAME": GameState(self),
        }
        self.current_state = self.states["TITLE"]

    def change_state(self, state_name):
        self.current_state = self.states[state_name]

    def run(self):
        while self.running:
            dt = self.clock.tick(settings.FPS) / 1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                self.current_state.handle_events(event)
            self.current_state.update(dt)
            self.current_state.draw(self.screen)
            pygame.display.flip()

if __name__ == "__main__":
    Game().run()
    pygame.quit()
