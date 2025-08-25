import pygame
from snake import settings
from snake.settings import settings_value
from snake.states import *
from snake.states.settings import SettingsState
from snake.utils.loader import AssetLoader


class Game:
    def __init__(self):
        pygame.init()
        self.settings = settings.settings_value
        self.loader = AssetLoader()
        flags = pygame.SCALED | pygame.HWSURFACE | pygame.DOUBLEBUF
        self.screen = pygame.display.set_mode((self.settings["screen_width"], self.settings["screen_height"]), flags, vsync=1)
        pygame.display.set_caption("Learn2Slither")
        self.clock  = pygame.time.Clock()
        self.running = True

        self.states = {
            "TITLE": TitleState(self),
            "GAME": GameState(self, self.settings),
            "SETTINGS": SettingsState(self, self.settings),
        }
        self.current_state = self.states["TITLE"]

    def change_state(self, state_name):
        if state_name == "GAME":
            print(self.settings["map_size"])
            self.states["GAME"] = GameState(self, self.settings)

        self.current_state = self.states[state_name]

    def run(self):
        while self.running:
            dt = self.clock.tick(self.settings["fps"]) / 1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if not self.current_state == self.states["TITLE"]:
                            self.current_state = self.states["TITLE"]
                self.current_state.handle_events(event)
            self.current_state.update(dt)
            self.current_state.draw(self.screen)
            pygame.display.flip()

if __name__ == "__main__":
    Game().run()
    pygame.quit()
