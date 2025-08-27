import pygame
from snake import settings
import argparse
from snake.eval import evaluate
from snake.states import *
from snake.train import train_model
from snake.utils.loader import AssetLoader


class Game:
    def __init__(self, program_settings):
        pygame.init()
        self.settings = program_settings
        self.loader = AssetLoader()
        flags = pygame.SCALED | pygame.HWSURFACE | pygame.DOUBLEBUF
        self.screen = pygame.display.set_mode((1080, 720), flags, vsync=1)
        pygame.display.set_caption("Learn2Slither")
        self.clock  = pygame.time.Clock()
        self.running = True
        self.episode_idx = 0
        self.game = GameState(self, self.settings)

    def run(self):
        while self.running and self.game._nb_sessions < self.settings["sessions"]:
            dt = self.clock.tick(60) / 1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                self.game.handle_events(event)
            self.game.update(dt)
            self.game.draw(self.screen)
            pygame.display.flip()

        if self.settings["save_path"]:
            self.game.agent.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save", type=str, help="Model path to save")
    parser.add_argument("--load", type=str, help="Load a saved game")
    parser.add_argument("--sessions", type=int, help="Number of sessions to train")
    parser.add_argument("--map_size", type=int, default=10, help="Size of the map")
    parser.add_argument("-train", action='store_true', help="Launch in training mode")
    parser.add_argument("-eval", action='store_true', help="Launch in evaluation mode")
    parser.add_argument("-step", action='store_true', help="Launch in step-by-step mode")
    parser.add_argument("-visual", action='store_true', help="Launch in visual mode")

    args = parser.parse_args()

    settings = settings.settings_value
    settings["map_size"] = args.map_size
    settings["train"] = args.train
    settings["step"] = args.step
    settings["visual"] = args.visual
    settings["save_path"] = args.save
    settings["load_path"] = args.load
    settings["sessions"] = args.sessions

    if args.visual:
        Game(settings).run()
    else:
        if args.eval:
            evaluate(settings["load_path"], settings["sessions"], settings["map_size"])
        if args.train:
            train_model(args.load, args.save, args.sessions)
    pygame.quit()


