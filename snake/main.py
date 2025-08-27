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

def validate_args(args):
    """Validate argument combinations and return error message if invalid"""
    primary_modes = [args.train, args.eval, args.visual]
    mode_count = sum(primary_modes)
    
    if mode_count == 0:
        return "Error: Must specify one mode: -train, -visual or -eval"

    if args.eval and args.train:
        return "Error: Cannot evaluate and train at the same time"

    if args.visual and args.eval:
        return "Error: Cannot visualize and evaluate at the same time"
    
    if args.eval:
        if not args.load:
            return "Error: --load required for evaluation mode"
        if not args.sessions:
            return "Error: --sessions required for evaluation mode"

    if args.map_size < 5 or args.map_size > 20:
        return "Error: Map size must be between 5 and 20"

    if args.sessions is not None and (args.sessions < 1 or args.sessions > 999999999):
        return "Error: Sessions must be between 1 and 999999999"

    # if args.visual and not args.train and not args.eval:
    #     if not args.load:
    #         return "Error: --load required for visual-only mode"
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save", type=str, help="Model path to save")
    parser.add_argument("--load", type=str, help="Load a saved model")
    parser.add_argument("--sessions", type=int, default=100, help="Number of sessions to train or eval")
    parser.add_argument("--map_size", type=int, default=10, help="Size of the map")
    parser.add_argument("-train", action='store_true', help="Launch in training mode")
    parser.add_argument("-eval", action='store_true', help="Launch in evaluation mode")
    parser.add_argument("-step", action='store_true', help="Launch in step-by-step mode")
    parser.add_argument("-visual", action='store_true', help="Launch in visual mode")

    args = parser.parse_args()
    
    error_msg = validate_args(args)
    if error_msg:
        print(error_msg)
        parser.print_help()
        exit(1)

    settings = settings.settings_value
    settings["map_size"] = args.map_size
    settings["train"] = args.train
    settings["step"] = args.step
    settings["visual"] = args.visual
    settings["save_path"] = args.save
    settings["load_path"] = args.load
    settings["sessions"] = args.sessions

    if args.visual and not args.train and not args.eval: # si je veux le visuel, sans train, et sans eval
        Game(settings).run()
    elif args.eval:
        evaluate(settings["load_path"], settings["sessions"], settings["map_size"])
    elif args.train:
        if args.visual:
            Game(settings).run()
        else:
            train_model(args.load, args.save, args.sessions)
    
    pygame.quit()


