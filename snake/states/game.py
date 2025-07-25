from typing import Tuple, Deque, Set

import pygame

from snake.action import Actions, get_coordinates_from_action, ActionResult, ActionState, index_to_action_tuple
from snake.agent import QLearningSnakeAgent
from snake.env import SnakeEnv
from snake.interpreter import Interpreter
from snake.states.base_state import BaseState
from snake.settings import SCREEN_WIDTH, SCREEN_HEIGHT
from snake.ui.animated_background import AnimatedGridBackground

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 200)
LIGHT_BLUE = (100, 150, 255)
GRAY = (128, 128, 128)
EMPTY = 0
WALL = 1
HEAD = 2
BODY = 3
GREEN_APPLE = 4
RED_APPLE = 5
Coordinate = Tuple[int, int]


class GameState(BaseState):

    def __init__(self, game):
        super().__init__(game)
        self.font = self.game.loader.load_font("8bitoperator_jve.ttf")
        self.animated_background = AnimatedGridBackground()

        self.grid_size = 10
        self.board_size = min(SCREEN_WIDTH, SCREEN_HEIGHT) * 0.8
        self.cell_size = self.board_size // self.grid_size
        self.board_size = self.cell_size * self.grid_size
        self.board_surface = self._build_board_surface()

        self.board_x = (SCREEN_WIDTH - self.board_size) // 2
        self.board_y = (SCREEN_HEIGHT - self.board_size) // 2

        self.env = SnakeEnv(10, 3, 1, 2)
        r_nothing = -1.23
        r_eat_green = 20.58
        r_eat_red = -28.16
        r_dead = -113.51
        self.interpreter = Interpreter(reward_nothing=r_nothing, reward_dead=r_dead, reward_red_apple=r_eat_red, reward_green_apple=r_eat_green)

        self.agent = QLearningSnakeAgent(filename="best_model.pkl")
        self.running = True

        self.snake_speed = 100
        self._snake_timer = 0.0
        self._bg_timer = 0.0
        self._paused = False

        def scale(img):
            return pygame.transform.scale(img, (self.cell_size, self.cell_size))

        self.snake_head_sprites = {
            (-1, 0): scale(self.game.loader.load_image("L_HEAD.png")),
            (1, 0): scale(self.game.loader.load_image("R_HEAD.png")),
            (0, -1): scale(self.game.loader.load_image("U_HEAD.png")),
            (0, 1): scale(self.game.loader.load_image("D_HEAD.png")),
        }

        self.snake_tail_sprites = {
            (-1, 0): scale(self.game.loader.load_image("R_TAIL.png")),
            ( 1, 0): scale(self.game.loader.load_image("L_TAIL.png")),
            ( 0,-1): scale(self.game.loader.load_image("D_TAIL.png")),
            ( 0, 1): scale(self.game.loader.load_image("U_TAIL.png")),
        }

        self.snake_body_sprites = {
            "v": scale(self.game.loader.load_image("U-D_BODY.png")),
            "h": scale(self.game.loader.load_image("L-R_BODY.png")),
        }

        self.snake_angle_body_sprites = {
            ((-1, 0), (0, -1)): scale(self.game.loader.load_image("U-L_BODY.png")),
            ((0, -1), (-1, 0)): scale(self.game.loader.load_image("U-L_BODY.png")),
            ((-1, 0), (0, 1)):  scale(self.game.loader.load_image("D-L_BODY.png")),
            ((0, 1), (-1, 0)):  scale(self.game.loader.load_image("D-L_BODY.png")),
            ((1, 0), (0, -1)):  scale(self.game.loader.load_image("U-R_BODY.png")),
            ((0, -1), (1, 0)):  scale(self.game.loader.load_image("U-R_BODY.png")),
            ((1, 0), (0, 1)):   scale(self.game.loader.load_image("D-R_BODY.png")),
            ((0, 1), (1, 0)):   scale(self.game.loader.load_image("D-R_BODY.png")),
        }

        self.apple_sprites = {
            "RED_APPLE": scale(self.game.loader.load_image("RED_APPLE.png")),
            "GREEN_APPLE": scale(self.game.loader.load_image("GREEN_APPLE.png")),
        }

    def set_grid_size(self, new_size):
        """Change le nombre de cellules de la grille"""
        self.grid_size = new_size
        self.cell_size = self.board_size // self.grid_size
        self.board_size = self.cell_size * self.grid_size

        self.board_x = (SCREEN_WIDTH - self.board_size) // 2
        self.board_y = (SCREEN_HEIGHT - self.board_size) // 2

    def draw_board(self, surface):
        surface.blit(self.board_surface, (self.board_x, self.board_y))


    def draw_snake(self, surface, snake: Deque[Coordinate], head_dir: Coordinate):
        for i, (col, row) in enumerate(snake):
            x = self.board_x + (col - 1) * self.cell_size
            y = self.board_y + (row - 1) * self.cell_size

            if i == 0:
                # tête
                sprite = self.snake_head_sprites[head_dir]

            elif i == len(snake) - 1:
                # queue
                prev_col, prev_row = snake[-2]
                tail_col, tail_row = snake[-1]
                tail_dir = (tail_col - prev_col, tail_row - prev_row)
                sprite = self.snake_tail_sprites[tail_dir]

            else:
                # corps
                before = snake[i - 1]
                curr   = snake[i]
                after  = snake[i + 1]
                sprite = self.get_body_sprite(before, curr, after)

            surface.blit(sprite, (x, y))


    def get_body_sprite(self, head: Coordinate, body: Coordinate, tail: Coordinate):
        din  = (head[0] - body[0], head[1] - body[1])
        dout = (tail[0] - body[0], tail[1] - body[1])

        if din == dout or din == (-dout[0], -dout[1]):
            if din[1] != 0:
                return self.snake_body_sprites["v"]
            else:
                return self.snake_body_sprites["h"]

        return self.snake_angle_body_sprites.get((din, dout), pygame.Surface((self.cell_size, self.cell_size)))


    def draw_apples(self, surface, apple_type, apple: Set[Coordinate]):
        sprite = self.apple_sprites[apple_type]

        for (x, y) in apple:
            x = self.board_x + (x - 1) * self.cell_size
            y = self.board_y + (y - 1) * self.cell_size
            surface.blit(sprite, (x, y))

    def _build_board_surface(self):
        surf = pygame.Surface((self.board_size, self.board_size))
        surf.fill(BLACK)

        pygame.draw.rect(
            surf,
            WHITE,
            pygame.Rect(0, 0, self.board_size, self.board_size),
            width=1
        )

        for i in range(1, self.grid_size):
            x = i * self.cell_size
            pygame.draw.line(surf, (128, 128, 128), (x, 0), (x, self.board_size), 1)
            pygame.draw.line(surf, (128, 128, 128), (0, x), (self.board_size, x), 1)

        return surf


    def handle_events(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self._paused = not self._paused
                return


    def update(self, dt):
        self._bg_timer += dt
        self._snake_timer += dt

        bg_interval = 1.0 / 30

        if self._bg_timer >= bg_interval:
            self._bg_timer -= bg_interval
            self.animated_background.update(dt)

        snake_interval = 1.0 / self.snake_speed

        if self._snake_timer >= snake_interval:
            self._snake_timer -=  snake_interval
            self._update_snake()


    def _update_snake(self):
        if self._paused:
            return

        state = self.interpreter.get_state(self.env.snake, self.env.board, self.env.direction)
        action_idx = self.agent.choose_action(state)
        self.env.direction = index_to_action_tuple(action_idx)
        result: ActionResult = self.env.step()
        # state = self.interpreter.get_state(self.env.snake, self.env.board, self.env.direction)
        # labels = [
        #     "danger_up", "danger_down", "danger_left", "danger_right",
        #     "obj_up", "obj_down", "obj_left", "obj_right",
        # ]
        # print("=" * 20)
        # print(f" taille labels{len(labels)}")
        # for label, value in zip(labels, state):
        #     print(f"{label}: [{value}]")

        if result.snake_length < 1 or result.action_state == ActionState.DEAD:
            print(f"Snake dead, max len: {result.snake_length}")
            self.env.reset()

    def draw(self, surface):
        self.animated_background.draw(surface)
        self.draw_board(surface)

        self.draw_apples(surface, "RED_APPLE", self.env.apples[5])
        self.draw_apples(surface, "GREEN_APPLE", self.env.apples[4])
        self.draw_snake(surface, self.env.snake, self.env.direction)
