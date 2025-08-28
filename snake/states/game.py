from typing import Tuple, Deque, Set

import pygame

from snake.action import index_to_string, ActionResult, ActionState, index_to_action_tuple
from snake.agent import QLearningSnakeAgent
from snake.env import SnakeEnv
from snake.interpreter import Interpreter
from snake.states.base_state import BaseState
from snake.ui.animated_background import AnimatedGridBackground

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 200)
LIGHT_BLUE = (100, 150, 255)
GRAY = (128, 128, 128)
DARK_RED = (150, 0, 0)
EMPTY = 0
WALL = 1
HEAD = 2
BODY = 3
GREEN_APPLE = 4
RED_APPLE = 5
Coordinate = Tuple[int, int]


class GameState(BaseState):

    def __init__(self, game, settings):
        super().__init__(game)
        self.settings = settings
        self.font = self.game.loader.load_font("8bitoperator_jve.ttf")
        self.animated_background = AnimatedGridBackground()

        self.grid_size = self.settings["map_size"]
        self.board_size = min(1080, 720) * 0.8
        self.cell_size = self.board_size // self.grid_size
        self.board_size = self.cell_size * self.grid_size
        self.board_surface = self._build_board_surface()

        self.board_x = (1080 - self.board_size) // 2
        self.board_y = (720 - self.board_size) // 2

        self.env = SnakeEnv(self.grid_size, 3, self.settings["red_apple_nbr"], self.settings["green_apple_nbr"])
        self.interpreter = Interpreter(reward_nothing=-1.14, reward_dead=-115, reward_green_apple=19.14, reward_red_apple=-21.96)

        self.agent = QLearningSnakeAgent(save_path=self.settings["save_path"], load_path=self.settings["load_path"], train=self.settings["train"])
        self.agent.calc_eps_decay(self.settings["sessions"])
        self.current_state = self.interpreter.get_state(self.env.snake, self.env.board)

        self.nb_sessions = 1
        self.snake_speed = 15

        self._snake_timer = 0.0
        self._bg_timer = 0.0
        self._paused = settings["step"]
        self._step_by_step = settings["step"]
        self._end_game = False
        self._nb_steps = 0

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

        self.board_x = (1080 - self.board_size) // 2
        self.board_y = (720 - self.board_size) // 2

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

    def draw_score(self, surface):
        snake_len = len(self.env.snake)
        score_text = f"Snake length: {snake_len}"
        score_surface = self.font.render(score_text, True, WHITE)
        score_rect = score_surface.get_rect(topleft=(10, 10))
        surface.blit(score_surface, score_rect)

    def draw_session_info(self, surface):
        session_text = f"Session: {self.nb_sessions}" + (f"/{self.settings['sessions']}" if self.settings['sessions'] > 0 else "")
        session_surface = self.font.render(session_text, True, WHITE)
        session_rect = session_surface.get_rect(topleft=(10, 30))
        surface.blit(session_surface, session_rect)

    def draw_end_screen(self, surface):
        score_text = f"Snake length: {len(self.env.snake)}"
        score_surface = self.font.render(score_text, True, WHITE)
        score_rect = score_surface.get_rect(center=(1080 // 2, 720 // 2))
        surface.blit(score_surface, score_rect)

    def draw_state(self, surface):
        state = self.interpreter.get_state(self.env.snake, self.env.board)

        danger_up, danger_down, danger_left, danger_right, obj_up, obj_down, obj_left, obj_right = state

        state_size = 30
        state_x = 1080 - state_size * 3
        state_y = 720 - state_size * 3

        directions = [
            (1, 0, "right"),
            (0, -1, "up"),
            (-1, 0, "left"),
            (0, 1, "down")
        ]

        dangers = [danger_right, danger_up, danger_left, danger_down]
        objects = [obj_right, obj_up, obj_left, obj_down]

        def resize_sprite(sprite):
            return pygame.transform.scale(sprite, (state_size, state_size))

        for i, (dx, dy, name) in enumerate(directions):
            cell_x = state_x + (dx + 1) * state_size
            cell_y = state_y + (dy + 1) * state_size

            pygame.draw.rect(surface, BLACK, (cell_x, cell_y, state_size, state_size))
            pygame.draw.rect(surface, WHITE, (cell_x, cell_y, state_size, state_size), 1)

            obj_type = objects[i]
            if obj_type == self.interpreter.OBJ_GREEN:
                resized_sprite = resize_sprite(self.apple_sprites["GREEN_APPLE"])
                surface.blit(resized_sprite, (cell_x, cell_y))
            elif obj_type == self.interpreter.OBJ_RED:
                resized_sprite = resize_sprite(self.apple_sprites["RED_APPLE"])
                surface.blit(resized_sprite, (cell_x, cell_y))
            elif obj_type == self.interpreter.OBJ_BODY:
                resized_sprite = resize_sprite(self.snake_body_sprites["h"])
                surface.blit(resized_sprite, (cell_x, cell_y))
            elif obj_type == self.interpreter.OBJ_WALL:
                text = self.font.render("W", True, WHITE)
                text_rect = text.get_rect(center=(cell_x + state_size // 2, cell_y + state_size // 2))
                surface.blit(text, text_rect)
            elif obj_type == self.interpreter.TAIL:
                resized_sprite = resize_sprite(self.snake_tail_sprites[(0, 1)])
                surface.blit(resized_sprite, (cell_x, cell_y))

            if dangers[i] == 1:
                text = self.font.render("!", True, DARK_RED)
                text_rect = text.get_rect(center=(cell_x + state_size // 2 + 7, cell_y + state_size // 2))
                surface.blit(text, text_rect)

    def reset(self):
        self.env.reset()
        self._snake_timer = 0.0
        self._nb_steps = 0
        self._end_game = False
        self.nb_sessions += 1
        if self.settings["train"]:
            self.agent.decay_epsilon()
        self.current_state = self.interpreter.get_state(self.env.snake, self.env.board)


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
            if self._end_game:
                self.reset()
                return
            if event.key == pygame.K_SPACE:
                self._step_by_step = not self._step_by_step
                return
            if event.key == pygame.K_RETURN:
                if self._step_by_step:
                    self._paused = False
                return
            if event.key == pygame.K_l:
                self.reset()
                return

    def update(self, dt):
        if self._nb_steps >= 500:
            self._end_game = True
            return

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
        if (self._step_by_step and self._paused) or self._end_game:
            return

        action_idx = self.agent.choose_action(self.current_state)
        self.interpreter.print_vision(self.env.board)
        print(index_to_string(action_idx))
        self.env.direction = index_to_action_tuple(action_idx)
        result: ActionResult = self.env.step()
        next_state = self.interpreter.get_state(self.env.snake, self.env.board)
        reward = self.interpreter.get_reward(result)

        if result.snake_length < 1 or result.action_state == ActionState.DEAD:
            self._end_game = True

        if self.settings["train"]:
            self.agent.update(self.current_state, action_idx, reward, next_state, self._end_game)

        self.current_state = next_state

        if self._step_by_step and not self._paused:
            self._paused = True

        self._nb_steps += 1

    def draw(self, surface):
        self.animated_background.draw(surface)
        if self._end_game:
            self.draw_end_screen(surface)
            return

        self.draw_board(surface)
        self.draw_score(surface)
        self.draw_session_info(surface)

        self.draw_apples(surface, "RED_APPLE", self.env.apples[5])
        self.draw_apples(surface, "GREEN_APPLE", self.env.apples[4])
        self.draw_snake(surface, self.env.snake, self.env.direction)

        if self._step_by_step:
            self.draw_state(surface)

