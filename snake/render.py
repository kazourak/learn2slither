import sys
from typing import Dict

import numpy as np
import pygame

import interpreter as interpreter
from action import Actions, ActionResult, get_coordinates_from_action
from env import SnakeEnv

# Constants
BOARD_SIZE = 10  # 10x10 grid
CELL_SIZE = 40   # size of each cell in pixels
WINDOW_SIZE = BOARD_SIZE * CELL_SIZE
FPS = 5

EMPTY = 0
WALL = 1
HEAD = 2
BODY = 3
GREEN_APPLE = 4
RED_APPLE = 5

# Colors\WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 150, 0)
DARK_BLUE = (0, 0, 150)
BLUE = (0, 0, 200)

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
clock = pygame.time.Clock()
pygame.display.set_caption("Snake Game")

def draw_grid():
    for y in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, DARK_GREEN, (y, 0), (y, WINDOW_SIZE))
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, DARK_GREEN, (0, x), (WINDOW_SIZE, x))

def draw_snake(env: SnakeEnv):
    for idx, (y, x) in enumerate(env.snake):
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, DARK_BLUE if idx == 0 else BLUE, rect)

def draw_apple(env: SnakeEnv, apple_type):
    apple_color = GREEN if apple_type == GREEN_APPLE else RED
    for (y, x) in env.apples[apple_type]:
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, apple_color, rect)


def render(env: SnakeEnv) -> None:
    char_map: Dict[int, str] = {
        EMPTY: "  ",
        WALL: "⬛",
        HEAD: "🟢",
        BODY: "🟩",
        GREEN_APPLE: "🍏",
        RED_APPLE: "🍎",
    }

    board: np.ndarray = env.board.astype(int)

    # Efface l’écran (ANSI)
    sys.stdout.write("\033[H\033[J")
    sys.stdout.flush()

    for row in board:
        print("".join(char_map.get(cell, "??") for cell in row))
    print()


if __name__ == "__main__":
    env = SnakeEnv(BOARD_SIZE, 3, 2, 1)
    total_score = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and env.direction != Actions.DOWN.value:
                    env.direction = get_coordinates_from_action(Actions.UP)
                elif event.key == pygame.K_DOWN and env.direction != Actions.UP.value:
                    env.direction = get_coordinates_from_action(Actions.DOWN)
                elif event.key == pygame.K_LEFT and env.direction != Actions.RIGHT.value:
                    env.direction = get_coordinates_from_action(Actions.LEFT)
                elif event.key == pygame.K_RIGHT and env.direction != Actions.LEFT.value:
                    env.direction = get_coordinates_from_action(Actions.RIGHT)

        result_action = env.step()
        total_score += interpreter.get_reward(result_action)
        if (
            result_action.replaced_cell in (BODY, WALL)
            or result_action.snake_length < 1
        ):
            print("💥 Collision ! La partie recommence...")
            env.reset()
        # screen.fill(BLACK)
        # draw_grid()
        # draw_snake(env)
        # draw_apple(env, GREEN_APPLE)
        # draw_apple(env, RED_APPLE)
        # pygame.display.flip()
        render(env)

        clock.tick(FPS)

# env = SnakeEnv(10, 3, 2, 1)
    # render(env)
    #
    # # Table de correspondance entre la saisie utilisateur et l’énumération Actions
    # input_to_action = {
    #     "W": Actions.UP,
    #     "S": Actions.DOWN,
    #     "A": Actions.LEFT,
    #     "D": Actions.RIGHT,
    # }
    #
    # total_score = 0
    #
    # while True:
    #     # Demande l’action à l’utilisateur
    #     user_choice = input("Votre action (UP / DOWN / LEFT / RIGHT) : ").strip().upper()
    #
    #     print(user_choice)
    #     if user_choice not in input_to_action:
    #         print("Action invalide, veuillez réessayer.")
    #         continue
    #
    #     action = input_to_action[user_choice]
    #     result_action: ActionResult = env.step(action)
    #     total_score += interpreter.get_reward(result_action)
    #
    #
    #     # Détection de collision ou de fin de partie
    #     if (
    #             result_action.replaced_cell in (BODY, WALL)
    #             or result_action.snake_length < 1
    #     ):
    #         print("💥 Collision ! La partie recommence...")
    #         env.reset()
    #
    #     render(env)
    #     print(f"Score : {total_score}")