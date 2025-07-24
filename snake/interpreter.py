from typing import Deque, Set

import numpy as np

from snake.action import ActionResult, ActionState
from snake.env import Coordinate

EMPTY = 0
WALL = 1
HEAD = 2
BODY = 3
GREEN_APPLE = 4
RED_APPLE = 5

def get_reward(result: ActionResult) -> float:
    if result.action_state == ActionState.NOTHING:
        return  -0.1
    elif result.action_state == ActionState.EAT_GREEN_APPLE:
        return  2.5
    elif result.action_state == ActionState.EAT_RED_APPLE:
        return -2.5
    elif result.action_state == ActionState.DEAD:
        return -5.0
    return 0.0


def get_state(snake: Deque[Coordinate], board: np.ndarray, current_direction: tuple) -> tuple:
    head_coord = snake[0]
    tail_coord = snake[-1]
    snake_without_tail = list(snake)[:-1]

    state = []

    directions = [(0,-1), (0,1), (-1,0), (1,0)]  # up, down, left, right

    for dx, dy in directions:
        next_pos = (head_coord[0] + dx, head_coord[1] + dy)

        danger = ((next_pos in snake_without_tail) or
                  next_pos[0] < 1 or next_pos[0] >= 11 or
                  next_pos[1] < 1 or next_pos[1] >= 11)
        state.append(int(danger))

        # TODO: change max-dist to grid size
        green_apple_close = apple_in_direction(board, head_coord, (dx, dy), GREEN_APPLE, tail_coord, 10)
        red_apple_close = apple_in_direction(board, head_coord, (dx, dy), RED_APPLE, tail_coord, 3)
        state.append(int(green_apple_close))
        state.append(int(red_apple_close))

    return tuple(state)

def apple_in_direction(board: np.ndarray, head_pos, direction, apple_type, tail_pos, max_dist=3):
    dx, dy = direction
    x, y = head_pos

    for distance in range(1, max_dist + 1):
        check_x = x + (dx * distance)
        check_y = y + (dy * distance)

        if check_x < 1 or check_x >= 11 or check_y < 1 or check_y >= 11:
            break

        if board[check_x, check_y] != EMPTY and tail_pos != (check_x, check_y):
            if board[check_x, check_y] == apple_type:
                return True
            return False

    return False
