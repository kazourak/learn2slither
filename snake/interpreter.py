
from typing import Deque, Tuple
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
        return  -2.5
    elif result.action_state == ActionState.EAT_GREEN_APPLE:
        return  25
    elif result.action_state == ActionState.EAT_RED_APPLE:
        return -25
    elif result.action_state == ActionState.DEAD:
        return -100
    return 0.0

# code d'objet dans la vision
OBJ_GREEN = 0
OBJ_RED   = 1
OBJ_BODY  = 2
OBJ_WALL  = 3

def get_state(
        snake: Deque[Tuple[int,int]],
        board: np.ndarray,
        current_direction: Tuple[int,int]
) -> Tuple[int,...]:
    """
    État = tuple de 12 entiers :
    [ danger_up, danger_down, danger_left, danger_right,
      obj_up,    obj_down,    obj_left,    obj_right ]
    """

    head_x, head_y = snake[0]
    body = set(list(snake)[:-1])
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    state = []

    for dx, dy in directions:
        nx, ny = head_x + dx, head_y + dy
        cell = board[ny][nx]
        is_wall = (cell == WALL)
        is_body = (nx, ny) in body
        state.append(int(is_wall or is_body))

    for dx, dy in directions:
        step = 1
        while True:
            x = head_x + dx * step
            y = head_y + dy * step
            cell = board[x][y]

            if (x, y) in body:
                obj = OBJ_BODY
                break
            elif cell == GREEN_APPLE:
                obj = OBJ_GREEN
                break
            elif cell == RED_APPLE:
                obj = OBJ_RED
                break
            elif cell == WALL:
                obj = OBJ_WALL
                break
            else:
                step += 1
        state.append(obj)

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
