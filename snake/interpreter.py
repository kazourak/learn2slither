from typing import Tuple, Deque, Set

from snake.action import ActionResult, ActionState
from snake.env import Coordinate

GREEN_APPLE = 4
RED_APPLE = 5

def get_reward(result: ActionResult) -> float:
    if result.action_state == ActionState.NOTHING:
        return -0.1
    elif result.action_state == ActionState.EAT_GREEN_APPLE:
        return 2
    elif result.action_state == ActionState.EAT_RED_APPLE:
        return -2
    elif result.action_state == ActionState.DEAD:
        return -10
    else:
        return 0


def get_state(snake: Deque[Coordinate], apples: dict[int, Set[Coordinate]]) -> tuple:
    head_coord = snake[0]
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
        green_apple_close = apple_in_direction(apples, head_coord, (dx, dy), GREEN_APPLE,  10)
        red_apple_close = apple_in_direction(apples, head_coord, (dx, dy), RED_APPLE, 3)
        state.append(int(green_apple_close))
        state.append(int(red_apple_close))

    labels = [
        "Danger up", "Green apple up", "Red apple up",
        "Danger down", "Green apple down", "Red apple down",
        "Danger left", "Green apple left", "Red apple left",
        "Danger right", "Green apple right", "Red apple right"
    ]

    return tuple(state)


def apple_in_direction(apples: dict[int, Set[Coordinate]], head_pos, direction, apple_type, max_dist=3):
    dx, dy = direction
    x, y = head_pos

    for distance in range(1, max_dist + 1):
        check_x = x + (dx * distance)
        check_y = y + (dy * distance)

        if check_x < 1 or check_x >= 11 or check_y < 1 or check_y >= 11:
            break

        if (check_x, check_y) in apples[apple_type]:
            return True

    return False
