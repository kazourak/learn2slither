from enum import Enum


class Actions(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class ActionResult:
    replaced_cell = None
    new_state = None
    snake_length = None

    def __init__(self, replaced_cell, new_state, snake_length):
        self.replaced_cell = replaced_cell
        self.new_state = new_state
        self.snake_length = snake_length


def get_coordinates_from_action(action):
    if action == Actions.UP:
        return 0, -1
    elif action == Actions.DOWN:
        return 0, 1
    elif action == Actions.LEFT:
        return -1, 0
    elif action == Actions.RIGHT:
        return 1, 0
    else:
        raise ValueError("Invalid action")
