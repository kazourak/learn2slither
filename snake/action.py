from enum import Enum


class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


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
        return -1, 0
    elif action == Actions.DOWN:
        return 1, 0
    elif action == Actions.LEFT:
        return 0, -1
    elif action == Actions.RIGHT:
        return 0, 1
    else:
        raise ValueError("Invalid action")
