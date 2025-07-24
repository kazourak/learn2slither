from enum import Enum


class Actions(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class ActionState(Enum):
    NOTHING = 0,
    EAT_RED_APPLE = 1,
    EAT_GREEN_APPLE = 2,
    DEAD = 3

class ActionResult:
    action_state = None
    new_state = None
    snake_length = None
    cause_death = None

    def __init__(self, action_state, new_state, snake_length, cause_death=None):
        self.action_state = action_state
        self.new_state = new_state
        self.snake_length = snake_length
        self.cause_death = cause_death


def index_to_action_tuple(index: int) -> tuple[int, int]:
    return list(Actions)[index].value


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
