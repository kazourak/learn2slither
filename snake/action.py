from enum import Enum


action_weights = {
    "EAT_GREEN_APPLE": 10,
    "EAT_RED_APPLE": -5,
    "NOTHING": -1,
    "DEAD": -100
}


class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class ActionResult:
    reward = 0
    new_state = None

    def __init__(self, reward, new_state):
        self.reward = reward
        self.new_state = new_state


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
