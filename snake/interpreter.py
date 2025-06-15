from action import ActionResult

action_weights = {
    "GREEN_APPLE": 10,
    "RED_APPLE": -5,
    "NOTHING": -1,
    "DEAD": -100
}


EMPTY = 0
WALL = 1
HEAD = 2
BODY = 3
GREEN_APPLE = 4
RED_APPLE = 5


def get_reward(action: ActionResult):
    reward_score = 0

    if action.snake_length < 1:
        reward_score += action_weights["DEAD"]

    if action.replaced_cell == EMPTY:
        reward_score += action_weights["NOTHING"]
    elif action.replaced_cell in (BODY, WALL):
        reward_score += action_weights["DEAD"]
    elif action.replaced_cell == GREEN_APPLE:
        reward_score += action_weights["GREEN_APPLE"]
    elif action.replaced_cell == RED_APPLE:
        reward_score += action_weights["RED_APPLE"]

    return reward_score
