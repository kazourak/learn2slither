from collections import deque
from typing import Deque, Tuple
import numpy as np

from snake.action import ActionResult, ActionState
from snake.env import Coordinate


class Interpreter:
    def __init__(
            self,
            reward_nothing: float = -2.5,
            reward_green_apple: float = 25.0,
            reward_red_apple: float = -30.0,
            reward_dead: float = -100.0,
    ):
        self.reward_nothing = reward_nothing
        self.reward_green_apple = reward_green_apple
        self.reward_red_apple = reward_red_apple
        self.reward_dead = reward_dead

        self.EMPTY = 0
        self.WALL = 1
        self.HEAD = 2
        self.BODY = 3
        self.GREEN_APPLE = 4
        self.RED_APPLE = 5

        self.OBJ_GREEN = 0
        self.OBJ_RED = 1
        self.OBJ_BODY = 2
        self.OBJ_WALL = 3

    def get_reward(self, result: ActionResult) -> float:
        if result.action_state == ActionState.NOTHING:
            return self.reward_nothing
        elif result.action_state == ActionState.EAT_GREEN_APPLE:
            return self.reward_green_apple
        elif result.action_state == ActionState.EAT_RED_APPLE:
            return self.reward_red_apple
        elif result.action_state == ActionState.DEAD:
            return self.reward_dead
        else:
            return 0.0

    def get_state(
            self,
            snake: Deque[Tuple[int, int]],
            board: np.ndarray,
            current_direction: Tuple[int, int]
    ) -> Tuple[int, ...]:
        head_x, head_y = snake[0]
        body = set(list(snake)[:-1])
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        state = []

        for dx, dy in directions:
            nx, ny = head_x + dx, head_y + dy
            cell = board[ny][nx]
            is_wall = (cell == self.WALL)
            is_body = (nx, ny) in body
            state.append(int(is_wall or is_body))

        for dx, dy in directions:
            step = 1
            while True:
                x = head_x + dx * step
                y = head_y + dy * step
                cell = board[x][y]

                if (x, y) in body:
                    obj = self.OBJ_BODY
                    break
                elif cell == self.GREEN_APPLE:
                    obj = self.OBJ_GREEN
                    break
                elif cell == self.RED_APPLE:
                    obj = self.OBJ_RED
                    break
                elif cell == self.WALL:
                    obj = self.OBJ_WALL
                    break
                step += 1
            state.append(obj)

        # for dx, dy in directions:
        #     step = 1
        #     while True:
        #         x = head_x + dx * step
        #         y = head_y + dy * step
        #         cell = board[x][y]
        #
        #         if (x, y) in body or cell == self.WALL:
        #             if step <= 3:
        #                 dist = 0
        #             elif step <= 6:
        #                 dist = 1
        #             else:
        #                 dist = 2
        #             break
        #         step += 1
        #     state.append(dist)

        return tuple(state)
