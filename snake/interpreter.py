from typing import Deque, Tuple
import numpy as np

from snake.action import ActionResult, ActionState


class Interpreter:
    def __init__(
            self,
            reward_nothing: float = -1.23,
            reward_green_apple: float = 20.58,
            reward_red_apple: float = -28.16,
            reward_dead: float = -113.51,
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
        self.TAIL = 4

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
            board: np.ndarray
    ) -> Tuple[int, ...]:
        head_x, head_y = snake[0]
        body = set(list(snake)[:-1])
        tail = snake[-1]
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

                if (x, y) == tail:
                    obj = self.TAIL
                    break
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

        return tuple(state)

    def print_vision(self, board: np.ndarray):
        head_pos = np.where(board == self.HEAD)
        if len(head_pos[0]) == 0:
            return

        head_y, head_x = head_pos[0][0], head_pos[1][0]

        height, width = board.shape

        symbols = {
            self.EMPTY: '0',
            self.WALL: 'W',
            self.HEAD: 'H',
            self.BODY: 'S',
            self.GREEN_APPLE: 'G',
            self.RED_APPLE: 'R'
        }

        for x in range(height):
            row = ""
            for y in range(width):
                if y == head_y or x == head_x:
                    cell_value = board[y][x]
                    row += symbols.get(cell_value, '?') + ' '
                else:
                    row += '  '
            print(row)


def snake_go_to_green_apple(state: tuple, new_state: tuple) -> bool:
    last_state = state[-4:]
    last_new_state = new_state[-4:]

    if 0 not in last_state or 0 not in last_new_state:
        return False

    for i in range(4):
        if last_state[i] == last_new_state[i] == 0:
            return True
    return False
