import random
from collections import deque
from typing import Deque, List, Optional, Set, Tuple

import numpy as np

from snake.action import ActionResult, ActionState


# Cell values
EMPTY = 0
WALL = 1
HEAD = 2
BODY = 3
GREEN_APPLE = 4
RED_APPLE = 5
Coordinate = Tuple[int, int]


class SnakeEnv:
    """
    Snake game environment with wall boundaries, apples, and snake movement.
    """

    def __init__(
            self,
            map_size: int,
            snake_start_length: int,
            red_apple_count: int,
            green_apple_count: int,
            seed: Optional[int] = None,
    ) -> None:
        if map_size < 3:
            raise ValueError("map_size must be at least 3 to allow for walls and play area.")

        if snake_start_length < 2:
            raise ValueError("snake_start_length must be at least 2.")
        random.seed(seed)
        self.map_size = map_size
        self.snake_start_length = snake_start_length
        self.red_apple_count = red_apple_count
        self.green_apple_count = green_apple_count

        self.board: np.ndarray = np.zeros((map_size + 2, map_size + 2), dtype=int)
        self.snake: Deque[Coordinate] = deque()
        self.apples: dict[int, Set[Coordinate]] = {RED_APPLE: set(), GREEN_APPLE: set()}

        self.reset()

    def get_state(self) -> np.ndarray:
        return self.board.copy()

    def reset(self) -> None:
        """
        Reset the game state: walls, snake, and apples.x
        """
        self.snake: Deque[Coordinate] = deque()
        self.apples: dict[int, Set[Coordinate]] = {RED_APPLE: set(), GREEN_APPLE: set()}
        self._init_walls()
        self._place_snake()
        self._place_apples(RED_APPLE, self.red_apple_count)
        self._place_apples(GREEN_APPLE, self.green_apple_count)

    def step(self) -> ActionResult:
        dx, dy = self.direction
        head_x, head_y = self.snake[0]
        tail = self.snake[-1]
        new_head = (head_x + dx, head_y + dy)
        cell = self.board[new_head]

        if cell in (WALL, BODY) and new_head != tail:
            return ActionResult(ActionState.DEAD, None, len(self.snake), cell)

        self.snake.appendleft(new_head)
        self.board[head_x, head_y] = BODY

        if cell == EMPTY or new_head == tail:
            tail = self.snake.pop()
            if new_head != tail:
                self.board[tail] = EMPTY
            self.board[new_head] = HEAD

        elif cell == GREEN_APPLE:
            self.board[new_head] = HEAD
            self.apples[cell].remove(new_head)
            self._place_apples(cell, 1)
            return ActionResult(ActionState.EAT_GREEN_APPLE, self.board.copy(), len(self.snake))

        elif cell == RED_APPLE:
            self.board[new_head] = HEAD
            self.apples[cell].remove(new_head)
            self._place_apples(cell, 1)
            for _ in range(2):
                if len(self.snake) > 1:
                    tail = self.snake.pop()
                    self.board[tail] = EMPTY
                else:
                    return ActionResult(RED_APPLE, self.board.copy(), 0)
            return ActionResult(ActionState.EAT_RED_APPLE, self.board.copy(), len(self.snake))

        return ActionResult(ActionState.NOTHING, self.board.copy(), len(self.snake))

    def _init_walls(self) -> None:
        """Initialize the boundary walls."""
        self.board.fill(EMPTY)
        self.board[0, :] = WALL
        self.board[-1, :] = WALL
        self.board[:, 0] = WALL
        self.board[:, -1] = WALL

    def _place_snake(self) -> None:
        """Generate a contiguous snake in a random orientation."""
        self.snake.clear()
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            try:
                points = _generate_snake_body(self.board, self.snake_start_length)
                self.snake = deque(points)
                head_x, head_y = points[0]
                neck_x, neck_y = points[1]
                self.direction = (head_x - neck_x, head_y - neck_y)
                return
            except RuntimeError:
                attempts += 1
        raise RuntimeError("Failed to place initial snake after multiple attempts.")

    def _place_apples(self, apple_type: int, count: int) -> None:
        """Place a given number of apples of type apple_type at random empty positions."""
        empties = self._available_positions()
        if count > len(empties):
            count = len(empties)

        for pos in random.sample(empties, count):
            x, y = pos
            self.board[x, y] = apple_type
            self.apples[apple_type].add(pos)

    def _available_positions(self) -> List[Coordinate]:
        """Return list of empty coordinates inside the boundary."""
        empties: List[Coordinate] = []
        for x in range(1, self.map_size - 1):
            for y in range(1, self.map_size - 1):
                if self.board[x, y] == EMPTY:
                    empties.append((x, y))
        return empties


def _generate_snake_body(
        board: np.ndarray, length: int
) -> List[Coordinate]:
    """Recursively generate a contiguous snake. Raises RuntimeError if placement fails."""
    if length < 1:
        raise ValueError("Length must be at least 1.")

    size = board.shape[0]
    # choose random starting cell inside walls
    x = random.randint(1, size - 2)
    y = random.randint(1, size - 2)
    if board[x, y] != EMPTY:
        return _generate_snake_body(board, length)

    path = [(x, y)]
    board[x, y] = HEAD

    for _ in range(length - 1):
        neighbors = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        ]
        random.shuffle(neighbors)
        placed = False
        for dx, dy in neighbors:
            nx, ny = path[-1][0] + dx, path[-1][1] + dy
            if board[nx, ny] == EMPTY:
                board[nx, ny] = BODY
                path.append((nx, ny))
                placed = True
                break
        if not placed:
            raise RuntimeError("No space to place the next snake segment.")
    return path
