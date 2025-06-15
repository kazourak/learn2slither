# Environment constants for the game
import random
from action import Actions, get_coordinates_from_action, ActionResult
from collections import deque

import numpy as np

EMPTY = 0
WALL = 1
HEAD = 2
BODY = 3
GREEN_APPLE = 4
RED_APPLE = 5


class SnakeEnv:
    def __init__(self, map_size: int, snake_start_length: int, red_apple_nb: int, green_apple_nb: int):
        self.snake = deque()
        # self.start_direction = None
        self.map = None
        self.map_size = map_size
        self.snake_start_length = snake_start_length
        self.red_apple_nb = red_apple_nb
        self.green_apple_nb = green_apple_nb
        self.apples = None
        self.reset()

    def reset(self):
        # Generate map and walls
        self.map = np.zeros((self.map_size + 2, self.map_size + 2))
        self.map[0, :] = WALL
        self.map[-1, :] = WALL
        self.map[:, 0] = WALL
        self.map[:, -1] = WALL

        # Generate the snake
        points = generate_contiguous_points(self.map, self.snake_start_length)
        self.snake = deque(points)

        # Set the snake starting direction
        # self.start_direction = (points[0][0] - points[1][0], points[0][1] - points[1][1])

        # Place apples
        self.apples = {RED_APPLE: set(), GREEN_APPLE: set()}
        self.place_apple(RED_APPLE, self.red_apple_nb)
        self.place_apple(GREEN_APPLE, self.green_apple_nb)
        pass


    def get_snake_length(self):
        return len(self.snake)


    def place_apple(self, apple_type, count):
        """
        Place an apple randomly in the map.
        """
        available_coordinates = self.get_available_coordinates()
        if count < 1 or len(available_coordinates) == 0:
            return None

        x, y = random.choice(available_coordinates)
        self.map[x, y] = apple_type
        self.apples[apple_type].add((x, y))
        return self.place_apple(apple_type, count - 1) if count > 0 else None


    def get_available_coordinates(self):
        """
        Get a list of available coordinates in the map.
        :return: List of available coordinates.
        """
        available_coordinates = []
        for x in range(1, self.map_size + 1):
            for y in range(1, self.map_size + 1):
                if self.map[x, y] == EMPTY:
                    available_coordinates.append((x, y))
        return available_coordinates


    def perform_action(self, action: Actions):
        action_to_perform = get_coordinates_from_action(action)
        new_head = (self.snake[0][0] + action_to_perform[0], self.snake[0][1] + action_to_perform[1])
        cell_value = self.map[new_head[0], new_head[1]]

        if cell_value == WALL:
            return ActionResult(WALL, None, self.get_snake_length())
        elif cell_value == BODY:
            return ActionResult(BODY, None, self.get_snake_length())
        elif cell_value == GREEN_APPLE:
            self.snake.appendleft(new_head)
            self.map[new_head[0], new_head[1]] = HEAD
            self.map[self.snake[1][0], self.snake[1][1]] = BODY
            self.apples[GREEN_APPLE].remove(new_head)
            self.place_apple(GREEN_APPLE, 1)
            return ActionResult(GREEN_APPLE, self.map, self.get_snake_length())
        elif cell_value == RED_APPLE:
            self.snake.appendleft(new_head)
            self.map[new_head[0], new_head[1]] = HEAD
            self.map[self.snake[1][0], self.snake[1][1]] = BODY
            for _ in range(2):
                if len(self.snake) > 0:
                    tail = self.snake.pop()
                    self.map[tail[0], tail[1]] = EMPTY
            self.apples[RED_APPLE].remove(new_head)
            self.place_apple(RED_APPLE, 1)
            return ActionResult(RED_APPLE, self.map, self.get_snake_length())
        elif cell_value == EMPTY:
            tail = self.snake.pop()
            self.map[tail[0], tail[1]] = EMPTY
            self.snake.appendleft(new_head)
            self.map[new_head[0], new_head[1]] = HEAD
            if len(self.snake) > 1:
                self.map[self.snake[1][0], self.snake[1][1]] = BODY
            return ActionResult(EMPTY, self.map, self.get_snake_length())
        return None


def generate_contiguous_points(board, length=3):
    head_x = random.randint(1, 9)
    head_y = random.randint(1, 9)
    board[head_x, head_y] = HEAD
    return [(head_x, head_y)] + get_next_point(board, head_x, head_y, length - 1)


def get_next_point(board, previous_x, previous_y, length):
    # DOWN, RIGHT, UP, LEFT
    directions = [(1,0), (0,1), (-1,0), (0,-1)]
    d = random.choice(directions)
    try:
        if board[previous_x + d[0], previous_y + d[1]] == EMPTY:
            board[previous_x + d[0], previous_y + d[1]] = BODY
            if length == 1:
                return [(previous_x + d[0], previous_y + d[1])]
            else:
                next_points = get_next_point(board, previous_x + d[0], previous_y + d[1], length - 1)
                return [(previous_x + d[0], previous_y + d[1])] + next_points
        else:
            return get_next_point(board, previous_x, previous_y, length)
    except IndexError:
        return get_next_point(board, previous_x, previous_y, length)



