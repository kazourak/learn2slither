from snake.action import index_to_action_tuple, ActionResult, ActionState
from snake.agent import QLearningSnakeAgent
from snake.env import SnakeEnv
from snake.interpreter import Interpreter

WALL = 1
BODY = 3

import statistics
import numpy as np
from tqdm import tqdm

def evaluate(model_path: str, episodes: int = 5000, map_size: int = 10, max_step: int = 2500):
    eat_green_apple = 0
    eat_red_apple = 0
    dead_by_wall = 0
    dead_by_body = 0
    dead_by_size = 0
    stopped = 0

    snake_lengths = []

    env = SnakeEnv(map_size, 3, 1, 2)
    agent = QLearningSnakeAgent(load_path=model_path, train=False)

    interpreter = Interpreter()

    for _ in tqdm(range(episodes), desc="Evaluating Episodes"):
        env.reset()
        step = 0

        while True:
            if step >= max_step:
                stopped += 1
                break

            state = interpreter.get_state(env.snake, env.board)
            action_idx = agent.choose_action(state)
            env.direction = index_to_action_tuple(action_idx)
            result: ActionResult = env.step()

            if result.action_state == ActionState.EAT_GREEN_APPLE:
                eat_green_apple += 1
            elif result.action_state == ActionState.EAT_RED_APPLE:
                if result.snake_length == 0:
                    dead_by_size += 1
                    snake_lengths.append(0)
                    env.reset()
                eat_red_apple += 1

            if result.action_state == ActionState.DEAD:
                if result.cause_death == WALL:
                    dead_by_wall += 1
                else:
                    dead_by_body += 1

                snake_lengths.append(result.snake_length)
                env.reset()
                break
            step += 1

    if snake_lengths:
        min_length = min(snake_lengths)
        max_length = max(snake_lengths)
        mean_length = statistics.mean(snake_lengths)
        median_length = statistics.median(snake_lengths)
        std_length = statistics.stdev(snake_lengths) if len(snake_lengths) > 1 else 0

        q1_length = np.percentile(snake_lengths, 25)
        q3_length = np.percentile(snake_lengths, 75)
    else:
        min_length = max_length = mean_length = median_length = std_length = 0
        q1_length = q3_length = 0

    print(f"Eat green apple: {eat_green_apple}")
    print(f"Eat red apple: {eat_red_apple}")
    print(f"Dead by wall: {dead_by_wall}")
    print(f"Dead by body: {dead_by_body}")
    print(f"Dead by size: {dead_by_size}")
    print(f"Stopped: {stopped}")
    print(f"Min snake length: {min_length}")
    print(f"Max snake length: {max_length}")
    print(f"Mean snake length: {mean_length:.2f}")
    print(f"Median snake length: {median_length}")
    print(f"Q1 (25th percentile): {q1_length}")
    print(f"Q3 (75th percentile): {q3_length}")
    print(f"Standard deviation: {std_length:.2f}")
    print(f"Total completed games: {len(snake_lengths)}")