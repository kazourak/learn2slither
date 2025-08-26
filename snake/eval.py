import time

import re
from snake.action import index_to_action_tuple, ActionResult, ActionState
from snake.agent import QLearningSnakeAgent
from snake.env import SnakeEnv
from snake.interpreter import Interpreter
import statistics
from tqdm import tqdm  # Added tqdm for progress indication

WALL = 1
BODY = 3

total_time = 0


import statistics
import numpy as np
from tqdm import tqdm

def evaluate(agent: QLearningSnakeAgent, env: SnakeEnv, interpreter: Interpreter, episodes=10000, max_step=10000):
    global total_time
    eat_green_apple = 0
    eat_red_apple = 0
    dead_by_wall = 0
    dead_by_body = 0
    dead_by_size = 0
    stopped = 0

    snake_lengths = []

    # Wrapping the iteration with tqdm for a progress bar
    for _ in tqdm(range(episodes), desc="Evaluating Episodes"):
        env.reset()
        step = 0

        while True:
            if step >= max_step:
                stopped += 1
                break

            state = interpreter.get_state(env.snake, env.board, env.direction)
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

    # Calcul des statistiques
    if snake_lengths:
        min_length = min(snake_lengths)
        max_length = max(snake_lengths)
        mean_length = statistics.mean(snake_lengths)
        median_length = statistics.median(snake_lengths)
        std_length = statistics.stdev(snake_lengths) if len(snake_lengths) > 1 else 0

        # Calcul des quartiles
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
    print(f"\n=== STATISTIQUES DES LONGUEURS ===")
    print(f"Min snake length: {min_length}")
    print(f"Max snake length: {max_length}")
    print(f"Mean snake length: {mean_length:.2f}")
    print(f"Median snake length: {median_length}")
    print(f"Q1 (25th percentile): {q1_length}")
    print(f"Q3 (75th percentile): {q3_length}")
    print(f"Standard deviation: {std_length:.2f}")
    print(f"Total completed games: {len(snake_lengths)}")

    return {
        'min_length': min_length,
        'max_length': max_length,
        'mean_length': mean_length,
        'median_length': median_length,
        'q1_length': q1_length,
        'q3_length': q3_length,
        'std_length': std_length,
        'total_games': len(snake_lengths)
    }




def extract_rewards_from_filename(filename):
    """Extrait les valeurs de récompenses depuis le nom du fichier après 'r_'"""
    pattern = r"r_([-\d.]+)_([-\d.]+)_([-\d.]+)_([-\d.]+?)(?:\.pkl|$)"
    match = re.search(pattern, filename)

    if match:
        r_nothing = float(match.group(1).rstrip('.'))
        r_eat_green = float(match.group(2).rstrip('.'))
        r_eat_red = float(match.group(3).rstrip('.'))
        r_dead = float(match.group(4).rstrip('.'))
        return r_nothing, r_eat_green, r_eat_red, r_dead
    else:
        return -1.23, 20.58, -28.16, -113.51

if __name__ == "__main__":
    env = SnakeEnv(10, 3, 1, 2)
    filename = "models_t/model_4_47.0_55.0_15.294262484889547_r_-1.21_17.81_-14.23_-115.00.pkl"
    agent = QLearningSnakeAgent(filename=filename)

    # Extraction automatique des récompenses depuis le nom du fichier
    r_nothing, r_eat_green, r_eat_red, r_dead = extract_rewards_from_filename(filename)

    print(f"Récompenses extraites: nothing={r_nothing}, green={r_eat_green}, red={r_eat_red}, dead={r_dead}")

    interpreter = Interpreter(reward_nothing=r_nothing, reward_dead=r_dead, reward_red_apple=r_eat_red, reward_green_apple=r_eat_green)

    evaluate(agent, env, interpreter, episodes=5000, max_step=1000)
    print(f"Total time: {total_time}")