import time

from snake.action import ActionResult, index_to_action_tuple, ActionState
from snake.agent import QLearningSnakeAgent
from snake.env import SnakeEnv
from snake.interpreter import get_state, get_reward

import numpy as np

def train(agent: QLearningSnakeAgent, env: SnakeEnv, num_episodes, max_steps_per_episode, model_path=None):
    snake_len = []
    for episode in range(num_episodes):
        env.reset()
        state = get_state(env.snake, env.apples)
        total_reward = 0.0
        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            action_idx: int = agent.choose_action(state)
            if episode > 20000:
                print_board(env.board)
                time.sleep(0.1)

            env.direction = index_to_action_tuple(action_idx)
            result: ActionResult = env.step()
            if result.action_state == ActionState.DEAD:
                done = True

            next_state = get_state(env.snake, env.apples)
            reward = get_reward(result)
            total_reward += reward

            agent.update(state, action_idx, reward, next_state, done)
            state = next_state
            step += 1

        snake_len.append(len(env.snake))
        agent.decay_epsilon()

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Total reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

        if (episode + 1) % 1000 == 0:
            avg_length = np.median(snake_len)
            print(f"➡️ Medianne de la taille du snake (épisodes {episode-999} à {episode}): {avg_length:.2f}")
            snake_len = []  # Réinitialisation

    if model_path:
        agent.save_model(model_path)
        print(f"Modèle sauvegardé dans {model_path}")

import os

# Cell types
EMPTY = 0
WALL = 1
HEAD = 2
BODY = 3
GREEN_APPLE = 4
RED_APPLE = 5

# ANSI escape sequences
_RESET = '\033[0m'
_BLACK_BG = '\033[40m'
_WHITE_FG = '\033[97m'
_GREEN_FG = '\033[92m'
_RED_FG = '\033[91m'
_YELLOW_FG = '\033[93m'
_BLUE_FG = '\033[94m'

# Mapping from cell value → colored string
_CELL_CHARS = {
    EMPTY:    f'{_BLACK_BG}   {_RESET}',                 # black background
    WALL:     f'{_BLACK_BG}{_WHITE_FG} # {_RESET}',      # white wall on black
    HEAD:     f'{_BLACK_BG}{_YELLOW_FG} H {_RESET}',     # yellow head
    BODY:     f'{_BLACK_BG}{_GREEN_FG} o {_RESET}',      # green body
    GREEN_APPLE: f'{_BLACK_BG}{_GREEN_FG} G {_RESET}',   # bright green apple
    RED_APPLE:   f'{_BLACK_BG}{_RED_FG} R {_RESET}',     # bright red apple
}

def print_board(board: np.ndarray, clear_screen: bool = True):
    """
    Print the snake game board to the terminal with colors.

    Args:
        board (np.ndarray): 2D array of ints encoding cell types.
        clear_screen (bool): If True, clears the terminal before printing.
    """
    if clear_screen:
        os.system('cls' if os.name == 'nt' else 'clear')

    for row in board:
        line = ''.join(_CELL_CHARS.get(int(cell), f'{_BLACK_BG}? {_RESET}') for cell in row)
        print(line)


if __name__ == '__main__':
    env = SnakeEnv(10, 3, 1, 2)
    agent = QLearningSnakeAgent()
    train(agent, env, num_episodes=100000, max_steps_per_episode=100000, model_path='snake.pkl')