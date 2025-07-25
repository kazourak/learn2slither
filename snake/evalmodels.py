import os
from snake.action import index_to_action_tuple, ActionResult, ActionState
from snake.agent import QLearningSnakeAgent
from snake.env import SnakeEnv
from snake.interpreter import Interpreter
import statistics
from tqdm import tqdm

WALL = 1
BODY = 3
MODELS_DIR = "test_models"
EVAL_RUNS = 10  # Number of evaluations per model
EPISODES = 1000
MAX_STEP = 1000


def evaluate(agent: QLearningSnakeAgent, env: SnakeEnv, interpreter: Interpreter, episodes=EPISODES, max_step=MAX_STEP):
    """
    Run one batch of episodes and return min, max, median snake lengths across episodes.
    """
    snake_lengths = []

    for _ in range(episodes):
        env.reset()
        step = 0
        while True:
            if step >= max_step:
                # Treat as zero-length if stopped
                snake_lengths.append(0)
                break

            state = interpreter.get_state(env.snake, env.board, env.direction)
            action_idx = agent.choose_action(state)
            env.direction = index_to_action_tuple(action_idx)
            result: ActionResult = env.step()

            if result.action_state == ActionState.DEAD:
                # Record length at death
                snake_lengths.append(result.snake_length)
                break
            step += 1

    min_length = min(snake_lengths) if snake_lengths else 0
    max_length = max(snake_lengths) if snake_lengths else 0
    median_length = statistics.median(snake_lengths) if snake_lengths else 0
    return min_length, max_length, median_length


if __name__ == "__main__":
    env = SnakeEnv(10, 3, 1, 2)
    r_nothing = -1.23
    r_eat_green = 20.58
    r_eat_red = -28.16
    r_dead = -113.51
    interpreter = Interpreter(reward_nothing=r_nothing, reward_dead=r_dead, reward_red_apple=r_eat_red, reward_green_apple=r_eat_green)

    model_files = [f for f in os.listdir(MODELS_DIR)
                   if os.path.isfile(os.path.join(MODELS_DIR, f)) and f.endswith('.pkl')]

    for model_file in model_files:
        model_path = os.path.join(MODELS_DIR, model_file)
        agent = QLearningSnakeAgent(filename=model_path)

        # Collect median results from multiple evaluation runs
        medians = []
        mins = []
        maxs = []

        print(f"\nEvaluating model: {model_file}")
        for run in tqdm(range(EVAL_RUNS), desc=f"Runs for {model_file}"):
            min_len, max_len, med_len = evaluate(agent, env, interpreter)
            mins.append(min_len)
            maxs.append(max_len)
            medians.append(med_len)

        avg_median = statistics.mean(medians)
        overall_min = min(mins)
        overall_max = max(maxs)

        print(f"Model: {model_file}")
        print(f"Average of medians over {EVAL_RUNS} runs: {avg_median:.2f}")
        print(f"Minimum snake length observed: {overall_min}")
        print(f"Maximum snake length observed: {overall_max}")
        print("-" * 50)
