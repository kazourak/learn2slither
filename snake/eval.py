from snake.action import index_to_action_tuple, ActionResult, ActionState
from snake.agent import QLearningSnakeAgent
from snake.env import SnakeEnv
from snake.interpreter import Interpreter
import statistics
from tqdm import tqdm  # Added tqdm for progress indication

WALL = 1
BODY = 3


def evaluate(agent: QLearningSnakeAgent, env: SnakeEnv, interpreter: Interpreter, episodes=10000, max_step=10000):
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

    min_length = min(snake_lengths) if snake_lengths else 0
    max_length = max(snake_lengths) if snake_lengths else 0
    median_length = statistics.median(snake_lengths) if snake_lengths else 0

    print(f"Eat green apple: {eat_green_apple}")
    print(f"Eat red apple: {eat_red_apple}")
    print(f"Dead by wall: {dead_by_wall}")
    print(f"Dead by body: {dead_by_body}")
    print(f"Dead by size: {dead_by_size}")
    print(f"Stopped: {stopped}")
    print(f"Min snake length: {min_length}")
    print(f"Max snake length: {max_length}")
    print(f"Median snake length: {median_length}")
    # print(snake_lengths)
    return min_length, max_length, median_length



if __name__ == "__main__":
    env = SnakeEnv(10, 3, 1, 2)
    agent = QLearningSnakeAgent(filename="best_model.pkl")
    r_nothing = -1.23
    r_eat_green = 20.58
    r_eat_red = -28.16
    r_dead = -113.51
    interpreter = Interpreter(reward_nothing=r_nothing, reward_dead=r_dead, reward_red_apple=r_eat_red, reward_green_apple=r_eat_green)

    evaluate(agent, env, interpreter, episodes=1000, max_step=1000)