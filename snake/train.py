from snake.action import ActionResult
from snake.agent import QLearningSnakeAgent
from snake.env import SnakeEnv
from snake.interpreter import get_state, get_reward


def train(agent: QLearningSnakeAgent, env: SnakeEnv, num_episodes, max_steps_per_episode, model_path=None):
    for episode in range(num_episodes):
        env.reset()
        state = get_state(env.snake, env.apples)
        total_reward = 0.0
        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            action_idx = agent.choose_action(state)

            env.direction = action_idx
            result: ActionResult = env.step()
            next_state = get_state(env.snake, env.apples)
            reward = get_reward(result)
            total_reward += reward

            agent.update(state, action_idx, reward, next_state, done)
            state = next_state
            step += 1

        agent.decay_epsilon()

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Total reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    if model_path:
        agent.save_model(model_path)
        print(f"Modèle sauvegardé dans {model_path}")

if __name__ == '__main__':
    env = SnakeEnv(10, 3, 1, 2)
    agent = QLearningSnakeAgent()
    train(agent, env, num_episodes=1000, max_steps_per_episode=500, model_path='snake.pkl')