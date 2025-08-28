from __future__ import annotations

from dataclasses import dataclass
from typing import List

from tqdm import trange

from snake.phases import optimal_cfg, get_standard_phases_cfg, basic_cfg, intensive_cfg
from snake.action import ActionResult, index_to_action_tuple, ActionState
from snake.agent import QLearningSnakeAgent
from snake.env import SnakeEnv
from snake.interpreter import Interpreter


@dataclass
class PhaseConfig:
    name: str
    episodes: int
    eps_start: float = 0.0
    eps_end: float = 0.0

def train_with_phases(
        agent: QLearningSnakeAgent,
        env: SnakeEnv,
        interpreter: Interpreter,
        phases: List[PhaseConfig],
        max_steps_per_episode: int
):
    global_episode = 0
    episode_rewards = []
    phase_stats = {}

    for phase in phases:
        agent.epsilon = phase.eps_start
        agent.eps_min = phase.eps_end
        agent.calc_eps_decay(phase.episodes)

        phase_rewards = []

        iterator = trange(
            phase.episodes,
            desc=f"Training session: {phase.name}",
            unit="ep",
            leave=True,
        )

        for local_ep in iterator:
            global_episode += 1
            env.reset()

            state = interpreter.get_state(env.snake, env.board)
            total_reward = 0.0
            done = False
            step = 0

            while not done and step < max_steps_per_episode:
                action_idx = agent.choose_action(state)

                env.direction = index_to_action_tuple(action_idx)
                result: ActionResult = env.step()
                if result.action_state == ActionState.DEAD:
                    done = True

                next_state = interpreter.get_state(env.snake, env.board)
                reward = interpreter.get_reward(result)
                total_reward += reward

                if agent.is_train:
                    agent.update(state, action_idx, reward, next_state, done)

                state = next_state
                step += 1

            agent.decay_epsilon()

            episode_rewards.append(total_reward)
            phase_rewards.append(total_reward)

    if agent.save_path:
        agent.save_model()
    return episode_rewards, phase_stats


def train_model(l_path: str, s_path: str, episodes: int | None, phase: str | None):
    env = SnakeEnv(10, 3, 1, 2)
    interpreter = Interpreter(reward_nothing=-1.14, reward_dead=-115, reward_green_apple=19.14, reward_red_apple=-21.96)
    agent = QLearningSnakeAgent(
        load_path=l_path, save_path=s_path, train=True
    )

    phases_to_use = get_phase(phase, episodes)

    episode_rewards, phase_stats = train_with_phases(
        agent=agent,
        env=env,
        interpreter=interpreter,
        phases=phases_to_use,
        max_steps_per_episode=5000
    )

    for name, stats in phase_stats.items():
        print(f"   {name}: {stats['avg_reward']:.2f} (avg), {stats['max_reward']:.2f} (max)")

def get_phase(phase: str | None, episodes: int):
    if phase is None:
        get_standard_phases_cfg(episodes)
    if phase == "basic":
        return basic_cfg
    elif phase == "intensive":
        return intensive_cfg
    elif phase == "optimal":
        return optimal_cfg
    else:
        return get_standard_phases_cfg(1)