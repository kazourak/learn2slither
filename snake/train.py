from __future__ import annotations

from dataclasses import dataclass
from typing import List

from tqdm import trange

from snake.action import ActionResult, index_to_action_tuple, ActionState
from snake.agent import QLearningSnakeAgent
from snake.env import SnakeEnv
from snake.interpreter import Interpreter

import numpy as np


@dataclass
class PhaseConfig:
    """
    Décrit une phase d'entraînement / d'évaluation.

    - name      : Name TQDM progress bar.
    - episodes  : Numbers of episodes to perform in this phase.
    - eps_start : Epsilon initial. Default: 0.
    - eps_end   : Final epsilon. Default: 0.
    - train     : True  -> Q-Learning actif (Exploration/Exploitation).
                  False -> Pas d'apprentissage, simple évaluation.
    """
    name: str
    episodes: int
    eps_start: float = 0.0
    eps_end: float = 0.0
    train: bool = True

    @property
    def eps_decay(self) -> float:
        if self.eps_start == 0 or self.episodes <= 1:
            return 1.0
        return (self.eps_end / self.eps_start) ** (1 / self.episodes)


def train_with_phases(
        agent: QLearningSnakeAgent,
        env: SnakeEnv,
        interpreter: Interpreter,
        phases: List[PhaseConfig],
        max_steps_per_episode: int,
        model_path: str | None = None,
):
    global_episode = 0
    episode_rewards = []
    phase_stats = {}

    for phase in phases:
        agent.epsilon = phase.eps_start
        agent.eps_decay = phase.eps_decay
        agent.is_train = phase.train

        phase_rewards = []

        iterator = trange(
            phase.episodes,
            desc=f"Phase: {phase.name}",
            unit="ep",
            leave=True,
        )

        for local_ep in iterator:
            global_episode += 1
            env.reset()

            state = interpreter.get_state(env.snake, env.board, env.direction)
            total_reward = 0.0
            done = False
            step = 0

            while not done and step < max_steps_per_episode:
                action_idx = agent.choose_action(state)

                env.direction = index_to_action_tuple(action_idx)
                result: ActionResult = env.step()
                if result.action_state == ActionState.DEAD:
                    done = True

                next_state = interpreter.get_state(env.snake, env.board, env.direction)
                reward = interpreter.get_reward(result)
                total_reward += reward

                if agent.is_train:
                    agent.update(state, action_idx, reward, next_state, done)

                state = next_state
                step += 1

            if agent.is_train:
                agent.decay_epsilon()

            episode_rewards.append(total_reward)
            phase_rewards.append(total_reward)

    if model_path:
        agent.save_model(model_path)
        print(f"✅ Modèle sauvegardé dans {model_path}")

    return episode_rewards, phase_stats


if __name__ == "__main__":
    env = SnakeEnv(10, 3, 1, 2)
    interpreter = Interpreter()
    agent = QLearningSnakeAgent(
        alpha=0.1,           # Learning rate
        gamma=0.95,          # Discount factor (plus élevé pour Snake)
    )

    phases_cfg = [
        PhaseConfig(
            name="Exploration initiale",
            episodes=70_000,
            eps_start=1.00,
            eps_end=0.70,
            train=True
        ),
        PhaseConfig(
            name="Exploration intensive",
            episodes=120_000,
            eps_start=0.70,
            eps_end=0.30,
            train=True
        ),
        PhaseConfig(
            name="Équilibrage Exp/Exp",
            episodes=95_000,
            eps_start=0.30,
            eps_end=0.10,
            train=True
        ),
        PhaseConfig(
            name="Exploitation dominante",
            episodes=70_000,
            eps_start=0.10,
            eps_end=0.02,
            train=True
        ),
        PhaseConfig(
            name="Fine-tuning",
            episodes=50_000,
            eps_start=0.02,
            eps_end=0.005,
            train=True
        ),
        PhaseConfig(
            name="Stabilisation",
            episodes=40_000,
            eps_start=0.005,
            eps_end=0.001,
            train=True
        )
    ]

    print("🐍 Démarrage de l'entraînement Snake Q-Learning")
    print(f"📋 Configuration: {len(phases_cfg)} phases, {sum(p.episodes for p in phases_cfg):,} épisodes totaux")
    print(f"🧠 Agent: α={agent.alpha}, γ={agent.gamma}")
    print("=" * 60)

    episode_rewards, phase_stats = train_with_phases(
        agent=agent,
        env=env,
        interpreter=interpreter,
        phases=phases_cfg,
        max_steps_per_episode=10000,
        model_path="snake_2.pkl",
    )

    print("🎉 Entraînement terminé!")
    print("📊 Résumé final:")
    for name, stats in phase_stats.items():
        print(f"   {name}: {stats['avg_reward']:.2f} (avg), {stats['max_reward']:.2f} (max)")