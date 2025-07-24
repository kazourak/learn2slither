from __future__ import annotations

from dataclasses import dataclass
from typing import List

from tqdm import trange

from snake.action import ActionResult, index_to_action_tuple, ActionState
from snake.agent import QLearningSnakeAgent
from snake.env import SnakeEnv
from snake.interpreter import get_state, get_reward

import numpy as np


@dataclass
class PhaseConfig:
    """
    Décrit une phase d’entraînement / d’évaluation.

    - name      : Name TQDM progress bar.
    - episodes  : Numbers of episodes to perform in this phase.
    - eps_start : Epsilon initial. Default: 0.
    - eps_end   : Final epsilon. Default: 0.
    - train     : True  -> Q-Learning actif (Exploration/Exploitation).
                  False -> Pas d’apprentissage, simple évaluation.
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
        phases: List[PhaseConfig],
        max_steps_per_episode: int,
        model_path: str | None = None,
):
    global_episode = 0

    for phase in phases:
        agent.epsilon = phase.eps_start
        agent.eps_decay = phase.eps_decay
        agent.is_train = phase.train

        iterator = trange(
            phase.episodes,
            desc=f"Phase: {phase.name}",
            unit="ep",
            leave=True,
        )

        for local_ep in iterator:
            global_episode += 1
            env.reset()

            state = get_state(env.snake, env.board, env.direction)
            total_reward = 0.0
            done = False
            step = 0

            while not done and step < max_steps_per_episode:
                action_idx = agent.choose_action(state)

                env.direction = index_to_action_tuple(action_idx)
                result: ActionResult = env.step()
                if result.action_state == ActionState.DEAD:
                    done = True

                next_state = get_state(env.snake, env.board, env.direction)
                reward = get_reward(result)
                total_reward += reward

                if agent.is_train:
                    agent.update(state, action_idx, reward, next_state, done)

                state = next_state
                step += 1

            if agent.is_train:
                agent.decay_epsilon()


            # if global_episode % 1000 == 0:
            #     iterator.write(
            #         f"Episode {global_episode} | "
            #         f"phase «{phase.name}» ({local_ep + 1}/{phase.episodes}) | "
            #         f"reward={total_reward:.2f} | "
            #         f"eps={agent.epsilon:.4f}"
            #     )

    if model_path:
        agent.save_model(model_path)
        print(f"✅ Modèle sauvegardé dans {model_path}")


if __name__ == "__main__":
    env = SnakeEnv(10, 3, 1, 2)
    agent = QLearningSnakeAgent()

    phases_cfg = [
        # 1. Warm‑up / Découverte aléatoire
        PhaseConfig(
            name="Warm‑up aléatoire",
            episodes=10_000,
            eps_start=1.00,
            eps_end=0.80,
            train=True
        ),

        # 2. Exploration intensive
        PhaseConfig(
            name="Exploration intensive",
            episodes=40_000,
            eps_start=0.80,
            eps_end=0.30,
            train=True
        ),

        # 3. Exploration modérée (annealing plus lentement)
        PhaseConfig(
            name="Exploration modérée",
            episodes=80_000,
            eps_start=0.30,
            eps_end=0.05,
            train=True
        ),

        # 4. Exploitation / Raffinement
        PhaseConfig(
            name="Exploitation poussée",
            episodes=30_000,
            eps_start=0.05,
            eps_end=0.01,
            train=True
        ),

        # 5. Stabilisation (epsilon fixe pour stabiliser les Q‑valeurs)
        PhaseConfig(
            name="Stabilisation",
            episodes=10_000,
            eps_start=0.01,
            eps_end=0.01,
            train=True
        ),

        # 6. Évaluation finale (greedy)
        PhaseConfig(
            name="Évaluation finale",
            episodes=5_000,
            eps_start=0.00,
            eps_end=0.00,
            train=False
        ),
    ]

    train_with_phases(
        agent=agent,
        env=env,
        phases=phases_cfg,
        max_steps_per_episode=10000,
        model_path="snake.pkl",
    )