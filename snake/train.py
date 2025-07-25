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

            episode_rewards.append(total_reward)
            phase_rewards.append(total_reward)

            if local_ep % 100 == 0 and len(phase_rewards) >= 100:
                avg_reward_100 = np.mean(phase_rewards[-100:])
                iterator.set_postfix({
                    'avg_reward_100': f'{avg_reward_100:.2f}',
                    'epsilon': f'{agent.epsilon:.4f}',
                    'q_states': len(agent.q_table) if hasattr(agent, 'q_table') else 0
                })

        phase_stats[phase.name] = {
            'avg_reward': np.mean(phase_rewards),
            'max_reward': np.max(phase_rewards),
            'min_reward': np.min(phase_rewards),
            'final_epsilon': agent.epsilon
        }

        print(f"📊 Phase '{phase.name}' terminée:")
        print(f"   Récompense moyenne: {phase_stats[phase.name]['avg_reward']:.2f}")
        print(f"   Récompense max: {phase_stats[phase.name]['max_reward']:.2f}")
        print(f"   Epsilon final: {phase_stats[phase.name]['final_epsilon']:.4f}")
        print(f"   États Q-table: {len(agent.q_table) if hasattr(agent, 'q_table') else 0}")
        print()

    if model_path:
        agent.save_model(model_path)
        print(f"✅ Modèle sauvegardé dans {model_path}")

    return episode_rewards, phase_stats


if __name__ == "__main__":
    env = SnakeEnv(10, 3, 1, 2)
    agent = QLearningSnakeAgent(
        alpha=0.1,           # Learning rate
        gamma=0.95,          # Discount factor (plus élevé pour Snake)
        epsilon=1.0,         # Sera géré par les phases
        eps_decay=0.999,     # Sera calculé automatiquement
        eps_min=0.001        # Minimum epsilon
    )

    phases_cfg = [
        PhaseConfig(
            name="🔍 Exploration initiale",
            episodes=15_000,
            eps_start=1.00,
            eps_end=0.70,
            train=True
        ),

        PhaseConfig(
            name="Exploration intensive",
            episodes=50_000,
            eps_start=0.70,
            eps_end=0.30,
            train=True
        ),

        PhaseConfig(
            name="Équilibrage Exp/Exp",
            episodes=75_000,
            eps_start=0.30,
            eps_end=0.10,
            train=True
        ),

        PhaseConfig(
            name="Exploitation dominante",
            episodes=40_000,
            eps_start=0.10,
            eps_end=0.02,
            train=True
        ),

        PhaseConfig(
            name="Fine-tuning",
            episodes=20_000,
            eps_start=0.02,
            eps_end=0.005,
            train=True
        ),

        PhaseConfig(
            name="Stabilisation",
            episodes=10_000,
            eps_start=0.005,
            eps_end=0.001,
            train=True
        ),

        PhaseConfig(
            name="finale",
            episodes=5_000,
            eps_start=0.0,
            eps_end=0.0,
            train=False
        ),
    ]

    print("🐍 Démarrage de l'entraînement Snake Q-Learning")
    print(f"📋 Configuration: {len(phases_cfg)} phases, {sum(p.episodes for p in phases_cfg):,} épisodes totaux")
    print(f"🧠 Agent: α={agent.alpha}, γ={agent.gamma}")
    print("=" * 60)

    episode_rewards, phase_stats = train_with_phases(
        agent=agent,
        env=env,
        phases=phases_cfg,
        max_steps_per_episode=10000,
        model_path="snake.pkl",
    )

    print("🎉 Entraînement terminé!")
    print("📊 Résumé final:")
    for name, stats in phase_stats.items():
        print(f"   {name}: {stats['avg_reward']:.2f} (avg), {stats['max_reward']:.2f} (max)")