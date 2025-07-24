from __future__ import annotations

import os
import shutil

from snake.agent import QLearningSnakeAgent
from snake.env import SnakeEnv
from snake.eval import evaluate
from snake.train import PhaseConfig, train_with_phases

if __name__ == "__main__":
    env = SnakeEnv(10, 3, 1, 2)
    agent = QLearningSnakeAgent(
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        eps_decay=0.999,
        eps_min=0.001
    )

    phases_cfg = [
        PhaseConfig(
            name="Exploration initiale",
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
        )
    ]

    max_model = 10000
    for i in range(max_model):
        model_path = f"./models/snake_optimized_{i}.pkl"
        episode_rewards, phase_stats = train_with_phases(
            agent=agent,
            env=env,
            phases=phases_cfg,
            max_steps_per_episode=10000,
            model_path=model_path,
        )
        result = evaluate(agent, env, episodes=5000, max_step=10000)

        new_filename = f"./models/{result[2]}_{result[0]}_{result[1]}_snake_optimized_{i}.pkl"
        if os.path.exists(model_path):
            shutil.move(model_path, new_filename)


