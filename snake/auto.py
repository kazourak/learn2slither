from __future__ import annotations

import os
import random
import shutil

from snake.agent import QLearningSnakeAgent
from snake.env import SnakeEnv
from snake.eval import evaluate
from snake.interpreter import Interpreter
from snake.train import PhaseConfig, train_with_phases

if __name__ == "__main__":
    env = SnakeEnv(10, 3, 1, 2)

    for i in range(500):
        r_nothing = random.uniform(-3.0, -1)
        r_eat_green = random.uniform(10, 50)
        r_eat_red = random.uniform(-50, -10)
        r_dead = random.uniform(-125, -75)
        # r_nothing = -1.23
        # r_eat_green = 20.58
        # r_eat_red = -28.16
        # r_dead = -113.51
        interpreter = Interpreter(reward_nothing=r_nothing, reward_dead=r_dead, reward_red_apple=r_eat_red, reward_green_apple=r_eat_green)
        # interpreter = Interpreter()

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

        model_path = f"./models/3_snake_optimized_{i}_6.pkl"
        episode_rewards, phase_stats = train_with_phases(
            agent=agent,
            env=env,
            interpreter=interpreter,
            phases=phases_cfg,
            max_steps_per_episode=10000,
            model_path=model_path,
        )

        result = evaluate(agent, env, interpreter, episodes=5000, max_step=2500)

        new_filename = f"./models/3_{result[2]}_{result[0]}_{result[1]}_snake_optimized_{i}_6_r_{r_nothing}_{r_eat_green}_{r_eat_red}_{r_dead}.pkl"
        if os.path.exists(model_path):
            shutil.move(model_path, new_filename)


