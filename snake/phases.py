from dataclasses import dataclass


@dataclass
class PhaseConfig:
    name: str
    episodes: int
    eps_start: float = 0.0
    eps_end: float = 0.0


basic_cfg = [
    PhaseConfig(
        name="Exploration",
        episodes=100_000,
        eps_start=1.00,
        eps_end=0.01,
    ),
    PhaseConfig(
        name="Learning",
        episodes=100_000,
        eps_start=0.01,
        eps_end=0.001,
    ),
    PhaseConfig(
        name="Exploitation",
        episodes=100_000,
        eps_start=0.001,
    )
]

intensive_cfg = [
    PhaseConfig(
        name="Intensive Exploration",
        episodes=250_000,
        eps_start=1.00,
        eps_end=0.1,
    ),
    PhaseConfig(
        name="Intensive Learning",
        episodes=250_000,
        eps_start=0.1,
        eps_end=0.001,
    ),
    PhaseConfig(
        name="Intensive Exploitation",
        episodes=500_000,
        eps_start=0.001,
    )
]

optimal_cfg = [
    PhaseConfig(
        name="Optimal Exploration 1",
        episodes=50_000,
        eps_start=1.00,
        eps_end=0.75,
    ),
    PhaseConfig(
        name="Optimal Exploration 2",
        episodes=100_000,
        eps_start=0.75,
        eps_end=0.5,
    ),
    PhaseConfig(
        name="Optimal Learning",
        episodes=200_000,
        eps_start=0.5,
        eps_end=0.1,
    ),
    PhaseConfig(
        name="Optimal Refinement",
        episodes=150_000,
        eps_start=0.1,
        eps_end=0.01,
    ),
    PhaseConfig(
        name="Optimal Exploitation",
        episodes=500_000,
        eps_start=0.01,
        eps_end=0.001,
    )
]

one_episode_cfg = [
    PhaseConfig(
        name="One Episode",
        episodes=1,
        eps_start=1.00,
        eps_end=1.00,
    )
]

phases = {
    'basic': basic_cfg,
    'intensive': intensive_cfg,
    'optimal': optimal_cfg,
}


def get_standard_phases_cfg(episodes: int):
    print(episodes)
    if episodes is None or episodes <= 1:
        return one_episode_cfg

    exploration_episodes = int(episodes * 0.35)  # 35%
    learning_episodes = int(episodes * 0.45)     # 45%
    exploitation_episodes = int(episodes * 0.20)  # 20%

    return [
        PhaseConfig(
            name="Exploration phase",
            episodes=exploration_episodes,
            eps_start=1.00,
            eps_end=0.5,
        ),
        PhaseConfig(
            name="Balanced Learning Phase",
            episodes=learning_episodes,
            eps_start=0.5,
            eps_end=0.1,
        ),
        PhaseConfig(
            name="Exploitation Phase",
            episodes=exploitation_episodes,
            eps_start=0.1,
            eps_end=0.01,
        )
    ]
