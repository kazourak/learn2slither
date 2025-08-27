from dataclasses import dataclass


@dataclass
class PhaseConfig:
    name: str
    episodes: int
    eps_start: float = 0.0
    eps_end: float = 0.0

optimal_phases_cfg = [
    PhaseConfig(
        name="Exploration initiale",
        episodes=70_000,
        eps_start=1.00,
        eps_end=0.70,
    ),
    PhaseConfig(
        name="Exploration intensive",
        episodes=120_000,
        eps_start=0.70,
        eps_end=0.30,
    ),
    PhaseConfig(
        name="Équilibrage Exp/Exp",
        episodes=95_000,
        eps_start=0.30,
        eps_end=0.10,
    ),
    PhaseConfig(
        name="Exploitation dominante",
        episodes=70_000,
        eps_start=0.10,
        eps_end=0.02,
    ),
    PhaseConfig(
        name="Fine-tuning",
        episodes=50_000,
        eps_start=0.02,
        eps_end=0.005,
    ),
    PhaseConfig(
        name="Stabilisation",
        episodes=40_000,
        eps_start=0.005,
        eps_end=0.001,
    )
]

intensive_cfg = [
    PhaseConfig(
        name="Phase 1: Exploration massive",
        episodes=100_000,
        eps_start=1.00,
        eps_end=0.50,
    ),
    PhaseConfig(
        name="Phase 2: Transition exploration",
        episodes=150_000,
        eps_start=0.50,
        eps_end=0.20,
    ),
    PhaseConfig(
        name="Phase 3: Apprentissage intensif",
        episodes=200_000,
        eps_start=0.20,
        eps_end=0.05,
    ),
    PhaseConfig(
        name="Phase 4: Raffinement stratégique",
        episodes=150_000,
        eps_start=0.05,
        eps_end=0.01,
    ),
    PhaseConfig(
        name="Phase 5: Optimisation fine",
        episodes=100_000,
        eps_start=0.01,
        eps_end=0.001,
    ),
    PhaseConfig(
        name="Phase 6: Consolidation",
        episodes=50_000,
        eps_start=0.001,
        eps_end=0.001,
    )
]

one_episode_cfg = [
    PhaseConfig(
        name="Exploration initiale",
        episodes=1,
        eps_start=1.00,
        eps_end=1.00,
    )
]

def get_standard_phases_cfg(episodes: int):
    if episodes <= 1:
        return one_episode_cfg

    exploration_episodes = int(episodes * 0.35)  # 35%
    learning_episodes = int(episodes * 0.45)     # 45%
    exploitation_episodes = int(episodes * 0.20) # 20%

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