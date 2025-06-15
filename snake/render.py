import random   # ← conservé pour éviter de casser d’autres parties éventuelles
import time
import sys
from typing import Dict

import numpy as np

import interpreter as interpreter
from action import Actions, ActionResult
from env import SnakeEnv

EMPTY = 0
WALL = 1
HEAD = 2
BODY = 3
GREEN_APPLE = 4
RED_APPLE = 5


def render(env: SnakeEnv) -> None:
    char_map: Dict[int, str] = {
        EMPTY: "  ",
        WALL: "⬛",
        HEAD: "🟢",
        BODY: "🟩",
        GREEN_APPLE: "🍏",
        RED_APPLE: "🍎",
    }

    board: np.ndarray = env.map.astype(int)

    # Efface l’écran (ANSI)
    sys.stdout.write("\033[H\033[J")
    sys.stdout.flush()

    for row in board:
        print("".join(char_map.get(cell, "??") for cell in row))
    print()


if __name__ == "__main__":
    env = SnakeEnv(10, 3, 2, 1)
    render(env)

    # Table de correspondance entre la saisie utilisateur et l’énumération Actions
    input_to_action = {
        "W": Actions.UP,
        "S": Actions.DOWN,
        "A": Actions.LEFT,
        "D": Actions.RIGHT,
    }

    total_score = 0

    while True:
        # Demande l’action à l’utilisateur
        user_choice = input("Votre action (UP / DOWN / LEFT / RIGHT) : ").strip().upper()

        print(user_choice)
        if user_choice not in input_to_action:
            print("Action invalide, veuillez réessayer.")
            continue

        action = input_to_action[user_choice]
        result_action: ActionResult = env.perform_action(action)
        total_score += interpreter.get_reward(result_action)


        # Détection de collision ou de fin de partie
        if (
                result_action.replaced_cell in (BODY, WALL)
                or result_action.snake_length < 1
        ):
            print("💥 Collision ! La partie recommence...")
            env.reset()

        render(env)
        print(f"Score : {total_score}")