from __future__ import annotations

import os
import random
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from snake.agent import QLearningSnakeAgent
from snake.env import SnakeEnv
from snake.eval import evaluate
from snake.phases import intensive_cfg
from snake.interpreter import Interpreter
from snake.train import PhaseConfig, train_with_phases


def train_single_model(args):
    model_id, use_random, base_path = args

    env = SnakeEnv(10, 3, 1, 2)

    if use_random:
        r_nothing = random.uniform(-1.5, -1)
        r_eat_green = random.uniform(15, 20)
        r_eat_red = random.uniform(-15, -10)
        r_dead = -115
    else:
        r_nothing = -1.23
        r_eat_green = 20.58
        r_eat_red = -28.16
        r_dead = -113.51

    interpreter = Interpreter(
        reward_nothing=r_nothing,
        reward_dead=r_dead,
        reward_red_apple=r_eat_red,
        reward_green_apple=r_eat_green
    )


    model_path = f"{base_path}/temp_model_{model_id}_{os.getpid()}.pkl"
    agent = QLearningSnakeAgent(save_path=model_path, train=True)

    try:
        _, _ = train_with_phases(
            agent=agent,
            env=env,
            interpreter=interpreter,
            phases=intensive_cfg,
            max_steps_per_episode=2500
        )

        result = evaluate(agent, env, interpreter, episodes=5000, max_step=2500)

        return {
            'model_id': model_id,
            'success': True,
            'result': result,
            'rewards': (r_nothing, r_eat_green, r_eat_red, r_dead),
            'filename': model_path
        }
    except Exception as e:
        return {
            'model_id': model_id,
            'success': False,
            'error': str(e)
        }


def main():
    print("=== Programme d'entraînement automatique de modèles Snake ===")

    num_models = int(input("Combien de modèles voulez-vous entraîner ? "))
    max_workers = int(input(f"Combien de processus simultanés ? (max recommandé: {mp.cpu_count()}) "))

    use_random = input("Utiliser des valeurs aléatoires pour les récompenses ? (o/n) ").lower().startswith('o')

    models_dir = "./models_test2"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    print(f"\nLancement de {num_models} entraînements avec {max_workers} processus simultanés...")
    print(f"Mode: {'Aléatoire' if use_random else 'Fixe'}")
    print("-" * 50)

    start_time = time.time()
    completed = 0

    args_list = [(i, use_random, models_dir) for i in range(num_models)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(train_single_model, args): args[0] for args in args_list}

        for future in as_completed(futures):
            model_id = futures[future]
            try:
                result = future.result()
                completed += 1

                if result['success']:
                    print(f"✓ Modèle {model_id} terminé ({completed}/{num_models}) - Score: {result['result'][0]:.2f}")
                else:
                    print(f"✗ Modèle {model_id} échoué ({completed}/{num_models}) - Erreur: {result['error']}")

            except Exception as e:
                completed += 1
                print(f"✗ Modèle {model_id} échoué ({completed}/{num_models}) - Exception: {str(e)}")

    elapsed_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Entraînement terminé en {elapsed_time:.1f} secondes")
    print(f"Modèles sauvegardés dans: {models_dir}/")


if __name__ == "__main__":
    main()


