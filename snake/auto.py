from __future__ import annotations

import os
import random
import shutil
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from snake.agent import QLearningSnakeAgent
from snake.env import SnakeEnv
from snake.eval import evaluate
from snake.interpreter import Interpreter
from snake.train import PhaseConfig, train_with_phases


def train_single_model(args):
    model_id, use_random, base_path = args
    
    env = SnakeEnv(10, 3, 1, 2)
    
    if use_random:
        r_nothing = random.uniform(-3.0, -1)
        r_eat_green = random.uniform(10, 50)
        r_eat_red = random.uniform(-50, -10)
        r_dead = random.uniform(-125, -75)
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

    temp_model_path = f"{base_path}/temp_model_{model_id}_{os.getpid()}.pkl"
    
    try:
        _, _ = train_with_phases(
            agent=agent,
            env=env,
            interpreter=interpreter,
            phases=phases_cfg,
            max_steps_per_episode=10000,
            model_path=temp_model_path,
        )

        result = evaluate(agent, env, interpreter, episodes=5000, max_step=2500)
        
        final_filename = f"{base_path}/model_{model_id}_{result[2]}_{result[0]}_{result[1]}_r_{r_nothing:.2f}_{r_eat_green:.2f}_{r_eat_red:.2f}_{r_dead:.2f}.pkl"
        
        if os.path.exists(temp_model_path):
            shutil.move(temp_model_path, final_filename)
            
        return {
            'model_id': model_id,
            'success': True,
            'result': result,
            'rewards': (r_nothing, r_eat_green, r_eat_red, r_dead),
            'filename': final_filename
        }
    except Exception as e:
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
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
    
    models_dir = "./models"
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


