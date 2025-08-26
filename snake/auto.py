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
        r_nothing = random.uniform(-1.5, -1)
        r_eat_green = random.uniform(15, 20)
        r_eat_red = random.uniform(-10, -30)
        r_dead = 115
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
        alpha=0.15,
        gamma=0.95,
        eps_min=0.001
    )

    phases_cfg = [
        PhaseConfig(
            name="Phase 1: Exploration massive",
            episodes=100_000,
            eps_start=1.00,
            eps_end=0.50,
            train=True
        ),
        PhaseConfig(
            name="Phase 2: Transition exploration",
            episodes=150_000,
            eps_start=0.50,
            eps_end=0.20,
            train=True
        ),
        PhaseConfig(
            name="Phase 3: Apprentissage intensif",
            episodes=200_000,
            eps_start=0.20,
            eps_end=0.05,
            train=True
        ),
        PhaseConfig(
            name="Phase 4: Raffinement stratégique",
            episodes=150_000,
            eps_start=0.05,
            eps_end=0.01,
            train=True
        ),
        PhaseConfig(
            name="Phase 5: Optimisation fine",
            episodes=100_000,
            eps_start=0.01,
            eps_end=0.001,
            train=True
        ),
        PhaseConfig(
            name="Phase 6: Consolidation",
            episodes=50_000,
            eps_start=0.001,
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
            max_steps_per_episode=5000,
            model_path=temp_model_path,
        )

        result = evaluate(agent, env, interpreter, episodes=5000, max_step=5000)
        
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
    
    models_dir = "./best_model"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    print(f"\nLancement de {num_models} entraînements avec {max_workers} processus simultanés...")
    print(f"Mode: {'Aléatoire' if use_random else 'Fixe'}")
    print("-" * 50)

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

if __name__ == "__main__":
    main()


