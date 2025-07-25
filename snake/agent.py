import numpy as np
import random
import pickle
from collections import defaultdict

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

class QLearningSnakeAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, eps_decay=0.9995,
                 eps_min=0.01, filename=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.is_train = True

        # Utilisation de numpy pour les valeurs par défaut
        self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))

        # Statistiques d'entraînement
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0,
            'epsilon_history': []
        }

        if filename:
            print('ok')
            self.load_model(filename)

    def choose_action(self, state: tuple):
        """Choisit une action selon la politique epsilon-greedy"""
        if self.is_train and random.random() < self.epsilon:
            return random.randrange(len(ACTIONS))

        q_values = self.q_table[state]
        print(q_values)
        return np.argmax(q_values)

    def update(self, state, action, reward, next_state, done):
        """
        Met à jour la Q-table selon la formule de Bellman
        Q(s,a) += alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        current = self.q_table[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action] = current + self.alpha * (target - current)

        # Mise à jour des stats
        if self.is_train:
            self.training_stats['total_reward'] += reward

    def decay_epsilon(self):
        """Décroissance d'epsilon avec minimum"""
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        if self.is_train:
            self.training_stats['epsilon_history'].append(self.epsilon)

    def end_episode(self):
        """À appeler à la fin de chaque épisode"""
        if self.is_train:
            self.training_stats['episodes'] += 1
            self.decay_epsilon()

    def set_training_mode(self, is_training=True):
        """Active/désactive le mode entraînement"""
        self.is_train = is_training

    def get_stats(self):
        """Retourne les statistiques d'entraînement"""
        return self.training_stats.copy()

    def get_q_table_size(self):
        """Retourne la taille de la Q-table"""
        return len(self.q_table)

    def save_model(self, path):
        """Sauvegarde le modèle et les statistiques"""
        data = {
            'q_table': dict(self.q_table),
            'training_stats': self.training_stats,
            'hyperparams': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'eps_decay': self.eps_decay,
                'eps_min': self.eps_min
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_model(self, path):
        """Charge un modèle sauvegardé"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            # Charger la Q-table
            if isinstance(data, dict) and 'q_table' in data:
                print('charge')
                # Nouveau format avec métadonnées
                self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS)),
                                           data['q_table'])
                if 'training_stats' in data:
                    self.training_stats = data['training_stats']
                if 'hyperparams' in data:
                    hp = data['hyperparams']
                    self.epsilon = hp.get('epsilon', self.epsilon)
            else:
                # Ancien format (seulement la Q-table)
                self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS)), data)

        except Exception as e:
            print(f"Erreur lors du chargement du modèle : {e}")
            # Réinitialiser avec des valeurs par défaut
            self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))

    def print_training_info(self):
        """Affiche des informations sur l'entraînement"""
        stats = self.training_stats
        print(f"Épisodes: {stats['episodes']}")
        print(f"Epsilon actuel: {self.epsilon:.4f}")
        print(f"Taille Q-table: {self.get_q_table_size()}")
        if stats['episodes'] > 0:
            avg_reward = stats['total_reward'] / stats['episodes']
            print(f"Récompense moyenne: {avg_reward:.2f}")