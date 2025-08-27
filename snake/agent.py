import numpy as np
import random
import pickle
from collections import defaultdict

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

class QLearningSnakeAgent:
    def __init__(self, alpha=0.15, gamma=0.95, epsilon=1.0, eps_decay=0.999,
                 eps_min=0, load_path=None, save_path=None, train=False):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        self.is_train = train
        self.save_path = save_path

        self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))

        self.training_stats = {
            'episodes': 0,
            'recent_rewards': [],
            'epsilon_history': []
        }

        if load_path:
            self.load_model(load_path)

    def calc_eps_decay(self, episodes: int):
        if self.epsilon == 0 or episodes <= 1:
            self.eps_decay = 1.0
        self.eps_decay = (self.eps_min / self.epsilon) ** (1 / episodes)

    def choose_action(self, state: tuple):
        if self.is_train and random.random() < self.epsilon:
            return random.randrange(len(ACTIONS))

        q_values = self.q_table[state]
        # Tie-breaking
        max_q = np.max(q_values)
        max_actions = np.where(q_values == max_q)[0]
        return np.random.choice(max_actions)

    def update(self, state, action, reward, next_state, done):
        """
        Q(s,a) += alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        current = self.q_table[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action] = current + self.alpha * (target - current)

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        if self.is_train:
            self.training_stats['epsilon_history'].append(self.epsilon)

    def end_episode(self, episode_reward=0):
        if self.is_train:
            self.training_stats['episodes'] += 1

            self.training_stats['recent_rewards'].append(episode_reward)
            if len(self.training_stats['recent_rewards']) > 1000:
                self.training_stats['recent_rewards'].pop(0)

    def get_stats(self):
        return self.training_stats.copy()

    def get_q_table_size(self):
        return len(self.q_table)

    def save_model(self):
        # print(self.q_table)
        if self.save_path is None:
            return

        data = {'q_table': dict(self.q_table)}

        with open(self.save_path, 'wb') as f:
            pickle.dump(data, f)

    def load_model(self, path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

                self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS)),
                                           data['q_table'])
                # print(self.q_table)

        except Exception as e:
            print(f"Erreur lors du chargement du modèle : {e}")
            self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))


    def print_training_info(self):
        stats = self.training_stats
        print(f"Épisodes: {stats['episodes']}")
        print(f"Epsilon actuel: {self.epsilon:.4f}")
        print(f"Taille Q-table: {self.get_q_table_size()}")
        if stats['recent_rewards']:
            avg_reward = np.mean(stats['recent_rewards'])
            print(f"Récompense moyenne (récente): {avg_reward:.2f}")