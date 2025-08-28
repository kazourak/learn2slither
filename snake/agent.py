import numpy as np
import random
import pickle
from collections import defaultdict

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

class QLearningSnakeAgent:
    def __init__(self, alpha=0.15, gamma=0.95, epsilon=1.0, eps_decay=0.1,
                 eps_min=0.001, load_path=None, save_path=None, train=False):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        self.is_train = train
        self.save_path = save_path

        self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))

        if load_path:
            self.load_model(load_path)

    def calc_eps_decay(self, episodes: int):
        if episodes == 0 or self.epsilon == 0:
            return

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

    def save_model(self):
        try:
            if self.save_path is None:
                return

            data = {'q_table': dict(self.q_table)}

            with open(self.save_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Model saved to {self.save_path}")
        except Exception as e:
            print(f"Error when saving model : {e}")

    def load_model(self, path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

                self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS)),
                                           data['q_table'])

        except Exception as e:
            print(f"Error when loading model : {e}")
            self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))
