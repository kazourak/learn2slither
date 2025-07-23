import numpy as np
import random
import pickle
from collections import defaultdict

ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

class QLearningSnakeAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, eps_decay=0.995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay

        self.q_table = defaultdict(lambda: [0.0] * len(ACTIONS))

    def choose_action(self, state: np.ndarray) -> tuple:
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)

        q_values = self.q_table[state]
        best_action_idx = int(np.argmax(q_values))
        return ACTIONS[best_action_idx]

    def update(self, state, action, reward, next_state, done):
        """
        Q(s,a) += alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        current = self.q_table[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.q_table[next_state])

        self.q_table[state][action] = current + self.alpha * (target - current)

    def decay_epsilon(self):
        self.epsilon *= self.eps_decay

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: [0.0]*len(ACTIONS), data)
