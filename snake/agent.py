import numpy as np
import random
import pickle
from collections import defaultdict

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

class QLearningSnakeAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, eps_decay=0.9995, filename=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.is_train = True

        self.q_table = defaultdict(lambda: [0.0] * len(ACTIONS))

        if filename:
            self.load_model(filename)

    def choose_action(self, state: tuple):
        if self.is_train and random.random() < self.epsilon:
            return random.randrange(len(ACTIONS))

        q_values = self.q_table[state]
        return int(np.argmax(q_values))

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
        self.is_train = False
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: [0.0]*len(ACTIONS), data)
