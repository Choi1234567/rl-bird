import numpy as np


class Q_learning:
    def __init__(self, env, gamma=0.5, alpha=0.5, epsilon=0.99):
        self.env = env
        self.q_table = np.zeros([env.total, env.actions])
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def get_action(self, state):
        self.epsilon *= self.epsilon
        if np.random.rand() > self.epsilon:
            return np.argmax(self.q_table[state])
        else:
            return np.random.randint(0, self.env.actions)

    def update(self, state, action, next_state, reward):
        error = reward + self.gamma * (self.q_table[next_state].max() - self.q_table[state][action])
        self.q_table[state][action] += self.alpha * error

