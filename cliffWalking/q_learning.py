import numpy as np
import random

class Q_learning:
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.9, n_planning=10):
        self.env = env
        self.q_table = np.zeros([env.total, env.actions])
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_planning = n_planning  #执行Q-planning的次数, 对应1次Q-learning
        self.model = dict()  # 环境模型

    def get_action(self, state):
        # self.epsilon *= self.epsilon
        if np.random.rand() < self.epsilon:
            return np.argmax(self.q_table[state])
        else:
            return np.random.randint(0, self.env.actions)

    def get_best_action(self, state):
        return np.argmax(self.q_table[state])

    def q_learning(self, state, action, next_state, reward):
        error = reward + self.gamma * (self.q_table[next_state].max() - self.q_table[state][action])
        self.q_table[state, action] += self.alpha * error


    def update(self, state, action, next_state, reward):
        self.q_learning(state, action, next_state, reward)
        self.model[(state, action)] = reward, next_state  # 将数据添加到模型中
        for _ in range(self.n_planning):  # Q-planning循环
            # 随机选择曾经遇到过的状态动作对
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, s_, r)

