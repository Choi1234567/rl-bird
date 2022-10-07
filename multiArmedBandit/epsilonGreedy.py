# ε-贪心算法
import numpy as np

from multiArmedBandit.bandit import Bandit


class EpsilonGreedyAgent:
    def __init__(self, bandit, epsilon, gamma):
        self.bandit = bandit
        self.epsilon = epsilon
        self.gamma = gamma
        self.estimate = np.zeros(self.bandit.nums)
        self.steps = np.zeros(self.bandit.nums)

    def step(self):
        self.epsilon *= self.gamma
        if np.random.rand() > self.epsilon:
            return self.update_state(np.argmax(self.estimate))
        else:
            return self.update_state(np.random.randint(0, self.bandit.nums))

    def update_state(self, k):
        reward = self.bandit.step(k)
        self.steps[k] += 1
        self.estimate[k] += 1 / self.steps[k] * (reward - self.estimate[k])
        return k, reward

    def run(self, max_step):
        for _ in range(max_step):
            k, reward = self.step()
            print(k, reward, self.estimate)
        return self.steps


def main():
    bandit = Bandit(10)
    agent = EpsilonGreedyAgent(bandit, 1, 0.99)
    agent.run(1000)
    print(bandit.arms)
    print(np.argmax(agent.steps), np.argmax(agent.estimate), bandit.get_best_arm(),
          np.argmax(agent.steps) == bandit.get_best_arm())


if __name__ == '__main__':
    main()
