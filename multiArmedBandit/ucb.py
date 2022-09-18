# 上置信界算法
import numpy as np

from multiArmedBandit.bandit import Bandit


class UCBAgent:
    def __init__(self, bandit, gamma):
        self.bandit = bandit
        self.gamma = gamma
        self.estimate = np.zeros(self.bandit.nums)
        self.steps = np.zeros(self.bandit.nums)

    def step(self):
        ucb = self.estimate + self.gamma * np.sqrt(
            np.log(np.sum(self.steps)) / (2 * (self.steps + 1)))  # 计算上置信界
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        return self.update_state(k)

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
    agent = UCBAgent(bandit, 1)
    agent.run(1000)
    print(np.argmax(agent.steps), np.argmax(agent.estimate), bandit.get_best_arm(),
          np.argmax(agent.steps) == bandit.get_best_arm())


if __name__ == '__main__':
    main()
