# 汤普森采样法
import numpy as np

from multiArmedBandit.bandit import Bandit


class UCBAgent:
    def __init__(self, bandit, gamma):
        self.bandit = bandit
        self._a = np.ones(self.bandit.nums)  # 列表,表示每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.nums)  # 列表,表示每根拉杆奖励为0的次数
        self.estimate = np.zeros(self.bandit.nums)
        self.steps = np.zeros(self.bandit.nums)


    def step(self):
        samples = np.random.beta(self._a, self._b)  # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples)  # 选出采样奖励最大的拉杆
        return self.update_state(k)

    def update_state(self, k):
        reward = self.bandit.step(k)
        self.steps[k] += 1
        self.estimate[k] += 1 / self.steps[k] * (reward - self.estimate[k])
        self._a[k] += reward  # 更新Beta分布的第一个参数
        self._b[k] += (1 - reward)  # 更新Beta分布的第二个参数
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
