import numpy as np


class Bandit:
    def __init__(self, K):
        self.nums = K
        self.arms = np.random.random(self.nums)
        self._best_arm = np.argmax(self.arms)
        self._best_prop = self.arms[self._best_arm]

    def get_best_arm(self):
        return self._best_arm

    def get_best_prop(self):
        return self._best_prop

    def step(self, k):
        if np.random.rand() < self.arms[k]:
            return 1
        else:
            return 0