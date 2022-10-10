import tkinter as tk
from time import sleep
import matplotlib.pyplot as plt

import numpy as np

from cliffWalking.q_learning import Q_learning


class CliffWalkingEnv:
    def __init__(self, column=12, row=4, holes=4):
        self.column = column
        self.row = row
        self.total = column * row
        self.cells = [-1] * self.total
        self.position = 0
        self.actions = 4
        for _ in range(holes):
            self.cells[np.random.randint(1, self.total - 1)] = -100
        self.cells[-1] = 0
        self.win = tk.Tk()
        self.win.title("悬崖漫步")
        self.win.geometry(str(row * 100 + 3) + "x" + str(column * 100 + 32)+"+100+100")
        self.info = tk.StringVar()
        self.info.set('游戏开始')
        self.label = tk.Label(self.win, textvariable=self.info, font=("仿宋", 20))
        self.label.pack(side='top', expand='yes', fill='both')
        self.label.focus_set()
        self.canvas = tk.Canvas(self.win, width=row * 100, height=column * 100, bg="white")
        self.canvas.pack()
        for i in range(column + 1):
            self.canvas.create_line(0, 100 * i, row * 100, 100 * i)
        for i in range(row + 1):
            self.canvas.create_line(100 * i, 0, 100 * i, column * 100)
        self.canvas.create_text(50, 50, text='😶', font=("仿宋", 75), tag="robot")
        for i in range(self.total):
            if self.cells[i] == -100:
                self.canvas.create_text(100 * (i % row) + 50, 100 * (i // row) + 50, text='x', font=("仿宋", 75))
            if self.cells[i] == 0:
                self.canvas.create_text(100 * (i % row) + 50, 100 * (i // row) + 50, text='o', font=("仿宋", 75))
        self.canvas.focus_set()
        self.canvas.bind('<Key>', self.on_key)

    def step(self, action):
        # print('开始执行动作%s' % action)
        # 0-上 1-下 2-左 3-右
        if action == 0:
            if self.position // self.row > 0:
                self.position = self.position - self.row
        if action == 1:
            if self.position // self.row < self.column - 1:
                self.position = self.position + self.row
        if action == 2:
            if self.position % self.row > 0:
                self.position = self.position - 1
        if action == 3:
            if self.position % self.row < self.row - 1:
                self.position = self.position + 1
        self.canvas.delete('robot')
        reward = self.cells[self.position]
        if reward == -1:
            self.canvas.create_text(100 * (self.position % self.row) + 50, 100 * (self.position // self.row) + 50, text='😶', font=("仿宋", 75), tag="robot")
        if reward == -100:
            self.canvas.create_text(100 * (self.position % self.row) + 50, 100 * (self.position // self.row) + 50, text='😭', font=("仿宋", 75), tag="robot")
        if reward == 0:
            self.canvas.create_text(100 * (self.position % self.row) + 50, 100 * (self.position // self.row) + 50, text='😁', font=("仿宋", 75), tag="robot")
        return self.position, reward, not reward == -1

    def on_key(self, evt):
        """响应所有键盘事件的函数"""
        if evt.keysym == 'Return':
            episode = 200
            q = Q_learning(self)
            state = 0
            res =[]
            for i in range(episode):
                total_reward = 0
                while True:
                    action = q.get_action(state)
                    next_state, reward, done = self.step(action)
                    total_reward += reward
                    q.update(state, action, next_state, reward)
                    state = next_state
                    self.win.update()

                    if done:
                        state = 0
                        res.append(total_reward)
                        self.info.set('第%s轮游戏结束,本次累计奖励%s' % (i, total_reward))
                        self.restart()
                        #sleep(0.5)
                        break
                    else:
                        pass
                        #sleep(0.01)
            plt.plot(range(1, episode + 1), res)
            plt.show()
            self.info.set('训练结束,开始展示最佳路线:')
            best_reward = 0
            while True:
                self.canvas.create_text(100 * (self.position % self.row) + 50, 100 * (self.position // self.row) + 50,
                                        text='😁', font=("仿宋", 75))
                action = q.get_best_action(state)
                next_state, reward, done = self.step(action)
                best_reward += reward
                q.update(state, action, next_state, reward)
                state = next_state
                self.win.update()
                sleep(2)
                if done:
                    state = 0
                    # self.restart()
                    self.info.set('最佳路线累计奖励为%s' % best_reward)
                    return

        next_state, reward, done = None, None, None
        if evt.keysym == 'Up':
            next_state, reward, done = self.step(0)
        if evt.keysym == 'Down':
            next_state, reward, done = self.step(1)
        if evt.keysym == 'Left':
            next_state, reward, done = self.step(2)
        if evt.keysym == 'Right':
            next_state, reward, done = self.step(3)
        self.info.set('当前奖励:%s 是否结束:%s' % (reward, done))
        if done:
            self.restart()
            self.info.set('游戏开始')

    def restart(self):
        self.canvas.delete('robot')
        self.canvas.create_text(50, 50, text='😶', font=("", 75), tag="robot")
        self.position = 0


if __name__ == '__main__':
    cw = CliffWalkingEnv(8, 12, 16)
    tk.mainloop()
