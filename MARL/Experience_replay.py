import numpy as np
import random
from collections import deque


# 经验回放
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, s, a, r, t, obs, avail_actions, filled):
        experience = [s, a, r, t, obs, avail_actions, np.array(filled)]
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


# 批采样  存放经验池的池子
class EpisodeBatch:
    def __init__(self, buffer_size):

        self.buffer = deque(maxlen=int(buffer_size))

    def add(self, replay_buffer):
        for exp in replay_buffer.buffer:
            if exp[3]:
                break
            else:
                self.buffer.append(exp)

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        if self.size() < batch_size:
            batch = random.sample(self.buffer, self.size())
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch, a_batch, r_batch, t_batch, obs_batch, available_actions_batch, filled_batch = [], [], [], [], [], [], []
        for s, a, r, t, obs, available_actions, filled in batch:
            s_batch.append(s)
            a_batch.append(a)
            r_batch.append(r)
            t_batch.append(t)
            obs_batch.append(obs)
            available_actions_batch.append(available_actions)
            filled_batch.append(filled)

        # 没有对s_batch 进行np类型转换
        filled_batch = np.array(filled_batch)
        r_batch = np.array(r_batch)
        t_batch = np.array(t_batch)
        a_batch = np.array(a_batch)
        obs_batch = np.array(obs_batch)
        available_actions_batch = np.array(available_actions_batch)

        return s_batch, a_batch, r_batch, t_batch, obs_batch, available_actions_batch, filled_batch