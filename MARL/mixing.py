import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from collections import deque
from policy import RNNAgent, IQL, QMIX, VGN, VDN, Idea, NoQmix, NoHistory


# 探索策略
class EpsilonGreedy:
    def __init__(self, action_nb, agent_nb, final_step, epsilon_start=float(1), epsilon_end=0.05):
        self.action_nb = action_nb
        self.agent_nb = agent_nb
        self.final_step = final_step
        self.epsilon = epsilon_start
        self.initial_epsilon = epsilon_start
        self.epsilon_end = epsilon_end

    # 动作选取
    def act(self, value_action, avail_actions):
        if np.random.random() > self.epsilon:
            action = value_action.argmax(axis=-1).detach().cpu().numpy()
        else:
            action = Categorical(avail_actions.to(torch.float32)).sample([1]).squeeze().to(torch.int64).detach().cpu().numpy()
        return action

    # 更新epsilon
    def epsilon_decay(self, step):
        progress = step / self.final_step

        decay = self.initial_epsilon - progress
        if decay <= self.epsilon_end:
            decay = self.epsilon_end
        self.epsilon = decay


# 经验回放
class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, s, a, r, t, obs, avail_actions, filled):
        experience = [s, a, r, t, obs, avail_actions, np.array(filled)]
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    # 采样
    def sample_batch(self, batch_size):
        batch = []

        for idx in range(batch_size):
            batch.append(self.buffer[idx])
        batch = np.array(batch, dtype=object)

        s_batch = np.array([_[0] for _ in batch], dtype='float32')
        a_batch = np.array([_[1] for _ in batch], dtype='float32')
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        obs_batch = np.array([_[4] for _ in batch], dtype='float32')
        available_actions_batch = np.array([_[5] for _ in batch], dtype='float32')
        filled_batch = np.array([_[6] for _ in batch], dtype='float32')

        return s_batch, a_batch, r_batch, t_batch, obs_batch, available_actions_batch, filled_batch

    def clear(self):
        self.buffer.clear()


# 批采样  存放经验池的池子
class EpisodeBatch:
    def __init__(self, buffer_size):

        self.buffer = deque(maxlen=int(buffer_size))

    def add(self, replay_buffer):
        self.buffer.append(replay_buffer)

    def _get_max_episode_len(self, batch):
        max_episode_len = 0

        for replay_buffer in batch:
            _, _, _, t, _, _, _ = replay_buffer.sample_batch(replay_buffer.size())
            for idx, t_idx in enumerate(t):
                if t_idx and idx > max_episode_len:
                    max_episode_len = idx + 1
                    break
        return max_episode_len

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        if self.size() < batch_size:
            batch = random.sample(self.buffer, self.size())
        else:
            batch = random.sample(self.buffer, batch_size)

        episode_len = self._get_max_episode_len(batch)

        s_batch, a_batch, r_batch, t_batch, obs_batch, available_actions_batch, filled_batch = [], [], [], [], [], [], []
        for replay_buffer in batch:
            s, a, r, t, obs, available_actions, filled = replay_buffer.sample_batch(episode_len)
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

        return s_batch, a_batch, r_batch, t_batch, obs_batch, available_actions_batch, filled_batch, episode_len


class Mixing:
    def __init__(self,
                 training,
                 is_cuda,
                 agent_nb,
                 obs_shape,
                 states_shape,
                 action_n, lr,
                 gamma=0.99,
                 batch_size=32,
                 replay_buffer_size=10000,
                 update_target_network_step=100,
                 final_step=50000,
                 algo='qmix'):
        self.algo = algo
        self.training = training
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_network = update_target_network_step
        self.hidden_states = None
        self.target_hidden_states = None
        self.agent_nb = agent_nb
        self.action_n = action_n
        self.obs_shape = obs_shape
        self.state_shape = states_shape
        self.is_cuda = is_cuda
        self.epsilon_greedy = EpsilonGreedy(action_n, agent_nb, final_step)
        self.episode_batch = EpisodeBatch(replay_buffer_size)

        self.agents = RNNAgent(obs_shape, n_actions=action_n)
        self.target_agents = RNNAgent(obs_shape, n_actions=action_n)

        if algo == 'qmix':
            self.mixer = QMIX(agent_nb, states_shape, mixing_embed_dim=32)
            self.target_mixer = QMIX(agent_nb, states_shape, mixing_embed_dim=32)
        elif algo == 'vdn':
            self.mixer = VDN()
            self.target_mixer = VDN()
        elif algo == 'iql':
            self.mixer = IQL()
            self.target_mixer = IQL()
        elif algo == 'vgn':
            self.mixer = VGN(agent_nb=agent_nb, obs_shape=obs_shape, hidden_dim=64, alpha=0.2, concat=True)
            self.target_mixer = VGN(agent_nb=agent_nb, obs_shape=obs_shape, hidden_dim=64, alpha=0.2, concat=True)
        elif algo == 'idea':
            self.mixer = Idea(agent_nb=agent_nb, state_shape=states_shape, feature_dim=self.agents.rnn_hidden_dim, hidden_dim=64, alpha=0.2, concat=True, mixing_embed_dim=32)
            self.target_mixer = Idea(agent_nb=agent_nb, state_shape=states_shape, feature_dim=self.agents.rnn_hidden_dim, hidden_dim=64, alpha=0.2, concat=True, mixing_embed_dim=32)
        elif algo == 'qatten':
            pass
        elif algo == 'NoQmix':
            self.mixer = NoQmix(agent_nb=agent_nb, state_shape=states_shape, feature_dim=self.agents.rnn_hidden_dim,
                              hidden_dim=64, alpha=0.2, concat=True, mixing_embed_dim=32)
            self.target_mixer = NoQmix(agent_nb=agent_nb, state_shape=states_shape,
                                     feature_dim=self.agents.rnn_hidden_dim, hidden_dim=64, alpha=0.2, concat=True,
                                     mixing_embed_dim=32)
        elif algo == 'NoHistory':
            self.mixer = NoHistory(agent_nb=agent_nb, state_shape=states_shape, feature_dim=self.obs_shape,
                                hidden_dim=64, alpha=0.2, concat=True, mixing_embed_dim=32)
            self.target_mixer = NoHistory(agent_nb=agent_nb, state_shape=states_shape,
                                       feature_dim=self.obs_shape, hidden_dim=64, alpha=0.2, concat=True,
                                       mixing_embed_dim=32)
        else:
            print("未定义该混和网络")
            print("未定义该混和网络")
            print("未定义该混和网络")
            raise RuntimeError
        # 是否使用GPU
        if self.is_cuda:
            self.agents = self.agents.cuda()
            self.target_agents = self.target_agents.cuda()
            self.mixer = self.mixer.cuda()
            self.target_mixer = self.target_mixer.cuda()

        self.target_agents.update(self.agents)
        self.target_mixer.update(self.mixer)

        self.params = list(self.agents.parameters())
        self.params += list(self.mixer.parameters())

        self.optimizer = torch.optim.RMSprop(params=self.params, lr=lr, eps=1e-8, alpha=0.99)
        # 如果使用图注意力机制
        self.adj = torch.as_tensor(np.ones((agent_nb, agent_nb)), dtype=torch.float32)

    def save_model(self, filename):
        torch.save(self.agents.state_dict(), filename)

    def load_model(self, filename):
        self.agents.load_state_dict(torch.load(filename))
        # 不进行dropout、不更新batchnorm的mean和var参数、不进行梯度反向传播，但梯度仍然会计算
        self.agents.eval()

    def _init_hidden_states(self, batch_size):
        self.hidden_states = self.agents.init_hidden().unsqueeze(0).expand([batch_size, self.agent_nb, -1])
        self.target_hidden_states = self.target_agents.init_hidden().unsqueeze(0).expand([batch_size, self.agent_nb, -1])
        if self.is_cuda:
            self.hidden_states = self.hidden_states.cuda()
            self.target_hidden_states = self.target_hidden_states.cuda()

    def decay_epsilon_greedy(self, global_steps):
        self.epsilon_greedy.epsilon_decay(global_steps)

    def on_reset(self, batch_size):
        self._init_hidden_states(batch_size)

    def update_targets(self, episode):
        if episode % self.update_target_network == 0 and self.training:
            self.target_agents.update(self.agents)
            self.target_mixer.update(self.mixer)
            pass

    def train(self):
        if self.training and self.episode_batch.size() > self.batch_size:
            for _ in range(2):
                self._init_hidden_states(self.batch_size)
                s_batch, a_batch, r_batch, t_batch, obs_batch, available_actions_batch, filled_batch, episode_len = self.episode_batch.sample_batch(
                    self.batch_size)

                r_batch = r_batch[:, :-1]
                a_batch = a_batch[:, :-1]
                t_batch = t_batch[:, :-1]
                filled_batch = filled_batch[:, :-1]

                mask = (1 - filled_batch) * (1 - t_batch.squeeze())

                r_batch = torch.as_tensor(r_batch, dtype=torch.float32)
                t_batch = torch.as_tensor(t_batch, dtype=torch.float32)
                mask = torch.as_tensor(mask, dtype=torch.float32)
                a_batch = torch.as_tensor(a_batch, dtype=torch.int64)
                if self.is_cuda:
                    r_batch = r_batch.cuda()
                    t_batch = t_batch.cuda()
                    mask = mask.cuda()
                    a_batch = a_batch.cuda()

                mac_out = []

                h = []
                target_h = []

                for t in range(episode_len):
                    obs = obs_batch[:, t]
                    obs = np.concatenate(obs, axis=0)
                    obs = torch.as_tensor(obs, dtype=torch.float32)
                    if self.is_cuda:
                        obs = obs.cuda()
                    agent_actions, self.hidden_states = self.agents(obs, self.hidden_states)
                    agent_actions = agent_actions.reshape([self.batch_size, self.agent_nb, -1])
                    h.append(self.hidden_states.reshape([self.batch_size, self.agent_nb, -1]))
                    mac_out.append(agent_actions)
                mac_out = torch.stack(mac_out, dim=1)
                h = torch.stack(h[:-1], dim=1)

                _a_batch = F.one_hot(a_batch.detach(), mac_out[:, :-1].shape[-1]).squeeze(-2)
                chosen_action_qvals = mac_out[:, :-1]
                chosen_action_qvals = chosen_action_qvals.multiply(_a_batch).sum(-1)

                target_mac_out = []

                for t in range(episode_len):
                    obs = obs_batch[:, t]
                    obs = np.concatenate(obs, axis=0)
                    obs = torch.as_tensor(obs, dtype=torch.float32)
                    if self.is_cuda:
                        obs = obs.cuda()
                    agent_actions, self.target_hidden_states = self.target_agents(obs, self.target_hidden_states)
                    agent_actions = agent_actions.reshape([self.batch_size, self.agent_nb, -1])
                    target_h.append(self.target_hidden_states.reshape([self.batch_size, self.agent_nb, -1]))
                    target_mac_out.append(agent_actions)
                target_mac_out = torch.stack(target_mac_out[1:], dim=1)
                target_h = torch.stack(target_h[1:], dim=1)

                available_actions_batch = torch.as_tensor(available_actions_batch)

                _condition_ = torch.zeros(target_mac_out.shape)
                _condition_ = _condition_ - 9999999
                if self.is_cuda:
                    available_actions_batch = available_actions_batch.cuda()
                    _condition_ = _condition_.cuda()
                    h = h.cuda()
                    target_h = target_h.cuda()
                target_mac_out = torch.where(available_actions_batch[:, 1:] == 0, _condition_, target_mac_out)

                target_max_qvals = torch.max(target_mac_out, dim=3)[0]

                states = torch.as_tensor(np.array(s_batch), dtype=torch.float32)
                obs_batch = torch.as_tensor(obs_batch, dtype=torch.float32)
                if self.is_cuda:
                    states = states.cuda()
                    self.adj = self.adj.cuda()
                    obs_batch = obs_batch.cuda()
                if self.algo == 'iql':
                    chosen_action_qvals = self.mixer(chosen_action_qvals)
                    target_max_qvals = self.target_mixer(target_max_qvals)
                elif self.algo == 'vdn':
                    chosen_action_qvals = self.mixer(chosen_action_qvals)
                    target_max_qvals = self.target_mixer(target_max_qvals)
                elif self.algo == 'qmix':
                    # 调用qmixer类 将agent的Qvalue值和当前state值传入
                    chosen_action_qvals = self.mixer(chosen_action_qvals, states[:, :-1])
                    target_max_qvals = self.target_mixer(target_max_qvals, states[:, 1:])
                elif self.algo == 'vgn':
                    chosen_action_qvals = self.mixer(obs_batch[:, :-1], chosen_action_qvals, self.adj)
                    target_max_qvals = self.target_mixer(obs_batch[:, 1:], target_max_qvals, self.adj)
                elif self.algo == 'idea':
                    chosen_action_qvals = self.mixer(h, chosen_action_qvals, self.adj, states[:, :-1])
                    target_max_qvals = self.target_mixer(target_h, target_max_qvals, self.adj, states[:, 1:])
                elif self.algo == 'NoQmix':
                    chosen_action_qvals = self.mixer(h, chosen_action_qvals, self.adj, states[:, :-1])
                    target_max_qvals = self.target_mixer(target_h, target_max_qvals, self.adj, states[:, 1:])
                elif self.algo == 'NoHistory':
                    chosen_action_qvals = self.mixer(obs_batch[:, :-1], chosen_action_qvals, self.adj, states[:, :-1])
                    target_max_qvals = self.target_mixer(obs_batch[:, 1:], target_max_qvals, self.adj, states[:, 1:])

                # 计算loss
                yi = r_batch + self.gamma * (1-t_batch.unsqueeze(-1)) * target_max_qvals
                td_error = (chosen_action_qvals - yi.detach())
                mask = mask.unsqueeze(-1).expand_as(td_error)
                masked_td_error = td_error * mask
                loss = (masked_td_error ** 2).sum() / mask.sum()
                if self.is_cuda:
                    loss = loss.cuda()
                print("loss:", loss.detach().cpu().numpy().item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def act(self, batch, obs, agents_available_actions):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        if self.is_cuda:
            obs = obs.cuda()
        value_action, self.hidden_states = self.agents(obs, self.hidden_states)
        condition = torch.zeros(value_action.shape)
        condition = condition - int(1e10)
        if self.is_cuda:
            condition = condition.cuda()
            agents_available_actions = agents_available_actions.cuda()
        value_action = torch.where(agents_available_actions == 0, condition, value_action)

        if self.training:
            value_action = self.epsilon_greedy.act(value_action, agents_available_actions)
        else:
            value_action = np.argmax(value_action.detach().cpu().numpy(), -1)

        value_action = value_action.reshape([batch, self.agent_nb, -1])
        return value_action


