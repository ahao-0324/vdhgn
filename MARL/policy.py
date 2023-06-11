import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class RNNAgent(nn.Module):

    def __init__(self, input_shape,  rnn_hidden_dim=64, n_actions=1):
        super(RNNAgent, self).__init__()

        print('input_shape: ', input_shape)

        self.input_shape = input_shape
        self.rnn_hidden_dim = rnn_hidden_dim
        self.n_actions = n_actions

        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)

    def init_hidden(self):
        return torch.zeros([1, self.rnn_hidden_dim])

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        # 将输入的hidden_state的列转换为GRUCell的输入纬度，否则报错
        h_in = hidden_state.reshape([-1, self.rnn_hidden_dim])
        h = self.rnn(x, h_in)
        actions = self.fc2(h)
        return actions, h

    def update(self, agent):
        self.load_state_dict(agent.state_dict())


class VGN(nn.Module):
    def __init__(self, agent_nb, obs_shape, hidden_dim, alpha, concat=True):
        super(VGN, self).__init__()
        self.agent_nb = agent_nb
        self.obs_shape = obs_shape           # 节点向量的特征维度
        self.hidden_dim = hidden_dim        # 经过GAT之后的特征维度
        self.alpha = alpha
        self.concat = concat
        # 定义可训练的参数，即论文中的a和W
        self.W = nn.Parameter(torch.zeros(size=(obs_shape, hidden_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)                    # xavier初始化，防止参数过大或过小，减少方差
        self.a = nn.Parameter(torch.zeros(size=(2 * hidden_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # 定义leakyReLU激活函数
        self.leakyReLU = nn.LeakyReLU(self.alpha)

    def forward(self, agent_obses, agent_qs, adj):
        '''
        :param input_h: [N, in_features]
        :param adj:     图的邻接矩阵,维度[N,N]
        :return:
        '''
        tmp = agent_obses.shape[0]
        agent_obses = agent_obses.reshape(-1, self.agent_nb, self.obs_shape)
        agent_qs = agent_qs.reshape(-1, self.agent_nb, 1)
        h = torch.matmul(agent_obses, self.W)         # [N, out_features]

        N = self.agent_nb
        input_concat = torch.cat([h.repeat(1, 1, N).reshape(-1, N * N, self.hidden_dim), h.repeat(1, N, 1)], dim=2).reshape(-1, N, N, 2 * self.hidden_dim)

        e = self.leakyReLU(torch.matmul(input_concat, self.a).squeeze(-1))

        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        q_tot = torch.matmul(attention, agent_qs).sum(1).reshape(tmp, -1)
        return q_tot.unsqueeze(-1)

    def update(self, net):
        self.load_state_dict(net.state_dict())


class QMIX(nn.Module):
    def __init__(self, n_agents, state_shape, mixing_embed_dim=32):
        super(QMIX, self).__init__()

        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape))
        self.embed_dim = mixing_embed_dim
        # 超网络定义
        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        # 数据预处理
        tmp = agent_qs.shape[0]
        states = states.reshape([-1, self.state_dim])
        agent_qs = agent_qs.reshape([-1, 1, self.n_agents])
        # 第一层超网络
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.reshape([-1, self.n_agents, self.embed_dim])
        b1 = b1.reshape([-1, 1, self.embed_dim])
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # 第二层超网络
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.reshape([-1, self.embed_dim, 1])

        v = self.V(states).reshape([-1, 1, 1])
        # 计算最后的输出
        y = torch.bmm(hidden, w_final) + v

        q_tot = y.reshape([tmp, -1, 1])
        return q_tot

    def update(self, net):
        self.load_state_dict(net.state_dict())


class IQL(nn.Module):
    def __init__(self):
        super(IQL, self).__init__()

    def forward(self, inputs):
        return inputs

    def update(self, net):
        self.load_state_dict(net.state_dict())


class VDN(nn.Module):
    def __init__(self):
        super(VDN, self).__init__()

    def forward(self, agent_qs):
        return agent_qs.sum(-1).unsqueeze(-1)

    def update(self, net):
        self.load_state_dict(net.state_dict())


class Idea(nn.Module):
    def __init__(self, agent_nb, state_shape, feature_dim, hidden_dim, alpha, concat=True, mixing_embed_dim=32):
        super(Idea, self).__init__()
        self.agent_nb = agent_nb
        self.feature_dim = feature_dim
        # 图注意力隐藏层特征维度
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.concat = concat
        self.state_dim = int(np.prod(state_shape))
        # 超网络隐藏层特征维度
        self.embed_dim = mixing_embed_dim
        # 定义可训练的参数，即论文中的a和W
        self.W = nn.Parameter(torch.zeros(size=(self.feature_dim, self.hidden_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化，防止参数过大或过小，减少方差
        self.a = nn.Parameter(torch.zeros(size=(2 * self.hidden_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # 定义超网络
        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.agent_nb)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )
        # 定义leakyReLU激活函数
        self.leakyReLU = nn.LeakyReLU(self.alpha)

    def forward(self, features, agent_qs, adj, states):
        '''
        :param input_h: [N, in_features]
        :param adj:     图的邻接矩阵,维度[N,N]
        :return:
        '''
        adj = torch.as_tensor(adj, dtype=torch.float32)
        states = states.reshape([-1, self.state_dim])
        tmp = features.shape[0]
        agent_obses = torch.as_tensor(features, dtype=torch.float32).reshape(-1, self.agent_nb, self.feature_dim)
        agent_qs = agent_qs.reshape(-1, self.agent_nb, 1)
        h = torch.matmul(agent_obses, self.W)  # [N, out_features]

        N = self.agent_nb
        input_concat = torch.cat([h.repeat(1, 1, N).reshape(-1, N * N, self.hidden_dim), h.repeat(1, N, 1)],
                                 dim=2).reshape(-1, N, N, 2 * self.hidden_dim)

        e = self.leakyReLU(torch.matmul(input_concat, self.a).squeeze(-1))

        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        q_tmp = torch.matmul(attention, agent_qs).reshape(-1, 1, self.agent_nb)
        # 第一层超网络
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.reshape([-1, self.agent_nb, self.embed_dim])
        b1 = b1.reshape([-1, 1, self.embed_dim])
        hidden = F.elu(torch.bmm(q_tmp, w1) + b1)
        # 第二层超网络
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.reshape([-1, self.embed_dim, 1])

        v = self.V(states).reshape([-1, 1, 1])
        # 计算最后的输出
        y = torch.bmm(hidden, w_final) + v

        q_tot = y.reshape([tmp, -1, 1])

        return q_tot

    def update(self, net):
        self.load_state_dict(net.state_dict())


class NoQmix(nn.Module):
    def __init__(self, agent_nb, state_shape, feature_dim, hidden_dim, alpha, concat=True, mixing_embed_dim=32):
        super(NoQmix, self).__init__()
        self.agent_nb = agent_nb
        self.feature_dim = feature_dim
        # 图注意力隐藏层特征维度
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.concat = concat
        self.state_dim = int(np.prod(state_shape))
        # 超网络隐藏层特征维度
        self.embed_dim = mixing_embed_dim
        # 定义可训练的参数，即论文中的a和W
        self.W = nn.Parameter(torch.zeros(size=(self.feature_dim, self.hidden_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化，防止参数过大或过小，减少方差
        self.a = nn.Parameter(torch.zeros(size=(2 * self.hidden_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # 定义超网络
        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.agent_nb)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )
        # 定义leakyReLU激活函数
        self.leakyReLU = nn.LeakyReLU(self.alpha)

    def forward(self, features, agent_qs, adj, states):
        '''
        :param input_h: [N, in_features]
        :param adj:     图的邻接矩阵,维度[N,N]
        :return:
        '''
        adj = torch.as_tensor(adj, dtype=torch.float32)
        states = states.reshape([-1, self.state_dim])
        tmp = features.shape[0]
        agent_obses = torch.as_tensor(features, dtype=torch.float32).reshape(-1, self.agent_nb, self.feature_dim)
        agent_qs = agent_qs.reshape(-1, self.agent_nb, 1)
        h = torch.matmul(agent_obses, self.W)  # [N, out_features]

        N = self.agent_nb
        input_concat = torch.cat([h.repeat(1, 1, N).reshape(-1, N * N, self.hidden_dim), h.repeat(1, N, 1)],
                                 dim=2).reshape(-1, N, N, 2 * self.hidden_dim)

        e = self.leakyReLU(torch.matmul(input_concat, self.a).squeeze(-1))

        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        q_tmp = torch.matmul(attention, agent_qs).reshape(-1, 1, self.agent_nb)
        q_tmp = torch.sum(q_tmp, dim=-1)
        # # 第一层超网络
        # w1 = torch.abs(self.hyper_w_1(states))
        # b1 = self.hyper_b_1(states)
        # w1 = w1.reshape([-1, self.agent_nb, self.embed_dim])
        # b1 = b1.reshape([-1, 1, self.embed_dim])
        # hidden = F.elu(torch.bmm(q_tmp, w1) + b1)
        # # 第二层超网络
        # w_final = torch.abs(self.hyper_w_final(states))
        # w_final = w_final.reshape([-1, self.embed_dim, 1])
        #
        # v = self.V(states).reshape([-1, 1, 1])
        # # 计算最后的输出
        # y = torch.bmm(hidden, w_final) + v

        q_tot = q_tmp.reshape([tmp, -1, 1])

        return q_tot

    def update(self, net):
        self.load_state_dict(net.state_dict())


class NoHistory(nn.Module):
    def __init__(self, agent_nb, state_shape, feature_dim, hidden_dim, alpha, concat=True, mixing_embed_dim=32):
        super(NoHistory, self).__init__()
        self.agent_nb = agent_nb
        self.feature_dim = feature_dim
        # 图注意力隐藏层特征维度
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.concat = concat
        self.state_dim = int(np.prod(state_shape))
        # 超网络隐藏层特征维度
        self.embed_dim = mixing_embed_dim
        # 定义可训练的参数，即论文中的a和W
        self.W = nn.Parameter(torch.zeros(size=(self.feature_dim, self.hidden_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化，防止参数过大或过小，减少方差
        self.a = nn.Parameter(torch.zeros(size=(2 * self.hidden_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # 定义超网络
        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.agent_nb)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )
        # 定义leakyReLU激活函数
        self.leakyReLU = nn.LeakyReLU(self.alpha)

    def forward(self, features, agent_qs, adj, states):
        '''
        :param input_h: [N, in_features]
        :param adj:     图的邻接矩阵,维度[N,N]
        :return:
        '''
        adj = torch.as_tensor(adj, dtype=torch.float32)
        states = states.reshape([-1, self.state_dim])
        tmp = features.shape[0]
        agent_obses = torch.as_tensor(features, dtype=torch.float32).reshape(-1, self.agent_nb, self.feature_dim)
        agent_qs = agent_qs.reshape(-1, self.agent_nb, 1)
        h = torch.matmul(agent_obses, self.W)  # [N, out_features]

        N = self.agent_nb
        input_concat = torch.cat([h.repeat(1, 1, N).reshape(-1, N * N, self.hidden_dim), h.repeat(1, N, 1)],
                                 dim=2).reshape(-1, N, N, 2 * self.hidden_dim)

        e = self.leakyReLU(torch.matmul(input_concat, self.a).squeeze(-1))

        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        q_tmp = torch.matmul(attention, agent_qs).reshape(-1, 1, self.agent_nb)
        # 第一层超网络
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.reshape([-1, self.agent_nb, self.embed_dim])
        b1 = b1.reshape([-1, 1, self.embed_dim])
        hidden = F.elu(torch.bmm(q_tmp, w1) + b1)
        # 第二层超网络
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.reshape([-1, self.embed_dim, 1])

        v = self.V(states).reshape([-1, 1, 1])
        # 计算最后的输出
        y = torch.bmm(hidden, w_final) + v

        q_tot = y.reshape([tmp, -1, 1])

        return q_tot

    def update(self, net):
        self.load_state_dict(net.state_dict())