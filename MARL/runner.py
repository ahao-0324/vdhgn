from smac.env import StarCraft2Env
import numpy as np
import torch
from multiprocessing import Process, Lock, Pipe
import mixing


class Transform:
    def transform(self, tensor):
        raise NotImplementedError

    def infer_output_info(self, vshape_in, dtype_in):
        raise NotImplementedError


class OneHot(Transform):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):
        y_onehot = dense_to_onehot(tensor, num_classes=self.out_dim).squeeze()
        return y_onehot.astype('float32')

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim, ), 'float32'


def env_run(scenario, id, child_conn, locker, replay_buffer_size):
    # 定义地图以及回放路径
    env = StarCraft2Env(map_name=scenario, replay_dir="./replay/")

    env_info = env.get_env_info()


    # 获取信息
    action_n = env_info["n_actions"]
    agent_nb = env_info["n_agents"]
    state_shape = env_info["state_shape"]
    obs_shape = env_info["obs_shape"] + action_n

    agent_id_one_hot = OneHot(agent_nb)
    actions_one_hot = OneHot(action_n)

    agent_id_one_hot_array = []
    for agent_id in range(agent_nb):
        agent_id_one_hot_array.append(agent_id_one_hot.transform(np.array([agent_id])))
    actions_one_hot_reset = np.zeros((agent_nb, action_n), dtype="float32")

    state_zeros = np.zeros(state_shape)
    obs_zeros = np.zeros((agent_nb, obs_shape))
    actions_zeros = np.zeros([agent_nb, 1])
    reward_zeros = 0
    agents_available_actions_zeros = np.zeros((agent_nb, action_n))
    agents_available_actions_zeros[:, 0] = 1

    child_conn.send(id)

    while True:
        while True:
            data = child_conn.recv()
            if data == 'save':
                env.save_replay()
                child_conn.send('save ok.')
            elif data == 'close':
                env.close()
                exit()
            else:
                break
        locker.acquire()
        env.reset()
        locker.release()

        episode_reward = 0
        episode_step = 0

        obs = np.array(env.get_obs())
        obs = np.concatenate([obs, actions_one_hot_reset], axis=-1)
        state = np.array(env.get_state())
        terminated = False
        if data == 'evaluate':
            while not terminated:
                agents_available_actions = []
                for agent_id in range(agent_nb):
                    agents_available_actions.append(env.get_avail_agent_actions(agent_id))
                child_conn.send(["eva_actions", obs, agents_available_actions])
                actions = child_conn.recv()
                reward, terminated, _ = env.step(actions)

                obs2 = np.array(env.get_obs())
                actions_one_hot_agents = []
                for action in actions:
                    actions_one_hot_agents.append(actions_one_hot.transform(np.array(action)))
                actions_one_hot_agents = np.array(actions_one_hot_agents)

                obs2 = np.concatenate([obs2, actions_one_hot_agents], axis=-1)

                episode_reward += reward
                episode_step += 1

                obs = obs2
            for _ in range(episode_step, replay_buffer_size):
                child_conn.send(["eva_actions", obs_zeros, agents_available_actions_zeros])
                child_conn.recv()
            print("--------evaluete_reward:{}".format(episode_reward))

            child_conn.send(["evaluate_end", episode_reward, env.win_counted])
            continue
        while not terminated:
            agents_available_actions = []
            for agent_id in range(agent_nb):
                agents_available_actions.append(env.get_avail_agent_actions(agent_id))
            child_conn.send(["actions", obs, agents_available_actions])
            actions = child_conn.recv()
            reward, terminated, _ = env.step(actions)

            agents_available_actions2 = []
            for agent_id in range(agent_nb):
                agents_available_actions2.append(env.get_avail_agent_actions(agent_id))

            obs2 = np.array(env.get_obs())
            actions_one_hot_agents = []
            for action in actions:
                actions_one_hot_agents.append(actions_one_hot.transform(np.array(action)))
            actions_one_hot_agents = np.array(actions_one_hot_agents)

            obs2 = np.concatenate([obs2, actions_one_hot_agents], axis=-1)
            state2 = np.array(env.get_state())

            child_conn.send(["replay_buffer", state, actions, [reward], terminated, obs, agents_available_actions, 0])

            episode_reward += reward
            episode_step += 1

            obs = obs2
            state = state2

        for _ in range(episode_step, replay_buffer_size):
            child_conn.send(["actions", obs_zeros, agents_available_actions_zeros])
            child_conn.send(["replay_buffer", state_zeros, actions_zeros, [reward_zeros], True, obs_zeros, agents_available_actions_zeros, 1])
            child_conn.recv()

        child_conn.send(["episode_end", episode_reward, episode_step, env.win_counted])


class Runner:
    def __init__(self, arglist, scenario, actors, algo, batch_size):
        env = StarCraft2Env(map_name=scenario, replay_dir='./replay')

        env_info = env.get_env_info()

        self.actors = actors
        self.scenario = scenario

        self.n_actions = env_info["n_actions"]
        self.n_agents = env_info["n_agents"]
        self.state_shape = env_info["state_shape"]
        self.obs_shape = env_info["obs_shape"] + self.n_actions
        self.episode_limit = env_info["episode_limit"]
        self.algo = mixing.Mixing(arglist.train, arglist.is_cuda, self.n_agents, self.obs_shape, self.state_shape, self.n_actions, lr=5e-4, replay_buffer_size=2e3, update_target_network_step=200, batch_size=batch_size, algo=algo)

        # 验证模式
        if arglist.train == False:
            self.algo.load_model('./saved/agents_' + str(arglist.load_episode_saved))
            print("Load model agent", str(arglist.load_episode_saved))

        self.episode_global_step = 0
        self.episode = 0

        self.process_com = []
        self.locker = Lock()
        # 每个actor都分配一个进程
        for idx in range(self.actors):
            parent_conn, child_conn = Pipe()
            Process(target=env_run, args=[self.scenario, idx, child_conn, self.locker, self.episode_limit]).start()
            self.process_com.append(parent_conn)

        for process_conn in self.process_com:
            process_id = process_conn.recv()
            print(process_id, " is ready!")

    def reset(self):
        self.algo.on_reset(self.actors)
        self.episodes = []
        self.episode_reward = []
        self.episode_step = []
        self.replay_buffers = []
        self.win_counted_array = []
        episode_managed = self.episode
        for _ in range(self.actors):
            self.episodes.append(episode_managed)
            self.episode_reward.append(0)
            self.episode_step.append(0)
            self.win_counted_array.append(False)
            self.replay_buffers.append(mixing.ReplayBuffer(self.episode_limit))
            episode_managed += 1
        for process_conn in self.process_com:
            process_conn.send("GO ! ")

    def run(self):
        episode_done = 0
        process_size = len(self.process_com)
        available_to_send = np.array([True for _ in range(self.actors)])
        while True:
            obs_batch = []
            available_batch = []
            actions = None
            for idx, process_conn in enumerate(self.process_com):
                data = process_conn.recv()
                if data[0] == 'actions':
                    obs_batch.append(data[1])
                    available_batch.append(np.array(data[2]))
                    if idx == process_size - 1:
                        obs_batch = np.concatenate(obs_batch, axis=0)
                        available_batch = np.concatenate(available_batch, axis=0)

                        actions = self.algo.act(self.actors, torch.as_tensor(obs_batch), torch.as_tensor(available_batch))
                elif data[0] == 'replay_buffer':
                    self.replay_buffers[idx].add(data[1], data[2], data[3], data[4], data[5], data[6], data[7])
                elif data[0] == 'episode_end':
                    self.episode_reward[idx] = data[1]
                    self.episode_step[idx] = data[2]
                    self.win_counted_array[idx] = data[3]
                    available_to_send[idx] = False
                    episode_done += 1
            if actions is not None:
                for idx_proc, process in enumerate(self.process_com):
                    if available_to_send[idx_proc]:
                        process.send(actions[idx_proc])
            if episode_done >= self.actors:
                break
        self.episode += self.actors
        self.episode_global_step += max(self.episode_step)
        self.algo.decay_epsilon_greedy(self.episode_global_step)
        return self.replay_buffers

    def save(self):
        for process in self.process_com:
            process.send('save')
            data = process.recv()
            print(data)

    def close(self):
        for process in self.process_com:
            process.send('close')

    def evaluate(self):
        self.algo.training = False
        self.algo.on_reset(self.actors)
        reward = 0
        victory = 0

        for process in self.process_com:
            process.send('evaluate')
        episode_done = 1
        process_size = len(self.process_com)
        available_to_send = np.array([True for _ in range(self.actors)])
        while True:
            obs_batch = []
            available_batch = []
            actions = None
            for idx, process_conn in enumerate(self.process_com):
                data = process_conn.recv()
                if data[0] == 'eva_actions':
                    obs_batch.append(data[1])
                    available_batch.append(np.array(data[2]))
                    if idx == process_size - 1:
                        obs_batch = np.concatenate(obs_batch, axis=0)
                        available_batch = np.concatenate(available_batch, axis=0)
                        actions = self.algo.act(self.actors, torch.as_tensor(obs_batch), torch.as_tensor(available_batch))
                elif data[0] == 'evaluate_end':
                    reward += data[1]
                    victory += data[2]
                    available_to_send[idx] = False
                    episode_done += 1
            if actions is not None:
                for idx_proc, process in enumerate(self.process_com):
                    if available_to_send[idx_proc]:
                        process.send(actions[idx_proc])
            if episode_done > self.actors:
                break

        self.algo.training = True
        reward /= self.actors
        victory /= self.actors
        return reward, victory


def dense_to_onehot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_onehot = np.zeros((num_labels, num_classes))
    # 展平的索引值对应相加，然后得到精准索引并修改label_onehot中的每一个值
    labels_onehot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_onehot


