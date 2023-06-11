import torch
import os
import runner
from torch.utils.tensorboard import SummaryWriter


class Args:
    def __init__(self, traning, scenario='6h_vs_8z', actors=1, algo='qmix', batch_size=32, is_cuda=False):
        self.scenario = scenario
        self.train = traning
        self.actors = actors
        self.algo = algo
        self.batch_size = batch_size
        self.is_cuda = is_cuda


def main(arglist):
    # 用于保存
    log_dir = './result/' + 'map--' + arglist.scenario + '-algo-' + arglist.algo

    writer = SummaryWriter(log_dir=log_dir, comment="Star Craft II")

    # 线程数
    actors = arglist.actors
    if not arglist.train:
        actors = 1
    env_runner = runner.Runner(arglist, arglist.scenario, actors, arglist.algo, arglist.batch_size)
    end_train = True
    cnt = 0
    while (arglist.train or env_runner.episode < 1) and end_train:
        env_runner.reset()
        replay_buffers = env_runner.run()
        for replay_buffer in replay_buffers:
            env_runner.algo.episode_batch.add(replay_buffer)
        env_runner.algo.train()
        for episode in env_runner.episodes:
            env_runner.algo.update_targets(episode)

        # # 保存agent网络参数
        # for idx, episode in enumerate(env_runner.episodes):
        #     if episode and episode % int(1e6) == 0 and arglist.train:
        #         env_runner.algo.save_model('./saved/agents_' + str(episode))
        #     if env_runner.episode_reward[idx] >= max_reward:
        #         max_reward = env_runner.episode_reward[idx]
        #         # env_runner.algo.save_model('./saved/agents_reward_' + str(env_runner.episode_reward[idx]) + '_' + str(episode))
        #         pass

        print(env_runner.win_counted_array)
        for idx, episode in enumerate(env_runner.episodes):
            print("Total reward in episode {} = {} and global step: {}".format(episode, env_runner.episode_reward[idx],
                                                                               env_runner.episode_global_step))
            if cnt == 200:
                end_train = False
            if arglist.train and episode and episode % 100 == 0:
                reward, victory = env_runner.evaluate()
                writer.add_scalar('Reward', torch.as_tensor(reward), cnt)
                writer.add_scalar('Victory', torch.as_tensor(victory), cnt)
                cnt += 1

    if not arglist.train:
        env_runner.save()

    env_runner.close()


if __name__ == '__main__':
    try:
        os.mkdir('./saved')
    except OSError:
        print("Creation of the directory failed")
    else:
        print("Successfully created the directory")

    is_training = True
    # 地图名
    map_names = ['2s_vs_1sc','3m', '8m', '2s3z', '3s_vs_5z', '5m_vs_6m', '2c_vs_64zg']
    # 算法名
    algoes = ['iql', 'vdn', 'qmix', 'vgn', 'idea']
    # 线程数
    actors = 10
    # 是否使用GPU
    is_cuda = True
    # for map in map_names:
    arglist = Args(is_training, '2s_vs_1sc', actors, 'NoQmix', batch_size=64, is_cuda=is_cuda)
    main(arglist)
    # for algo in algoes:
    #     arglist = Args(is_training, '3s5z_vs_3s6z', actors, algo, batch_size=64, is_cuda=is_cuda)
    #     main(arglist)
    #     # algo = 'vgn'
    #     # arglist = Args(is_training, scenario, actors, algo, batch_size=10)
    #     # main(arglist)