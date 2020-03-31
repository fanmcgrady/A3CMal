# coding=UTF-8
# ! /usr/bin/python
import argparse
import linecache
import sys

# add rl
import time

sys.path.append('.')
# add winner
sys.path.append('../kaggle_Microsoft_malware_full/')
# add novel
sys.path.append('../novel_feature')
from collections import defaultdict

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
from chainer import optimizers
from chainerrl import experiments, explorers, misc, links
from chainerrl.replay_buffers import *
from tools import interface

from chainerrl.agents import a3c
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.optimizers import rmsprop_async
from chainerrl import policies
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl import v_function

from env import sha256_holdout, MAXTURNS
from action import action as manipulate
from tools.interface import *
from tools.hook.plot_hook import PlotHook
import logging

ACTION_LOOKUP = {i: act for i, act in enumerate(manipulate.ACTION_TABLE.keys())}

net_layers = [256, 64]

# 用于快速调用chainerrl的训练方法，参数如下：
# 1、命令行启动visdom
# ➜  ~ source activate new
# (new) ➜  ~ python -m visdom.server -p 8888
# 2、运行train
# python train.py

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', type=int, default=8)
    parser.add_argument('--arch', type=str, default='FFSoftmax',
                        choices=('FFSoftmax', 'FFMellowmax'))
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--outdir', type=str, default='models')
    parser.add_argument('--t-max', type=int, default=5)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--gpu', action='store_false')
    parser.add_argument('--final-exploration-steps', type=int, default=10 ** 4)
    parser.add_argument('--start-epsilon', type=float, default=1.0)
    parser.add_argument('--end-epsilon', type=float, default=0.1)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--update-interval', type=int, default=1)
    parser.add_argument('--eval-n-runs', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--test-random', action='store_true')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--rmsprop-epsilon', type=float, default=1e-1)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--description', type=str, default='')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Set a random seed used in ChainerRL.
    # If you use more than one processes, the results will be no longer
    # deterministic even with the same random seed.
    misc.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.processes) + args.seed * args.processes
    assert process_seeds.max() < 2 ** 32

    class A3CFFSoftmax(chainer.ChainList, a3c.A3CModel):
        """An example of A3C feedforward softmax policy."""

        def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
            self.pi = policies.SoftmaxPolicy(
                model=links.MLP(ndim_obs, n_actions, hidden_sizes))
            self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
            super().__init__(self.pi, self.v)

        def pi_and_v(self, state):
            return self.pi(state), self.v(state)

    class A3CFFMellowmax(chainer.ChainList, a3c.A3CModel):
        """An example of A3C feedforward mellowmax policy."""

        def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
            self.pi = policies.MellowmaxPolicy(
                model=links.MLP(ndim_obs, n_actions, hidden_sizes))
            self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
            super().__init__(self.pi, self.v)

        def pi_and_v(self, state):
            return self.pi(state), self.v(state)

    def make_env(process_idx, test):
        train_env = gym.make('malware-v0')
        test_env = gym.make('malware-test-v0')
        env = train_env if not test else test_env
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        if args.monitor and process_idx == 0:
            env = chainerrl.wrappers.Monitor(env, args.outdir)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = chainerrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        if args.render and process_idx == 0 and not test:
            env = chainerrl.wrappers.Render(env)
        return env

    # 创建ddqn agent
    def create_a3c_agent():
        # env = gym.make('malware-v0')
        # obs_size = env.observation_space.shape[1]
        # action_space = env.action_space
        # n_actions = action_space.n
        obs_size = 8006
        n_actions = 6

        # Switch policy types accordingly to action space types
        if args.arch == 'FFSoftmax':
            model = A3CFFSoftmax(obs_size, n_actions)
        elif args.arch == 'FFMellowmax':
            model = A3CFFMellowmax(obs_size, n_actions)

        if args.gpu:
            pass
        #     model.to_gpu(0)

        opt = rmsprop_async.RMSpropAsync(
            lr=args.lr, eps=args.rmsprop_epsilon, alpha=0.99)
        opt.setup(model)

        agent = a3c.A3C(model, opt, t_max=args.t_max, gamma=0.99,
                        beta=args.beta)

        return agent

    # 开始训练
    def train_agent():
        agent = create_a3c_agent()

        step_q_hook = PlotHook('Average Q Value (Step)', plot_index=0, xlabel='train step',
                               ylabel='Average Q Value (Step)')
        step_loss_hook = PlotHook('Average Loss (Step)', plot_index=1, xlabel='train step',
                                  ylabel='Average Loss (Step)')

        experiments.train_agent_async(
            agent=agent,
            outdir=args.outdir,
            processes=args.processes,
            make_env=make_env,
            profile=args.profile,
            steps=args.steps,
            eval_n_steps=args.eval_n_runs,
            eval_n_episodes=None,
            eval_interval=args.eval_interval,
            successful_score=8.5,
            global_step_hooks=[step_q_hook, step_loss_hook],
            max_episode_len=MAXTURNS)

    # 获取保存的模型目录
    def get_latest_model_dir_from(basedir):
        best_model = os.path.join(basedir, 'best')
        if os.path.exists(best_model):
            return best_model
        else:
            assert False, "No best models!"

    # 动作评估，测试时使用
    def evaluate(action_function, cm_name):
        success = []
        misclassified = []
        label_map = interface.get_original_label()
        cm_dict_before = {}
        cm_dict_after = {}
        for i, sha256 in enumerate(sha256_holdout):
            # 创建字典存放测试后的{文件——>类别}对应关系
            success_dict = defaultdict(list)
            bytez = interface.fetch_file(sha256, test=True)
            label = interface.get_label_local(bytez)
            cm_dict_before[sha256] = label
            cm_dict_after[sha256] = label   # 先记录原始的，改成功后再更新

            if label != label_map[sha256]:
                misclassified.append(sha256)
                continue  # already misclassified, move along

            action_list = []
            for _ in range(MAXTURNS):
                action = action_function(bytez)
                action_list.append(action)
                success_dict[sha256].append(action)
                bytez = manipulate.modify_without_breaking(bytez, action)
                new_label = interface.get_label_local(bytez)
                if new_label != env.label_map[sha256]:
                    # 如果改成功了，记录
                    cm_dict_after[sha256] = new_label
                    success.append(success_dict)
                    break

            print("{}:{}->{}".format(i + 1, sha256, action_list))

        # 绘制cm
        interface.draw_after_train(cm_dict_before, cm_dict_after, cm_name)

        return success, misclassified  # evasion accuracy is len(success) / len(sha256_holdout)

    if not args.test:
        print("training...")

        # 反复多次重新训练模型，避免手工操作
        for _ in range(args.rounds):
            start_time = time.time()
            args.outdir = experiments.prepare_output_dir(
                args, args.outdir, argv=sys.argv)

            train_agent()

            # 训练结束
            # with open(os.path.join(args.outdir, 'scores.txt'), 'a') as f:
            #     f.write(
            #         "total_turn/episode->{}({}/{})\n".format(env.total_turn / env.episode, env.total_turn, env.episode))
            #
            #     success_count = 0
            #     for k, v in env.history.items():
            #         if v['evaded']:
            #             success_count += 1
            #
            #     f.write("success count->{}/{}\n".format(success_count, len(env.history.keys())))
            #
            # # 保存history
            # with open(os.path.join(args.outdir, 'history.txt'), 'a') as f:
            #     f.write("{}".format(env.history))
            #
            # 保存history
            with open(os.path.join(args.outdir, 'time.txt'), 'a') as f:
                f.write('Time elapsed {} hours.\n'.format((time.time() - start_time) / 3600))

            # 标识成功失败
            dirs = os.listdir(args.outdir)

            with open(os.path.join(args.outdir, 'scores.txt'), 'r') as f:
                lines = f.readlines()
                last = lines[-1]
                elements = last.strip('\n').split('\t')
                step = elements[0]
                success_score = elements[3]

            # 训练提前结束，标识成功
            success_flag = False
            for file in dirs:
                if file.endswith('_finish') and not file.startswith(str(args.steps)):
                    success_flag = True
                    break

            os.rename(args.outdir, '{}-{}-{}{}'.format(args.outdir.split('.')[0], step, success_score, '-success' if success_flag else ''))

            # 重置outdir到models
            args.outdir = 'models'
    else:
        print("testing...")
        model_fold = os.path.join(args.outdir, args.load)
        scores_file = os.path.join(model_fold, 'scores.txt')

        env = gym.make('malware-test-v0')

        # baseline: choose actions at random
        if args.test_random:
            random_action = lambda bytez: np.random.choice(list(manipulate.ACTION_TABLE.keys()))
            random_success, misclassified = evaluate(random_action, 'random')
            total = len(sha256_holdout) - len(misclassified)  # don't count misclassified towards success

            with open(scores_file, 'a') as f:
                random_result = "random: {}({}/{})\n".format(len(random_success) / total, len(random_success), total)
                f.write(random_result)

        total = len(sha256_holdout)

        def agent_policy(agent):
            def f(bytez):
                # first, get features from bytez
                feats = interface.feature_extractor.get_state(bytez)
                action_index = agent.act(feats)
                return ACTION_LOOKUP[action_index]

            return f

        # ddqn

        agent = create_a3c_agent()
        mm = get_latest_model_dir_from(model_fold)
        agent.load(mm)
        success, _ = evaluate(agent_policy(agent), 'test')
        blackbox_result = "black: {}({}/{})".format(len(success) / total, len(success), total)
        with open(scores_file, 'a') as f:
            f.write("{}->{}\n".format(mm, blackbox_result))


if __name__ == '__main__':
    main()
