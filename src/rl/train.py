# coding=UTF-8
# ! /usr/bin/python
import argparse
import linecache
import sys

# add rl
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
from chainerrl import experiments, explorers, misc
from chainerrl.replay_buffers import *

from env import sha256_holdout, MAXTURNS
from action import action as manipulate
from tools.interface import *
from tools.hook.plot_hook import PlotHook

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
    parser.add_argument('--outdir', type=str, default='models')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--final-exploration-steps', type=int, default=10 ** 4)
    parser.add_argument('--start-epsilon', type=float, default=1.0)
    parser.add_argument('--end-epsilon', type=float, default=0.1)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--prioritized-replay', action='store_false')
    parser.add_argument('--episodic-replay', action='store_true')
    parser.add_argument('--replay-start-size', type=int, default=1000)
    parser.add_argument('--target-update-interval', type=int, default=10 ** 2)
    parser.add_argument('--target-update-method', type=str, default='hard')
    parser.add_argument('--soft-update-tau', type=float, default=1e-2)
    parser.add_argument('--update-interval', type=int, default=1)
    parser.add_argument('--eval-n-runs', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--minibatch-size', type=int, default=None)
    parser.add_argument('--test-random', action='store_true')
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()

    # q函数
    class QFunction(chainer.Chain):
        def __init__(self, obs_size, n_actions, n_hidden_channels=None):
            super(QFunction, self).__init__()
            if n_hidden_channels is None:
                n_hidden_channels = net_layers
            net = []
            inpdim = obs_size
            for i, n_hid in enumerate(n_hidden_channels):
                net += [('l{}'.format(i), L.Linear(inpdim, n_hid))]
                # net += [('norm{}'.format(i), L.BatchNormalization(n_hid))]
                net += [('_act{}'.format(i), F.relu)]
                net += [('_dropout{}'.format(i), F.dropout)]
                inpdim = n_hid

            net += [('output', L.Linear(inpdim, n_actions))]

            with self.init_scope():
                for n in net:
                    if not n[0].startswith('_'):
                        setattr(self, n[0], n[1])

            self.forward = net

        def __call__(self, x, test=False):
            """
            Args:
                x (ndarray or chainer.Variable): An observation
                test (bool): a flag indicating whether it is in test mode
            """
            for n, f in self.forward:
                if not n.startswith('_'):
                    x = getattr(self, n)(x)
                elif n.startswith('_dropout'):
                    x = f(x, 0.1)
                else:
                    x = f(x)

            return chainerrl.action_value.DiscreteActionValue(x)

    # 创建ddqn agent
    def create_ddqn_agent(env, args):
        obs_size = env.observation_space.shape[1]
        action_space = env.action_space
        n_actions = action_space.n

        # q_func = q_functions.FCStateQFunctionWithDiscreteAction(
        #     obs_size, n_actions,
        #     n_hidden_channels=args.n_hidden_channels,
        #     n_hidden_layers=args.n_hidden_layers)
        q_func = QFunction(obs_size, n_actions)
        if args.gpu:
            q_func.to_gpu(0)

        # Draw the computational graph and save it in the output directory.
        if not args.test and not args.gpu:
            chainerrl.misc.draw_computational_graph(
                [q_func(np.zeros_like(env.observation_space, dtype=np.float32)[None])],
                os.path.join(args.outdir, 'model'))

        # Use epsilon-greedy for exploration
        explorer = explorers.LinearDecayEpsilonGreedy(
            args.start_epsilon, args.end_epsilon, args.final_exploration_steps,
            action_space.sample)
        # explorer = explorers.Boltzmann()
        # explorer = explorers.ConstantEpsilonGreedy(
        #     epsilon=0.3, random_action_func=env.action_space.sample)

        opt = optimizers.Adam()
        opt.setup(q_func)

        rbuf_capacity = 5 * 10 ** 3
        if args.episodic_replay:
            if args.minibatch_size is None:
                args.minibatch_size = 4
            if args.prioritized_replay:
                betasteps = (args.steps - args.replay_start_size) // args.update_interval
                rbuf = PrioritizedEpisodicReplayBuffer(rbuf_capacity, betasteps=betasteps)
            else:
                rbuf = EpisodicReplayBuffer(rbuf_capacity)
        else:
            if args.minibatch_size is None:
                args.minibatch_size = 32
            if args.prioritized_replay:
                betasteps = (args.steps - args.replay_start_size) // args.update_interval
                rbuf = PrioritizedReplayBuffer(rbuf_capacity, betasteps=betasteps)
            else:
                rbuf = ReplayBuffer(rbuf_capacity)

        # Chainer only accepts numpy.float32 by default, make sure
        # a converter as a feature extractor function phi.
        phi = lambda x: x.astype(np.float32, copy=False)

        agent = chainerrl.agents.DoubleDQN(q_func, opt, rbuf, gamma=args.gamma,
                                           explorer=explorer, replay_start_size=args.replay_start_size,
                                           target_update_interval=args.target_update_interval,
                                           update_interval=args.update_interval,
                                           phi=phi, minibatch_size=args.minibatch_size,
                                           target_update_method=args.target_update_method,
                                           soft_update_tau=args.soft_update_tau,
                                           # episodic_update=args.episodic_replay,
                                           episodic_update_len=16)

        return agent

    # 开始训练
    def train_agent(args):
        env = gym.make('malware-v0')
        test_env = gym.make('malware-test-v0')
        print("max turns is {}".format(env.maxturns))
        # np.random.seed(123)
        env.seed(123)
        # Set a random seed used in ChainerRL
        misc.set_random_seed(123)

        agent = create_ddqn_agent(env, args)

        step_q_hook = PlotHook('Average Q Value (Step)', plot_index=0, xlabel='train step',
                               ylabel='Average Q Value (Step)')
        step_loss_hook = PlotHook('Average Loss (Step)', plot_index=1, xlabel='train step',
                                  ylabel='Average Loss (Step)')
        episode_q_hook = PlotHook('Average Q Value (Episode)', plot_index=2, xlabel='train episode',
                                  ylabel='Average Q Value (Episode)')
        episode_loss_hook = PlotHook('Average Loss (Episode)', plot_index=3, xlabel='train episode',
                                     ylabel='Average Loss (Episode)')
        episode_finish_hook = PlotHook('Steps to finish (train)', plot_index=4, xlabel='train episode',
                                       ylabel='Steps to finish (train)')
        test_finish_hook = PlotHook('Steps to finish (test)', plot_index=5, xlabel='test episode',
                                    ylabel='Steps to finish (test)')
        test_scores_hook = PlotHook('success rate', plot_index=6, xlabel='test epoch', ylabel='success rate')
        # test_scores_hook = TrainingScoresHook('scores.txt', args.outdir)

        chainerrl.experiments.train_agent_with_evaluation(
            agent, env,
            steps=args.steps,  # Train the graduation_agent for this many rounds steps
            train_max_episode_len=env.maxturns,  # Maximum length of each episodes
            eval_interval=args.eval_interval,  # Evaluate the graduation_agent after every 1000 steps
            eval_n_steps=args.eval_n_runs,  # 100 episodes are sampled for each evaluation
            eval_n_episodes=None,
            outdir=args.outdir,  # Save everything to 'result' directory
            step_hooks=[step_q_hook, step_loss_hook],
            successful_score=9,
            eval_env=test_env
        )

        # episode_hooks = [episode_q_hook, episode_loss_hook, episode_finish_hook],
        # test_hooks = [test_scores_hook, test_finish_hook],

        # 保证训练一轮就成功的情况下能成功打印scores.txt文件
        # test_scores_hook(None, None, 1000)

        return env, agent

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
            bytez = interface.fetch_file(sha256)
            label, _ = interface.get_label_local(bytez)
            cm_dict_before[sha256] = label
            if label != label_map[sha256]:
                misclassified.append(sha256)
                continue  # already misclassified, move along

            action_list = []
            for _ in range(MAXTURNS):
                action = action_function(bytez)
                action_list.append(action)
                success_dict[sha256].append(action)
                bytez = manipulate.modify_without_breaking(bytez, action)
                new_label, new_state = interface.get_label_local(bytez)
                if new_label != env.label_map[sha256]:
                    # 如果改成功了，记录
                    cm_dict_after[sha256] = new_label
                    success.append(success_dict)
                    break

            print("{}:{}->{}".format(i + 1, sha256, action_list))

            # 说明改了MAXTURN次还没成功，记录原始标签
            if sha256 not in cm_dict_after:
                cm_dict_after[sha256] = env.label_map[sha256]

        # 绘制cm
        interface.draw_after_train(cm_dict_before, cm_dict_after, cm_name)

        return success, misclassified  # evasion accuracy is len(success) / len(sha256_holdout)

    interface = Interface(args.test)

    if not args.test:
        print("training...")

        # 反复多次重新训练模型，避免手工操作
        for _ in range(args.rounds):
            start_time = time.time()
            args.outdir = experiments.prepare_output_dir(
                args, args.outdir, argv=sys.argv)

            env, agent = train_agent(args)

            # 训练结束
            with open(os.path.join(args.outdir, 'scores.txt'), 'a') as f:
                f.write(
                    "total_turn/episode->{}({}/{})\n".format(env.total_turn / env.episode, env.total_turn, env.episode))

                success_count = 0
                for k, v in env.history.items():
                    if v['evaded']:
                        success_count += 1

                f.write("success count->{}/{}\n".format(success_count, len(env.history.keys())))

            # 保存history
            with open(os.path.join(args.outdir, 'history.txt'), 'a') as f:
                f.write("{}".format(env.history))

            # 保存history
            with open(os.path.join(args.outdir, 'time.txt'), 'a') as f:
                f.write('Time elapsed {} hours.\n'.format((time.time() - start_time) / 3600))

            # 标识成功失败
            dirs = os.listdir(args.outdir)

            with open(os.path.join(args.outdir, 'scores.txt'), 'r') as f:
                lines = f.readlines()
                last = lines[-3]
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
                feats = interface.get_state(bytez)
                action_index = agent.act(feats)
                return ACTION_LOOKUP[action_index]

            return f

        # ddqn

        agent = create_ddqn_agent(env, args)
        mm = get_latest_model_dir_from(model_fold)
        agent.load(mm)
        success, _ = evaluate(agent_policy(agent), 'test')
        blackbox_result = "black: {}({}/{})".format(len(success) / total, len(success), total)
        with open(scores_file, 'a') as f:
            f.write("{}->{}\n".format(mm, blackbox_result))


if __name__ == '__main__':
    main()
