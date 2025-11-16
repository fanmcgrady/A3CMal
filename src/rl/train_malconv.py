# coding=UTF-8
"""
Train A3CMal to attack MalConv model
使用A3CMal攻击MalConv模型的训练脚本
"""
import argparse
import os
import sys
import time

# 设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append('.')
sys.path.append('../kaggle_Microsoft_malware_full/')
sys.path.append('../novel_feature')

from collections import defaultdict
from tools import interface_malconv as interface

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
from chainer import optimizers
from chainerrl import experiments, explorers, misc
from chainerrl.replay_buffers import *

# 导入MalConv环境
from env.malware.malware_env_malconv import ACTION_LOOKUP
from action import manipulate2 as manipulate
from tools.hook.plot_hook import PlotHook

net_layers = [256, 64]


def main():
    parser = argparse.ArgumentParser(description='Train A3CMal to attack MalConv')
    parser.add_argument('--outdir', type=str, default='models_malconv')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--final-exploration-steps', type=int, default=10 ** 4)
    parser.add_argument('--start-epsilon', type=float, default=1.0)
    parser.add_argument('--end-epsilon', type=float, default=0.1)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--prioritized-replay', action='store_false')
    parser.add_argument('--episodic-replay', action='store_true')
    parser.add_argument('--replay-start-size', type=int, default=500)  # 降低以加快开始
    parser.add_argument('--target-update-interval', type=int, default=10 ** 2)
    parser.add_argument('--target-update-method', type=str, default='hard')
    parser.add_argument('--soft-update-tau', type=float, default=1e-2)
    parser.add_argument('--update-interval', type=int, default=1)
    parser.add_argument('--eval-n-runs', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--minibatch-size', type=int, default=None)
    parser.add_argument('--test-random', action='store_true')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--description', type=str, default='MalConv Attack')
    args = parser.parse_args()

    # Q函数网络
    class QFunction(chainer.Chain):
        def __init__(self, obs_size, n_actions, n_hidden_channels=None):
            super(QFunction, self).__init__()
            if n_hidden_channels is None:
                n_hidden_channels = net_layers
            
            net = []
            inpdim = obs_size
            for i, n_hid in enumerate(n_hidden_channels):
                net += [('l{}'.format(i), L.Linear(inpdim, n_hid))]
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
            for n, f in self.forward:
                if not n.startswith('_'):
                    x = getattr(self, n)(x)
                elif n.startswith('_dropout'):
                    x = f(x, 0.1)
                else:
                    x = f(x)
            return chainerrl.action_value.DiscreteActionValue(x)

    # 创建DDQN agent
    def create_ddqn_agent(env, args):
        obs_size = env.observation_space.shape[1]
        action_space = env.action_space
        n_actions = action_space.n

        print(f"Creating DDQN agent:")
        print(f"  Observation size: {obs_size}")
        print(f"  Action space: {n_actions}")

        q_func = QFunction(obs_size, n_actions)
        if args.gpu:
            q_func.to_gpu(0)

        # ε-greedy探索
        explorer = explorers.LinearDecayEpsilonGreedy(
            args.start_epsilon, args.end_epsilon, args.final_exploration_steps,
            action_space.sample)

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

        phi = lambda x: x.astype(np.float32, copy=False)

        agent = chainerrl.agents.DoubleDQN(
            q_func, opt, rbuf, gamma=args.gamma,
            explorer=explorer, replay_start_size=args.replay_start_size,
            target_update_interval=args.target_update_interval,
            update_interval=args.update_interval,
            phi=phi, minibatch_size=args.minibatch_size,
            target_update_method=args.target_update_method,
            soft_update_tau=args.soft_update_tau,
            episodic_update_len=16
        )

        return agent

    # 训练函数
    def train_agent(args):
        print("\n" + "="*80)
        print(" Starting A3CMal Training to Attack MalConv")
        print("="*80)
        
        # 导入环境
        from env.malware.malware_env_malconv import MalwareEnvMalConv
        
        # 创建训练和测试环境
        train_samples = interface.get_available_sha256(test=False)
        test_samples = interface.get_available_sha256(test=True)
        
        print(f"\nDataset:")
        print(f"  Training samples: {len(train_samples)}")
        print(f"  Test samples: {len(test_samples)}")
        
        env = MalwareEnvMalConv(
            sha256list=train_samples,
            random_sample=True,
            maxturns=60,
            cache=True,
            test=False
        )
        
        test_env = MalwareEnvMalConv(
            sha256list=test_samples,
            random_sample=True,
            maxturns=60,
            cache=True,
            test=True
        )
        
        print(f"\nEnvironment:")
        print(f"  Max turns: {env.maxturns}")
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space.n} actions")
        
        # 设置随机种子
        env.seed(123)
        misc.set_random_seed(123)

        agent = create_ddqn_agent(env, args)

        # Hook for plotting (disabled for MalConv to save time)
        step_q_hook = PlotHook('Average Q Value (Step)', plot_index=0, xlabel='train step',
                               ylabel='Average Q Value (Step)')
        step_loss_hook = PlotHook('Average Loss (Step)', plot_index=1, xlabel='train step',
                                  ylabel='Average Loss (Step)')

        print(f"\nTraining parameters:")
        print(f"  Steps: {args.steps}")
        print(f"  Replay start size: {args.replay_start_size}")
        print(f"  Eval interval: {args.eval_interval}")
        print(f"  Minibatch size: {args.minibatch_size}")
        print(f"  Epsilon: {args.start_epsilon} → {args.end_epsilon}")
        
        print("\n" + "="*80)
        print(" Training started...")
        print("="*80 + "\n")

        chainerrl.experiments.train_agent_with_evaluation(
            agent, env,
            steps=args.steps,
            train_max_episode_len=env.maxturns,
            eval_interval=args.eval_interval,
            eval_n_steps=args.eval_n_runs,
            eval_n_episodes=None,
            outdir=args.outdir,
            step_hooks=[step_q_hook, step_loss_hook],
            successful_score=8,
            eval_env=test_env
        )

        return env, agent

    if not args.test:
        print("Training mode activated...")
        
        for _ in range(args.rounds):
            start_time = time.time()
            args.outdir = experiments.prepare_output_dir(
                args, args.outdir, argv=sys.argv)

            env, agent = train_agent(args)

            # 保存训练统计
            with open(os.path.join(args.outdir, 'scores.txt'), 'a') as f:
                f.write(f"total_turn/episode->{env.total_turn / env.episode}"
                       f"({env.total_turn}/{env.episode})\n")

                success_count = sum(1 for v in env.history.values() if v['evaded'])
                f.write(f"success count->{success_count}/{len(env.history.keys())}\n")
                
                # 统计平均置信度下降
                conf_changes = []
                for v in env.history.values():
                    if v['initial_confidence'] is not None and v['final_confidence'] is not None:
                        change = v['initial_confidence'] - v['final_confidence']
                        conf_changes.append(change)
                
                if conf_changes:
                    avg_conf_change = sum(conf_changes) / len(conf_changes)
                    f.write(f"avg confidence drop->{avg_conf_change:.4f}\n")

            # 保存历史
            with open(os.path.join(args.outdir, 'history.txt'), 'a') as f:
                f.write(f"{env.history}")

            # 保存时间
            elapsed = (time.time() - start_time) / 3600
            with open(os.path.join(args.outdir, 'time.txt'), 'a') as f:
                f.write(f'Time elapsed {elapsed:.2f} hours.\n')

            print(f"\n{'='*80}")
            print(f" Training completed!")
            print(f"  Time: {elapsed:.2f} hours")
            print(f"  Success rate: {success_count}/{len(env.history.keys())}")
            print(f"  Output dir: {args.outdir}")
            print(f"{'='*80}")

            # 重置outdir
            args.outdir = 'models_malconv'
    else:
        print("Testing mode not yet implemented for MalConv")
        # TODO: 实现测试模式


if __name__ == '__main__':
    main()

