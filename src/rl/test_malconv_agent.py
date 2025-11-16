# coding=UTF-8
"""
测试训练好的Agent在测试集上的规避率
Test the trained Agent on test set and calculate evasion rate
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

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
from chainer import optimizers

from tools import interface_malconv as interface
from action import manipulate2

net_layers = [256, 64]


def main():
    parser = argparse.ArgumentParser(description='Test trained Agent on MalConv')
    parser.add_argument('--model-dir', type=str, required=True, 
                       help='Path to trained model directory')
    parser.add_argument('--maxturns', type=int, default=60,
                       help='Maximum modification turns per sample')
    parser.add_argument('--output', type=str, default='malconv_test_results.txt',
                       help='Output file for results')
    args = parser.parse_args()

    print("="*80)
    print(" Testing Trained Agent to Attack MalConv")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model directory: {args.model_dir}")
    print(f"  Max turns: {args.maxturns}")
    print(f"  Output file: {args.output}")

    # Q函数定义（与训练时相同）
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

    # 加载测试样本
    test_samples = interface.get_available_sha256(test=True)
    label_map = interface.get_original_label()
    
    print(f"\n测试集:")
    print(f"  样本数: {len(test_samples)}")

    # 动作表
    ACTION_LOOKUP = {i: act for i, act in enumerate(manipulate2.ACTION_TABLE.keys())}
    n_actions = len(ACTION_LOOKUP)
    
    # 创建Agent（需要知道obs_size）
    # 先读一个样本来获取特征维度
    from state.state_malconv import FeatureExtractMalConv
    feature_extractor = FeatureExtractMalConv()
    
    sample_bytez = interface.fetch_file(test_samples[0], test=True)
    sample_features = feature_extractor.get_state(sample_bytez)
    obs_size = sample_features.shape[1]
    
    print(f"\n模型配置:")
    print(f"  Observation size: {obs_size}")
    print(f"  Action space: {n_actions}")
    
    # 创建Q函数
    q_func = QFunction(obs_size, n_actions)
    
    # 创建优化器（测试时不需要，但创建Agent需要）
    opt = optimizers.Adam()
    opt.setup(q_func)
    
    # 创建Agent
    phi = lambda x: x.astype(np.float32, copy=False)
    
    agent = chainerrl.agents.DoubleDQN(
        q_func, opt, 
        chainerrl.replay_buffers.ReplayBuffer(1000), 
        gamma=0.99,
        explorer=chainerrl.explorers.ConstantEpsilonGreedy(
            epsilon=0.0,  # 测试时不探索
            random_action_func=lambda: np.random.randint(n_actions)
        ),
        replay_start_size=1,
        target_update_interval=1,
        update_interval=1,
        phi=phi,
        minibatch_size=1
    )
    
    # 加载训练好的模型
    print(f"\n加载模型...")
    try:
        agent.load(args.model_dir)
        print(f"✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 开始测试
    print(f"\n{'='*80}")
    print(" 开始测试")
    print(f"{'='*80}\n")
    
    results = []
    success_count = 0
    total_turns = 0
    
    start_time = time.time()
    
    for idx, sample_name in enumerate(test_samples, 1):
        print(f"[{idx}/{len(test_samples)}] {sample_name}")
        
        # 读取样本
        try:
            bytez = interface.fetch_file(sample_name, test=True)
        except Exception as e:
            print(f"  ✗ 读取失败: {e}")
            results.append({
                'sample': sample_name,
                'success': False,
                'turns': 0,
                'actions': [],
                'error': str(e)
            })
            continue
        
        # 检查原始标签
        original_label = interface.get_label_local(bytez)
        original_conf = interface.get_confidence(bytez)
        
        print(f"  原始: label={original_label}, conf={original_conf:.4f}")
        
        if original_label != label_map[sample_name]:
            print(f"  ⚠ 已误分类，跳过")
            continue
        
        # 使用Agent进行攻击
        actions_taken = []
        evaded = False
        
        for turn in range(args.maxturns):
            # 提取特征
            features = feature_extractor.get_state(bytez)
            
            # Agent选择动作
            action_idx = agent.act(features)
            action_name = ACTION_LOOKUP[action_idx]
            actions_taken.append(action_name)
            
            # 执行动作
            try:
                bytez = bytes(manipulate2.modify_without_breaking(bytez, [action_name]))
            except Exception as e:
                print(f"  ✗ 动作执行失败: {e}")
                break
            
            # 检测
            new_label = interface.get_label_local(bytez)
            new_conf = interface.get_confidence(bytez)
            
            # 检查是否成功免杀
            if new_label == '0':  # 成功变成良性
                evaded = True
                print(f"  ✓ 免杀成功! turns={turn+1}, conf={new_conf:.4f}")
                print(f"    动作序列: {actions_taken}")
                success_count += 1
                total_turns += turn + 1
                break
        
        if not evaded:
            final_conf = interface.get_confidence(bytez)
            print(f"  ✗ 免杀失败 (最终conf={final_conf:.4f})")
        
        # 记录结果
        results.append({
            'sample': sample_name,
            'success': evaded,
            'turns': len(actions_taken) if evaded else args.maxturns,
            'actions': actions_taken,
            'original_conf': original_conf,
            'final_conf': interface.get_confidence(bytez) if not evaded else new_conf
        })
        
        print()
    
    elapsed_time = time.time() - start_time
    
    # 计算统计信息
    tested_count = len([r for r in results if 'error' not in r])
    evasion_rate = (success_count / tested_count * 100) if tested_count > 0 else 0
    avg_turns = (total_turns / success_count) if success_count > 0 else 0
    
    # 打印总结
    print(f"{'='*80}")
    print(" 测试完成")
    print(f"{'='*80}")
    print(f"\n总体统计:")
    print(f"  测试样本数: {tested_count}")
    print(f"  成功免杀数: {success_count}")
    print(f"  规避率: {evasion_rate:.2f}%")
    print(f"  平均修改次数: {avg_turns:.1f}")
    print(f"  测试时间: {elapsed_time/60:.1f} 分钟")
    
    # 保存详细结果
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" MalConv Attack Test Results\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model: {args.model_dir}\n")
        f.write(f"Test samples: {tested_count}\n")
        f.write(f"Successful evasions: {success_count}\n")
        f.write(f"Evasion rate: {evasion_rate:.2f}%\n")
        f.write(f"Average turns: {avg_turns:.1f}\n")
        f.write(f"Test time: {elapsed_time/60:.1f} minutes\n")
        f.write("\n" + "="*80 + "\n\n")
        
        f.write("Detailed Results:\n\n")
        for i, result in enumerate(results, 1):
            f.write(f"[{i}] {result['sample']}\n")
            if 'error' in result:
                f.write(f"  Error: {result['error']}\n")
            else:
                f.write(f"  Success: {result['success']}\n")
                f.write(f"  Turns: {result['turns']}\n")
                f.write(f"  Original conf: {result['original_conf']:.4f}\n")
                f.write(f"  Final conf: {result['final_conf']:.4f}\n")
                if result['success']:
                    f.write(f"  Actions: {result['actions']}\n")
            f.write("\n")
    
    print(f"\n详细结果已保存到: {args.output}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

