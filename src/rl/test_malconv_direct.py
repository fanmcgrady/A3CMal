#!/usr/bin/env python
# coding=UTF-8
"""
直接测试PE修改对MalConv的影响（无需训练Agent）
Test PE modifications against MalConv directly (without training Agent)
"""
import os
import sys

# 设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append('.')
sys.path.append('../novel_feature')

from tools import interface_malconv as interface
from action import manipulate2
import random

print("="*80)
print(" 直接测试PE修改攻击MalConv（无需训练Agent）")
print("="*80)

# 获取样本
samples = interface.get_available_sha256(test=False)
print(f"\n找到 {len(samples)} 个样本")

# 选择一个样本
test_sample = samples[0]
print(f"\n测试样本: {test_sample}")

# 读取原始文件
original_bytez = interface.fetch_file(test_sample)
print(f"文件大小: {len(original_bytez):,} 字节")

# 原始检测
orig_label = interface.get_label_local(original_bytez)
orig_conf = interface.get_confidence(original_bytez)

print(f"\n【原始文件】")
print(f"  预测: {orig_label} ({'恶意' if orig_label == '1' else '良性'})")
print(f"  置信度: {orig_conf:.4f}")

# 可用的动作
actions = list(manipulate2.ACTION_TABLE.keys())
print(f"\n可用的修改动作: {actions}")

# 测试每个动作的效果
print(f"\n{'='*80}")
print(" 测试各个动作的免杀效果")
print(f"{'='*80}")

results = []
for action_name in actions:
    try:
        print(f"\n[{action_name}]")
        
        # 应用动作
        modified_bytez = manipulate2.modify_without_breaking(original_bytez, [action_name])
        
        if len(modified_bytez) == len(original_bytez):
            print(f"  ⚠ 文件未改变，跳过")
            continue
        
        # 检测修改后的文件
        mod_label = interface.get_label_local(modified_bytez)
        mod_conf = interface.get_confidence(modified_bytez)
        
        # 计算变化
        conf_change = orig_conf - mod_conf
        conf_pct = (conf_change / orig_conf * 100) if orig_conf > 0 else 0
        
        print(f"  修改后: {mod_label} ({'恶意' if mod_label == '1' else '良性'})")
        print(f"  置信度: {mod_conf:.4f}")
        print(f"  置信度变化: {conf_change:+.4f} ({conf_pct:+.1f}%)")
        
        # 判断效果
        if mod_label == '0' and orig_label == '1':
            print(f"  ✅ 免杀成功！从恶意变成良性")
            results.append((action_name, conf_change, True))
        elif conf_change > 0.05:
            print(f"  ✓ 部分有效：显著降低了检测置信度")
            results.append((action_name, conf_change, False))
        elif conf_change > 0:
            print(f"  ～ 轻微效果：略微降低了检测置信度")
            results.append((action_name, conf_change, False))
        else:
            print(f"  ✗ 无效果")
            results.append((action_name, conf_change, False))
            
    except Exception as e:
        print(f"  ✗ 执行失败: {str(e)[:50]}")

# 总结
print(f"\n{'='*80}")
print(" 测试总结")
print(f"{'='*80}")

if results:
    # 按效果排序
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n动作效果排名:")
    for i, (action, change, success) in enumerate(results, 1):
        status = "✅ 完全免杀" if success else f"置信度降低 {change:.4f}"
        print(f"  {i}. {action:20s} - {status}")
    
    best_action, best_change, best_success = results[0]
    print(f"\n最有效的动作: {best_action}")
    if best_success:
        print(f"  🎉 成功实现免杀！")
    else:
        print(f"  可降低置信度 {best_change:.4f}")
        print(f"  💡 提示: 组合多个动作可能更有效")

print(f"\n{'='*80}")
print(" 如果想让Agent自动学习最优策略，请运行:")
print("   python train_malconv.py --steps 5000")
print(f"{'='*80}")

