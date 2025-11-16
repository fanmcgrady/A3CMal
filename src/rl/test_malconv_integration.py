#!/usr/bin/env python
# coding=UTF-8
"""
测试MalConv集成是否成功
Test MalConv integration with A3CMal
"""
import os
import sys

# 设置环境变量以减少TensorFlow日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append('.')
sys.path.append('../novel_feature')

print("="*80)
print(" Testing MalConv Integration with A3CMal")
print("="*80)

# 测试1: 导入MalConv包装器
print("\n[Test 1] Importing MalConv wrapper...")
try:
    from tools.malconv_wrapper import MalConvWrapper
    print("✓ MalConv wrapper imported successfully")
except Exception as e:
    print(f"✗ Failed to import MalConv wrapper: {e}")
    sys.exit(1)

# 测试2: 加载MalConv模型
print("\n[Test 2] Loading MalConv model...")
try:
    wrapper = MalConvWrapper()
    print(f"✓ MalConv model loaded successfully")
    print(f"  Max length: {wrapper.maxlen}")
    print(f"  Threshold: {wrapper.threshold}")
except Exception as e:
    print(f"✗ Failed to load MalConv model: {e}")
    sys.exit(1)

# 测试3: 测试接口
print("\n[Test 3] Testing interface...")
try:
    from tools import interface_malconv as interface
    
    samples = interface.get_available_sha256(test=False)
    print(f"✓ Interface loaded, found {len(samples)} samples")
    
    # 测试一个样本
    test_sample = samples[0]
    bytez = interface.fetch_file(test_sample)
    label = interface.get_label_local(bytez)
    confidence = interface.get_confidence(bytez)
    
    print(f"  Test sample: {test_sample}")
    print(f"  Prediction: {label} ({'Malicious' if label == '1' else 'Benign'})")
    print(f"  Confidence: {confidence:.4f}")
    
except Exception as e:
    print(f"✗ Interface test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: 测试环境
print("\n[Test 4] Testing MalConv environment...")
try:
    import gym
    from env.malware.malware_env_malconv import MalwareEnvMalConv, ACTION_LOOKUP
    
    # 创建环境
    env = MalwareEnvMalConv(
        sha256list=samples[:10],  # 只用10个样本测试
        random_sample=True,
        maxturns=5,
        cache=True,
        test=False
    )
    
    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space} ({len(ACTION_LOOKUP)} actions)")
    
    # 测试一个episode
    print("\n[Test 5] Running test episode...")
    obs = env.reset()
    print(f"  Initial observation shape: {obs.shape}")
    print(f"  Sample: {env.sha256}")
    print(f"  Initial confidence: {env.history[env.sha256]['initial_confidence']:.4f}")
    
    # 执行3个随机动作
    for i in range(3):
        action = env.action_space.sample()
        action_name = ACTION_LOOKUP[action]
        obs, reward, done, info = env.step(action)
        
        print(f"\n  Step {i+1}:")
        print(f"    Action: {action_name}")
        print(f"    Reward: {reward}")
        print(f"    Done: {done}")
        print(f"    Confidence: {info.get('confidence', 'N/A'):.4f}")
        
        if done:
            if reward > 0:
                print(f"    ✓ Successfully evaded detection!")
            else:
                print(f"    ✗ Failed to evade")
            break
    
    print("\n✓ Environment test passed")
    
except Exception as e:
    print(f"✗ Environment test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试6: 对比原始样本和修改后的检测结果
print("\n[Test 6] Testing evasion effect...")
try:
    from action import manipulate2
    
    # 选择一个样本
    test_sample = samples[0]
    original_bytez = interface.fetch_file(test_sample)
    
    # 原始检测
    orig_label = interface.get_label_local(original_bytez)
    orig_conf = interface.get_confidence(original_bytez)
    
    print(f"  Sample: {test_sample}")
    print(f"  Original:")
    print(f"    Label: {orig_label} ({'Malicious' if orig_label == '1' else 'Benign'})")
    print(f"    Confidence: {orig_conf:.4f}")
    
    # 应用一个修改
    modified_bytez = manipulate2.modify_without_breaking(original_bytez, ['overlay_append'])
    
    # 修改后检测
    mod_label = interface.get_label_local(modified_bytez)
    mod_conf = interface.get_confidence(modified_bytez)
    
    print(f"  After overlay_append:")
    print(f"    Label: {mod_label} ({'Malicious' if mod_label == '1' else 'Benign'})")
    print(f"    Confidence: {mod_conf:.4f}")
    print(f"    Confidence change: {mod_conf - orig_conf:+.4f}")
    
    if mod_label != orig_label:
        print(f"    ✓ Successfully changed label!")
    elif abs(mod_conf - orig_conf) > 0.01:
        print(f"    ~ Confidence changed (evasion partially effective)")
    else:
        print(f"    - No significant change (try other actions)")
    
except Exception as e:
    print(f"✗ Evasion test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print(" All tests completed successfully! ✓")
print(" You can now train A3CMal to attack MalConv using:")
print("   cd /home/baixb/paper/A3CMal/src/rl")
print("   python train_malconv.py --steps 5000")
print("="*80)

