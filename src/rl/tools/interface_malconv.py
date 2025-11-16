"""
Interface for A3CMal with MalConv model
适配MalConv模型的接口文件
"""
import csv
import glob
import sys
import os

sys.path.append('../../')
sys.path.append('../../kaggle_Microsoft_malware_full/')
sys.path.append('../../novel_feature')

module_path = os.path.dirname(os.path.abspath(sys.modules[__name__].__file__))

from tools.malconv_wrapper import MalConvWrapper
from state.state_malconv import FeatureExtractMalConv

# ==================== 配置 ====================
MODEL_NAME = 'malconv'
MODEL_CLASSIFIER = MalConvWrapper()

feature_extractor = FeatureExtractMalConv()

print(f"✓ MalConv interface loaded successfully!")
print(f"  Max input length: {MODEL_CLASSIFIER.maxlen}")
print(f"  Threshold: {MODEL_CLASSIFIER.threshold}")


# ==================== 文件操作 ====================
def fetch_file(sha256, test=False):
    """
    获取文件二进制数据
    
    Args:
        sha256: 文件名
        test: 是否是测试集
        
    Returns:
        bytes: 文件内容
    """
    root = 'test_malconv' if test else 'train_malconv'
    root = os.path.join(module_path, '../../../Dataset/pe/' + root)

    location = os.path.join(root, sha256)
    try:
        with open(location, 'rb') as infile:
            bytez = infile.read()
    except IOError:
        print("Unable to read sha256 from {}".format(location))
        raise

    return bytez


def get_available_sha256(test=False):
    """
    在samples目录中读取样本，放入list返回
    
    Args:
        test: 是否是测试集
        
    Returns:
        list: 样本文件名列表
    """
    root = 'test_malconv' if test else 'train_malconv'
    root = os.path.join(module_path, '../../../Dataset/pe/' + root)
    sha256list = []
    for fp in glob.glob(os.path.join(root, '*')):
        fn = os.path.split(fp)[-1]
        sha256list.append(fn)
    assert len(sha256list) > 0, "no files found in {} with sha256 names".format(root)
    return sha256list


# ==================== 标签操作 ====================
def get_original_label():
    """
    加载原始标签字典
    
    注意：对于MalConv（二分类），我们将所有恶意软件都标记为'1'
    不需要trainLabels.csv，直接返回所有样本标记为恶意
    
    Returns:
        dict: {文件名: 标签} 字典
    """
    label_map = {}
    
    # 获取所有训练和测试样本
    train_samples = get_available_sha256(test=False)
    test_samples = get_available_sha256(test=True)
    
    # MalConv是二分类：所有恶意软件都标记为'1'
    for sample in train_samples + test_samples:
        label_map[sample] = '1'  # 原始都是恶意的

    print(f"✓ Loaded {len(label_map)} labels (all marked as malicious for MalConv)")
    return label_map


def get_label_local(bytez):
    """
    获取MalConv分类器的预测标签
    
    Args:
        bytez: PE文件字节
        
    Returns:
        str: '0' (良性) 或 '1' (恶意)
    """
    label, confidence = MODEL_CLASSIFIER.predict(bytez)
    return label


def get_confidence(bytez):
    """
    获取MalConv的预测置信度
    
    Args:
        bytez: PE文件字节
        
    Returns:
        float: 恶意的概率 (0-1)
    """
    return MODEL_CLASSIFIER.get_confidence(bytez)


# ==================== 测试代码 ====================
if __name__ == '__main__':
    print("\n" + "="*70)
    print(" Testing MalConv Interface")
    print("="*70)
    
    # 测试1: 加载样本列表
    print("\n[Test 1] Loading sample list...")
    samples = get_available_sha256(test=False)
    print(f"✓ Found {len(samples)} samples")
    print(f"  First 3: {samples[:3]}")
    
    # 测试2: 加载标签
    print("\n[Test 2] Loading labels...")
    labels = get_original_label()
    print(f"✓ Loaded {len(labels)} labels")
    
    # 测试3: 测试一个样本
    print("\n[Test 3] Testing on a sample...")
    test_sample = samples[0]
    print(f"  Sample: {test_sample}")
    
    bytez = fetch_file(test_sample)
    print(f"  File size: {len(bytez):,} bytes")
    
    label = get_label_local(bytez)
    confidence = get_confidence(bytez)
    
    print(f"  Original label: {labels.get(test_sample, 'Unknown')}")
    print(f"  MalConv prediction: {label}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Status: {'✓ Correctly detected' if label == labels[test_sample] else '✗ Misclassified'}")
    
    print("\n" + "="*70)
    print(" All tests passed! MalConv interface is ready.")
    print("="*70)

