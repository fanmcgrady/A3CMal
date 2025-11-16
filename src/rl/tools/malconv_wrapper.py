#!/usr/bin/python
"""
MalConv模型包装器，用于A3CMal框架
"""
import os
import sys
import numpy as np

# 添加malpatch路径以导入MalConv
sys.path.insert(0, 'models/old_model')

# 创建简化版的MalConv包装，去掉secml依赖
from keras.models import load_model
from keras.optimizers import SGD
from keras import metrics

MODULE_PATH = 'models/old_model'
MODEL_PATH = os.path.join(MODULE_PATH, 'malconv.h5')


class SimpleMalConv:
    """简化的MalConv类，去掉secml依赖"""
    def __init__(self):
        self.batch_size = 100
        self.input_dim = 257
        self.padding_char = 256
        self.malicious_threshold = 0.5
        print(f"Loading MalConv model from: {MODEL_PATH}")
        self.model = load_model(MODEL_PATH)
        _, self.maxlen, self.embedding_size = self.model.layers[1].output_shape
        
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=1e-3),
            metrics=[metrics.binary_accuracy],
        )
    
    def extract(self, bytez):
        b = np.ones((self.maxlen,), dtype=np.int16) * self.padding_char
        bytez = np.frombuffer(bytez[: self.maxlen], dtype=np.uint8)
        b[: len(bytez)] = bytez
        return b
    
    def predict_sample(self, bytez):
        return self.model.predict(bytez.reshape(1, -1), verbose=0).item()


class MalConvWrapper:
    """
    包装MalConv模型，适配A3CMal的接口
    """
    def __init__(self):
        print("Loading MalConv model...")
        self.model = SimpleMalConv()
        self.maxlen = self.model.maxlen
        self.threshold = self.model.malicious_threshold
        print(f"MalConv loaded: maxlen={self.maxlen}, threshold={self.threshold}")
    
    def extract_features(self, bytez):
        """
        提取特征（对MalConv来说就是padding后的字节序列）
        
        Args:
            bytez: PE文件的原始字节
            
        Returns:
            numpy array: shape (1, maxlen)
        """
        return self.model.extract(bytez).reshape(1, -1)
    
    def predict(self, bytez):
        """
        预测文件是否为恶意软件
        
        Args:
            bytez: PE文件的原始字节
            
        Returns:
            label (str): '0' = 良性, '1' = 恶意
            confidence (float): 预测置信度
        """
        # 提取特征
        features = self.extract_features(bytez)
        
        # 预测
        pred_prob = self.model.predict_sample(self.model.extract(bytez))
        
        # 二分类
        label = '1' if pred_prob >= self.threshold else '0'
        
        return label, pred_prob
    
    def get_confidence(self, bytez):
        """
        获取预测置信度
        
        Args:
            bytez: PE文件的原始字节
            
        Returns:
            float: 恶意的概率 (0-1)
        """
        features = self.extract_features(bytez)
        return self.model.predict_sample(self.model.extract(bytez))


if __name__ == '__main__':
    # 测试代码
    wrapper = MalConvWrapper()
    
    # 测试一个样本
    test_file = '/home/baixb/paper/A3CMal/Dataset/pe/train/Email-Worm.Win32.Bagle.fr'
    with open(test_file, 'rb') as f:
        bytez = f.read()
    
    label, confidence = wrapper.predict(bytez)
    print(f"\nTest Result:")
    print(f"  Label: {label} ({'Malicious' if label == '1' else 'Benign'})")
    print(f"  Confidence: {confidence:.4f}")

