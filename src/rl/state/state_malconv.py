"""
Feature extraction for MalConv model
MalConv不需要传统的特征工程，直接使用padding后的原始字节
"""
import numpy as np


class FeatureExtractMalConv:
    """
    MalConv特征提取器
    实际上就是将字节序列padding到固定长度
    """
    def __init__(self, maxlen=1048576, padding_char=256):
        """
        Args:
            maxlen: 最大长度 (默认1MB)
            padding_char: padding字符的值
        """
        self.maxlen = maxlen
        self.padding_char = padding_char
    
    def get_state(self, bytez):
        """
        获取状态（对MalConv来说就是padding后的字节序列）
        
        Args:
            bytez: PE文件的原始字节
            
        Returns:
            numpy array: shape (1, maxlen)，值域[0, 256]
        """
        # 创建padding数组
        b = np.ones((self.maxlen,), dtype=np.int16) * self.padding_char
        
        # 填充实际字节（截断到maxlen）
        bytez_array = np.frombuffer(bytez[:self.maxlen], dtype=np.uint8)
        b[:len(bytez_array)] = bytez_array
        
        # 返回二维数组以符合gym的observation_space格式
        return b.reshape(1, -1)


if __name__ == '__main__':
    # 测试代码
    print("Testing FeatureExtractMalConv...")
    
    extractor = FeatureExtractMalConv()
    
    # 创建测试数据
    test_bytes = b'MZ\x90\x00' * 100  # 模拟PE文件头
    
    state = extractor.get_state(test_bytes)
    
    print(f"Input size: {len(test_bytes)} bytes")
    print(f"Output shape: {state.shape}")
    print(f"Output dtype: {state.dtype}")
    print(f"First 10 values: {state[0, :10]}")
    print(f"Last 10 values: {state[0, -10:]}")
    print(f"Unique values: {len(np.unique(state))}")
    
    print("\n✓ FeatureExtractMalConv test passed!")

