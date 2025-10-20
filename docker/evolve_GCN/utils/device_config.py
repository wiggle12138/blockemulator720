"""
设备配置模块
"""
import torch

# 全局设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_device():
    """获取当前设备"""
    return device

def print_device_info():
    """打印设备信息"""
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")