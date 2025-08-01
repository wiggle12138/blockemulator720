"""
第二步多尺度对比学习配置文件
支持真实时间步的参数配置
"""

import time
from typing import Dict, Any, Optional


class Step2Config:
    """第二步多尺度对比学习配置类"""
    
    def __init__(self):
        """初始化默认配置"""
        # 基础配置
        self.base_config = {
            # 模型架构参数
            'input_dim': 64,           # 修复：匹配第一步f_classic的64维输出
            'hidden_dim': 32,          # 修复：相应调整隐藏层维度
            'time_dim': 16,            # 时间嵌入维度
            'output_dim': 32,          # 修复：相应调整输出嵌入维度
            
            # 对比学习参数
            'k_ratio': 0.9,            # 采样比例
            'alpha': 0.3,              # 图级别损失权重
            'beta': 0.4,               # 节点级别损失权重
            'gamma': 0.3,              # 子图级别损失权重
            'tau': 0.09,               # 温度参数
            
            # 训练参数
            'lr': 0.02,                # 学习率
            'weight_decay': 9e-6,      # 权重衰减
            'epochs': 100,             # 训练轮数
            'batch_size': 32,          # 批次大小
            
            # 时序参数
            'time_window': 5,          # 时间窗口大小
            'max_timestamp': 10000,    # 最大时间戳
            'use_real_timestamps': True, # 是否使用真实时间戳
            
            # 图结构参数
            'num_node_types': 5,       # 节点类型数量
            'num_edge_types': 3,       # 边类型数量
            
            # 处理参数
            'device': 'auto',          # 设备选择: 'auto', 'cpu', 'cuda'
            'save_intermediate': True,  # 是否保存中间结果
            'verbose': True            # 是否输出详细信息
        }
        
        # 实时处理配置
        self.realtime_config = {
            'cache_temporal_data': True,    # 是否缓存时序数据
            'max_cache_size': 20,           # 最大缓存大小
            'processing_timeout': 30.0,     # 处理超时时间(秒)
            'memory_limit_mb': 2048,        # 内存限制(MB)
            'adaptive_batch_size': True,    # 自适应批次大小
            'min_batch_size': 8,            # 最小批次大小
            'max_batch_size': 64            # 最大批次大小
        }
        
        # BlockEmulator集成配置
        self.blockemulator_config = {
            'step1_input_format': {
                'f_classic': '[N, 128]',   # 经典特征格式
                'f_graph': '[N, 96]',      # 图特征格式
                'f_reduced': '[N, 64]',    # 精简特征格式
                'node_mapping': 'Dict',    # 节点映射格式
                'metadata': 'Dict'         # 元数据格式
            },
            'timestamp_source': 'blockemulator',  # 时间戳来源
            'feature_scaling': 'standard',        # 特征缩放方法
            'adjacency_construction': 'similarity' # 邻接矩阵构建方法
        }
    
    def get_config(self, mode: str = 'default') -> Dict[str, Any]:
        """
        获取指定模式的配置
        
        Args:
            mode: 配置模式
                - 'default': 默认配置
                - 'realtime': 实时处理配置
                - 'training': 训练优化配置
                - 'inference': 推理优化配置
                - 'debug': 调试配置
        
        Returns:
            配置字典
        """
        config = self.base_config.copy()
        
        if mode == 'realtime':
            config.update(self.realtime_config)
            # 实时模式优化
            config.update({
                'epochs': 20,           # 减少训练轮数
                'batch_size': 16,       # 减少批次大小
                'time_window': 3,       # 减少时间窗口
                'save_intermediate': False
            })
            
        elif mode == 'training':
            # 训练模式优化
            config.update({
                'epochs': 300,          # 增加训练轮数
                'batch_size': 32,       # 标准批次大小
                'lr': 0.01,             # 稍微降低学习率
                'time_window': 10,      # 增加时间窗口
                'save_intermediate': True
            })
            
        elif mode == 'inference':
            # 推理模式优化
            config.update({
                'epochs': 1,            # 仅推理，不训练
                'batch_size': 64,       # 增大批次以提高效率
                'save_intermediate': False,
                'verbose': False
            })
            
        elif mode == 'debug':
            # 调试模式
            config.update({
                'epochs': 5,            # 快速训练
                'batch_size': 8,        # 小批次便于调试
                'verbose': True,
                'save_intermediate': True,
                'time_window': 2
            })
        
        # 设备自动选择
        if config['device'] == 'auto':
            import torch
            config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        return config
    
    def get_blockemulator_integration_config(self) -> Dict[str, Any]:
        """获取BlockEmulator集成专用配置"""
        config = self.get_config('realtime')
        config.update(self.blockemulator_config)
        
        # BlockEmulator特定优化
        config.update({
            'adaptive_time_window': True,    # 自适应时间窗口
            'node_count_adaptive': True,     # 节点数量自适应
            'feature_validation': True,     # 特征验证
            'error_resilience': True        # 错误恢复能力
        })
        
        return config
    
    def create_custom_config(self, 
                           input_dim: Optional[int] = None,
                           hidden_dim: Optional[int] = None,
                           time_window: Optional[int] = None,
                           use_real_timestamps: Optional[bool] = None,
                           **kwargs) -> Dict[str, Any]:
        """
        创建自定义配置
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            time_window: 时间窗口大小
            use_real_timestamps: 是否使用真实时间戳
            **kwargs: 其他参数
        
        Returns:
            自定义配置字典
        """
        config = self.get_config('default')
        
        # 更新指定参数
        if input_dim is not None:
            config['input_dim'] = input_dim
        if hidden_dim is not None:
            config['hidden_dim'] = hidden_dim
        if time_window is not None:
            config['time_window'] = time_window
        if use_real_timestamps is not None:
            config['use_real_timestamps'] = use_real_timestamps
        
        # 更新其他参数
        config.update(kwargs)
        
        return config
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        验证配置的有效性
        
        Args:
            config: 要验证的配置
        
        Returns:
            是否有效
        """
        required_keys = [
            'input_dim', 'hidden_dim', 'time_dim', 'time_window',
            'k_ratio', 'alpha', 'beta', 'gamma', 'tau'
        ]
        
        # 检查必需的键
        for key in required_keys:
            if key not in config:
                print(f"配置验证失败: 缺少必需键 '{key}'")
                return False
        
        # 检查数值范围
        validations = [
            ('input_dim', lambda x: x > 0),
            ('hidden_dim', lambda x: x > 0),
            ('time_dim', lambda x: x > 0),
            ('time_window', lambda x: x > 0),
            ('k_ratio', lambda x: 0 < x <= 1),
            ('alpha', lambda x: x >= 0),
            ('beta', lambda x: x >= 0),
            ('gamma', lambda x: x >= 0),
            ('tau', lambda x: x > 0)
        ]
        
        for key, validator in validations:
            if not validator(config[key]):
                print(f"配置验证失败: '{key}' 值无效: {config[key]}")
                return False
        
        # 检查权重和是否合理
        weight_sum = config['alpha'] + config['beta'] + config['gamma']
        if weight_sum == 0:
            print("配置验证失败: 所有损失权重都为0")
            return False
        
        print("✓ 配置验证通过")
        return True
    
    def print_config(self, config: Dict[str, Any]):
        """打印配置信息"""
        print("=== 第二步多尺度对比学习配置 ===")
        
        # 分类打印
        categories = {
            '模型架构': ['input_dim', 'hidden_dim', 'time_dim', 'output_dim'],
            '对比学习': ['k_ratio', 'alpha', 'beta', 'gamma', 'tau'],
            '训练参数': ['lr', 'weight_decay', 'epochs', 'batch_size'],
            '时序参数': ['time_window', 'max_timestamp', 'use_real_timestamps'],
            '处理参数': ['device', 'save_intermediate', 'verbose']
        }
        
        for category, keys in categories.items():
            print(f"\n{category}:")
            for key in keys:
                if key in config:
                    print(f"  {key}: {config[key]}")
        
        # 打印其他参数
        other_keys = set(config.keys()) - set().union(*categories.values())
        if other_keys:
            print(f"\n其他参数:")
            for key in sorted(other_keys):
                print(f"  {key}: {config[key]}")


# 预定义配置实例
step2_config = Step2Config()

# 导出常用配置
DEFAULT_CONFIG = step2_config.get_config('default')
REALTIME_CONFIG = step2_config.get_config('realtime')
TRAINING_CONFIG = step2_config.get_config('training')
INFERENCE_CONFIG = step2_config.get_config('inference')
DEBUG_CONFIG = step2_config.get_config('debug')
BLOCKEMULATOR_CONFIG = step2_config.get_blockemulator_integration_config()


def get_step2_config(mode: str = 'default', **kwargs) -> Dict[str, Any]:
    """
    便捷函数：获取第二步配置
    
    Args:
        mode: 配置模式
        **kwargs: 自定义参数
    
    Returns:
        配置字典
    """
    config = step2_config.get_config(mode)
    config.update(kwargs)
    
    if not step2_config.validate_config(config):
        raise ValueError("配置验证失败")
    
    return config


if __name__ == "__main__":
    # 演示不同配置模式
    print("=== 第二步配置演示 ===")
    
    modes = ['default', 'realtime', 'training', 'inference', 'debug']
    
    for mode in modes:
        print(f"\n--- {mode.upper()} 模式配置 ---")
        config = get_step2_config(mode)
        step2_config.print_config(config)
    
    # 演示自定义配置
    print(f"\n--- 自定义配置演示 ---")
    custom_config = step2_config.create_custom_config(
        input_dim=256,
        hidden_dim=128,
        time_window=8,
        use_real_timestamps=True,
        epochs=150
    )
    step2_config.print_config(custom_config)
