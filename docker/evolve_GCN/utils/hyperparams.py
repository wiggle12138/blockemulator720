"""
动态超参数调节器
"""
import numpy as np


class HyperparameterUpdater:
    """动态超参数调节器"""

    def __init__(self):
        self.params = {
            'alpha': 0.1,  # 特征融合权重
            'lambda': 0.1,  # 对比损失权重
            'K_base': 3,  # 基础分片数
            'tau': 1.0,  # 温度参数
            'balance_weight': 1.0,
            'cross_weight': 1.0,
            'security_weight': 1.5,
            'migrate_weight': 0.5
        }

    def update_hyperparams(self, performance_metrics):
        """根据性能指标动态调节参数"""
        balance_score = performance_metrics.get('balance_score', 0.5)
        cross_tx_rate = performance_metrics.get('cross_tx_rate', 0.1)
        cross_increase_count = performance_metrics.get('cross_increase_count', 0)

        # 负载不均衡时的调节
        if balance_score < 0.3:
            self.params['alpha'] *= 1.1  # 增强图特征作用
            self.params['lambda'] += 0.02  # 加大对比约束

        # 跨片交易率持续升高时的调节
        if cross_increase_count >= 3:
            self.params['K_base'] += 1

        # 限制参数范围
        self.params['alpha'] = np.clip(self.params['alpha'], 0.05, 0.5)
        self.params['lambda'] = np.clip(self.params['lambda'], 0.01, 0.3)
        self.params['K_base'] = np.clip(self.params['K_base'], 2, 10)

        return self.params

    def get_params(self):
        """获取当前参数"""
        return self.params.copy()

    def reset_params(self):
        """重置参数到初始值"""
        self.__init__()