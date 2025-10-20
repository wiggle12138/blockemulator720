"""
第四步反馈控制器 - 与第三步EvolveGCN衔接
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .performance_evaluator import PerformanceEvaluator
from .importance_analyzer import FeatureImportanceAnalyzer
from .feature_evolution import DynamicFeatureEvolution

class FeedbackController:
    """第四步反馈控制器 - 与第三步EvolveGCN衔接"""

    def __init__(self, feature_dims: Dict[str, int], config: Dict[str, Any] = None):
        self.feature_dims = feature_dims
        self.config = config or self._default_config()

        # 初始化各个组件
        self.performance_evaluator = PerformanceEvaluator(feature_dims)
        self.importance_analyzer = FeatureImportanceAnalyzer(feature_dims)
        self.feature_evolution = None  # 延迟初始化

        # 状态管理
        self.feedback_history = []
        self.evolution_enabled = True

    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'feedback_weight': 1.0,
            'evolution_threshold': 0.1,
            'max_feedback_history': 100,
            'enable_evolution': True,
            'performance_weights': {
                'balance': 0.4,
                'cross_shard': 0.3,
                'security': 0.3
            }
        }

    def process_feedback(self, features: Dict[str, torch.Tensor],
                         shard_assignments: torch.Tensor,
                         edge_index: torch.Tensor = None,
                         evolve_gcn_model: nn.Module = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        处理反馈并优化特征空间

        Args:
            features: 分层特征 {'hardware': tensor, 'topology': tensor, ...}
            shard_assignments: 分片分配结果 [num_nodes, num_shards] 或 [num_nodes]
            edge_index: 边索引 [2, num_edges] (可选)
            evolve_gcn_model: EvolveGCN模型 (可选，用于梯度分析)

        Returns:
            feedback_signal: 反馈信号 [3] (负载均衡度, 跨片交易率, 安全阈值)
            evolved_features: 演化后的特征空间
        """
        print(f"\n=== 第四步反馈处理开始 ===")
        print(f"输入特征层: {list(features.keys())}")
        print(f"分片分配形状: {shard_assignments.shape}")

        # 1. 性能评估
        print("1. 执行性能评估...")
        performance_metrics = self.performance_evaluator(
            features, shard_assignments, edge_index
        )

        print("性能指标:")
        for key, value in performance_metrics.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")
            else:
                print(f"  {key}: {value:.4f}")

        # 2. 特征重要性分析
        print("2. 执行特征重要性分析...")
        importance_matrix = self.importance_analyzer.analyze_importance(
            features, performance_metrics, evolve_gcn_model
        )

        print("特征重要性:")
        for layer, scores in importance_matrix.items():
            print(f"  {layer}: {scores['combined']:.4f}")

        # 3. 特征空间演化 (如果启用)
        evolved_features = features  # 默认不变

        if self.config['enable_evolution']:
            print("3. 执行特征空间演化...")

            # 延迟初始化特征演化器
            if self.feature_evolution is None:
                self.feature_evolution = DynamicFeatureEvolution(features)

            # 检查是否需要演化
            if self._should_evolve(performance_metrics, importance_matrix):
                evolved_features = self.feature_evolution.evolve_feature_space(
                    importance_matrix, performance_metrics
                )
                print("特征演化完成")
            else:
                print("跳过特征演化（不满足演化条件）")
        else:
            print("3. 特征演化已禁用")

        # 4. 生成反馈信号
        feedback_signal = self._generate_feedback_signal(performance_metrics)

        # 5. 更新历史记录
        self._update_feedback_history(performance_metrics, importance_matrix, feedback_signal)

        print(f"反馈信号: {feedback_signal}")
        print(f"=== 第四步反馈处理完成 ===\n")

        return feedback_signal, evolved_features

    def _generate_feedback_signal(self, performance_metrics: Dict[str, torch.Tensor]) -> torch.Tensor:
        """生成反馈信号给第三步"""
        weights = self.config['performance_weights']

        # 提取三个核心指标
        balance_score = performance_metrics.get('balance_score', torch.tensor(0.5))
        cross_tx_rate = performance_metrics.get('cross_tx_rate', torch.tensor(0.5))
        security_score = performance_metrics.get('security_score', torch.tensor(0.8))

        # 转换为标量
        if isinstance(balance_score, torch.Tensor):
            balance_score = balance_score.item()
        if isinstance(cross_tx_rate, torch.Tensor):
            cross_tx_rate = cross_tx_rate.item()
        if isinstance(security_score, torch.Tensor):
            security_score = security_score.item()

        # 构建反馈信号
        feedback_signal = torch.tensor([
            balance_score * weights['balance'],
            cross_tx_rate * weights['cross_shard'],
            security_score * weights['security']
        ], dtype=torch.float32)

        return feedback_signal

    def _should_evolve(self, performance_metrics: Dict[str, torch.Tensor],
                       importance_matrix: Dict[str, Dict[str, float]]) -> bool:
        """判断是否应该进行特征演化"""
        # 检查性能是否低于阈值
        threshold = self.config['evolution_threshold']

        for metric_name, metric_value in performance_metrics.items():
            if isinstance(metric_value, torch.Tensor):
                value = metric_value.item()
            else:
                value = metric_value

            # 如果任何性能指标低于阈值，触发演化
            if metric_name in ['balance_score', 'security_score'] and value < (1 - threshold):
                return True
            elif metric_name == 'cross_tx_rate' and value > threshold:  # 跨片交易率高也需要优化
                return True

        # 检查特征重要性变化
        if len(self.feedback_history) > 0:
            prev_importance = self.feedback_history[-1].get('importance_matrix', {})
            for layer in importance_matrix:
                if layer in prev_importance:
                    prev_score = prev_importance[layer].get('combined', 0.5)
                    curr_score = importance_matrix[layer].get('combined', 0.5)
                    if abs(curr_score - prev_score) > threshold:
                        return True

        return False

    def _update_feedback_history(self, performance_metrics: Dict[str, torch.Tensor],
                                 importance_matrix: Dict[str, Dict[str, float]],
                                 feedback_signal: torch.Tensor):
        """更新反馈历史记录"""
        history_entry = {
            'performance_metrics': {
                key: value.item() if isinstance(value, torch.Tensor) else value
                for key, value in performance_metrics.items()
            },
            'importance_matrix': importance_matrix,
            'feedback_signal': feedback_signal.tolist(),
            'timestamp': len(self.feedback_history)
        }

        self.feedback_history.append(history_entry)

        # 限制历史记录长度
        max_history = self.config['max_feedback_history']
        if len(self.feedback_history) > max_history:
            self.feedback_history.pop(0)

    def get_feedback_statistics(self) -> Dict[str, Any]:
        """获取反馈统计信息"""
        if not self.feedback_history:
            return {}

        # 计算性能指标的统计信息
        stats = {}

        # 收集所有性能指标
        all_metrics = {}
        for entry in self.feedback_history:
            for metric_name, value in entry['performance_metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        # 计算统计量
        for metric_name, values in all_metrics.items():
            if values:
                stats[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }

        return stats

    def reset(self):
        """重置反馈控制器状态"""
        self.feedback_history.clear()
        if self.feature_evolution is not None:
            self.feature_evolution = None
        print("反馈控制器状态已重置")