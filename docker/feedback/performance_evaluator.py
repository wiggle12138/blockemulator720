"""
层次化分片性能评估器
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict

class PerformanceEvaluator(nn.Module):
    """层次化分片性能评估器"""

    def __init__(self, feature_dims: Dict[str, int]):
        super().__init__()
        self.feature_dims = feature_dims

        # 指标-特征层映射权重 (可学习参数)
        self.layer_weights = nn.ParameterDict({
            'hardware_to_balance': nn.Parameter(torch.tensor(0.6)),
            'topo_to_balance': nn.Parameter(torch.tensor(0.4)),
            'semantic_to_cross': nn.Parameter(torch.tensor(0.7)),
            'topo_to_cross': nn.Parameter(torch.tensor(0.3)),
        })

        # 性能指标预测器
        self.metric_predictors = nn.ModuleDict({
            'balance': nn.Sequential(
                nn.Linear(feature_dims.get('hardware', 17) + feature_dims.get('topology', 20), 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ),
            'cross_shard': nn.Sequential(
                nn.Linear(feature_dims.get('semantic', 32) + feature_dims.get('topology', 20), 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ),
            'security': nn.Sequential(
                nn.Linear(feature_dims.get('consensus', 3), 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        })

        # 历史状态缓存
        self.history_window = 24
        self.performance_history = []

    def forward(self, features: Dict[str, torch.Tensor],
                shard_assignments: torch.Tensor,
                edge_index: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        层次化性能评估

        Args:
            features: 分层特征字典 {'hardware': tensor, 'topology': tensor, ...}
            shard_assignments: 分片分配结果 [num_nodes, num_shards] 或 [num_nodes]
            edge_index: 边索引 [2, num_edges] (可选)

        Returns:
            performance_metrics: 性能指标字典
        """
        metrics = {}

        # 处理分片分配格式
        if shard_assignments.dim() == 2:
            hard_assignment = torch.argmax(shard_assignments, dim=1)
        else:
            hard_assignment = shard_assignments

        num_nodes = hard_assignment.size(0)
        num_shards = int(torch.max(hard_assignment).item()) + 1

        # 1. 负载均衡度评估
        metrics['balance_score'] = self._compute_load_balance(
            hard_assignment, features, num_shards
        )

        # 2. 跨片交易率评估 (如果有边信息)
        if edge_index is not None:
            metrics['cross_tx_rate'] = self._compute_cross_shard_rate(
                hard_assignment, edge_index, features
            )
        else:
            # 基于特征相似度估算跨片交易率
            metrics['cross_tx_rate'] = self._estimate_cross_shard_rate(
                hard_assignment, features
            )

        # 3. 安全阈值评估
        metrics['security_score'] = self._compute_security_threshold(
            hard_assignment, features, num_shards
        )

        # 4. 动态权重调整
        metrics = self._apply_entropy_weights(metrics)

        # 5. 更新历史记录
        self._update_history(metrics)

        return metrics

    def _compute_load_balance(self, hard_assignment: torch.Tensor,
                              features: Dict[str, torch.Tensor],
                              num_shards: int) -> torch.Tensor:
        """计算负载均衡度"""
        # 基础分片大小均衡
        shard_sizes = torch.zeros(num_shards, device=hard_assignment.device)
        for s in range(num_shards):
            shard_sizes[s] = torch.sum(hard_assignment == s).float()

        # 考虑硬件权重的负载
        if 'hardware' in features:
            weighted_sizes = torch.zeros(num_shards, device=hard_assignment.device)
            hardware_features = features['hardware']

            for s in range(num_shards):
                shard_mask = (hard_assignment == s)
                if torch.sum(shard_mask) > 0:
                    # 使用CPU和内存特征计算权重
                    cpu_weight = torch.mean(hardware_features[shard_mask, 0])  # CPU cores
                    mem_weight = torch.mean(hardware_features[shard_mask, 1])  # Memory
                    weighted_sizes[s] = shard_sizes[s] * (cpu_weight + mem_weight) / 2
            shard_sizes = weighted_sizes

        # 计算均衡度 (1 - 变异系数)
        mean_size = torch.mean(shard_sizes)
        std_size = torch.std(shard_sizes)
        balance_score = 1.0 - (std_size / (mean_size + 1e-8))

        return torch.clamp(balance_score, 0.0, 1.0)

    def _compute_cross_shard_rate(self, hard_assignment: torch.Tensor,
                                  edge_index: torch.Tensor,
                                  features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算跨片交易率"""
        u, v = edge_index[0], edge_index[1]
        cross_shard_mask = (hard_assignment[u] != hard_assignment[v])

        base_cross_rate = torch.sum(cross_shard_mask).float() / edge_index.size(1)
        return torch.clamp(base_cross_rate, 0.0, 1.0)

    def _estimate_cross_shard_rate(self, hard_assignment: torch.Tensor,
                                   features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """基于特征相似度估算跨片交易率"""
        if 'topology' not in features:
            return torch.tensor(0.5, device=hard_assignment.device)

        # 使用拓扑特征估算
        topo_features = features['topology']
        num_shards = int(torch.max(hard_assignment).item()) + 1

        cross_shard_similarity = 0.0
        total_pairs = 0

        for s1 in range(num_shards):
            for s2 in range(s1 + 1, num_shards):
                mask1 = (hard_assignment == s1)
                mask2 = (hard_assignment == s2)

                if torch.sum(mask1) > 0 and torch.sum(mask2) > 0:
                    feat1 = torch.mean(topo_features[mask1], dim=0)
                    feat2 = torch.mean(topo_features[mask2], dim=0)
                    similarity = torch.nn.functional.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0))
                    cross_shard_similarity += similarity.item()
                    total_pairs += 1

        if total_pairs > 0:
            avg_similarity = cross_shard_similarity / total_pairs
            # 相似度越高，跨片交易率越高
            estimated_rate = torch.tensor(avg_similarity, device=hard_assignment.device)
        else:
            estimated_rate = torch.tensor(0.5, device=hard_assignment.device)

        return torch.clamp(estimated_rate, 0.0, 1.0)

    def _compute_security_threshold(self, hard_assignment: torch.Tensor,
                                    features: Dict[str, torch.Tensor],
                                    num_shards: int) -> torch.Tensor:
        """计算安全阈值"""
        if 'consensus' not in features:
            return torch.tensor(0.8, device=hard_assignment.device)

        consensus_features = features['consensus']
        min_security = 1.0

        for s in range(num_shards):
            shard_mask = (hard_assignment == s)
            if torch.sum(shard_mask) > 0:
                shard_consensus = consensus_features[shard_mask]
                # 使用共识参与率作为安全指标
                security_score = torch.mean(shard_consensus[:, 0])  # 参与率
                min_security = min(min_security, security_score.item())

        return torch.tensor(min_security, device=hard_assignment.device)

    def _apply_entropy_weights(self, metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """基于熵权法的动态权重调整"""
        if len(self.performance_history) < 3:
            return metrics

        # 简化的熵权法实现
        history_tensors = []
        for hist in self.performance_history[-min(self.history_window, len(self.performance_history)):]:
            hist_values = [hist[key].item() for key in ['balance_score', 'cross_tx_rate', 'security_score'] if key in hist]
            if hist_values:
                history_tensors.append(hist_values)

        if len(history_tensors) > 1:
            history_array = np.array(history_tensors)
            # 计算信息熵权重
            weights = self._compute_entropy_weights(history_array)

            metric_names = ['balance_score', 'cross_tx_rate', 'security_score']
            for i, metric_name in enumerate(metric_names):
                if metric_name in metrics and i < len(weights):
                    metrics[metric_name] *= weights[i]

        return metrics

    def _compute_entropy_weights(self, data: np.ndarray) -> np.ndarray:
        """计算信息熵权重"""
        # 标准化数据
        data_norm = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0) + 1e-8)

        # 计算熵
        entropies = []
        for j in range(data_norm.shape[1]):
            p = data_norm[:, j] / np.sum(data_norm[:, j])
            p = p[p > 0]  # 避免log(0)
            entropy = -np.sum(p * np.log(p))
            entropies.append(entropy)

        entropies = np.array(entropies)
        weights = (1 - entropies) / np.sum(1 - entropies)
        return weights

    def _update_history(self, metrics: Dict[str, torch.Tensor]):
        """更新历史记录"""
        # 转换为可存储的格式
        history_entry = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                history_entry[key] = value.detach().clone()
            else:
                history_entry[key] = value

        self.performance_history.append(history_entry)

        # 保持历史记录窗口大小
        if len(self.performance_history) > self.history_window:
            self.performance_history.pop(0)