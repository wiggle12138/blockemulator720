"""
第四步：分片结果反馈优化特征空间 - 支持6类原始特征
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import normalized_mutual_info_score
from typing import Dict, List, Tuple, Any, Optional
import json
from collections import defaultdict

class PerformanceEvaluator(nn.Module):
    """层次化分片性能评估器 - 支持6类原始特征"""

    def __init__(self, feature_dims: Dict[str, int]):
        super().__init__()
        self.feature_dims = feature_dims

        # 验证必需的6类特征
        required_features = ['hardware', 'onchain_behavior', 'network_topology',
                             'dynamic_attributes', 'heterogeneous_type', 'categorical']

        for feature in required_features:
            if feature not in feature_dims:
                print(f"[WARNING] 缺失特征类别: {feature}，使用默认维度")

        # 获取实际特征维度，设置默认值
        self.hw_dim = feature_dims.get('hardware', 17)
        self.onchain_dim = feature_dims.get('onchain_behavior', 17)
        self.topology_dim = feature_dims.get('network_topology', 20)
        self.dynamic_dim = feature_dims.get('dynamic_attributes', 13)
        self.hetero_dim = feature_dims.get('heterogeneous_type', 17)
        self.categorical_dim = feature_dims.get('categorical', 15)

        print(f"📋 PerformanceEvaluator 支持的6类特征:")
        print(f"  - hardware: {self.hw_dim}维")
        print(f"  - onchain_behavior: {self.onchain_dim}维")
        print(f"  - network_topology: {self.topology_dim}维")
        print(f"  - dynamic_attributes: {self.dynamic_dim}维")
        print(f"  - heterogeneous_type: {self.hetero_dim}维")
        print(f"  - categorical: {self.categorical_dim}维")

        # 6类特征对应的权重参数
        self.feature_weights = nn.ParameterDict({
            # 负载均衡相关权重
            'hw_to_balance': nn.Parameter(torch.tensor(0.4)),           # 硬件→负载均衡
            'topology_to_balance': nn.Parameter(torch.tensor(0.3)),     # 拓扑→负载均衡
            'dynamic_to_balance': nn.Parameter(torch.tensor(0.3)),      # 动态属性→负载均衡

            # 跨片交易相关权重
            'categorical_to_cross': nn.Parameter(torch.tensor(0.4)),    # 分类→跨片
            'topology_to_cross': nn.Parameter(torch.tensor(0.3)),       # 拓扑→跨片
            'hetero_to_cross': nn.Parameter(torch.tensor(0.3)),         # 异构→跨片

            # 安全性相关权重
            'onchain_to_security': nn.Parameter(torch.tensor(0.6)),     # 链上行为→安全
            'hetero_to_security': nn.Parameter(torch.tensor(0.4)),      # 异构类型→安全

            # 一致性相关权重
            'onchain_to_consensus': nn.Parameter(torch.tensor(0.5)),    # 链上行为→共识
            'dynamic_to_consensus': nn.Parameter(torch.tensor(0.5)),    # 动态属性→共识
        })

        # 基于6类特征的性能指标计算层
        self.metric_calculators = nn.ModuleDict({
            # 负载均衡：硬件+拓扑+动态属性
            'balance': nn.Linear(self.hw_dim + self.topology_dim + self.dynamic_dim, 1),

            # 跨片交易：分类+拓扑+异构类型
            'cross_shard': nn.Linear(self.categorical_dim + self.topology_dim + self.hetero_dim, 1),

            # 安全性：链上行为+异构类型
            'security': nn.Linear(self.onchain_dim + self.hetero_dim, 1),

            # 共识时延：链上行为+动态属性
            'consensus': nn.Linear(self.onchain_dim + self.dynamic_dim, 1),

            # 特征融合质量：所有6类特征
            'fusion_quality': nn.Linear(
                self.hw_dim + self.onchain_dim + self.topology_dim +
                self.dynamic_dim + self.hetero_dim + self.categorical_dim, 1
            ),
        })

        # 6类特征的专门评估器
        self.feature_evaluators = nn.ModuleDict({
            'hardware_evaluator': nn.Sequential(
                nn.Linear(self.hw_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            ),
            'onchain_evaluator': nn.Sequential(
                nn.Linear(self.onchain_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ),
            'topology_evaluator': nn.Sequential(
                nn.Linear(self.topology_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ),
            'dynamic_evaluator': nn.Sequential(
                nn.Linear(self.dynamic_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            ),
            'hetero_evaluator': nn.Sequential(
                nn.Linear(self.hetero_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ),
            'categorical_evaluator': nn.Sequential(
                nn.Linear(self.categorical_dim, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
                nn.Sigmoid()
            ),
        })

        # 历史状态缓存
        self.history_window = 24
        self.performance_history = []

        # 6类特征质量历史
        self.feature_quality_history = {
            'hardware': [],
            'onchain_behavior': [],
            'network_topology': [],
            'dynamic_attributes': [],
            'heterogeneous_type': [],
            'categorical': []
        }

    def forward(self, features: Dict[str, torch.Tensor],
                shard_assignments: torch.Tensor,
                edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        基于6类原始特征的层次化性能评估

        Args:
            features: 6类原始特征字典
            shard_assignments: 分片分配结果 [num_nodes, num_shards]
            edge_index: 边索引 [2, num_edges]

        Returns:
            performance_metrics: 性能指标字典
        """
        print(f" 评估输入特征: {list(features.keys())}")

        # 验证并提取6类特征
        extracted_features = self._extract_six_feature_types(features)

        metrics = {}

        try:
            # 1. 基于硬件+拓扑+动态属性的负载均衡评估
            balance_features = self._combine_features_for_balance(extracted_features)
            metrics['balance_score'] = self._compute_load_balance(
                shard_assignments, balance_features, extracted_features
            )

            # 2. 基于分类+拓扑+异构的跨片交易评估
            cross_features = self._combine_features_for_cross_shard(extracted_features)
            metrics['cross_tx_rate'] = self._compute_cross_shard_rate(
                shard_assignments, edge_index, cross_features, extracted_features
            )

            # 3. 基于链上行为+异构类型的安全性评估
            security_features = self._combine_features_for_security(extracted_features)
            metrics['security_score'] = self._compute_security_threshold(
                shard_assignments, security_features, extracted_features
            )

            # 4. 基于链上行为+动态属性的共识时延评估
            consensus_features = self._combine_features_for_consensus(extracted_features)
            metrics['consensus_latency'] = self._compute_consensus_latency(
                consensus_features, extracted_features
            )

            # 5. 6类特征的融合质量评估
            metrics['fusion_quality'] = self._compute_fusion_quality(
                extracted_features, shard_assignments
            )

            # 6. 各类特征的独立质量评估
            feature_quality_scores = self._evaluate_individual_feature_quality(
                extracted_features, shard_assignments
            )
            metrics.update(feature_quality_scores)

            # 7. 特征间协同性评估
            metrics['feature_synergy'] = self._compute_feature_synergy(
                extracted_features, shard_assignments
            )

            # 8. 动态权重调整
            metrics = self._apply_six_feature_entropy_weights(metrics, extracted_features)

            # 9. 更新历史记录
            self._update_feature_quality_history(feature_quality_scores)
            self._update_history(metrics)

        except Exception as e:
            print(f"[WARNING] 6类特征性能评估出错: {e}")
            # 返回默认指标
            device = shard_assignments.device
            metrics = self._get_default_metrics(device)

        return metrics

    def _extract_six_feature_types(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """提取并验证6类特征"""
        extracted = {}

        # 获取第一个特征的设备和批次大小用于生成占位符
        if features:
            sample_tensor = next(iter(features.values()))
            device = sample_tensor.device
            batch_size = sample_tensor.size(0)
        else:
            device = torch.device('cpu')
            batch_size = 100

        # 提取6类特征，如果缺失则生成占位符
        feature_specs = {
            'hardware': self.hw_dim,
            'onchain_behavior': self.onchain_dim,
            'network_topology': self.topology_dim,
            'dynamic_attributes': self.dynamic_dim,
            'heterogeneous_type': self.hetero_dim,
            'categorical': self.categorical_dim
        }

        for feature_name, expected_dim in feature_specs.items():
            if feature_name in features:
                extracted[feature_name] = features[feature_name]
                print(f"[SUCCESS] 提取特征: {feature_name} {features[feature_name].shape}")
            else:
                # 生成有意义的占位符
                placeholder = self._generate_feature_placeholder(
                    feature_name, batch_size, expected_dim, device
                )
                extracted[feature_name] = placeholder
                print(f"[CONFIG] 生成占位符: {feature_name} {placeholder.shape}")

        return extracted

    def _generate_feature_placeholder(self, feature_name: str, batch_size: int,
                                      dim: int, device: torch.device) -> torch.Tensor:
        """为缺失的特征类别生成有意义的占位符"""
        if feature_name == 'hardware':
            # 硬件特征：CPU、内存、存储等，通常值较高
            return torch.rand(batch_size, dim, device=device) * 0.6 + 0.3  # [0.3, 0.9]
        elif feature_name == 'onchain_behavior':
            # 链上行为：交易处理、共识参与等，中等变化
            return torch.rand(batch_size, dim, device=device) * 0.5 + 0.4  # [0.4, 0.9]
        elif feature_name == 'network_topology':
            # 网络拓扑：连接性、延迟等，相对稳定
            return torch.rand(batch_size, dim, device=device) * 0.4 + 0.3  # [0.3, 0.7]
        elif feature_name == 'dynamic_attributes':
            # 动态属性：负载、状态等，变化较大
            return torch.rand(batch_size, dim, device=device) * 0.8 + 0.1  # [0.1, 0.9]
        elif feature_name == 'heterogeneous_type':
            # 异构类型：节点类型、角色等，离散性较强
            return torch.rand(batch_size, dim, device=device) * 0.3 + 0.2  # [0.2, 0.5]
        elif feature_name == 'categorical':
            # 分类特征：类别、标签等，通常较为稳定
            return torch.rand(batch_size, dim, device=device) * 0.5 + 0.25 # [0.25, 0.75]
        else:
            return torch.rand(batch_size, dim, device=device) * 0.5 + 0.25

    def _combine_features_for_balance(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """组合硬件+拓扑+动态属性特征用于负载均衡评估"""
        hw_weighted = features['hardware'] * self.feature_weights['hw_to_balance']
        topo_weighted = features['network_topology'] * self.feature_weights['topology_to_balance']
        dynamic_weighted = features['dynamic_attributes'] * self.feature_weights['dynamic_to_balance']

        return torch.cat([hw_weighted, topo_weighted, dynamic_weighted], dim=1)

    def _combine_features_for_cross_shard(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """组合分类+拓扑+异构特征用于跨片交易评估"""
        cat_weighted = features['categorical'] * self.feature_weights['categorical_to_cross']
        topo_weighted = features['network_topology'] * self.feature_weights['topology_to_cross']
        hetero_weighted = features['heterogeneous_type'] * self.feature_weights['hetero_to_cross']

        return torch.cat([cat_weighted, topo_weighted, hetero_weighted], dim=1)

    def _combine_features_for_security(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """组合链上行为+异构类型特征用于安全性评估"""
        onchain_weighted = features['onchain_behavior'] * self.feature_weights['onchain_to_security']
        hetero_weighted = features['heterogeneous_type'] * self.feature_weights['hetero_to_security']

        return torch.cat([onchain_weighted, hetero_weighted], dim=1)

    def _combine_features_for_consensus(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """组合链上行为+动态属性特征用于共识评估"""
        onchain_weighted = features['onchain_behavior'] * self.feature_weights['onchain_to_consensus']
        dynamic_weighted = features['dynamic_attributes'] * self.feature_weights['dynamic_to_consensus']

        return torch.cat([onchain_weighted, dynamic_weighted], dim=1)

    def _compute_load_balance(self, shard_assignments: torch.Tensor,
                              balance_features: torch.Tensor,
                              original_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """基于硬件+拓扑+动态属性计算负载均衡度"""
        hard_assignment = torch.argmax(shard_assignments, dim=1)
        num_shards = shard_assignments.size(1)

        # 计算每个分片的负载
        shard_loads = torch.zeros(num_shards, device=shard_assignments.device)

        for s in range(num_shards):
            shard_mask = (hard_assignment == s)
            shard_size = torch.sum(shard_mask).item()

            if shard_size > 0:
                # 基础节点数负载
                base_load = float(shard_size)

                # 硬件能力加权（硬件能力高的节点负载能力强）
                hw_capability = torch.mean(original_features['hardware'][shard_mask]).item()

                # 网络拓扑影响（连接性好的节点通信开销低）
                topo_efficiency = torch.mean(original_features['network_topology'][shard_mask]).item()

                # 动态属性影响（当前负载状态）
                dynamic_load = torch.mean(original_features['dynamic_attributes'][shard_mask]).item()

                # 综合负载计算
                effective_load = base_load * (1.0 - hw_capability * 0.3) * (1.0 - topo_efficiency * 0.2) * (1.0 + dynamic_load * 0.5)
                shard_loads[s] = effective_load

        # 负载均衡度计算（基于变异系数）
        mean_load = torch.mean(shard_loads)
        std_load = torch.std(shard_loads)
        balance_score = 1.0 - (std_load / (mean_load + 1e-8))

        return torch.clamp(balance_score, 0.0, 1.0)

    def _compute_cross_shard_rate(self, shard_assignments: torch.Tensor,
                                  edge_index: torch.Tensor,
                                  cross_features: torch.Tensor,
                                  original_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """基于分类+拓扑+异构特征计算跨片交易率"""
        hard_assignment = torch.argmax(shard_assignments, dim=1)

        u, v = edge_index[0], edge_index[1]

        # 确保索引有效
        valid_mask = (u < len(hard_assignment)) & (v < len(hard_assignment))
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=shard_assignments.device)

        u_valid, v_valid = u[valid_mask], v[valid_mask]
        cross_shard_mask = (hard_assignment[u_valid] != hard_assignment[v_valid])

        total_edges = valid_mask.sum().item()
        cross_edges = cross_shard_mask.sum().item()
        base_cross_rate = cross_edges / max(total_edges, 1)

        if cross_edges > 0:
            cross_u = u_valid[cross_shard_mask]
            cross_v = v_valid[cross_shard_mask]

            # 确保特征索引有效
            max_idx = min(cross_features.size(0) - 1, len(original_features['categorical']) - 1)
            cross_u_clamped = torch.clamp(cross_u, 0, max_idx)
            cross_v_clamped = torch.clamp(cross_v, 0, max_idx)

            # 分类特征差异（语义距离）
            cat_diff = torch.norm(
                original_features['categorical'][cross_u_clamped] -
                original_features['categorical'][cross_v_clamped], dim=1
            ).mean()

            # 异构类型差异
            hetero_diff = torch.norm(
                original_features['heterogeneous_type'][cross_u_clamped] -
                original_features['heterogeneous_type'][cross_v_clamped], dim=1
            ).mean()

            # 拓扑距离影响
            topo_penalty = torch.norm(
                original_features['network_topology'][cross_u_clamped] -
                original_features['network_topology'][cross_v_clamped], dim=1
            ).mean()

            # 综合跨片开销
            semantic_penalty = (cat_diff * 0.4 + hetero_diff * 0.3 + topo_penalty * 0.3).item()
            adjusted_rate = base_cross_rate * (1.0 + semantic_penalty * 0.2)
        else:
            adjusted_rate = base_cross_rate

        return torch.clamp(torch.tensor(adjusted_rate, device=shard_assignments.device), 0.0, 1.0)

    def _compute_security_threshold(self, shard_assignments: torch.Tensor,
                                    security_features: torch.Tensor,
                                    original_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """基于链上行为+异构类型计算安全阈值"""
        hard_assignment = torch.argmax(shard_assignments, dim=1)
        num_shards = shard_assignments.size(1)

        min_security = 1.0

        for s in range(num_shards):
            shard_mask = (hard_assignment == s)
            shard_size = torch.sum(shard_mask).item()

            if shard_size > 0:
                # 链上行为安全性（共识参与率、成功率等）
                onchain_security = torch.mean(original_features['onchain_behavior'][shard_mask]).item()

                # 异构类型多样性（类型越多样，安全性越高）
                hetero_features = original_features['heterogeneous_type'][shard_mask]
                hetero_diversity = torch.std(hetero_features, dim=0).mean().item()

                # 分片大小安全性（过小或过大的分片都不安全）
                size_factor = min(shard_size / 10.0, 1.0) * (1.0 - max(shard_size - 50, 0) / 100.0)

                # 综合安全分数
                shard_security = onchain_security * 0.6 + hetero_diversity * 0.2 + size_factor * 0.2
                min_security = min(min_security, shard_security)

        return torch.tensor(max(min_security, 0.0), device=shard_assignments.device)

    def _compute_consensus_latency(self, consensus_features: torch.Tensor,
                                   original_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """基于链上行为+动态属性计算共识时延"""
        # 链上行为中的共识效率
        onchain_efficiency = torch.mean(original_features['onchain_behavior']).item()

        # 动态属性中的当前负载状态
        current_load = torch.mean(original_features['dynamic_attributes']).item()

        # 共识时延计算（效率高、负载低 → 时延低）
        base_latency = 1.0 - onchain_efficiency  # 效率越高，基础时延越低
        load_penalty = current_load * 0.3        # 负载越高，时延增加

        total_latency = base_latency + load_penalty

        return torch.tensor(max(0.0, min(total_latency, 1.0)), device=consensus_features.device)

    def _compute_fusion_quality(self, features: Dict[str, torch.Tensor],
                                shard_assignments: torch.Tensor) -> torch.Tensor:
        """计算6类特征的融合质量"""
        # 将所有6类特征拼接
        all_features = torch.cat([
            features['hardware'],
            features['onchain_behavior'],
            features['network_topology'],
            features['dynamic_attributes'],
            features['heterogeneous_type'],
            features['categorical']
        ], dim=1)

        # 使用融合质量计算器
        fusion_score = self.metric_calculators['fusion_quality'](all_features)

        return torch.sigmoid(fusion_score.mean())  # 归一化到[0,1]

    def _evaluate_individual_feature_quality(self, features: Dict[str, torch.Tensor],
                                             shard_assignments: torch.Tensor) -> Dict[str, torch.Tensor]:
        """评估每类特征的独立质量"""
        quality_scores = {}

        for feature_name, feature_tensor in features.items():
            if feature_name in self.feature_evaluators:
                evaluator = self.feature_evaluators[feature_name]
                quality_score = evaluator(feature_tensor).mean()
                quality_scores[f'{feature_name}_quality'] = quality_score

        return quality_scores

    def _compute_feature_synergy(self, features: Dict[str, torch.Tensor],
                                 shard_assignments: torch.Tensor) -> torch.Tensor:
        """计算6类特征间的协同性"""
        # 计算特征间的相关性
        feature_list = list(features.values())
        synergy_scores = []

        for i in range(len(feature_list)):
            for j in range(i+1, len(feature_list)):
                # 计算特征对的相似性
                feat_i_mean = torch.mean(feature_list[i], dim=1)  # [N]
                feat_j_mean = torch.mean(feature_list[j], dim=1)  # [N]

                # 皮尔逊相关系数
                correlation = torch.corrcoef(torch.stack([feat_i_mean, feat_j_mean]))[0, 1]
                synergy_scores.append(abs(correlation))

        if synergy_scores:
            return torch.stack(synergy_scores).mean()
        else:
            return torch.tensor(0.5, device=shard_assignments.device)

    def _apply_six_feature_entropy_weights(self, metrics: Dict[str, torch.Tensor],
                                           features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """基于6类特征历史的熵权调整"""
        if len(self.performance_history) < 3:
            return metrics

        try:
            # 基于特征质量历史调整权重
            for feature_name in features.keys():
                quality_key = f'{feature_name}_quality'
                if quality_key in metrics and feature_name in self.feature_quality_history:
                    history = self.feature_quality_history[feature_name]
                    if len(history) >= 3:
                        # 计算质量稳定性
                        stability = 1.0 - np.std(history[-5:]) if len(history) >= 5 else 0.5
                        # 调整对应指标权重
                        metrics[quality_key] = metrics[quality_key] * stability

        except Exception as e:
            print(f"[WARNING] 6类特征熵权调整失败: {e}")

        return metrics

    def _update_feature_quality_history(self, quality_scores: Dict[str, torch.Tensor]):
        """更新特征质量历史"""
        for key, score in quality_scores.items():
            if key.endswith('_quality'):
                feature_name = key.replace('_quality', '')
                if feature_name in self.feature_quality_history:
                    self.feature_quality_history[feature_name].append(score.item())
                    # 保持历史长度
                    if len(self.feature_quality_history[feature_name]) > self.history_window:
                        self.feature_quality_history[feature_name].pop(0)

    def _get_default_metrics(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """返回默认的性能指标"""
        return {
            'balance_score': torch.tensor(0.5, device=device),
            'cross_tx_rate': torch.tensor(0.2, device=device),
            'security_score': torch.tensor(0.8, device=device),
            'consensus_latency': torch.tensor(0.1, device=device),
            'fusion_quality': torch.tensor(0.6, device=device),
            'hardware_quality': torch.tensor(0.7, device=device),
            'onchain_behavior_quality': torch.tensor(0.6, device=device),
            'network_topology_quality': torch.tensor(0.8, device=device),
            'dynamic_attributes_quality': torch.tensor(0.5, device=device),
            'heterogeneous_type_quality': torch.tensor(0.7, device=device),
            'categorical_quality': torch.tensor(0.6, device=device),
            'feature_synergy': torch.tensor(0.5, device=device)
        }

    def _update_history(self, metrics: Dict[str, torch.Tensor]):
        """更新历史记录"""
        self.performance_history.append(metrics.copy())
        if len(self.performance_history) > self.history_window:
            self.performance_history.pop(0)


class FeatureImportanceAnalyzer:
    """6类特征重要性分析器"""

    def __init__(self, feature_dims: Dict[str, int]):
        self.feature_dims = feature_dims
        self.importance_history = defaultdict(list)

        # 6类特征名称
        self.six_features = ['hardware', 'onchain_behavior', 'network_topology',
                             'dynamic_attributes', 'heterogeneous_type', 'categorical']

    def analyze_importance(self, features: Dict[str, torch.Tensor],
                           performance_scores: Dict[str, torch.Tensor],
                           model: Optional[nn.Module] = None) -> Dict[str, Dict[str, float]]:
        """
        分析6类特征的重要性

        Returns:
            layer_importance: 6类特征重要性矩阵
        """
        print(f" 分析6类特征重要性: {list(features.keys())}")

        importance_matrix = {}

        try:
            # 1. 梯度重要性分析
            gradient_importance = self._gradient_importance_analysis(
                features, performance_scores, model
            )

            # 2. 互信息重要性分析
            mutual_info_importance = self._mutual_info_analysis(
                features, performance_scores
            )

            # 3. 方差贡献分析
            variance_importance = self._variance_contribution_analysis(
                features, performance_scores
            )

            # 4. 特征消融重要性分析
            ablation_importance = self._ablation_importance_analysis(
                features, performance_scores
            )

            # 5. 综合重要性评分
            for feature_name in self.six_features:
                if feature_name in features:
                    importance_matrix[feature_name] = {
                        'gradient': gradient_importance.get(feature_name, 0.0),
                        'mutual_info': mutual_info_importance.get(feature_name, 0.0),
                        'variance': variance_importance.get(feature_name, 0.0),
                        'ablation': ablation_importance.get(feature_name, 0.0),
                        'combined': self._combine_six_feature_importance(
                            gradient_importance.get(feature_name, 0.0),
                            mutual_info_importance.get(feature_name, 0.0),
                            variance_importance.get(feature_name, 0.0),
                            ablation_importance.get(feature_name, 0.0)
                        )
                    }
                else:
                    # 为缺失的特征提供默认重要性
                    importance_matrix[feature_name] = {
                        'gradient': 0.3, 'mutual_info': 0.3, 'variance': 0.3, 'ablation': 0.3, 'combined': 0.3
                    }

        except Exception as e:
            print(f"[WARNING] 6类特征重要性分析出错: {e}")
            # 返回默认重要性
            for feature_name in self.six_features:
                importance_matrix[feature_name] = {
                    'gradient': 0.5, 'mutual_info': 0.5, 'variance': 0.5, 'ablation': 0.5, 'combined': 0.5
                }

        return importance_matrix

    def _gradient_importance_analysis(self, features: Dict[str, torch.Tensor],
                                      performance_scores: Dict[str, torch.Tensor],
                                      model: Optional[nn.Module] = None) -> Dict[str, float]:
        """基于梯度的6类特征重要性分析"""
        importance_scores = {}

        try:
            for feature_name in self.six_features:
                if feature_name in features:
                    feature_tensor = features[feature_name].detach().requires_grad_(True)

                    # 计算性能分数总和
                    main_scores = ['balance_score', 'cross_tx_rate', 'security_score', 'consensus_latency']
                    total_score = sum(performance_scores[k] for k in main_scores if k in performance_scores)

                    if isinstance(total_score, torch.Tensor) and total_score.requires_grad:
                        total_score.backward(retain_graph=True)

                        if feature_tensor.grad is not None:
                            grad_norm = torch.norm(feature_tensor.grad, p=2).item()
                            importance_scores[feature_name] = grad_norm
                        else:
                            importance_scores[feature_name] = 0.0
                    else:
                        importance_scores[feature_name] = 0.0
                else:
                    importance_scores[feature_name] = 0.0

        except Exception as e:
            print(f"[WARNING] 梯度重要性分析失败: {e}")
            for feature_name in self.six_features:
                importance_scores[feature_name] = 0.5

        return importance_scores

    def _mutual_info_analysis(self, features: Dict[str, torch.Tensor],
                              performance_scores: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """6类特征的互信息重要性分析"""
        importance_scores = {}

        try:
            # 准备目标变量
            main_scores = ['balance_score', 'cross_tx_rate', 'security_score']
            score_values = []
            for score_name in main_scores:
                if score_name in performance_scores:
                    score_values.append(performance_scores[score_name].detach().cpu())

            if score_values:
                target = torch.stack(score_values).mean(dim=0).numpy()

                for feature_name in self.six_features:
                    if feature_name in features:
                        feature_np = features[feature_name].detach().cpu().numpy()

                        # 计算每个维度与目标的互信息
                        mi_scores = []
                        max_dims = min(feature_np.shape[1], 8)  # 限制维度数
                        for dim in range(max_dims):
                            try:
                                mi = mutual_info_regression(
                                    feature_np[:, dim].reshape(-1, 1),
                                    target
                                )[0]
                                mi_scores.append(mi)
                            except:
                                mi_scores.append(0.0)

                        importance_scores[feature_name] = np.mean(mi_scores) if mi_scores else 0.0
                    else:
                        importance_scores[feature_name] = 0.0
            else:
                for feature_name in self.six_features:
                    importance_scores[feature_name] = 0.0

        except Exception as e:
            print(f"[WARNING] 互信息分析失败: {e}")
            for feature_name in self.six_features:
                importance_scores[feature_name] = 0.5

        return importance_scores

    def _variance_contribution_analysis(self, features: Dict[str, torch.Tensor],
                                        performance_scores: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """6类特征的方差贡献分析"""
        importance_scores = {}

        try:
            for feature_name in self.six_features:
                if feature_name in features:
                    feature_tensor = features[feature_name]

                    # 计算特征方差
                    feature_var = torch.var(feature_tensor, dim=0).mean().item()

                    # 计算特征与性能指标的协方差
                    feature_mean = torch.mean(feature_tensor, dim=1)  # [N]

                    covariances = []
                    for score_name, score_tensor in performance_scores.items():
                        if score_name in ['balance_score', 'cross_tx_rate', 'security_score']:
                            try:
                                if isinstance(score_tensor, torch.Tensor):
                                    if score_tensor.dim() == 0:  # 标量
                                        score_expanded = score_tensor.expand(len(feature_mean))
                                    else:
                                        score_expanded = score_tensor

                                    cov = torch.cov(torch.stack([feature_mean, score_expanded]))[0, 1].item()
                                    covariances.append(abs(cov))
                            except:
                                continue

                    avg_covariance = np.mean(covariances) if covariances else 0.0
                    importance_scores[feature_name] = feature_var * avg_covariance
                else:
                    importance_scores[feature_name] = 0.0

        except Exception as e:
            print(f"[WARNING] 方差贡献分析失败: {e}")
            for feature_name in self.six_features:
                importance_scores[feature_name] = 0.5

        return importance_scores

    def _ablation_importance_analysis(self, features: Dict[str, torch.Tensor],
                                      performance_scores: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """6类特征的消融重要性分析"""
        importance_scores = {}

        try:
            # 基准性能（使用所有特征）
            baseline_score = sum(score.item() if isinstance(score, torch.Tensor) else score
                                 for key, score in performance_scores.items()
                                 if key in ['balance_score', 'cross_tx_rate', 'security_score'])

            for feature_name in self.six_features:
                if feature_name in features:
                    # 模拟移除该特征后的性能下降
                    # 这里简化为基于特征统计量的估计
                    feature_contribution = torch.mean(features[feature_name]).item()

                    # 不同特征类型的权重不同
                    feature_weights = {
                        'hardware': 0.3,
                        'onchain_behavior': 0.25,
                        'network_topology': 0.2,
                        'dynamic_attributes': 0.1,
                        'heterogeneous_type': 0.1,
                        'categorical': 0.05
                    }

                    weight = feature_weights.get(feature_name, 0.1)
                    estimated_drop = baseline_score * weight * feature_contribution
                    importance_scores[feature_name] = estimated_drop
                else:
                    importance_scores[feature_name] = 0.0

        except Exception as e:
            print(f"[WARNING] 消融分析失败: {e}")
            for feature_name in self.six_features:
                importance_scores[feature_name] = 0.5

        return importance_scores

    def _combine_six_feature_importance(self, gradient: float, mutual_info: float,
                                        variance: float, ablation: float) -> float:
        """组合6类特征的重要性分数"""
        # 不同分析方法的权重
        weights = [0.3, 0.3, 0.2, 0.2]  # gradient, mutual_info, variance, ablation
        scores = [gradient, mutual_info, variance, ablation]

        # 归一化分数到[0,1]
        normalized_scores = []
        for score in scores:
            normalized_scores.append(max(0.0, min(1.0, score)))

        combined = sum(w * s for w, s in zip(weights, normalized_scores))
        return max(0.0, min(1.0, combined))


class FeedbackController:
    """第四步反馈控制器 - 支持6类原始特征"""

    def __init__(self, feature_dims: Dict[str, int]):
        self.performance_evaluator = PerformanceEvaluator(feature_dims)
        self.importance_analyzer = FeatureImportanceAnalyzer(feature_dims)
        self.feature_evolution = None

        # 6类特征名称
        self.six_features = ['hardware', 'onchain_behavior', 'network_topology',
                             'dynamic_attributes', 'heterogeneous_type', 'categorical']

        print(f"[SUCCESS] FeedbackController 初始化完成，支持6类原始特征:")
        for feature in self.six_features:
            dim = feature_dims.get(feature, 0)
            print(f"  - {feature}: {dim}维")

    def process_feedback(self, features: Dict[str, torch.Tensor],
                         shard_assignments: torch.Tensor,
                         edge_index: torch.Tensor,
                         evolve_gcn_model: Optional[nn.Module] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        处理6类原始特征的反馈并优化特征空间

        Args:
            features: 6类原始特征字典
            shard_assignments: 分片分配结果
            edge_index: 边索引
            evolve_gcn_model: EvolveGCN模型

        Returns:
            feedback_signal: 反馈信号 [4] (负载均衡, 跨片率, 安全性, 特征质量)
            evolved_features: 优化后的特征空间
        """
        print(f" 处理6类原始特征反馈 - 输入特征: {list(features.keys())}")

        try:
            # 1. 基于6类特征的性能评估
            performance_metrics = self.performance_evaluator(
                features, shard_assignments, edge_index
            )

            # 2. 6类特征重要性分析
            importance_matrix = self.importance_analyzer.analyze_importance(
                features, performance_metrics, evolve_gcn_model
            )

            # 3. 6类特征空间演化
            if self.feature_evolution is None:
                self.feature_evolution = SixFeatureEvolution(features)

            evolved_features = self.feature_evolution.evolve_six_feature_space(
                importance_matrix, performance_metrics
            )

            # 4. 生成增强的反馈信号
            feedback_signal = torch.tensor([
                performance_metrics['balance_score'].item(),
                performance_metrics['cross_tx_rate'].item(),
                performance_metrics['security_score'].item(),
                performance_metrics['fusion_quality'].item()  # 添加特征融合质量
            ], device=shard_assignments.device)

            print(f"[SUCCESS] 6类特征反馈处理完成")
            print(f"   反馈信号: {[f'{x:.3f}' for x in feedback_signal.tolist()]}")
            print(f"   重要性最高的特征: {max(importance_matrix.keys(), key=lambda k: importance_matrix[k]['combined'])}")

        except Exception as e:
            print(f"[WARNING] 6类特征反馈处理出错: {e}")
            # 返回默认值
            device = shard_assignments.device
            feedback_signal = torch.tensor([0.5, 0.2, 0.8, 0.6], device=device)
            evolved_features = features

        return feedback_signal, evolved_features


class SixFeatureEvolution:
    """6类特征空间演化器"""

    def __init__(self, initial_features: Dict[str, torch.Tensor]):
        self.current_features = initial_features.copy()
        self.feature_history = [initial_features.copy()]
        self.six_features = ['hardware', 'onchain_behavior', 'network_topology',
                             'dynamic_attributes', 'heterogeneous_type', 'categorical']

    def evolve_six_feature_space(self, importance_matrix: Dict[str, Dict[str, float]],
                                 performance_metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """演化6类特征空间"""
        evolved_features = self.current_features.copy()

        # 基于重要性矩阵调整6类特征
        for feature_name in self.six_features:
            if feature_name in evolved_features and feature_name in importance_matrix:
                importance = importance_matrix[feature_name]['combined']

                # 根据重要性调整特征
                if importance > 0.8:
                    # 高重要性：增强特征
                    evolved_features[feature_name] = evolved_features[feature_name] * 1.15
                elif importance > 0.6:
                    # 中等重要性：轻微增强
                    evolved_features[feature_name] = evolved_features[feature_name] * 1.05
                elif importance < 0.3:
                    # 低重要性：降权
                    evolved_features[feature_name] = evolved_features[feature_name] * 0.9

                # 特征特定的调整策略
                evolved_features[feature_name] = self._apply_feature_specific_evolution(
                    feature_name, evolved_features[feature_name], performance_metrics
                )

        self.current_features = evolved_features
        self.feature_history.append(evolved_features.copy())

        return evolved_features

    def _apply_feature_specific_evolution(self, feature_name: str, feature_tensor: torch.Tensor,
                                          performance_metrics: Dict[str, torch.Tensor]) -> torch.Tensor:
        """应用特征特定的演化策略"""
        if feature_name == 'hardware':
            # 硬件特征：如果负载不均衡，增强硬件差异化
            if 'balance_score' in performance_metrics and performance_metrics['balance_score'] < 0.5:
                feature_tensor = feature_tensor * 1.1

        elif feature_name == 'onchain_behavior':
            # 链上行为：如果安全性低，增强共识相关特征
            if 'security_score' in performance_metrics and performance_metrics['security_score'] < 0.6:
                feature_tensor = feature_tensor * 1.08

        elif feature_name == 'network_topology':
            # 网络拓扑：如果跨片率高，增强拓扑特征
            if 'cross_tx_rate' in performance_metrics and performance_metrics['cross_tx_rate'] > 0.3:
                feature_tensor = feature_tensor * 1.05

        return feature_tensor