"""
动态特征空间演化器
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Callable
from collections import defaultdict
import copy

class DynamicFeatureEvolution:
    """动态特征空间演化器"""

    def __init__(self, initial_features: Dict[str, torch.Tensor],
                 evolution_config: Dict[str, Any] = None):
        self.current_features = copy.deepcopy(initial_features)
        self.original_features = copy.deepcopy(initial_features)
        self.feature_history = [copy.deepcopy(initial_features)]
        self.checkpoints = []
        self.evolution_config = evolution_config or self._default_config()
        self.evolution_rules = self._initialize_expert_rules()

    def _default_config(self) -> Dict[str, Any]:
        """默认演化配置"""
        return {
            'momentum': 0.9,
            'learning_rate': 0.1,
            'pruning_threshold': {
                'hardware': 0.7,
                'consensus': 0.5,
                'topology': 0.5,
                'default': 0.5
            },
            'max_dimension_reduction': 0.5,
            'checkpoint_interval': 5,
            'anomaly_threshold': 0.2
        }

    def evolve_feature_space(self, importance_matrix: Dict[str, Dict[str, float]],
                             performance_metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        动态特征空间演化

        Args:
            importance_matrix: 特征重要性矩阵
            performance_metrics: 性能指标

        Returns:
            evolved_features: 演化后的特征空间
        """
        print(f"\n=== 开始特征空间演化 ===")

        # 1. 分层特征剪枝
        pruned_features = self._hierarchical_feature_pruning(
            self.current_features, importance_matrix
        )
        print(f"特征剪枝完成")

        # 2. 应用专家规则约束
        rule_constrained_features = self._apply_expert_rules(
            pruned_features, performance_metrics
        )
        print(f"专家规则应用完成")

        # 3. 跨层协同优化
        optimized_features = self._cross_layer_optimization(
            rule_constrained_features, importance_matrix
        )
        print(f"跨层优化完成")

        # 4. 动量平滑更新
        smoothed_features = self._momentum_update(
            optimized_features
        )
        print(f"动量更新完成")

        # 5. 异常检测和回滚
        if self._detect_performance_anomaly(performance_metrics):
            print("检测到性能异常，执行回滚")
            final_features = self._rollback_to_checkpoint()
        else:
            final_features = smoothed_features
            self._save_checkpoint(final_features)

        # 6. 更新状态
        self.current_features = final_features
        self.feature_history.append(copy.deepcopy(final_features))

        # 限制历史记录长度
        if len(self.feature_history) > 50:
            self.feature_history.pop(0)

        print(f"特征空间演化完成")
        return final_features

    def _hierarchical_feature_pruning(self, features: Dict[str, torch.Tensor],
                                      importance_matrix: Dict[str, Dict[str, float]]) -> Dict[str, torch.Tensor]:
        """分层特征剪枝"""
        pruned_features = {}
        config = self.evolution_config

        for layer_name, layer_features in features.items():
            importance = importance_matrix.get(layer_name, {}).get('combined', 0.0)

            # 获取该层的剪枝阈值
            threshold = config['pruning_threshold'].get(layer_name,
                                                        config['pruning_threshold']['default'])

            # 计算全局最大重要性
            max_importance = max([
                layer_info.get('combined', 0.0)
                for layer_info in importance_matrix.values()
            ], default=1.0)

            threshold_value = threshold * max_importance

            if importance > threshold_value:
                # 保留完整特征
                pruned_features[layer_name] = layer_features.clone()
            else:
                # 进行维度缩减而不是完全移除
                original_dim = layer_features.size(1) if layer_features.dim() > 1 else layer_features.size(0)
                max_reduction = config['max_dimension_reduction']
                reduction_ratio = max(1 - (importance / threshold_value), max_reduction)

                new_dim = max(int(original_dim * (1 - reduction_ratio)), 1)

                if layer_features.dim() == 2:
                    # 保留最重要的维度（基于方差）
                    feature_vars = torch.var(layer_features, dim=0)
                    _, top_indices = torch.topk(feature_vars, new_dim)
                    pruned_features[layer_name] = layer_features[:, top_indices]
                else:
                    pruned_features[layer_name] = layer_features[:new_dim]

                print(f"  {layer_name}: {original_dim} -> {new_dim} 维度")

        return pruned_features

    def _apply_expert_rules(self, features: Dict[str, torch.Tensor],
                            performance_metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """应用专家规则约束"""
        constrained_features = copy.deepcopy(features)

        for rule in self.evolution_rules:
            try:
                constrained_features = rule(constrained_features, performance_metrics)
            except Exception as e:
                print(f"Warning: Expert rule failed: {e}")
                continue

        return constrained_features

    def _cross_layer_optimization(self, features: Dict[str, torch.Tensor],
                                  importance_matrix: Dict[str, Dict[str, float]]) -> Dict[str, torch.Tensor]:
        """跨层协同优化"""
        optimized_features = copy.deepcopy(features)

        # 计算层间相似度并进行协同调整
        layer_names = list(features.keys())

        for i, layer1 in enumerate(layer_names):
            for j, layer2 in enumerate(layer_names[i+1:], i+1):
                if layer1 in features and layer2 in features:
                    # 计算层间相似度
                    sim = self._compute_layer_similarity(
                        features[layer1], features[layer2]
                    )

                    # 根据相似度和重要性调整特征
                    importance1 = importance_matrix.get(layer1, {}).get('combined', 0.5)
                    importance2 = importance_matrix.get(layer2, {}).get('combined', 0.5)

                    if sim > 0.8 and abs(importance1 - importance2) > 0.3:
                        # 高相似度但重要性差异大，进行特征融合
                        self._fuse_similar_layers(
                            optimized_features, layer1, layer2,
                            importance1, importance2
                        )

        return optimized_features

    def _momentum_update(self, new_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """动量平滑更新"""
        momentum = self.evolution_config['momentum']
        smoothed_features = {}

        for layer_name, new_feat in new_features.items():
            if layer_name in self.current_features:
                old_feat = self.current_features[layer_name]

                # 维度匹配检查
                if old_feat.shape == new_feat.shape:
                    smoothed_features[layer_name] = (
                            momentum * old_feat + (1 - momentum) * new_feat
                    )
                else:
                    # 维度不匹配时，使用新特征
                    smoothed_features[layer_name] = new_feat
                    print(f"维度变化 {layer_name}: {old_feat.shape} -> {new_feat.shape}")
            else:
                smoothed_features[layer_name] = new_feat

        return smoothed_features

    def _detect_performance_anomaly(self, performance_metrics: Dict[str, torch.Tensor]) -> bool:
        """检测性能异常"""
        if len(self.feature_history) < 2:
            return False

        # 检查性能是否突然下降
        threshold = self.evolution_config['anomaly_threshold']

        for metric_name, current_value in performance_metrics.items():
            if isinstance(current_value, torch.Tensor):
                current_val = current_value.item()
            else:
                current_val = current_value

            # 与历史平均值比较
            if len(self.feature_history) >= 3:
                # 简单的异常检测：当前值比历史平均值低threshold以上
                if current_val < (1 - threshold):
                    return True

        return False

    def _save_checkpoint(self, features: Dict[str, torch.Tensor]):
        """保存检查点"""
        checkpoint = {
            'features': copy.deepcopy(features),
            'timestamp': len(self.feature_history)
        }
        self.checkpoints.append(checkpoint)

        # 限制检查点数量
        max_checkpoints = 10
        if len(self.checkpoints) > max_checkpoints:
            self.checkpoints.pop(0)

    def _rollback_to_checkpoint(self) -> Dict[str, torch.Tensor]:
        """回滚到最近的检查点"""
        if self.checkpoints:
            latest_checkpoint = self.checkpoints[-1]
            return copy.deepcopy(latest_checkpoint['features'])
        else:
            return copy.deepcopy(self.original_features)

    def _initialize_expert_rules(self) -> List[Callable]:
        """初始化专家规则"""
        rules = []

        # 规则1: 硬件特征保护
        def hardware_protection_rule(features, metrics):
            if 'hardware' in features:
                # 确保硬件特征不被过度剪枝
                min_dim = max(features['hardware'].size(1) // 2, 5)
                if features['hardware'].size(1) < min_dim:
                    # 从原始特征恢复
                    if 'hardware' in self.original_features:
                        features['hardware'] = self.original_features['hardware'][:, :min_dim]
            return features

        # 规则2: 安全阈值保护
        def security_protection_rule(features, metrics):
            if 'security_score' in metrics and metrics['security_score'] < 0.5:
                # 安全分数低时，增强共识特征
                if 'consensus' in features:
                    features['consensus'] = features['consensus'] * 1.2
            return features

        # 规则3: 负载均衡优化
        def load_balance_rule(features, metrics):
            if 'balance_score' in metrics and metrics['balance_score'] < 0.6:
                # 负载不均衡时，增强硬件和拓扑特征
                for layer in ['hardware', 'topology']:
                    if layer in features:
                        features[layer] = features[layer] * 1.1
            return features

        rules.extend([hardware_protection_rule, security_protection_rule, load_balance_rule])
        return rules

    def _compute_layer_similarity(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """计算层间相似度"""
        try:
            # 如果维度不同，使用平均值比较
            if feat1.shape != feat2.shape:
                mean1 = torch.mean(feat1)
                mean2 = torch.mean(feat2)
                return float(torch.abs(mean1 - mean2))

            # 计算余弦相似度
            feat1_flat = feat1.flatten()
            feat2_flat = feat2.flatten()

            similarity = torch.nn.functional.cosine_similarity(
                feat1_flat.unsqueeze(0), feat2_flat.unsqueeze(0)
            )
            return float(similarity)

        except Exception:
            return 0.0

    def _fuse_similar_layers(self, features: Dict[str, torch.Tensor],
                             layer1: str, layer2: str,
                             importance1: float, importance2: float):
        """融合相似的层"""
        try:
            feat1 = features[layer1]
            feat2 = features[layer2]

            # 根据重要性计算权重
            total_importance = importance1 + importance2
            if total_importance > 0:
                w1 = importance1 / total_importance
                w2 = importance2 / total_importance
            else:
                w1 = w2 = 0.5

            # 维度适配
            if feat1.shape[1] == feat2.shape[1]:
                # 维度相同，直接加权平均
                fused_feat = w1 * feat1 + w2 * feat2
                features[layer1] = fused_feat
                # 保留重要性更高的层名
                if importance1 >= importance2:
                    features[layer1] = fused_feat
                else:
                    features[layer2] = fused_feat

            print(f"融合层 {layer1} 和 {layer2}, 权重: {w1:.3f}, {w2:.3f}")

        except Exception as e:
            print(f"层融合失败: {e}")

    def process_step1_features(self, extraction_info: Dict[str, torch.Tensor],
                               performance_metrics: Dict[str, torch.Tensor],
                               epoch: int = 0) -> Dict[str, Any]:
        """
        处理第一步的特征提取信息并生成反馈指导 (新增方法)

        Args:
            extraction_info: 第一步的特征提取信息
            performance_metrics: 性能指标
            epoch: 当前轮次

        Returns:
            step1_guidance: 给第一步的反馈指导
        """
        print(f"\n=== 处理第一步特征并生成反馈指导 (Epoch {epoch}) ===")

        # 1. 分析99维原始特征的6层表现
        layered_features = extraction_info.get('layered_features', {})
        layered_importance = self._analyze_layered_importance(layered_features, performance_metrics)

        # 2. 分析时序特征表现
        sequence_features = extraction_info.get('sequence_features')
        sequence_importance = self._analyze_sequence_importance(sequence_features, performance_metrics)

        # 3. 分析图结构特征表现
        graph_features = extraction_info.get('graph_structure_features')
        graph_importance = self._analyze_graph_importance(graph_features, performance_metrics)

        # 4. 生成分层反馈指导
        step1_guidance = {
            'epoch': epoch,
            'guidance_type': 'layered_99dim_feedback',

            # 99维原始特征的6层权重调整
            'layer_weight_adjustments': self._generate_layer_weight_adjustments(
                layered_importance, performance_metrics
            ),

            # 99维原始特征的6层维度选择
            'layer_dimension_selection': self._generate_layer_dimension_selection(
                layered_features, layered_importance
            ),

            # 99维原始特征的6层增强因子
            'layer_enhancement_factors': self._generate_layer_enhancement_factors(
                layered_importance, performance_metrics
            ),

            # 32维时序特征调整
            'temporal_focus_adjustment': self._generate_temporal_adjustment(
                sequence_importance, performance_metrics
            ),

            # 10维图结构特征调整
            'graph_structure_adjustment': self._generate_graph_adjustment(
                graph_importance, performance_metrics
            ),

            # 141->128投影调整
            'projection_adjustment': self._generate_projection_adjustment(
                performance_metrics, epoch
            )
        }

        print(f"[SUCCESS] 第一步反馈指导生成完成")
        return step1_guidance

    def _analyze_layered_importance(self, layered_features: Dict[str, torch.Tensor],
                                    performance_metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """分析99维原始特征6层的重要性"""
        layered_importance = {}

        for layer_name, layer_features in layered_features.items():
            # 基于方差分析重要性
            feature_variance = torch.var(layer_features, dim=0).mean().item()

            # 基于性能指标调整重要性
            performance_factor = 1.0

            if layer_name == 'hardware':
                # 硬件特征与负载均衡相关
                if 'balance_score' in performance_metrics:
                    balance_score = performance_metrics['balance_score'].item()
                    performance_factor = 0.5 + balance_score  # [0.5, 1.5]

            elif layer_name == 'onchain_behavior':
                # 链上行为与安全性相关
                if 'security_score' in performance_metrics:
                    security_score = performance_metrics['security_score'].item()
                    performance_factor = 0.5 + security_score

            elif layer_name == 'network_topology':
                # 网络拓扑与跨片交易相关
                if 'cross_tx_rate' in performance_metrics:
                    cross_tx_rate = performance_metrics['cross_tx_rate'].item()
                    performance_factor = 1.5 - cross_tx_rate  # 跨片交易率低时重要性高

            # 计算综合重要性
            base_importance = min(feature_variance, 1.0)  # 归一化方差
            final_importance = base_importance * performance_factor

            layered_importance[layer_name] = np.clip(final_importance, 0.1, 2.0)

        return layered_importance

    def _analyze_sequence_importance(self, sequence_features: torch.Tensor,
                                     performance_metrics: Dict[str, torch.Tensor]) -> float:
        """分析32维时序特征的重要性"""
        if sequence_features is None:
            return 0.5

        # 基于时序特征的方差和性能指标
        seq_variance = torch.var(sequence_features, dim=0).mean().item()

        # 时序特征通常与系统稳定性相关
        stability_factor = 1.0
        if 'balance_score' in performance_metrics:
            balance_score = performance_metrics['balance_score'].item()
            stability_factor = balance_score

        importance = seq_variance * stability_factor
        return np.clip(importance, 0.1, 2.0)

    def _analyze_graph_importance(self, graph_features: torch.Tensor,
                                  performance_metrics: Dict[str, torch.Tensor]) -> float:
        """分析10维图结构特征的重要性"""
        if graph_features is None:
            return 0.5

        # 基于图结构特征的方差和网络性能
        graph_variance = torch.var(graph_features, dim=0).mean().item()

        # 图结构与网络拓扑和跨片交易相关
        network_factor = 1.0
        if 'cross_tx_rate' in performance_metrics:
            cross_tx_rate = performance_metrics['cross_tx_rate'].item()
            network_factor = 1.2 - cross_tx_rate  # 跨片交易率低时图结构重要性高

        importance = graph_variance * network_factor
        return np.clip(importance, 0.1, 2.0)

    def _generate_layer_weight_adjustments(self, layered_importance: Dict[str, float],
                                           performance_metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """生成6层权重调整"""
        adjustments = {}

        # 基于重要性和性能生成权重调整
        for layer_name, importance in layered_importance.items():
            base_weight = 1.0

            # 根据性能指标调整权重
            if layer_name == 'hardware' and 'balance_score' in performance_metrics:
                balance_score = performance_metrics['balance_score'].item()
                if balance_score < 0.6:  # 负载不均衡时增强硬件权重
                    base_weight *= 1.2

            elif layer_name == 'onchain_behavior' and 'security_score' in performance_metrics:
                security_score = performance_metrics['security_score'].item()
                if security_score < 0.7:  # 安全性低时增强行为权重
                    base_weight *= 1.3

            elif layer_name == 'network_topology' and 'cross_tx_rate' in performance_metrics:
                cross_tx_rate = performance_metrics['cross_tx_rate'].item()
                if cross_tx_rate > 0.3:  # 跨片交易率高时增强拓扑权重
                    base_weight *= 1.1

            # 结合重要性计算最终权重
            final_weight = base_weight * importance
            adjustments[layer_name] = np.clip(final_weight, 0.5, 2.0)

        return adjustments

    def _generate_layer_dimension_selection(self, layered_features: Dict[str, torch.Tensor],
                                            layered_importance: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """生成6层维度选择"""
        dimension_selection = {}

        for layer_name, layer_features in layered_features.items():
            importance = layered_importance.get(layer_name, 0.5)
            current_dim = layer_features.size(1)

            # 根据重要性决定维度保留比例
            if importance > 1.0:
                selection_ratio = 1.0  # 重要性高时保留所有维度
            elif importance > 0.7:
                selection_ratio = 0.9  # 重要性中等时保留90%
            else:
                selection_ratio = max(0.6, importance)  # 重要性低时至少保留60%

            dimension_selection[layer_name] = {
                'selection_ratio': selection_ratio,
                'current_dim': current_dim,
                'importance': importance
            }

        return dimension_selection

    def _generate_layer_enhancement_factors(self, layered_importance: Dict[str, float],
                                            performance_metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """生成6层增强因子"""
        enhancement_factors = {}

        for layer_name, importance in layered_importance.items():
            # 基础增强因子
            if importance > 1.0:
                factor = 1.0 + (importance - 1.0) * 0.2  # 重要性高时增强
            else:
                factor = 1.0 - (1.0 - importance) * 0.2  # 重要性低时减弱

            # 基于性能微调
            if layer_name == 'hardware':
                balance_score = performance_metrics.get('balance_score', torch.tensor(0.5)).item()
                factor *= (0.8 + 0.4 * balance_score)  # 负载均衡影响硬件增强

            elif layer_name == 'onchain_behavior':
                security_score = performance_metrics.get('security_score', torch.tensor(0.8)).item()
                factor *= (0.7 + 0.6 * security_score)  # 安全性影响行为增强

            enhancement_factors[layer_name] = np.clip(factor, 0.7, 1.5)

        return enhancement_factors

    def _generate_temporal_adjustment(self, sequence_importance: float,
                                      performance_metrics: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """生成32维时序特征调整"""
        adjustment = {
            'focus_adjustment_factor': sequence_importance,
        }

        # 生成时序权重 (32维)
        if 'balance_score' in performance_metrics:
            balance_score = performance_metrics['balance_score'].item()
            # 负载不均衡时更关注近期时序
            if balance_score < 0.6:
                temporal_weights = torch.cat([
                    torch.ones(16) * 0.8,  # 前16维权重降低
                    torch.ones(16) * 1.2   # 后16维权重提高 (更关注近期)
                ])
            else:
                temporal_weights = torch.ones(32)
        else:
            temporal_weights = torch.ones(32)

        adjustment['temporal_weights'] = temporal_weights
        return adjustment

    def _generate_graph_adjustment(self, graph_importance: float,
                                   performance_metrics: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """生成10维图结构特征调整"""
        adjustment = {
            'structure_enhancement': graph_importance,
        }

        # 连接性关注度
        if 'cross_tx_rate' in performance_metrics:
            cross_tx_rate = performance_metrics['cross_tx_rate'].item()
            # 跨片交易率高时更关注连接性
            connectivity_focus = 1.0 + cross_tx_rate * 0.5
        else:
            connectivity_focus = 1.0

        adjustment['connectivity_focus'] = connectivity_focus
        return adjustment

    def _generate_projection_adjustment(self, performance_metrics: Dict[str, torch.Tensor],
                                        epoch: int) -> Dict[str, Any]:
        """生成141->128投影调整"""
        adjustment = {}

        # 基于整体性能决定是否使用注意力机制
        avg_performance = 0.0
        metric_count = 0

        for metric_name, metric_value in performance_metrics.items():
            if metric_name in ['balance_score', 'security_score']:
                avg_performance += metric_value.item()
                metric_count += 1
            elif metric_name == 'cross_tx_rate':
                avg_performance += (1.0 - metric_value.item())  # 跨片交易率越低越好
                metric_count += 1

        if metric_count > 0:
            avg_performance /= metric_count
        else:
            avg_performance = 0.5

        # 性能较低时使用注意力机制
        adjustment['use_attention'] = avg_performance < 0.6

        # 权重因子
        adjustment['weight_factor'] = np.clip(avg_performance + 0.5, 0.7, 1.3)

        return adjustment

