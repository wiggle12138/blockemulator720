"""
自适应特征提取器 - 支持第四步反馈指导
"""
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
from .feature_extractor import UnifiedFeatureExtractor, ComprehensiveFeatureExtractor
from .sliding_window_extractor import EnhancedSequenceFeatureEncoder
from .nodeInitialize import Node
import copy
import numpy as np

class AdaptiveFeatureExtractor(nn.Module):
    """自适应特征提取器 - 接受第四步反馈并调整99维原始特征"""

    def __init__(self):
        super().__init__()

        # 原始特征提取组件
        self.comprehensive_extractor = ComprehensiveFeatureExtractor()
        self.sequence_encoder = EnhancedSequenceFeatureEncoder()
        self.graph_encoder = None  # 将从原始实现中获取

        # 6层特征的反馈适配器 (对应99维原始特征的分解)
        self.layer_adapters = nn.ModuleDict({


            'hardware': LayerFeedbackAdapter(17, 'hardware'),           # 硬件规格特征 (0:17)
            'onchain_behavior': LayerFeedbackAdapter(17, 'onchain_behavior'),  # 链上行为特征 (17:34)
            'network_topology': LayerFeedbackAdapter(20, 'network_topology'),  # 网络拓扑特征 (34:54)
            'dynamic_attributes': LayerFeedbackAdapter(13, 'dynamic_attributes'), # 动态属性特征 (54:67)
            'heterogeneous_type': LayerFeedbackAdapter(17, 'heterogeneous_type'), # 异构类型特征 (67:84)
            'categorical': LayerFeedbackAdapter(15, 'categorical'),     # 分类特征 (84:99)
        })

        # 时序和图结构特征适配器
        self.sequence_adapter = SequenceFeedbackAdapter(32)  # 时序特征 32维
        self.graph_adapter = GraphFeedbackAdapter(10)       # 图结构特征 10维

        # 最终投影适配器
        self.projection_adapter = ProjectionAdapter(141, 128)

        # 反馈状态管理
        self.feedback_state = {
            'mode': 'cold_start',  # cold_start, warm_feedback, stable_feedback
            'epoch': 0,
            'last_guidance': None,
            'adaptation_history': []
        }

        # 特征层结构定义
        self.feature_layer_mapping = {
            'hardware': {'slice': slice(0, 17), 'dim': 17},
            'onchain_behavior': {'slice': slice(17, 34), 'dim': 17},
            'network_topology': {'slice': slice(34, 54), 'dim': 20},
            'dynamic_attributes': {'slice': slice(54, 67), 'dim': 13},
            'heterogeneous_type': {'slice': slice(67, 84), 'dim': 17},
            'categorical': {'slice': slice(84, 99), 'dim': 15}
        }

    def forward(self, nodes: List[Node],
                step4_guidance: Optional[Dict[str, Any]] = None,
                epoch: int = 0) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        自适应特征提取主流程

        Args:
            nodes: 节点列表
            step4_guidance: 第四步反馈指导
            epoch: 当前训练轮次

        Returns:
            f_classic: 最终的经典特征 [N, 128]
            f_graph: 图特征 [N, feature_dim]
            extraction_info: 提取过程信息
        """
        self.feedback_state['epoch'] = epoch
        self._update_feedback_mode(step4_guidance, epoch)

        # 1. 提取99维原始特征并分解为6层
        comprehensive_features = self.comprehensive_extractor.extract_features(nodes)  # [N, 99]
        layered_original = self._decompose_99dim_to_6layers(comprehensive_features)

        # 2. 提取32维时序特征
        sequence_features = self.sequence_encoder(nodes)  # [N, 32]

        # 3. 提取10维图结构特征 (从原始实现获取)
        if self.graph_encoder is None:
            self.graph_encoder = self._get_graph_encoder()
        graph_structure_features = self.graph_encoder(nodes)  # [N, 10]

        # 4. 应用第四步反馈调整
        if step4_guidance and self.feedback_state['mode'] != 'cold_start':
            # 调整99维原始特征的6层
            adjusted_layered = self._apply_layered_feedback(layered_original, step4_guidance)
            # 调整32维时序特征
            adjusted_sequence = self.sequence_adapter(sequence_features, step4_guidance)
            # 调整10维图结构特征
            adjusted_graph_structure = self.graph_adapter(graph_structure_features, step4_guidance)
        else:
            # 冷启动模式，不应用反馈
            adjusted_layered = layered_original
            adjusted_sequence = sequence_features
            adjusted_graph_structure = graph_structure_features

        # 5. 重构99维原始特征
        reconstructed_99dim = self._reconstruct_6layers_to_99dim(adjusted_layered)  # [N, 99]

        # 6. 拼接为141维特征
        f_classic_raw = torch.cat([
            reconstructed_99dim,        # [N, 99] - 调整后的原始特征
            adjusted_sequence,          # [N, 32] - 调整后的时序特征
            adjusted_graph_structure    # [N, 10] - 调整后的图结构特征
        ], dim=1)  # [N, 141]

        # 7. 自适应投影到128维
        f_classic = self.projection_adapter(f_classic_raw, step4_guidance)  # [N, 128]

        # 8. 图特征提取 (保持与原始实现一致)
        f_graph = self._extract_graph_features(nodes)

        # 9. 构建提取信息
        extraction_info = {
            'original_99dim': comprehensive_features,
            'layered_features': adjusted_layered,
            'sequence_features': adjusted_sequence,
            'graph_structure_features': adjusted_graph_structure,
            'reconstructed_99dim': reconstructed_99dim,
            'raw_141dim': f_classic_raw,
            'final_128dim': f_classic,
            'feedback_applied': step4_guidance is not None,
            'feedback_mode': self.feedback_state['mode'],
            'epoch': epoch
        }

        # 10. 记录适应历史
        if step4_guidance:
            self._record_adaptation(step4_guidance, extraction_info)

        return f_classic, f_graph, extraction_info

    def _decompose_99dim_to_6layers(self, features_99dim: torch.Tensor) -> Dict[str, torch.Tensor]:
        """将99维原始特征分解为6层"""
        layered_features = {}

        for layer_name, layer_info in self.feature_layer_mapping.items():
            slice_range = layer_info['slice']
            layered_features[layer_name] = features_99dim[:, slice_range]

        print(f"  分解99维特征 -> 6层: {[(k, v.shape) for k, v in layered_features.items()]}")
        return layered_features

    def _reconstruct_6layers_to_99dim(self, layered_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """将6层特征重构为99维"""
        feature_parts = []

        layer_order = ['hardware', 'onchain_behavior', 'network_topology',
                       'dynamic_attributes', 'heterogeneous_type', 'categorical']

        for layer_name in layer_order:
            if layer_name in layered_features:
                feature_parts.append(layered_features[layer_name])
            else:
                # 缺失层用零填充
                expected_dim = self.feature_layer_mapping[layer_name]['dim']
                batch_size = next(iter(layered_features.values())).size(0)
                device = next(iter(layered_features.values())).device
                zero_features = torch.zeros(batch_size, expected_dim, device=device)
                feature_parts.append(zero_features)

        reconstructed = torch.cat(feature_parts, dim=1)
        print(f"  重构6层 -> 99维: {reconstructed.shape}")
        return reconstructed

    def _apply_layered_feedback(self, layered_features: Dict[str, torch.Tensor],
                                step4_guidance: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """应用第四步反馈到6层特征"""
        adjusted_features = {}

        # 从反馈指导中提取各层调整信息
        layer_weight_adjustments = step4_guidance.get('layer_weight_adjustments', {})
        layer_dimension_selection = step4_guidance.get('layer_dimension_selection', {})
        layer_enhancement_factors = step4_guidance.get('layer_enhancement_factors', {})

        for layer_name, layer_features in layered_features.items():
            if layer_name in self.layer_adapters:
                adapter = self.layer_adapters[layer_name]

                # 构建该层的反馈信息
                layer_guidance = {
                    'weight_adjustment': layer_weight_adjustments.get(layer_name, 1.0),
                    'dimension_selection': layer_dimension_selection.get(layer_name, {}),
                    'enhancement_factor': layer_enhancement_factors.get(layer_name, 1.0),
                    'feedback_mode': self.feedback_state['mode']
                }

                # 应用适配器调整
                adjusted_features[layer_name] = adapter(layer_features, layer_guidance)

                print(f"    应用反馈到 {layer_name}: 权重={layer_guidance['weight_adjustment']:.3f}, "
                      f"增强={layer_guidance['enhancement_factor']:.3f}")
            else:
                adjusted_features[layer_name] = layer_features

        return adjusted_features

    def _update_feedback_mode(self, step4_guidance: Optional[Dict[str, Any]], epoch: int):
        """更新反馈模式"""
        old_mode = self.feedback_state['mode']

        if epoch == 0:
            self.feedback_state['mode'] = 'cold_start'
        elif epoch < 5 and step4_guidance is not None:
            self.feedback_state['mode'] = 'warm_feedback'
        elif epoch >= 5 and step4_guidance is not None:
            self.feedback_state['mode'] = 'stable_feedback'

        if old_mode != self.feedback_state['mode']:
            print(f"  反馈模式切换: {old_mode} -> {self.feedback_state['mode']}")

    def _get_graph_encoder(self):
        """获取图结构编码器 (从原始实现)"""
        # 这里需要根据实际的图结构编码器实现进行调整
        class GraphStructureEncoder(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, nodes: List[Node]) -> torch.Tensor:
                # 简化实现，实际应该从原始的UnifiedFeatureExtractor中获取
                batch_size = len(nodes)
                return torch.randn(batch_size, 10)  # [N, 10]

        return GraphStructureEncoder()

    def _extract_graph_features(self, nodes: List[Node]) -> torch.Tensor:
        """提取图特征 (保持与原始实现一致)"""
        # 这里应该调用原始的图特征提取逻辑
        # 简化实现
        batch_size = len(nodes)
        return torch.randn(batch_size, 64)  # 假设图特征维度为64

    def _record_adaptation(self, step4_guidance: Dict[str, Any], extraction_info: Dict[str, Any]):
        """记录适应历史"""
        adaptation_record = {
            'epoch': self.feedback_state['epoch'],
            'mode': self.feedback_state['mode'],
            'guidance_summary': {
                'has_layer_weights': 'layer_weight_adjustments' in step4_guidance,
                'has_dimension_selection': 'layer_dimension_selection' in step4_guidance,
                'has_enhancement_factors': 'layer_enhancement_factors' in step4_guidance,
                'layer_count': len(step4_guidance.get('layer_weight_adjustments', {}))
            },
            'feature_shapes': {
                'original_99dim': list(extraction_info['original_99dim'].shape),
                'final_128dim': list(extraction_info['final_128dim'].shape)
            }
        }

        self.feedback_state['adaptation_history'].append(adaptation_record)
        self.feedback_state['last_guidance'] = step4_guidance

        # 限制历史长度
        if len(self.feedback_state['adaptation_history']) > 50:
            self.feedback_state['adaptation_history'].pop(0)

    def get_adaptation_report(self) -> Dict[str, Any]:
        """获取适应报告"""
        return {
            'current_mode': self.feedback_state['mode'],
            'current_epoch': self.feedback_state['epoch'],
            'total_adaptations': len(self.feedback_state['adaptation_history']),
            'has_last_guidance': self.feedback_state['last_guidance'] is not None,
            'layer_adapters_count': len(self.layer_adapters),
            'feature_mapping': self.feature_layer_mapping
        }


class LayerFeedbackAdapter(nn.Module):
    """单层特征反馈适配器"""

    def __init__(self, feature_dim: int, layer_type: str):
        super().__init__()
        self.feature_dim = feature_dim
        self.layer_type = layer_type

        # 权重调整参数
        self.weight_factor = nn.Parameter(torch.ones(1))

        # 维度选择掩码 (可学习)
        self.dimension_mask = nn.Parameter(torch.ones(feature_dim))

        # 增强变换
        self.enhancement_layer = nn.Linear(feature_dim, feature_dim)
        if self.enhancement_layer.weight.size(0) == self.enhancement_layer.weight.size(1):
            torch.nn.init.eye_(self.enhancement_layer.weight)
        else:
            torch.nn.init.xavier_uniform_(self.enhancement_layer.weight)
        nn.init.zeros_(self.enhancement_layer.bias)

    def forward(self, features: torch.Tensor, guidance: Dict[str, Any]) -> torch.Tensor:
        """应用反馈指导调整特征"""
        adjusted = features

        # 1. 权重调整
        weight_adj = guidance.get('weight_adjustment', 1.0)
        if weight_adj != 1.0:
            adjusted = adjusted * weight_adj

        # 2. 维度选择 (基于guidance和学习到的掩码)
        dimension_info = guidance.get('dimension_selection', {})
        if 'selection_ratio' in dimension_info:
            selection_ratio = dimension_info['selection_ratio']
            if selection_ratio < 1.0:
                # 使用学习到的维度掩码
                mask = torch.sigmoid(self.dimension_mask)
                # 根据selection_ratio调整掩码强度
                threshold = torch.quantile(mask, 1.0 - selection_ratio)
                binary_mask = (mask >= threshold).float()
                adjusted = adjusted * binary_mask.unsqueeze(0)

        # 3. 特征增强
        enhancement_factor = guidance.get('enhancement_factor', 1.0)
        if enhancement_factor != 1.0:
            enhanced = self.enhancement_layer(adjusted)
            # 混合原始特征和增强特征
            mix_ratio = min(abs(enhancement_factor - 1.0), 0.5)  # 限制增强幅度
            if enhancement_factor > 1.0:
                adjusted = (1 - mix_ratio) * adjusted + mix_ratio * enhanced
            else:
                adjusted = (1 + mix_ratio) * adjusted - mix_ratio * enhanced

        return adjusted


class SequenceFeedbackAdapter(nn.Module):
    """时序特征反馈适配器"""

    def __init__(self, feature_dim: int = 32):
        super().__init__()
        self.feature_dim = feature_dim

        # 时序权重调整
        self.temporal_weights = nn.Parameter(torch.ones(feature_dim))

    def forward(self, features: torch.Tensor, guidance: Dict[str, Any]) -> torch.Tensor:
        """应用时序反馈调整"""
        adjusted = features

        temporal_adjustment = guidance.get('temporal_focus_adjustment', {})

        if 'focus_adjustment_factor' in temporal_adjustment:
            factor = temporal_adjustment['focus_adjustment_factor']
            adjusted = adjusted * factor

        if 'temporal_weights' in temporal_adjustment:
            temp_weights = temporal_adjustment['temporal_weights']
            if isinstance(temp_weights, torch.Tensor) and temp_weights.size(0) == self.feature_dim:
                adjusted = adjusted * temp_weights.unsqueeze(0)

        return adjusted


class GraphFeedbackAdapter(nn.Module):
    """图结构特征反馈适配器"""

    def __init__(self, feature_dim: int = 10):
        super().__init__()
        self.feature_dim = feature_dim

        # 图结构增强层
        self.structure_enhancer = nn.Linear(feature_dim, feature_dim)

        if self.structure_enhancer.weight.size(0) == self.structure_enhancer.weight.size(1):
            torch.nn.init.eye_(self.structure_enhancer.weight)
        else:
            torch.nn.init.xavier_uniform_(self.structure_enhancer.weight)

    def forward(self, features: torch.Tensor, guidance: Dict[str, Any]) -> torch.Tensor:
        """应用图结构反馈调整"""
        adjusted = features

        graph_adjustment = guidance.get('graph_structure_adjustment', {})

        if 'structure_enhancement' in graph_adjustment:
            enhancement = graph_adjustment['structure_enhancement']
            if enhancement != 1.0:
                enhanced = self.structure_enhancer(features)
                mix_ratio = min(abs(enhancement - 1.0), 0.3)
                adjusted = (1 - mix_ratio) * features + mix_ratio * enhanced

        if 'connectivity_focus' in graph_adjustment:
            focus = graph_adjustment['connectivity_focus']
            adjusted = adjusted * focus

        return adjusted


class ProjectionAdapter(nn.Module):
    """投影层反馈适配器"""

    def __init__(self, input_dim: int = 141, output_dim: int = 128):
        super().__init__()

        # 基础投影层
        self.base_projection = nn.Linear(input_dim, output_dim)

        # 反馈指导的注意力调整
        self.feedback_attention = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=4, batch_first=True
        )

        # 自适应权重
        self.adaptive_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, features: torch.Tensor, guidance: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """自适应投影"""
        # 基础投影
        projected = self.base_projection(features)  # [N, 128]

        # 如果有投影调整指导
        if guidance and 'projection_adjustment' in guidance:
            proj_adj = guidance['projection_adjustment']

            # 应用注意力调整
            if 'use_attention' in proj_adj and proj_adj['use_attention']:
                projected_seq = projected.unsqueeze(1)  # [N, 1, 128]
                attended, _ = self.feedback_attention(
                    projected_seq, projected_seq, projected_seq
                )
                projected = attended.squeeze(1)  # [N, 128]

            # 应用自适应权重
            if 'weight_factor' in proj_adj:
                weight_factor = proj_adj['weight_factor']
                projected = projected * weight_factor

        return projected