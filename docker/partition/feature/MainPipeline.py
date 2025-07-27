"""
主要流水线
"""
from datetime import datetime

import torch
import torch.nn as nn
import json
import csv
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
def smart_import():
    """智能导入函数"""
    try:
        # 直接导入模块，失败时立即报错
        from nodeInitialize import Node, load_nodes_from_csv
        from feature_extractor import UnifiedFeatureExtractor
        from feature_fusion import FeatureFusionPipeline
        from config import FeatureDimensions
        from adaptive_feature_extractor import AdaptiveFeatureExtractor
        return Node, load_nodes_from_csv, UnifiedFeatureExtractor, FeatureFusionPipeline, FeatureDimensions, AdaptiveFeatureExtractor
    except ImportError as e:
        raise ImportError(f"主要模块导入失败: {e}")
            return Node, load_nodes_from_csv, UnifiedFeatureExtractor, FeatureFusionPipeline, FeatureDimensions, AdaptiveFeatureExtractor
        except ImportError:
            # 方式3：直接导入（脚本模式）
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, current_dir)
            
            from nodeInitialize import Node, load_nodes_from_csv
            from feature_extractor import UnifiedFeatureExtractor
            from feature_fusion import FeatureFusionPipeline
            from config import FeatureDimensions
            from adaptive_feature_extractor import AdaptiveFeatureExtractor
            return Node, load_nodes_from_csv, UnifiedFeatureExtractor, FeatureFusionPipeline, FeatureDimensions, AdaptiveFeatureExtractor

# 执行智能导入
Node, load_nodes_from_csv, UnifiedFeatureExtractor, FeatureFusionPipeline, FeatureDimensions, AdaptiveFeatureExtractor = smart_import()

class Pipeline:
    """第一步完整流水线 - 使用全面特征"""

    def __init__(self, use_fusion: bool = True, save_adjacency: bool = True):
        """
        初始化流水线

        Args:
            use_fusion: 是否使用特征融合
            save_adjacency: 是否保存邻接矩阵信息
        """
        self.use_fusion = use_fusion
        self.save_adjacency = save_adjacency  # 新增参数
        self.dims = FeatureDimensions()
        self.adaptive_mode = False
        self.adaptive_extractor = None

        # 初始化各个组件
        self.feature_extractor = UnifiedFeatureExtractor()

        if self.use_fusion:
            self.fusion_pipeline = FeatureFusionPipeline(
                classic_dim=self.dims.CLASSIC_DIM,
                graph_dim=self.dims.GRAPH_OUTPUT_DIM,
                fused_dim=self.dims.FUSED_DIM
            )

        print(f"Step1Pipeline初始化完成 - 融合模式: {use_fusion}, 邻接矩阵保存: {save_adjacency}")
        print(f"配置维度 - Classic: {self.dims.CLASSIC_DIM}, Graph: {self.dims.GRAPH_OUTPUT_DIM}, Fused: {self.dims.FUSED_DIM}")

    def extract_features(self, nodes: List[Node]) -> Dict[str, torch.Tensor]:
        """
        执行完整的特征提取流程

        Args:
            nodes: 节点列表

        Returns:
            results: 包含F_classic, F_graph, F_fused的字典
        """
        print(f"开始处理 {len(nodes)} 个节点的全面特征提取...")

        # 提取F_classic和F_graph
        f_classic, f_graph = self.feature_extractor(nodes)

        results = {
            'f_classic': f_classic,   # [N, 128]
            'f_graph': f_graph,       # [N, 96]
            'nodes': nodes
        }

        print(f"基础特征提取完成 - F_classic: {f_classic.shape}, F_graph: {f_graph.shape}")

        # 新增：保存邻接矩阵信息
        if self.save_adjacency:
            self._save_adjacency_information()

        # 如果启用融合
        if self.use_fusion:
            print("开始特征融合...")
            f_fused, contrastive_loss = self.fusion_pipeline(f_classic, f_graph)
            results['f_fused'] = f_fused  # [N, 256]
            results['contrastive_loss'] = contrastive_loss
            print(f"特征融合完成 - F_fused: {f_fused.shape}, 对比损失: {contrastive_loss:.4f}")

        return results

    def _save_adjacency_information(self):
        """保存邻接矩阵相关信息"""
        print(f"\n=== 保存邻接矩阵信息 ===")

        try:
            # 1. 保存原始图构建的邻接矩阵
            graph_builder = self.feature_extractor.graph_feature_extractor.graph_builder
            if hasattr(graph_builder, 'adjacency_info') and graph_builder.adjacency_info:
                graph_builder.save_adjacency_matrices("step1_adjacency")
            else:
                print("警告: 图构建器中没有邻接矩阵信息")

            # 2. 保存RGCN中间表示
            rgcn_extractor = self.feature_extractor.graph_feature_extractor
            if hasattr(rgcn_extractor, 'intermediate_representations') and rgcn_extractor.intermediate_representations:
                rgcn_extractor.save_rgcn_representations("step1_rgcn")
            else:
                print("警告: RGCN提取器中没有中间表示信息")

        except Exception as e:
            print(f"保存邻接矩阵信息时出错: {e}")

    def load_and_extract(self, csv_file: str) -> Dict[str, torch.Tensor]:
        """
        从CSV文件加载数据并提取特征

        Args:
            csv_file: CSV文件路径

        Returns:
            特征提取结果
        """
        # 加载节点数据
        nodes = load_nodes_from_csv(csv_file)
        print(f"成功从 {csv_file} 加载 {len(nodes)} 个节点")

        # 提取特征
        results = self.extract_features(nodes)

        print(f"\n=== 最终特征提取结果 ===")
        print(f"F_classic shape: {results['f_classic'].shape}")
        print(f"F_graph shape: {results['f_graph'].shape}")

        if 'f_fused' in results:
            print(f"F_fused shape: {results['f_fused'].shape}")
            print(f"对比学习损失: {results['contrastive_loss']:.4f}")

        return results

    def save_features(self, results: Dict[str, torch.Tensor], save_path: str = "step1_comprehensive_features.pt"):
        """
        保存特征提取结果

        Args:
            results: 特征提取结果
            save_path: 保存路径
        """
        # 准备保存的数据（排除nodes对象）
        save_data = {}
        for key, value in results.items():
            if key != 'nodes' and isinstance(value, torch.Tensor):
                save_data[key] = value
            elif key == 'contrastive_loss':
                save_data[key] = value

        # 添加元数据
        save_data['metadata'] = {
            'num_nodes': results['f_classic'].shape[0],
            'classic_dim': results['f_classic'].shape[1],
            'graph_dim': results['f_graph'].shape[1],
            'use_fusion': 'f_fused' in results,
            'feature_version': 'comprehensive_v1.0'
        }

        if 'f_fused' in results:
            save_data['metadata']['fused_dim'] = results['f_fused'].shape[1]

        torch.save(save_data, save_path)
        print(f"特征数据已保存到: {save_path}")
        return save_path

    def save_features_with_adjacency(self, results: Dict[str, torch.Tensor], base_name: str = "step1_features"):
        """
        保存特征和邻接矩阵信息

        Args:
            results: 特征提取结果
            base_name: 基础文件名
        """
        # 1. 保存常规特征
        feature_path = self.save_features(results, f"{base_name}.pt")

        # 2. 保存可读格式特征
        self.save_readable_features(results, base_name)

        # 3. 保存邻接矩阵信息（如果启用）
        if self.save_adjacency:
            self._save_comprehensive_adjacency_info(base_name)

        return feature_path

    def _save_comprehensive_adjacency_info(self, base_name: str):
        """
        保存综合的邻接矩阵信息

        Args:
            base_name: 基础文件名
        """
        print(f"\n=== 保存综合邻接矩阵信息 ===")

        try:
            # 获取图构建器和RGCN提取器
            graph_builder = self.feature_extractor.graph_feature_extractor.graph_builder
            rgcn_extractor = self.feature_extractor.graph_feature_extractor

            # 整合所有邻接矩阵信息
            comprehensive_info = {
                'generation_info': {
                    'timestamp': str(pd.Timestamp.now()),
                    'pipeline_config': {
                        'use_fusion': self.use_fusion,
                        'save_adjacency': self.save_adjacency,
                        'classic_dim': self.dims.CLASSIC_DIM,
                        'graph_dim': self.dims.GRAPH_OUTPUT_DIM,
                        'fused_dim': self.dims.FUSED_DIM if self.use_fusion else None
                    }
                }
            }

            # 添加原始图信息
            if hasattr(graph_builder, 'adjacency_info') and graph_builder.adjacency_info:
                comprehensive_info['original_graph'] = graph_builder.adjacency_info

            # 添加RGCN处理信息
            if hasattr(rgcn_extractor, 'intermediate_representations') and rgcn_extractor.intermediate_representations:
                comprehensive_info['rgcn_processing'] = {
                    'layer_statistics': rgcn_extractor.intermediate_representations['layer_statistics'],
                    'tensor_shapes': {
                        'input_shape': list(rgcn_extractor.intermediate_representations['rgcn_input'].shape),
                        'hidden_shape': list(rgcn_extractor.intermediate_representations['rgcn_hidden'].shape),
                        'output_shape': list(rgcn_extractor.intermediate_representations['rgcn_output'].shape)
                    }
                }

            # 保存综合信息
            with open(f"{base_name}_adjacency_comprehensive.json", 'w', encoding='utf-8') as f:
                json.dump(comprehensive_info, f, indent=2, ensure_ascii=False, default=str)

            print(f"  ✓ 综合邻接矩阵信息: {base_name}_adjacency_comprehensive.json")

            # 保存PyTorch格式的完整数据
            if (hasattr(graph_builder, 'adjacency_info') and graph_builder.adjacency_info and
                    hasattr(rgcn_extractor, 'intermediate_representations') and rgcn_extractor.intermediate_representations):

                complete_adjacency_data = {
                    'original_adjacency_matrix': torch.from_numpy(graph_builder.adjacency_info['adjacency_matrix_dense']),
                    'original_relation_matrix': torch.from_numpy(graph_builder.adjacency_info['relation_matrix']),
                    'original_edge_index': torch.from_numpy(graph_builder.adjacency_info['edge_index_coo']),
                    'original_edge_type': torch.from_numpy(graph_builder.adjacency_info['edge_type']),
                    'rgcn_input_features': rgcn_extractor.intermediate_representations['rgcn_input'],
                    'rgcn_hidden_features': rgcn_extractor.intermediate_representations['rgcn_hidden'],
                    'rgcn_output_features': rgcn_extractor.intermediate_representations['rgcn_output'],
                    'metadata': comprehensive_info['generation_info']
                }

                torch.save(complete_adjacency_data, f"{base_name}_adjacency_complete.pt")
                print(f"  ✓ 完整邻接矩阵数据: {base_name}_adjacency_complete.pt")

        except Exception as e:
            print(f"保存综合邻接矩阵信息时出错: {e}")
            import traceback
            traceback.print_exc()

    def save_readable_features(self, results: Dict[str, torch.Tensor], base_name: str = "features"):
        """
        保存可读格式的特征数据

        Args:
            results: 特征提取结果
            base_name: 基础文件名
        """
        print(f"\n=== 保存可读格式特征 ===")

        # 1. 单独保存每种特征类型
        self._save_individual_features(results, base_name)

        # 2. 保存特征统计信息
        self._save_feature_statistics(results, f"{base_name}_statistics.json")


        print(f"[SUCCESS] 可读格式特征已保存完成")


    def _save_individual_features(self, results: Dict[str, torch.Tensor], base_name: str):
        """单独保存每种特征类型"""
        print(f"单独保存各特征类型...")

        # 保存 F_classic 特征
        self._save_single_feature_type(
            results['f_classic'],
            f"{base_name}_f_classic.csv",
            "f_classic"
        )

        # 保存 F_graph 特征
        self._save_single_feature_type(
            results['f_graph'],
            f"{base_name}_f_graph.csv",
            "f_graph"
        )

        # 保存 F_fused 特征（如果存在）
        if 'f_fused' in results:
            self._save_single_feature_type(
                results['f_fused'],
                f"{base_name}_f_fused.csv",
                "f_fused"
            )

        # 保存各特征类型的详细统计
        self._save_individual_statistics(results, base_name)

    def _save_single_feature_type(self, tensor: torch.Tensor, csv_path: str, feature_type: str):
        """保存单一特征类型到CSV"""
        print(f"  保存 {feature_type}: {csv_path}")

        # 转换为numpy数组
        feature_array = tensor.detach().cpu().numpy()
        num_nodes, num_features = feature_array.shape

        # 创建DataFrame
        data = {}
        data['node_id'] = list(range(num_nodes))

        # 添加特征列
        for i in range(num_features):
            data[f'{feature_type}_{i:03d}'] = feature_array[:, i]

        df = pd.DataFrame(data)

        # 保存为CSV
        df.to_csv(csv_path, index=False, float_format='%.6f')

        # 打印统计信息
        print(f"    ✓ 形状: {feature_array.shape}")
        print(f"    ✓ 数值范围: [{np.min(feature_array):.3f}, {np.max(feature_array):.3f}]")
        print(f"    ✓ 均值: {np.mean(feature_array):.3f}, 标准差: {np.std(feature_array):.3f}")

    def _save_individual_statistics(self, results: Dict[str, torch.Tensor], base_name: str):
        """保存各特征类型的详细统计信息"""
        stats_path = f"{base_name}_individual_statistics.json"
        print(f"  保存详细统计: {stats_path}")

        detailed_stats = {
            'generation_time': str(pd.Timestamp.now()),
            'feature_analysis': {}
        }

        # 分析每种特征类型
        for feature_name, tensor in results.items():
            if isinstance(tensor, torch.Tensor):
                feature_array = tensor.detach().cpu().numpy()

                # 计算详细统计
                stats = {
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'overall_stats': self._compute_detailed_stats(feature_array),
                    'dimension_wise_stats': self._compute_dimension_wise_stats(feature_array),
                    'scale_analysis': self._analyze_feature_scale(feature_array)
                }

                detailed_stats['feature_analysis'][feature_name] = stats

        # 保存详细统计
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_stats, f, indent=2, ensure_ascii=False)

        print(f"    ✓ 详细统计已保存")


    def _compute_detailed_stats(self, feature_array: np.ndarray) -> Dict[str, Any]:
        """计算详细的统计信息"""
        return {
            'mean': float(np.mean(feature_array)),
            'std': float(np.std(feature_array)),
            'min': float(np.min(feature_array)),
            'max': float(np.max(feature_array)),
            'median': float(np.median(feature_array)),
            'q25': float(np.percentile(feature_array, 25)),
            'q75': float(np.percentile(feature_array, 75)),
            'q05': float(np.percentile(feature_array, 5)),
            'q95': float(np.percentile(feature_array, 95)),
            'zero_ratio': float((feature_array == 0).sum() / feature_array.size),
            'non_zero_ratio': float((feature_array != 0).sum() / feature_array.size),
            'nan_count': int(np.isnan(feature_array).sum()),
            'inf_count': int(np.isinf(feature_array).sum()),
            'outlier_ratio_iqr': self._compute_outlier_ratio_iqr(feature_array),
            'outlier_ratio_zscore': self._compute_outlier_ratio_zscore(feature_array)
        }

    def _compute_dimension_wise_stats(self, feature_array: np.ndarray) -> List[Dict[str, float]]:
        """计算每个维度的统计信息"""
        if len(feature_array.shape) != 2:
            return []

        dimension_stats = []
        for i in range(feature_array.shape[1]):
            dim_data = feature_array[:, i]
            dim_stats = {
                'dimension': i,
                'mean': float(np.mean(dim_data)),
                'std': float(np.std(dim_data)),
                'min': float(np.min(dim_data)),
                'max': float(np.max(dim_data)),
                'range': float(np.max(dim_data) - np.min(dim_data)),
                'zero_ratio': float((dim_data == 0).sum() / len(dim_data))
            }
            dimension_stats.append(dim_stats)

        return dimension_stats

    def _analyze_feature_scale(self, feature_array: np.ndarray) -> Dict[str, Any]:
        """分析特征尺度问题"""
        if len(feature_array.shape) != 2:
            return {}

        # 计算每个维度的尺度
        dimension_scales = []
        for i in range(feature_array.shape[1]):
            dim_data = feature_array[:, i]
            scale = np.std(dim_data)
            dimension_scales.append(scale)

        dimension_scales = np.array(dimension_scales)

        analysis = {
            'scale_range': {
                'min_scale': float(np.min(dimension_scales)),
                'max_scale': float(np.max(dimension_scales)),
                'scale_ratio': float(np.max(dimension_scales) / (np.min(dimension_scales) + 1e-8))
            },
            'scale_distribution': {
                'scale_mean': float(np.mean(dimension_scales)),
                'scale_std': float(np.std(dimension_scales)),
                'scale_cv': float(np.std(dimension_scales) / (np.mean(dimension_scales) + 1e-8))
            },
            'recommendations': self._get_scaling_recommendations(dimension_scales)
        }

        return analysis

    def _compute_outlier_ratio_iqr(self, data: np.ndarray) -> float:
        """使用IQR方法计算异常值比例"""
        q25 = np.percentile(data, 25)
        q75 = np.percentile(data, 75)
        iqr = q75 - q25

        if iqr == 0:
            return 0.0

        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        outliers = (data < lower_bound) | (data > upper_bound)
        return float(outliers.sum() / len(data))

    def _compute_outlier_ratio_zscore(self, data: np.ndarray, threshold: float = 3.0) -> float:
        """使用Z-score方法计算异常值比例"""
        if np.std(data) == 0:
            return 0.0

        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        outliers = z_scores > threshold
        return float(outliers.sum() / len(data))

    def _get_scaling_recommendations(self, dimension_scales: np.ndarray) -> List[str]:
        """基于尺度分析给出标准化建议"""
        recommendations = []

        scale_ratio = np.max(dimension_scales) / (np.min(dimension_scales) + 1e-8)

        if scale_ratio > 1000:
            recommendations.append("强烈建议进行标准化 - 特征尺度差异极大 (>1000倍)")
        elif scale_ratio > 100:
            recommendations.append("建议进行标准化 - 特征尺度差异较大 (>100倍)")
        elif scale_ratio > 10:
            recommendations.append("考虑进行标准化 - 特征尺度有差异 (>10倍)")
        else:
            recommendations.append("特征尺度相对合理")

        # 检查是否有零方差特征
        zero_var_count = (dimension_scales < 1e-8).sum()
        if zero_var_count > 0:
            recommendations.append(f"发现 {zero_var_count} 个零方差特征，建议移除")

        # 检查尺度分布
        cv = np.std(dimension_scales) / (np.mean(dimension_scales) + 1e-8)
        if cv > 1.0:
            recommendations.append("特征尺度分布不均匀，建议使用RobustScaler")

        return recommendations

    def analyze_and_recommend_scaling(self, results: Dict[str, torch.Tensor]):
        """分析特征并给出标准化建议"""
        print(f"\n=== 特征尺度分析和建议 ===")

        for feature_name, tensor in results.items():
            if isinstance(tensor, torch.Tensor):
                feature_array = tensor.detach().cpu().numpy()
                analysis = self._analyze_feature_scale(feature_array)

                print(f"\n{feature_name} 特征分析:")
                print(f"  形状: {feature_array.shape}")
                print(f"  尺度范围: {analysis['scale_range']['min_scale']:.3f} - {analysis['scale_range']['max_scale']:.3f}")
                print(f"  尺度比例: {analysis['scale_range']['scale_ratio']:.1f}")
                print(f"  建议:")
                for rec in analysis['recommendations']:
                    print(f"    - {rec}")

    def _save_feature_statistics(self, results: Dict[str, torch.Tensor], stats_path: str):
        """保存特征统计信息"""
        print(f"保存统计信息: {stats_path}")

        statistics = {
            'generation_time': str(pd.Timestamp.now()),
            'feature_types': {
                'f_classic': self._compute_tensor_stats(results['f_classic']),
                'f_graph': self._compute_tensor_stats(results['f_graph'])
            }
        }

        if 'f_fused' in results:
            statistics['feature_types']['f_fused'] = self._compute_tensor_stats(results['f_fused'])

        # 添加整体统计
        statistics['overall'] = {
            'total_nodes': int(results['f_classic'].shape[0]),
            'total_features': int(results['f_classic'].shape[1] + results['f_graph'].shape[1]),
            'memory_usage_mb': self._estimate_memory_usage(results)
        }

        if 'f_fused' in results:
            statistics['overall']['total_features'] += int(results['f_fused'].shape[1])

        # 保存统计信息
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)

        print(f"  ✓ 统计信息已保存")

    def _compute_tensor_stats(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """计算张量的统计信息"""
        tensor_np = tensor.detach().cpu().numpy()

        return {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'mean': float(np.mean(tensor_np)),
            'std': float(np.std(tensor_np)),
            'min': float(np.min(tensor_np)),
            'max': float(np.max(tensor_np)),
            'median': float(np.median(tensor_np)),
            'q25': float(np.percentile(tensor_np, 25)),
            'q75': float(np.percentile(tensor_np, 75)),
            'zero_ratio': float((tensor_np == 0).sum() / tensor_np.size),
            'non_zero_ratio': float((tensor_np != 0).sum() / tensor_np.size),
            'nan_count': int(np.isnan(tensor_np).sum()),
            'inf_count': int(np.isinf(tensor_np).sum())
        }

    def _estimate_memory_usage(self, results: Dict[str, torch.Tensor]) -> float:
        """估算内存使用量（MB）"""
        total_elements = 0
        for key, tensor in results.items():
            if isinstance(tensor, torch.Tensor):
                total_elements += tensor.numel()

        # 假设每个元素4字节（float32）
        return (total_elements * 4) / (1024 * 1024)

    def enable_adaptive_mode(self):
        """启用自适应模式"""
        self.adaptive_mode = True
        self.adaptive_extractor = AdaptiveFeatureExtractor()
        print("[SUCCESS] Pipeline自适应模式已启用")

    def extract_features_with_feedback(self, nodes: List[Node],
                                       step4_guidance: Optional[Dict[str, Any]] = None,
                                       epoch: int = 0) -> Dict[str, torch.Tensor]:
        """
        带反馈的特征提取 (新增方法)

        Args:
            nodes: 节点列表
            step4_guidance: 第四步反馈指导
            epoch: 当前训练轮次

        Returns:
            特征提取结果字典
        """
        if self.adaptive_mode and self.adaptive_extractor is not None:
            print(f"[CONFIG] 使用自适应特征提取 (Epoch {epoch})")

            # 使用自适应提取器
            f_classic, f_graph, extraction_info = self.adaptive_extractor(
                nodes, step4_guidance, epoch
            )

            results = {
                'f_classic': f_classic,
                'f_graph': f_graph,
                'nodes': nodes,
                'extraction_info': extraction_info,
                'adaptive_mode': True
            }

        else:
            print(f"[CONFIG] 使用标准特征提取 (Epoch {epoch})")

            # 使用原始提取器
            f_classic, f_graph = self.feature_extractor(nodes)

            results = {
                'f_classic': f_classic,
                'f_graph': f_graph,
                'nodes': nodes,
                'adaptive_mode': False
            }

        # 特征融合 (如果启用)
        if self.use_fusion:
            print(" 执行特征融合...")
            f_fused, contrastive_loss = self.fusion_pipeline(results['f_classic'], results['f_graph'])
            results['f_fused'] = f_fused
            results['contrastive_loss'] = contrastive_loss

        # 保存邻接矩阵信息
        if self.save_adjacency:
            self._save_adjacency_information()

        return results

    def get_adaptive_status(self) -> Dict[str, Any]:
        """获取自适应状态"""
        if self.adaptive_mode and self.adaptive_extractor is not None:
            return self.adaptive_extractor.get_adaptation_report()
        else:
            return {'adaptive_mode': False, 'message': 'Adaptive mode not enabled'}

    def save_adaptive_history(self, filepath: str = "adaptive_history.json"):
        """保存自适应历史"""
        if self.adaptive_mode and self.adaptive_extractor is not None:
            import json

            history_data = {
                'pipeline_config': {
                    'use_fusion': self.use_fusion,
                    'save_adjacency': self.save_adjacency,
                    'adaptive_mode': self.adaptive_mode
                },
                'adaptive_status': self.get_adaptive_status(),
                'timestamp': str(datetime.now()) if 'datetime' in globals() else 'unknown'
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False, default=str)

            print(f"📁 自适应历史已保存: {filepath}")
        else:
            print("[WARNING] 自适应模式未启用，无法保存历史")


def main():
    """全面特征提取流程"""
    print("=== 全面特征提取流水线演示 ===\n")

    # 初始化流水线（启用邻接矩阵保存）
    pipeline = Pipeline(use_fusion=True, save_adjacency=True)

    # 选择数据文件
    csv_files = ["large_samples.csv"]

    for csv_file in csv_files:
        try:
            print(f"\n{'='*50}")
            print(f"处理文件: {csv_file}")
            print(f"{'='*50}")

            # 提取特征
            results = pipeline.load_and_extract(csv_file)

            # 使用新的保存方法（包含邻接矩阵）
            base_name = f"step1_{csv_file.replace('.csv', '')}"
            pipeline.save_features_with_adjacency(results, base_name)

            # 分析特征尺度并给出建议
            pipeline.analyze_and_recommend_scaling(results)

            # 简单验证
            print(f"\n特征质量检查:")
            print(f"- F_classic 非零元素比例: {(results['f_classic'] != 0).float().mean():.3f}")
            print(f"- F_graph 非零元素比例: {(results['f_graph'] != 0).float().mean():.3f}")

            if 'f_fused' in results:
                print(f"- F_fused 非零元素比例: {(results['f_fused'] != 0).float().mean():.3f}")

            print(f"[SUCCESS] {csv_file} 处理完成\n")

        except Exception as e:
            print(f"[ERROR] 处理 {csv_file} 时出错: {e}\n")
            import traceback
            traceback.print_exc()
            continue

    print("=== 全面特征提取流水线演示完成 ===")

if __name__ == "__main__":
    main()