"""
分层特征重要性分析器
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from typing import Dict, List, Tuple, Any
from collections import defaultdict

class FeatureImportanceAnalyzer:
    """分层特征重要性分析器"""

    def __init__(self, feature_dims: Dict[str, int]):
        self.feature_dims = feature_dims
        self.importance_history = defaultdict(list)

    def analyze_importance(self, features: Dict[str, torch.Tensor],
                           performance_scores: Dict[str, torch.Tensor],
                           model: nn.Module = None) -> Dict[str, Dict[str, float]]:
        """
        分析特征重要性

        Args:
            features: 分层特征字典
            performance_scores: 性能评分字典
            model: 相关模型（用于梯度分析）

        Returns:
            layer_importance: 层次化特征重要性矩阵
        """
        importance_matrix = {}

        # 1. 梯度反向传播分析
        if model is not None:
            gradient_importance = self._gradient_importance_analysis(
                features, performance_scores, model
            )
        else:
            gradient_importance = {layer: 0.0 for layer in features.keys()}

        # 2. 跨层互信息计算
        mutual_info_importance = self._mutual_info_analysis(
            features, performance_scores
        )

        # 3. 特征方差分析
        variance_importance = self._variance_analysis(features)

        # 4. 综合重要性评分
        for layer in features.keys():
            importance_matrix[layer] = {
                'gradient': gradient_importance.get(layer, 0.0),
                'mutual_info': mutual_info_importance.get(layer, 0.0),
                'variance': variance_importance.get(layer, 0.0),
                'combined': self._combine_importance_scores(
                    gradient_importance.get(layer, 0.0),
                    mutual_info_importance.get(layer, 0.0),
                    variance_importance.get(layer, 0.0)
                )
            }

            # 更新历史记录
            self.importance_history[layer].append(importance_matrix[layer]['combined'])

        return importance_matrix

    def _gradient_importance_analysis(self, features: Dict[str, torch.Tensor],
                                      performance_scores: Dict[str, torch.Tensor],
                                      model: nn.Module) -> Dict[str, float]:
        """梯度反向传播分析"""
        importance_scores = {}

        # 计算总体性能分数
        total_score = torch.stack(list(performance_scores.values())).mean()

        for layer_name, layer_features in features.items():
            if not layer_features.requires_grad:
                layer_features.requires_grad_(True)

            try:
                # 清零梯度
                if layer_features.grad is not None:
                    layer_features.grad.zero_()

                # 计算梯度
                total_score.backward(retain_graph=True)

                if layer_features.grad is not None:
                    # 计算梯度的L2范数作为重要性
                    grad_norm = torch.norm(layer_features.grad, p=2).item()
                    importance_scores[layer_name] = grad_norm
                else:
                    importance_scores[layer_name] = 0.0

            except Exception as e:
                print(f"Warning: Gradient analysis failed for {layer_name}: {e}")
                importance_scores[layer_name] = 0.0

        return importance_scores

    def _mutual_info_analysis(self, features: Dict[str, torch.Tensor],
                              performance_scores: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """跨层互信息计算"""
        importance_scores = {}

        # 将性能分数合并为目标变量
        target_scores = []
        for score in performance_scores.values():
            if isinstance(score, torch.Tensor):
                target_scores.append(score.detach().cpu().numpy())
            else:
                target_scores.append(np.array([score]))

        if not target_scores:
            return {layer: 0.0 for layer in features.keys()}

        target = np.mean(target_scores, axis=0)
        if target.ndim == 0:
            target = np.array([target])

        for layer_name, layer_features in features.items():
            try:
                layer_np = layer_features.detach().cpu().numpy()

                # 如果特征是二维的，计算每个维度与目标的互信息
                if layer_np.ndim == 2:
                    mi_scores = []
                    for dim in range(min(layer_np.shape[1], 10)):  # 限制维度数量
                        try:
                            mi = mutual_info_regression(
                                layer_np[:, dim].reshape(-1, 1),
                                target
                            )[0]
                            mi_scores.append(mi)
                        except:
                            mi_scores.append(0.0)

                    # 平均互信息作为层重要性
                    importance_scores[layer_name] = np.mean(mi_scores) if mi_scores else 0.0
                else:
                    # 一维特征直接计算
                    try:
                        mi = mutual_info_regression(
                            layer_np.reshape(-1, 1),
                            target
                        )[0]
                        importance_scores[layer_name] = mi
                    except:
                        importance_scores[layer_name] = 0.0

            except Exception as e:
                print(f"Warning: Mutual info analysis failed for {layer_name}: {e}")
                importance_scores[layer_name] = 0.0

        return importance_scores

    def _variance_analysis(self, features: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """特征方差分析"""
        variance_scores = {}

        for layer_name, layer_features in features.items():
            try:
                # 计算特征的方差
                if layer_features.dim() == 2:
                    # 对每个维度计算方差，然后取平均
                    variances = torch.var(layer_features, dim=0)
                    avg_variance = torch.mean(variances).item()
                else:
                    avg_variance = torch.var(layer_features).item()

                variance_scores[layer_name] = avg_variance

            except Exception as e:
                print(f"Warning: Variance analysis failed for {layer_name}: {e}")
                variance_scores[layer_name] = 0.0

        return variance_scores

    def _combine_importance_scores(self, gradient_score: float,
                                   mutual_info_score: float,
                                   variance_score: float) -> float:
        """综合重要性评分"""
        # 标准化各个分数
        scores = np.array([gradient_score, mutual_info_score, variance_score])

        # 避免除零错误
        if np.sum(scores) == 0:
            return 0.0

        # 加权平均 (可以调整权重)
        weights = np.array([0.4, 0.4, 0.2])  # 梯度和互信息权重更高
        combined_score = np.average(scores, weights=weights)

        return float(combined_score)

    def get_importance_trends(self, layer_name: str, window_size: int = 5) -> List[float]:
        """获取特定层的重要性趋势"""
        if layer_name not in self.importance_history:
            return []

        history = self.importance_history[layer_name]
        if len(history) < window_size:
            return history

        # 计算滑动平均
        trends = []
        for i in range(len(history) - window_size + 1):
            window_avg = np.mean(history[i:i + window_size])
            trends.append(window_avg)

        return trends