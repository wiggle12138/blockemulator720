"""
数据处理器
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """全面特征数据处理器"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.is_fitted = False

    def create_one_hot(self, value: str, categories: List[str]) -> List[float]:
        """
        创建one-hot编码

        Args:
            value: 要编码的值
            categories: 类别列表

        Returns:
            one-hot编码列表
        """
        encoding = [0.0] * len(categories)

        if value in categories:
            idx = categories.index(value)
            encoding[idx] = 1.0

        return encoding

    def adaptive_scaling(self, features: np.ndarray, feature_type: str = 'mixed') -> np.ndarray:
        """
        自适应特征缩放

        Args:
            features: 特征矩阵 [N, D]
            feature_type: 特征类型 ('numeric', 'categorical', 'mixed')

        Returns:
            缩放后的特征
        """
        if feature_type not in self.scalers:
            if feature_type == 'numeric':
                # 数值特征使用RobustScaler（对异常值不敏感）
                self.scalers[feature_type] = RobustScaler()
            elif feature_type == 'categorical':
                # 分类特征使用MinMaxScaler
                self.scalers[feature_type] = MinMaxScaler()
            else:  # mixed
                # 混合特征使用StandardScaler
                self.scalers[feature_type] = StandardScaler()

        if not self.is_fitted:
            scaled_features = self.scalers[feature_type].fit_transform(features)
        else:
            scaled_features = self.scalers[feature_type].transform(features)

        return scaled_features

    def quantile_transform(self, features: np.ndarray, feature_name: str = 'default') -> np.ndarray:
        """
        分位数变换（适用于长尾分布）

        Args:
            features: 特征矩阵 [N, D]
            feature_name: 特征名称

        Returns:
            变换后的特征
        """
        scaler_key = f'quantile_{feature_name}'

        if scaler_key not in self.scalers:
            self.scalers[scaler_key] = QuantileTransformer(
                output_distribution='uniform',
                random_state=42
            )

        if not self.is_fitted:
            transformed_features = self.scalers[scaler_key].fit_transform(features)
        else:
            transformed_features = self.scalers[scaler_key].transform(features)

        return transformed_features

    def power_transform(self, features: np.ndarray, method: str = 'yeo-johnson') -> np.ndarray:
        """
        幂变换（处理偏态分布）

        Args:
            features: 特征矩阵 [N, D]
            method: 变换方法

        Returns:
            变换后的特征
        """
        from sklearn.preprocessing import PowerTransformer

        if 'power_transformer' not in self.scalers:
            self.scalers['power_transformer'] = PowerTransformer(
                method=method,
                standardize=True
            )

        if not self.is_fitted:
            transformed_features = self.scalers['power_transformer'].fit_transform(features)
        else:
            transformed_features = self.scalers['power_transformer'].transform(features)

        return transformed_features

    def comprehensive_preprocessing(self,
                                    numeric_features: np.ndarray,
                                    categorical_features: np.ndarray,
                                    sequence_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        全面的特征预处理

        Args:
            numeric_features: 数值特征 [N, D1]
            categorical_features: 分类特征 [N, D2]
            sequence_features: 序列特征 [N, T, D3]

        Returns:
            处理后的特征元组
        """
        # 处理数值特征
        if numeric_features.size > 0:
            # 检查是否需要幂变换
            if self._needs_power_transform(numeric_features):
                numeric_processed = self.power_transform(numeric_features)
            else:
                numeric_processed = self.adaptive_scaling(numeric_features, 'numeric')
        else:
            numeric_processed = numeric_features

        # 处理分类特征
        if categorical_features.size > 0:
            categorical_processed = self.adaptive_scaling(categorical_features, 'categorical')
        else:
            categorical_processed = categorical_features

        # 处理序列特征
        if sequence_features.size > 0:
            sequence_processed = self._process_sequence_features(sequence_features)
        else:
            sequence_processed = sequence_features

        return numeric_processed, categorical_processed, sequence_processed

    def _needs_power_transform(self, features: np.ndarray) -> bool:
        """检查是否需要幂变换"""
        from scipy import stats

        # 计算偏度
        skewness_threshold = 1.0
        for i in range(features.shape[1]):
            col = features[:, i]
            if len(np.unique(col)) > 1:  # 避免常数列
                skewness = abs(stats.skew(col))
                if skewness > skewness_threshold:
                    return True

        return False

    def _process_sequence_features(self, sequence_features: np.ndarray) -> np.ndarray:
        """处理序列特征"""
        N, T, D = sequence_features.shape

        # 重塑为2D进行标准化
        reshaped = sequence_features.reshape(-1, D)
        scaled = self.adaptive_scaling(reshaped, 'mixed')

        # 重塑回原始形状
        processed = scaled.reshape(N, T, D)

        return processed

    def dimension_reduction(self,
                            features: np.ndarray,
                            target_dim: int,
                            method: str = 'pca') -> np.ndarray:
        """
        维度约简

        Args:
            features: 输入特征 [N, D]
            target_dim: 目标维度
            method: 约简方法

        Returns:
            约简后的特征 [N, target_dim]
        """
        if features.shape[1] <= target_dim:
            return features

        reducer_key = f'{method}_reducer'

        if reducer_key not in self.scalers:
            if method == 'pca':
                self.scalers[reducer_key] = PCA(
                    n_components=target_dim,
                    random_state=42
                )
            else:
                raise ValueError(f"Unknown reduction method: {method}")

        if not self.is_fitted:
            reduced_features = self.scalers[reducer_key].fit_transform(features)
        else:
            reduced_features = self.scalers[reducer_key].transform(features)

        return reduced_features

    def handle_missing_values(self, features: np.ndarray, strategy: str = 'mean') -> np.ndarray:
        """
        处理缺失值

        Args:
            features: 特征矩阵
            strategy: 填充策略

        Returns:
            处理后的特征
        """
        from sklearn.impute import SimpleImputer

        imputer_key = f'imputer_{strategy}'

        if imputer_key not in self.scalers:
            self.scalers[imputer_key] = SimpleImputer(strategy=strategy)

        if not self.is_fitted:
            imputed_features = self.scalers[imputer_key].fit_transform(features)
        else:
            imputed_features = self.scalers[imputer_key].transform(features)

        return imputed_features

    def detect_and_handle_outliers(self,
                                   features: np.ndarray,
                                   method: str = 'iqr',
                                   factor: float = 1.5) -> np.ndarray:
        """
        检测和处理异常值

        Args:
            features: 特征矩阵
            method: 检测方法 ('iqr', 'zscore')
            factor: 阈值因子

        Returns:
            处理后的特征
        """
        processed_features = features.copy()

        for i in range(features.shape[1]):
            col = features[:, i]

            if method == 'iqr':
                Q1 = np.percentile(col, 25)
                Q3 = np.percentile(col, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR

                # 截断异常值
                processed_features[:, i] = np.clip(col, lower_bound, upper_bound)

            elif method == 'zscore':
                mean_val = np.mean(col)
                std_val = np.std(col)
                z_scores = np.abs((col - mean_val) / std_val)

                # 截断z-score > factor的值
                outlier_mask = z_scores > factor
                processed_features[outlier_mask, i] = mean_val

        return processed_features

    def feature_selection(self,
                          features: np.ndarray,
                          target: np.ndarray = None,
                          method: str = 'variance',
                          k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        特征选择

        Args:
            features: 特征矩阵
            target: 目标变量（可选）
            method: 选择方法
            k: 选择的特征数量

        Returns:
            选择后的特征和选择掩码
        """
        from sklearn.feature_selection import (
            VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
        )

        selector_key = f'selector_{method}'

        if selector_key not in self.scalers:
            if method == 'variance':
                self.scalers[selector_key] = VarianceThreshold(threshold=0.01)
            elif method == 'kbest_f' and target is not None:
                k = k or min(features.shape[1], 50)
                self.scalers[selector_key] = SelectKBest(f_classif, k=k)
            elif method == 'mutual_info' and target is not None:
                k = k or min(features.shape[1], 50)
                self.scalers[selector_key] = SelectKBest(mutual_info_classif, k=k)
            else:
                # 默认返回原特征
                return features, np.ones(features.shape[1], dtype=bool)

        if not self.is_fitted:
            if target is not None and method in ['kbest_f', 'mutual_info']:
                selected_features = self.scalers[selector_key].fit_transform(features, target)
            else:
                selected_features = self.scalers[selector_key].fit_transform(features)
        else:
            selected_features = self.scalers[selector_key].transform(features)

        # 获取选择掩码
        if hasattr(self.scalers[selector_key], 'get_support'):
            mask = self.scalers[selector_key].get_support()
        else:
            mask = np.ones(features.shape[1], dtype=bool)

        return selected_features, mask

    def fit(self):
        """标记处理器为已拟合状态"""
        self.is_fitted = True

    def reset(self):
        """重置处理器"""
        self.scalers = {}
        self.encoders = {}
        self.is_fitted = False

    def get_feature_stats(self, features: np.ndarray) -> Dict[str, float]:
        """获取特征统计信息"""
        stats = {
            'mean': float(np.mean(features)),
            'std': float(np.std(features)),
            'min': float(np.min(features)),
            'max': float(np.max(features)),
            'median': float(np.median(features)),
            'q25': float(np.percentile(features, 25)),
            'q75': float(np.percentile(features, 75)),
            'missing_ratio': float(np.isnan(features).sum() / features.size),
            'zero_ratio': float((features == 0).sum() / features.size)
        }

        return stats

def main():
    """测试数据处理器"""
    print("测试数据处理器...")

    # 生成测试数据
    np.random.seed(42)
    N, D = 100, 20

    # 数值特征（包含一些偏态分布）
    numeric_features = np.random.exponential(2, (N, D//2))

    # 分类特征
    categorical_features = np.random.randint(0, 5, (N, D//4))

    # 序列特征
    sequence_features = np.random.randn(N, 10, D//4)

    # 初始化处理器
    processor = DataProcessor()

    # 全面预处理
    num_processed, cat_processed, seq_processed = processor.comprehensive_preprocessing(
        numeric_features.astype(float),
        categorical_features.astype(float),
        sequence_features
    )

    processor.fit()

    print(f"数值特征: {numeric_features.shape} -> {num_processed.shape}")
    print(f"分类特征: {categorical_features.shape} -> {cat_processed.shape}")
    print(f"序列特征: {sequence_features.shape} -> {seq_processed.shape}")

    # 特征统计
    stats = processor.get_feature_stats(num_processed)
    print(f"处理后数值特征统计: {stats}")

    print("测试完成!")

if __name__ == "__main__":
    main()