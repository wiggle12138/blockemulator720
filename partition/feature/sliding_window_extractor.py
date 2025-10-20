"""
滑动窗口特征提取器
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import sys
from pathlib import Path

# 添加当前目录到系统路径
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 直接导入模块
try:
    from nodeInitialize import Node
except ImportError as e:
    raise ImportError(f"nodeInitialize导入失败: {e}")

try:
    from config import FeatureDimensions, EncodingMaps
except ImportError as e:
    raise ImportError(f"config导入失败: {e}")

class EnhancedSequenceFeatureEncoder(nn.Module):
    """增强的时序特征编码器"""

    def __init__(self, input_size: int = 5, hidden_size: int = 32):
        super().__init__()
        self.dims = FeatureDimensions()
        self.encodings = EncodingMaps()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTM编码器
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.1)

        # 滑动窗口提取器
        self.sliding_window_extractor = SlidingWindowExtractor()

        # 模式识别器
        self.pattern_recognizer = PatternRecognizer()

        print(f"EnhancedSequenceFeatureEncoder初始化 - 输入维度: {input_size}, 隐藏维度: {hidden_size}")

    def forward(self, nodes: List[Node]) -> torch.Tensor:
        """
        编码时序特征

        Args:
            nodes: 节点列表

        Returns:
            编码后的时序特征 [N, hidden_size]
        """
        sequence_data = []

        for node in nodes:
            # 提取基础时序数据
            sequences = self._extract_comprehensive_sequences(node)
            sequence_data.append(sequences)

        # 转换为张量
        sequence_tensor = torch.tensor(sequence_data, dtype=torch.float32)

        # LSTM编码
        _, (hidden, _) = self.lstm(sequence_tensor)
        result = hidden[-1]  # 取最后一层的隐藏状态

        print(f"EnhancedSequenceFeatureEncoder输出维度: {result.shape}")
        return result

    def _extract_comprehensive_sequences(self, node: Node) -> List[List[float]]:
        """提取全面的时序数据"""
        sequences = []
        max_len = self.dims.MAX_SEQUENCE_LENGTH

        # 模拟生成历史数据，之后需要根据真实历史数据进行修改
        try:
            # 提取多个时序维度的数据
            time_series_features = self._get_time_series_features(node)

            # 生成更真实的历史序列
            for i in range(max_len):
                # 添加时间依赖性
                time_factor = i / max_len

                # 添加趋势
                trend_factor = time_factor * 0.15  # 更强的趋势

                # 添加周期性（日周期和周周期）
                daily_cycle = 0.1 * np.sin(2 * np.pi * i / 24)  # 日周期
                weekly_cycle = 0.05 * np.sin(2 * np.pi * i / (24 * 7))  # 周周期

                # 添加随机噪声
                noise_factor = 0.03 * np.random.normal(0, 1)

                # 添加突发事件
                spike_probability = 0.05  # 5%概率出现突发
                spike_factor = 0.0
                if np.random.random() < spike_probability:
                    spike_factor = 0.3 * np.random.random()

                # 合成每个时间点的数据
                point_data = []
                for base_value in time_series_features:
                    # 应用所有因子
                    total_factor = 1 + trend_factor + daily_cycle + weekly_cycle + noise_factor + spike_factor
                    new_value = max(0, base_value * total_factor)
                    point_data.append(new_value)

                sequences.append(point_data)

        except Exception as e:
            print(f"Warning: Error extracting comprehensive sequences: {e}")
            sequences = [[0.0] * self.input_size for _ in range(max_len)]

        return sequences

    def _get_time_series_features(self, node: Node) -> List[float]:
        """获取时序特征的基准值"""
        features = []

        try:
            da = node.DynamicAttributes

            # CPU和内存使用率
            features.append(getattr(da.Compute, 'CPUUsage', 0))
            features.append(getattr(da.Compute, 'MemUsage', 0))

            # 网络延迟
            features.append(getattr(da.Network, 'AvgLatency', 0))

            # 交易频率
            features.append(getattr(da.Transactions, 'Frequency', 0))

            # 声誉分数
            features.append(getattr(da.Reputation, 'ReputationScore', 0))

        except Exception as e:
            print(f"Warning: Error getting time series features: {e}")
            features = [0.0] * self.input_size

        # 确保维度正确
        while len(features) < self.input_size:
            features.append(0.0)

        return features[:self.input_size]

class SlidingWindowExtractor:
    """滑动窗口统计量提取器"""

    def __init__(self):
        self.window_sizes = [5, 10, 20]  # 多种窗口尺度
        self.statistics = [
            'mean', 'std', 'min', 'max', 'median',
            'q25', 'q75', 'skewness', 'kurtosis', 'range'
        ]

    def extract_sliding_window_features(self, time_series: List[float]) -> List[float]:
        """
        提取滑动窗口统计特征

        Args:
            time_series: 时序数据

        Returns:
            滑动窗口特征列表
        """
        features = []

        if not time_series or len(time_series) < 5:
            # 返回零特征
            total_features = len(self.window_sizes) * len(self.statistics)
            return [0.0] * total_features

        time_series = np.array(time_series)

        for window_size in self.window_sizes:
            window_features = self._extract_window_statistics(time_series, window_size)
            features.extend(window_features)

        return features

    def _extract_window_statistics(self, time_series: np.ndarray, window_size: int) -> List[float]:
        """提取单个窗口尺度的统计特征"""
        features = []

        if len(time_series) < window_size:
            return [0.0] * len(self.statistics)

        # 计算滑动窗口统计量
        windowed_stats = []

        for i in range(len(time_series) - window_size + 1):
            window = time_series[i:i + window_size]

            stats = {
                'mean': np.mean(window),
                'std': np.std(window),
                'min': np.min(window),
                'max': np.max(window),
                'median': np.median(window),
                'q25': np.percentile(window, 25),
                'q75': np.percentile(window, 75),
                'skewness': self._compute_skewness(window),
                'kurtosis': self._compute_kurtosis(window),
                'range': np.max(window) - np.min(window)
            }

            windowed_stats.append(stats)

        # 对所有窗口的统计量再次统计
        for stat_name in self.statistics:
            values = [ws[stat_name] for ws in windowed_stats]
            if values:
                # 取均值作为该统计量的代表值
                features.append(float(np.mean(values)))
            else:
                features.append(0.0)

        return features

    def _compute_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        if len(data) < 3:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0

        return float(np.mean(((data - mean) / std) ** 3))

    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        if len(data) < 4:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0

        return float(np.mean(((data - mean) / std) ** 4)) - 3.0

    def extract_pattern_features(self, time_series: List[float]) -> Dict[str, float]:
        """提取高级模式特征"""
        if not time_series or len(time_series) < 5:
            return {
                'periodicity': 0.0,
                'stability': 0.0,
                'anomaly_score': 0.0,
                'auto_correlation': 0.0
            }

        features = {}

        # 周期性检测（简化版）
        features['periodicity'] = self._detect_periodicity(time_series)

        # 稳定性度量
        features['stability'] = self._compute_stability(time_series)

        # 异常检测分数
        features['anomaly_score'] = self._compute_anomaly_score(time_series)

        # 自相关性
        features['auto_correlation'] = self._compute_autocorrelation(time_series)

        return features

    def _detect_periodicity(self, time_series: List[float]) -> float:
        """检测周期性"""
        if len(time_series) < 10:
            return 0.0

        # 简化的周期性检测：计算自相关
        correlations = []
        for lag in range(1, min(len(time_series) // 2, 20)):
            corr = self._compute_lag_correlation(time_series, lag)
            correlations.append(abs(corr))

        return float(max(correlations)) if correlations else 0.0

    def _compute_stability(self, time_series: List[float]) -> float:
        """计算稳定性"""
        if len(time_series) < 2:
            return 1.0

        # 使用变异系数的倒数作为稳定性指标
        mean_val = np.mean(time_series)
        std_val = np.std(time_series)

        if mean_val == 0:
            return 1.0 if std_val == 0 else 0.0

        cv = std_val / abs(mean_val)  # 变异系数
        stability = 1.0 / (1.0 + cv)  # 稳定性

        return float(stability)

    def _compute_anomaly_score(self, time_series: List[float]) -> float:
        """计算异常分数"""
        if len(time_series) < 3:
            return 0.0

        # 使用IQR方法检测异常
        q25 = np.percentile(time_series, 25)
        q75 = np.percentile(time_series, 75)
        iqr = q75 - q25

        if iqr == 0:
            return 0.0

        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        outliers = [x for x in time_series if x < lower_bound or x > upper_bound]
        anomaly_ratio = len(outliers) / len(time_series)

        return float(anomaly_ratio)

    def _compute_autocorrelation(self, time_series: List[float]) -> float:
        """计算自相关性"""
        if len(time_series) < 3:
            return 0.0

        # 计算lag=1的自相关
        return self._compute_lag_correlation(time_series, 1)

    def _compute_lag_correlation(self, time_series: List[float], lag: int) -> float:
        """计算指定滞后的相关性"""
        if len(time_series) <= lag:
            return 0.0

        x = np.array(time_series[:-lag])
        y = np.array(time_series[lag:])

        if len(x) == 0 or np.std(x) == 0 or np.std(y) == 0:
            return 0.0

        correlation = np.corrcoef(x, y)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0

class PatternRecognizer(nn.Module):
    """模式识别器"""

    def __init__(self, sequence_length: int = 50):
        super().__init__()
        self.sequence_length = sequence_length

        # 简单的1D卷积网络用于模式识别
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 8)  # 输出8维模式特征

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        识别时序模式

        Args:
            sequences: [batch_size, seq_len, feature_dim]

        Returns:
            模式特征 [batch_size, 8]
        """
        # 对每个特征维度单独进行模式识别，然后平均
        batch_size, seq_len, feature_dim = sequences.shape
        pattern_features = []

        for i in range(feature_dim):
            # 提取单个特征维度 [batch_size, seq_len]
            single_feature = sequences[:, :, i].unsqueeze(1)  # [batch_size, 1, seq_len]

            # 卷积操作
            x = torch.relu(self.conv1(single_feature))
            x = torch.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)  # [batch_size, 32]
            x = self.fc(x)  # [batch_size, 8]

            pattern_features.append(x)

        # 平均所有特征维度的模式特征
        result = torch.stack(pattern_features, dim=0).mean(dim=0)

        return result

def main():
    """测试增强的时序特征提取器"""
    from nodeInitialize import load_nodes_from_csv

    print("测试增强的时序特征提取器...")

    # 加载测试数据
    nodes = load_nodes_from_csv("small_samples.csv")

    if nodes:
        # 初始化提取器
        encoder = EnhancedSequenceFeatureEncoder()

        # 提取特征
        features = encoder([nodes[0]])
        print(f"输出特征维度: {features.shape}")
        print(f"特征值样例: {features[0][:10].tolist()}")

        # 测试滑动窗口提取器
        extractor = SlidingWindowExtractor()
        test_series = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2]

        window_features = extractor.extract_sliding_window_features(test_series)
        print(f"滑动窗口特征数量: {len(window_features)}")

        pattern_features = extractor.extract_pattern_features(test_series)
        print(f"模式特征: {pattern_features}")

    print("测试完成!")

if __name__ == "__main__":
    main()