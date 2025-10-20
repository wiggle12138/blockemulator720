"""
时间卷积网络 - 用于预测最优分片数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvNet(nn.Module):
    """时间卷积网络 - 预测最优分片数"""

    def __init__(self, input_dim=3, hidden_dim=64, max_shards=10):
        super().__init__()
        self.max_shards = max_shards

        # 1D卷积层序列
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim // 2, 1, kernel_size=1)

        # 全连接预测层
        self.predictor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, history_states):
        """
        Args:
            history_states: [batch_size, seq_len, features] 或 [seq_len, features]
        Returns:
            predicted_shards: 预测的分片数
        """
        if history_states.dim() == 2:
            history_states = history_states.unsqueeze(0)  # [1, seq_len, features]

        # 确保数据类型一致
        history_states = history_states.float()

        # 转置为 [batch_size, features, seq_len]
        x = history_states.transpose(1, 2)

        # 时间卷积
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        # 预测分片数
        shard_ratio = self.predictor(x)
        predicted_shards = torch.clamp(
            shard_ratio * self.max_shards + 2,  # 最少2个分片
            min=2, max=self.max_shards
        )

        return predicted_shards.squeeze()