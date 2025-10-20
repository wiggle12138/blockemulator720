"""
特征融合模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
from .config import FeatureDimensions

class FeatureFusionPipeline(nn.Module):
    """特征融合流水线"""

    def __init__(self, classic_dim: int = 128, graph_dim: int = 96, fused_dim: int = 256):
        super().__init__()
        self.classic_dim = classic_dim
        self.graph_dim = graph_dim
        self.fused_dim = fused_dim

        # 多头注意力融合
        self.multihead_fusion = MultiHeadAttentionFusion(
            classic_dim=classic_dim,
            graph_dim=graph_dim,
            hidden_dim=fused_dim // 2,
            num_heads=8
        )

        # 对比学习模块
        self.contrastive_learner = ContrastiveLearner(
            classic_dim=classic_dim,
            graph_dim=graph_dim,
            projection_dim=fused_dim // 4
        )

        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(classic_dim + graph_dim + fused_dim // 2, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fused_dim, fused_dim)
        )

        print(f"FeatureFusionPipeline初始化 - Classic: {classic_dim}, Graph: {graph_dim}, Fused: {fused_dim}")

    def forward(self, f_classic: torch.Tensor, f_graph: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        特征融合前向传播

        Args:
            f_classic: [N, classic_dim]
            f_graph: [N, graph_dim]

        Returns:
            f_fused: [N, fused_dim]
            contrastive_loss: 对比学习损失
        """
        batch_size = f_classic.size(0)

        # 1. 多头注意力融合
        attention_fused = self.multihead_fusion(f_classic, f_graph)  # [N, fused_dim//2]

        # 2. 对比学习
        contrastive_loss = self.contrastive_learner(f_classic, f_graph)

        # 3. 最终融合
        combined = torch.cat([f_classic, f_graph, attention_fused], dim=1)
        f_fused = self.final_fusion(combined)  # [N, fused_dim]

        return f_fused, contrastive_loss

class MultiHeadAttentionFusion(nn.Module):
    """多头注意力特征融合"""

    def __init__(self, classic_dim: int, graph_dim: int, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # 投影层
        self.classic_proj = nn.Linear(classic_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)

        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, f_classic: torch.Tensor, f_graph: torch.Tensor) -> torch.Tensor:
        """
        多头注意力融合

        Args:
            f_classic: [N, classic_dim]
            f_graph: [N, graph_dim]

        Returns:
            fused_features: [N, hidden_dim]
        """
        # 投影到统一维度
        classic_proj = self.classic_proj(f_classic)  # [N, hidden_dim]
        graph_proj = self.graph_proj(f_graph)       # [N, hidden_dim]

        # 添加序列维度用于注意力计算
        classic_seq = classic_proj.unsqueeze(1)  # [N, 1, hidden_dim]
        graph_seq = graph_proj.unsqueeze(1)      # [N, 1, hidden_dim]

        # 将两种特征作为键值对
        key_value = torch.cat([classic_seq, graph_seq], dim=1)  # [N, 2, hidden_dim]

        # 使用经典特征作为查询
        query = classic_seq  # [N, 1, hidden_dim]

        # 多头注意力
        attn_output, _ = self.multihead_attn(
            query=query,
            key=key_value,
            value=key_value
        )  # [N, 1, hidden_dim]

        # 移除序列维度并投影
        fused = attn_output.squeeze(1)  # [N, hidden_dim]
        output = self.output_proj(fused)

        return output

class ContrastiveLearner(nn.Module):
    """对比学习模块"""

    def __init__(self, classic_dim: int, graph_dim: int, projection_dim: int = 64, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

        # 投影头
        self.classic_projector = nn.Sequential(
            nn.Linear(classic_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

        self.graph_projector = nn.Sequential(
            nn.Linear(graph_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, f_classic: torch.Tensor, f_graph: torch.Tensor) -> float:
        """
        计算对比学习损失

        Args:
            f_classic: [N, classic_dim]
            f_graph: [N, graph_dim]

        Returns:
            contrastive_loss: 对比学习损失
        """
        # 投影到对比学习空间
        z_classic = self.classic_projector(f_classic)  # [N, projection_dim]
        z_graph = self.graph_projector(f_graph)        # [N, projection_dim]

        # L2标准化
        z_classic = F.normalize(z_classic, dim=1)
        z_graph = F.normalize(z_graph, dim=1)

        # 计算相似度矩阵
        similarity = torch.matmul(z_classic, z_graph.T) / self.temperature  # [N, N]

        # 对比学习损失（InfoNCE）
        batch_size = similarity.size(0)
        labels = torch.arange(batch_size, device=similarity.device)

        # 对称损失
        loss_classic = F.cross_entropy(similarity, labels)
        loss_graph = F.cross_entropy(similarity.T, labels)

        contrastive_loss = (loss_classic + loss_graph) / 2

        return contrastive_loss.item()

class AdaptiveFeatureFusion(nn.Module):
    """自适应特征融合"""

    def __init__(self, classic_dim: int, graph_dim: int, fused_dim: int):
        super().__init__()

        # 特征重要性评估
        self.importance_net = nn.Sequential(
            nn.Linear(classic_dim + graph_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, 2),  # 输出两个权重：classic和graph
            nn.Softmax(dim=1)
        )

        # 特征变换
        self.classic_transform = nn.Linear(classic_dim, fused_dim)
        self.graph_transform = nn.Linear(graph_dim, fused_dim)

        # 最终融合
        self.final_layer = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU()
        )

    def forward(self, f_classic: torch.Tensor, f_graph: torch.Tensor) -> torch.Tensor:
        """
        自适应特征融合

        Args:
            f_classic: [N, classic_dim]
            f_graph: [N, graph_dim]

        Returns:
            fused_features: [N, fused_dim]
        """
        # 评估特征重要性
        combined_input = torch.cat([f_classic, f_graph], dim=1)
        weights = self.importance_net(combined_input)  # [N, 2]

        classic_weight = weights[:, 0:1]  # [N, 1]
        graph_weight = weights[:, 1:2]    # [N, 1]

        # 变换特征
        classic_transformed = self.classic_transform(f_classic)  # [N, fused_dim]
        graph_transformed = self.graph_transform(f_graph)       # [N, fused_dim]

        # 加权融合
        fused = classic_weight * classic_transformed + graph_weight * graph_transformed

        # 最终处理
        output = self.final_layer(fused)

        return output

def create_fusion_pipeline(classic_dim: int = 128, graph_dim: int = 96, fused_dim: int = 256) -> FeatureFusionPipeline:
    """
    创建特征融合流水线

    Args:
        classic_dim: 经典特征维度
        graph_dim: 图特征维度
        fused_dim: 融合特征维度

    Returns:
        特征融合流水线
    """
    return FeatureFusionPipeline(
        classic_dim=classic_dim,
        graph_dim=graph_dim,
        fused_dim=fused_dim
    )

def main():
    """测试特征融合模块"""
    print("测试特征融合模块...")

    # 模拟数据
    batch_size = 10
    classic_dim = 128
    graph_dim = 96
    fused_dim = 256

    f_classic = torch.randn(batch_size, classic_dim)
    f_graph = torch.randn(batch_size, graph_dim)

    # 创建融合流水线
    fusion_pipeline = create_fusion_pipeline(classic_dim, graph_dim, fused_dim)

    # 测试融合
    f_fused, contrastive_loss = fusion_pipeline(f_classic, f_graph)

    print(f"输入维度 - Classic: {f_classic.shape}, Graph: {f_graph.shape}")
    print(f"输出维度 - Fused: {f_fused.shape}")
    print(f"对比学习损失: {contrastive_loss:.4f}")

    print("测试完成!")

if __name__ == "__main__":
    main()