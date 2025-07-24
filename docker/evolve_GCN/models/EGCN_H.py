"""
EvolveGCN-H模型实现
基于EvolveGCN-O的增强版本，支持异构图演化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Optional, Tuple, List


class EvolveGCNH(nn.Module):
    """
    EvolveGCN-H: 异构图演化神经网络
    基于EvolveGCN-O架构，增加了异构节点和边的处理能力
    """
    
    def __init__(self, args=None, input_dim=None, hidden_dim=None, output_dim=None, num_layers=2, dropout=0.1):
        super(EvolveGCNH, self).__init__()
        
        # 支持两种初始化方式：args对象或直接参数
        if args is not None:
            self.input_dim = getattr(args, 'feats_per_node', 64)
            self.hidden_dim = getattr(args, 'layer_1_feats', 32) 
            self.output_dim = getattr(args, 'num_classes', 8)
            self.num_layers = getattr(args, 'num_layers', 2)
            self.dropout = getattr(args, 'dropout', 0.1)
        else:
            self.input_dim = input_dim or 64
            self.hidden_dim = hidden_dim or 32
            self.output_dim = output_dim or 8
            self.num_layers = num_layers
            self.dropout = dropout
        
        # 特征投影层
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # 多层GCN结构
        self.gcn_layers = nn.ModuleList()
        self.weight_updaters = nn.ModuleList()
        
        # 第一层
        self.gcn_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
        self.weight_updaters.append(nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True))
        
        # 中间层
        for _ in range(self.num_layers - 2):
            self.gcn_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
            self.weight_updaters.append(nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True))
        
        # 输出层
        if self.num_layers > 1:
            self.gcn_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
            self.weight_updaters.append(nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True))
        
        # 输出投影
        self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)
        
        # 节点类型嵌入（异构支持）
        self.node_type_embedding = nn.Embedding(10, self.hidden_dim // 4)  # 支持10种节点类型
        
        # 边类型嵌入（异构支持）
        self.edge_type_embedding = nn.Embedding(5, self.hidden_dim // 4)   # 支持5种边类型
        
        # 时序注意力机制
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # LSTM隐藏状态
        self.hidden_states = [None] * num_layers
        
    def reset_parameters(self):
        """重置模型参数"""
        for layer in self.gcn_layers:
            layer.reset_parameters()
        
        for updater in self.weight_updaters:
            for name, param in updater.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
    
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor,
                node_types: Optional[torch.Tensor] = None,
                edge_types: Optional[torch.Tensor] = None,
                temporal_step: int = 0) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            node_types: 节点类型 [num_nodes]
            edge_types: 边类型 [num_edges]
            temporal_step: 时间步
            
        Returns:
            output: 输出特征 [num_nodes, output_dim]
            embeddings: 各层嵌入列表
        """
        # 输入投影
        h = self.input_projection(x)
        
        # 添加节点类型信息（如果提供）
        if node_types is not None:
            node_type_emb = self.node_type_embedding(node_types)
            # 扩展到匹配hidden_dim
            node_type_emb = F.pad(node_type_emb, (0, h.size(1) - node_type_emb.size(1)))
            h = h + node_type_emb
        
        embeddings = []
        
        # 多层GCN传播
        for layer_idx in range(self.num_layers):
            # 时序权重更新
            h_seq = h.unsqueeze(0)  # [1, num_nodes, hidden_dim]
            
            if self.hidden_states[layer_idx] is None:
                self.hidden_states[layer_idx] = (
                    torch.zeros(1, h.size(0), self.hidden_dim, device=h.device),
                    torch.zeros(1, h.size(0), self.hidden_dim, device=h.device)
                )
            
            # LSTM更新权重
            updated_h, self.hidden_states[layer_idx] = self.weight_updaters[layer_idx](
                h_seq, self.hidden_states[layer_idx]
            )
            updated_h = updated_h.squeeze(0)  # [num_nodes, hidden_dim]
            
            # GCN传播
            h = self.gcn_layers[layer_idx](updated_h, edge_index)
            
            # 层归一化
            h = self.layer_norms[layer_idx](h)
            
            # 激活函数
            h = F.relu(h)
            
            # Dropout
            h = self.dropout_layer(h)
            
            embeddings.append(h.clone())
        
        # 时序注意力（如果有多个时间步）
        if temporal_step > 0 and len(embeddings) > 1:
            # 将嵌入堆叠为序列
            embedding_seq = torch.stack(embeddings[-3:], dim=1)  # 最近3层
            attended_h, _ = self.temporal_attention(
                embedding_seq, embedding_seq, embedding_seq
            )
            h = attended_h.mean(dim=1)  # 平均池化
        
        # 输出投影
        output = self.output_projection(h)
        
        return output, embeddings
    
    def evolve_weights(self, performance_feedback: torch.Tensor):
        """
        基于性能反馈演化权重
        
        Args:
            performance_feedback: 性能反馈信号 [batch_size, feedback_dim]
        """
        # 简化的权重演化机制
        for layer_idx in range(self.num_layers):
            if self.hidden_states[layer_idx] is not None:
                h_state, c_state = self.hidden_states[layer_idx]
                
                # 基于反馈调整隐藏状态
                feedback_effect = performance_feedback.mean(dim=0).unsqueeze(0).unsqueeze(0)
                feedback_effect = F.pad(feedback_effect, (0, h_state.size(2) - feedback_effect.size(2)))
                
                # 轻微调整隐藏状态
                adjustment_factor = 0.01  # 小的调整因子
                h_state = h_state + adjustment_factor * feedback_effect
                c_state = c_state + adjustment_factor * feedback_effect
                
                self.hidden_states[layer_idx] = (h_state, c_state)
    
    def get_node_embeddings(self) -> List[torch.Tensor]:
        """获取各层节点嵌入"""
        return [h.detach().clone() for h in self.hidden_states if h is not None]
    
    def reset_hidden_states(self):
        """重置LSTM隐藏状态"""
        self.hidden_states = [None] * self.num_layers


class EvolveGCNHSharding(nn.Module):
    """
    基于EvolveGCN-H的分片模块
    """
    
    def __init__(self, 
                 node_dim: int, 
                 hidden_dim: int, 
                 num_shards: int,
                 num_layers: int = 3):
        super(EvolveGCNHSharding, self).__init__()
        
        self.num_shards = num_shards
        
        # 核心EvolveGCN-H模型
        self.evolve_gcn = EvolveGCNH(
            input_dim=node_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # 分片分类器
        self.shard_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_shards),
            nn.Softmax(dim=-1)
        )
        
        # 负载均衡损失权重
        self.balance_weight = 0.1
        
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor,
                node_types: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播进行分片预测
        
        Args:
            x: 节点特征
            edge_index: 边索引
            node_types: 节点类型
            
        Returns:
            shard_probs: 分片概率 [num_nodes, num_shards]
            node_embeddings: 节点嵌入 [num_nodes, hidden_dim]
        """
        # 获取节点嵌入
        node_embeddings, _ = self.evolve_gcn(x, edge_index, node_types)
        
        # 预测分片分配
        shard_probs = self.shard_classifier(node_embeddings)
        
        return shard_probs, node_embeddings
    
    def compute_loss(self, 
                     shard_probs: torch.Tensor, 
                     true_shards: torch.Tensor,
                     edge_index: torch.Tensor) -> torch.Tensor:
        """
        计算分片损失（包含负载均衡项）
        
        Args:
            shard_probs: 预测的分片概率
            true_shards: 真实分片标签
            edge_index: 边索引
            
        Returns:
            total_loss: 总损失
        """
        # 分类损失
        classification_loss = F.cross_entropy(shard_probs, true_shards)
        
        # 负载均衡损失
        shard_counts = torch.sum(shard_probs, dim=0)
        ideal_count = shard_probs.size(0) / self.num_shards
        balance_loss = torch.var(shard_counts) / (ideal_count ** 2)
        
        # 跨分片边损失
        predicted_shards = torch.argmax(shard_probs, dim=1)
        cross_shard_edges = (predicted_shards[edge_index[0]] != predicted_shards[edge_index[1]]).float()
        cross_shard_loss = torch.mean(cross_shard_edges)
        
        # 总损失
        total_loss = (classification_loss + 
                     self.balance_weight * balance_loss + 
                     0.05 * cross_shard_loss)
        
        return total_loss
    
    def predict_shards(self, 
                      x: torch.Tensor, 
                      edge_index: torch.Tensor,
                      node_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        预测节点分片分配
        
        Returns:
            predicted_shards: 预测的分片ID [num_nodes]
        """
        with torch.no_grad():
            shard_probs, _ = self.forward(x, edge_index, node_types)
            predicted_shards = torch.argmax(shard_probs, dim=1)
        
        return predicted_shards
