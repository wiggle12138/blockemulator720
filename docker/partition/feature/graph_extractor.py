"""
图特征提取模块
包含GraphStructureEncoder和GraphFeatureExtractor的完整实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
import numpy as np
import sys
import os

# 修复相对导入问题
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 使用安全导入
try:
    from .feature_dimensions import FeatureDimensions
except ImportError:
    try:
        from feature_dimensions import FeatureDimensions
    except ImportError:
        from config import FeatureDimensions

try:
    from .nodeInitialize import Node
except ImportError:
    from nodeInitialize import Node

try:
    from .heterogeneous_graph_builder import HeterogeneousGraphBuilder
except ImportError:
    from heterogeneous_graph_builder import HeterogeneousGraphBuilder

try:
    from .relation_types import RelationTypes
except ImportError:
    from relation_types import RelationTypes

try:
    from torch_geometric.nn import RGCNConv
except ImportError:
    print("Warning: torch_geometric not available, using simplified RGCN implementation")
    # 简化的RGCN实现
    class RGCNConv(nn.Module):
        def __init__(self, in_channels, out_channels, num_relations):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.num_relations = num_relations
            self.weight = nn.Parameter(torch.randn(num_relations, in_channels, out_channels))
            
        def forward(self, x, edge_index, edge_type):
            if edge_index.size(1) == 0:
                return torch.zeros(x.size(0), self.out_channels, device=x.device)
            
            out = torch.zeros(x.size(0), self.out_channels, device=x.device)
            for rel in range(self.num_relations):
                mask = edge_type == rel
                if mask.sum() > 0:
                    rel_edges = edge_index[:, mask]
                    if rel_edges.size(1) > 0:
                        src, dst = rel_edges
                        out[dst] += torch.mm(x[src], self.weight[rel])
            return out


class GraphStructureEncoder(nn.Module):
    """图结构特征编码器"""

    def __init__(self, input_dim: int = 10, output_dim: int = 10):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=2, batch_first=True),
            num_layers=2
        )
        self.graph_builder = HeterogeneousGraphBuilder()
        self.dims = FeatureDimensions()

    def forward(self, nodes: List[Node]) -> torch.Tensor:
        """编码图结构特征"""
        graph_data = []

        for node in nodes:
            neighbors_features = self._extract_neighbor_features(node)
            graph_data.append(neighbors_features)

        graph_tensor = torch.tensor(graph_data, dtype=torch.float32)
        encoded = self.transformer(graph_tensor)
        result = encoded.mean(dim=1)

        print(f"GraphStructureEncoder输出维度: {result.shape}")
        return result

    def _extract_neighbor_features(self, node: Node) -> List[List[float]]:
        """提取邻居特征"""
        try:
            connections = node.NetworkTopology.Connections
            intra_shard_conn = getattr(connections, 'IntraShardConn', 0)
            inter_shard_conn = getattr(connections, 'InterShardConn', 0)
            weighted_degree = getattr(connections, 'WeightedDegree', 0)
            active_conn = getattr(connections, 'ActiveConn', 0)

            neighbor_features = []
            for i in range(self.dims.MAX_NEIGHBORS):
                if i < active_conn:
                    features = [
                        weighted_degree / max(1, active_conn),
                        1.0 if i < intra_shard_conn else 0.0,
                        1.0 if i < inter_shard_conn else 0.0,
                        np.random.random(),
                        np.random.random(),
                        0.0, 0.0, 0.0, 0.0, 0.0
                    ]
                else:
                    features = [0.0] * self.dims.NEIGHBOR_FEATURE_DIM

                # 确保特征维度正确
                while len(features) < self.dims.NEIGHBOR_FEATURE_DIM:
                    features.append(0.0)
                features = features[:self.dims.NEIGHBOR_FEATURE_DIM]
                neighbor_features.append(features)

            return neighbor_features

        except Exception as e:
            print(f"Warning: Error extracting neighbor features: {e}")
            # 返回默认特征
            return [[0.0] * self.dims.NEIGHBOR_FEATURE_DIM for _ in range(self.dims.MAX_NEIGHBORS)]


class GraphFeatureExtractor(nn.Module):
    """图特征提取器（使用RGCN）- 增加邻接矩阵保存功能"""

    def __init__(self):
        super().__init__()
        self.dims = FeatureDimensions()
        self.graph_builder = HeterogeneousGraphBuilder()

        # RGCN layers
        self.rgcn1 = RGCNConv(
            self.dims.CLASSIC_DIM,           # 从配置文件读取
            self.dims.CLASSIC_DIM,           # 从配置文件读取
            RelationTypes.NUM_RELATIONS      # 从配置文件读取
        )
        self.rgcn2 = RGCNConv(
            self.dims.CLASSIC_DIM,           # 从配置文件读取
            self.dims.GRAPH_OUTPUT_DIM,      # 从配置文件读取
            RelationTypes.NUM_RELATIONS      # 从配置文件读取
        )
        self.dropout = nn.Dropout(0.2)
        
        # 保存中间表示
        self.intermediate_representations = {}

    def forward(self, classic_features: torch.Tensor, nodes: List[Node]) -> torch.Tensor:
        """使用RGCN提取图特征"""
        print(f"GraphFeatureExtractor输入维度: {classic_features.shape}")

        # 构建图结构
        edge_index, edge_type = self.graph_builder.build_graph(nodes)

        # 如果没有边，返回零特征
        if edge_index.size(1) == 0:
            print("Warning: No edges found in graph, returning zero features")
            result = torch.zeros(classic_features.size(0), self.dims.GRAPH_OUTPUT_DIM)
            
            # 保存空图信息
            self._save_intermediate_representations(
                input_features=classic_features,
                hidden_features=torch.zeros(classic_features.size(0), self.dims.CLASSIC_DIM),
                output_features=result,
                edge_index=edge_index,
                edge_type=edge_type
            )
            
            print(f"GraphFeatureExtractor输出维度: {result.shape}")
            return result

        # RGCN前向传播
        h = self.rgcn1(classic_features, edge_index, edge_type)
        h = F.relu(h)
        h = self.dropout(h)

        graph_features = self.rgcn2(h, edge_index, edge_type)

        # 保存中间表示
        self._save_intermediate_representations(
            input_features=classic_features,
            hidden_features=h,
            output_features=graph_features,
            edge_index=edge_index,
            edge_type=edge_type
        )

        print(f"GraphFeatureExtractor输出维度: {graph_features.shape}")
        return graph_features

    def _save_intermediate_representations(self, input_features: torch.Tensor, 
                                         hidden_features: torch.Tensor,
                                         output_features: torch.Tensor,
                                         edge_index: torch.Tensor,
                                         edge_type: torch.Tensor):
        """保存中间表示用于分析"""
        self.intermediate_representations = {
            'input_features': input_features.detach().clone(),
            'hidden_features': hidden_features.detach().clone(),
            'output_features': output_features.detach().clone(),
            'edge_index': edge_index.detach().clone(),
            'edge_type': edge_type.detach().clone(),
            'num_nodes': input_features.size(0),
            'num_edges': edge_index.size(1)
        }

    def get_adjacency_info(self) -> Dict[str, Any]:
        """获取邻接矩阵信息"""
        if not self.intermediate_representations:
            return {}
            
        edge_index = self.intermediate_representations['edge_index']
        edge_type = self.intermediate_representations['edge_type']
        num_nodes = self.intermediate_representations['num_nodes']
        
        # 构建邻接矩阵
        adjacency_matrix = torch.zeros(num_nodes, num_nodes)
        if edge_index.size(1) > 0:
            src, dst = edge_index
            adjacency_matrix[src, dst] = 1.0
        
        return {
            'edge_index': edge_index,
            'edge_type': edge_type,
            'adjacency_matrix': adjacency_matrix,
            'num_nodes': num_nodes,
            'num_edges': edge_index.size(1)
        }

    def save_adjacency_matrices(self, filename_prefix: str):
        """保存邻接矩阵到文件"""
        try:
            adjacency_info = self.get_adjacency_info()
            if adjacency_info:
                import os
                os.makedirs('outputs/adjacency_matrices', exist_ok=True)
                
                # 保存邻接矩阵
                torch.save(adjacency_info['adjacency_matrix'], 
                          f'outputs/adjacency_matrices/{filename_prefix}_adjacency.pt')
                
                # 保存边信息
                torch.save({
                    'edge_index': adjacency_info['edge_index'],
                    'edge_type': adjacency_info['edge_type']
                }, f'outputs/adjacency_matrices/{filename_prefix}_edges.pt')
                
                print(f"邻接矩阵保存成功: {filename_prefix}")
            else:
                print("没有邻接矩阵信息可保存")
        except Exception as e:
            print(f"保存邻接矩阵失败: {e}")
