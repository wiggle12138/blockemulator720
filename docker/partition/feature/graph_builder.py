"""
异构图构建器
"""
import torch
import numpy as np
import json
from typing import List, Dict, Tuple, Any
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
    from config import RelationTypes, NodeTypes
except ImportError as e:
    raise ImportError(f"config导入失败: {e}")

class HeterogeneousGraphBuilder:
    def __init__(self):

        self.relation_types = {
            'compete': RelationTypes.COMPETE,
            'serve': RelationTypes.SERVE,
            'validate': RelationTypes.VALIDATE,
            # 'cooperate': RelationTypes.COOPERATE,
            # 'connect': RelationTypes.CONNECT,
            # 'communicate': RelationTypes.COMMUNICATE
        }
        self.node_types = NodeTypes.TYPES
        
        # 新增：保存邻接矩阵信息
        self.adjacency_info = {}

    def build_graph(self, nodes: List[Node]) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建有向异构图"""
        edges = []
        edge_types = []
        relation_details = []

        # 创建节点ID到索引的映射
        node_id_to_idx = {i: i for i in range(len(nodes))}

        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i == j:  # 避免自环
                    continue

                # 检查有向关系 (i → j)
                relation_type, relation_reason = self._determine_relation_type_with_reason(node1, node2, i, j)

                if relation_type is not None:
                    edges.append([i, j])  # 有向边：i → j
                    edge_types.append(relation_type)

                    # 记录关系详情
                    relation_details.append({
                        'source': i,
                        'target': j,
                        'relation_type': relation_type,
                        'relation_name': self._get_relation_name(relation_type),
                        'reason': relation_reason,
                        'source_type': getattr(node1.HeterogeneousType, 'NodeType', 'unknown'),
                        'target_type': getattr(node2.HeterogeneousType, 'NodeType', 'unknown')
                    })

        if not edges:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_type = torch.tensor(edge_types, dtype=torch.long)

        # 保存邻接矩阵信息
        self._save_adjacency_info(nodes, edge_index, edge_type, relation_details)

        return edge_index, edge_type

    def _determine_relation_type_with_reason(self, node1: Node, node2: Node, idx1: int, idx2: int) -> Tuple[int, str]:
        """
        根据节点类型和属性确定关系类型，并返回原因

        Args:
            node1, node2: 两个节点
            idx1, idx2: 节点索引

        Returns:
            (关系类型ID, 关系原因)，如果无关系则返回(None, None)
        """

        type1 = getattr(node1.HeterogeneousType, 'NodeType', 'unknown')
        type2 = getattr(node2.HeterogeneousType, 'NodeType', 'unknown')

        # 矿工间竞争关系
        if type1 == 'miner' and type2 == 'miner':
            return self.relation_types['compete'], "miner_competition"

        # 存储节点服务轻节点
        elif type1 == 'storage' and type2 == 'light_node':
            return self.relation_types['serve'], "storage_service"

        # 验证关系
        elif type1 == 'validator' and type2 == 'full_node':
            return self.relation_types['validate'], "validation_relation"

        # 验证者协作关系
        # elif type1 == 'validator' and type2 == 'validator':
        #     return self.relation_types['cooperate'], "validator_cooperation"


        # 基于地理位置的连接关系
        # elif self._are_geographically_close(node1, node2):
        #     return self.relation_types['connect'], "geographic_proximity"

        # 基于网络拓扑的通信关系
        # elif self._have_communication_link(node1, node2):
        #     return self.relation_types['communicate'], "communication_link"

        return None, None

    def _get_relation_name(self, relation_type: int) -> str:
        """获取关系类型名称"""
        for name, value in self.relation_types.items():
            if value == relation_type:
                return name
        return "unknown"

    def _save_adjacency_info(self, nodes: List[Node], edge_index: torch.Tensor, 
                       edge_type: torch.Tensor, relation_details: List[Dict]):
        """保存有向图的邻接矩阵信息"""
        num_nodes = len(nodes)

        # 构建有向邻接矩阵
        adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        relation_matrix = torch.full((num_nodes, num_nodes), -1, dtype=torch.long)

        if edge_index.size(1) > 0:
            # 有向图：只在指定方向填充
            for i in range(edge_index.size(1)):
                src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
                rel_type = edge_type[i].item()

                # 有向边：只填充 src → tgt 方向
                adjacency_matrix[src, tgt] = 1.0
                relation_matrix[src, tgt] = rel_type

        # 保存信息（添加有向图标识）
        self.adjacency_info = {
            'num_nodes': num_nodes,
            'num_edges': edge_index.size(1),
            'edge_index_coo': edge_index.cpu().numpy(),
            'edge_type': edge_type.cpu().numpy(),
            'adjacency_matrix_dense': adjacency_matrix.cpu().numpy(),
            'relation_matrix': relation_matrix.cpu().numpy(),
            'relation_details': relation_details,
            'relation_statistics': self._compute_relation_statistics(edge_type, relation_details),
            'node_degree_statistics': self._compute_directed_degree_statistics(adjacency_matrix),  # 修改为有向度统计
            'node_types': [getattr(node.HeterogeneousType, 'NodeType', 'unknown') for node in nodes],
            'graph_type': 'directed'  # 新增：标识图类型
        }

    def _compute_relation_statistics(self, edge_type: torch.Tensor, relation_details: List[Dict]) -> Dict:
        """计算关系统计信息"""
        if len(relation_details) == 0:
            return {'total_relations': 0, 'relation_counts': {}, 'relation_percentages': {}}
        
        # 统计各类关系数量
        relation_counts = {}
        for detail in relation_details:
            rel_name = detail['relation_name']
            relation_counts[rel_name] = relation_counts.get(rel_name, 0) + 1
        
        # 计算百分比
        total = len(relation_details)
        relation_percentages = {k: (v / total) * 100 for k, v in relation_counts.items()}
        
        return {
            'total_relations': total,
            'relation_counts': relation_counts,
            'relation_percentages': relation_percentages
        }

    def _compute_directed_degree_statistics(self, adjacency_matrix: torch.Tensor) -> Dict:
        """计算有向图的度统计信息"""
        # 出度和入度
        out_degrees = adjacency_matrix.sum(dim=1).cpu().numpy()  # 每行和
        in_degrees = adjacency_matrix.sum(dim=0).cpu().numpy()   # 每列和
        total_degrees = out_degrees + in_degrees

        return {
            'out_degrees': out_degrees.tolist(),
            'in_degrees': in_degrees.tolist(),
            'total_degrees': total_degrees.tolist(),
            'average_out_degree': float(np.mean(out_degrees)),
            'average_in_degree': float(np.mean(in_degrees)),
            'max_out_degree': int(np.max(out_degrees)),
            'max_in_degree': int(np.max(in_degrees)),
            'min_out_degree': int(np.min(out_degrees)),
            'min_in_degree': int(np.min(in_degrees)),
            'out_degree_std': float(np.std(out_degrees)),
            'in_degree_std': float(np.std(in_degrees))
        }

    def save_adjacency_matrices(self, base_name: str = "adjacency"):
        """
        保存邻接矩阵到文件

        Args:
            base_name: 基础文件名
        """
        if not hasattr(self, 'adjacency_info') or not self.adjacency_info:
            print("警告: 没有邻接矩阵信息可保存")
            return

        print(f"保存原始邻接矩阵信息...")
        
        # 1. 保存为 .pt 格式（PyTorch 张量）
        torch_data = {
            'edge_index': torch.from_numpy(self.adjacency_info['edge_index_coo']),
            'edge_type': torch.from_numpy(self.adjacency_info['edge_type']),
            'adjacency_matrix': torch.from_numpy(self.adjacency_info['adjacency_matrix_dense']),
            'relation_matrix': torch.from_numpy(self.adjacency_info['relation_matrix']),
            'metadata': {
                'num_nodes': self.adjacency_info['num_nodes'],
                'num_edges': self.adjacency_info['num_edges'],
                'node_types': self.adjacency_info['node_types']
            }
        }
        
        torch.save(torch_data, f"{base_name}_raw.pt")
        print(f"  ✓ PyTorch格式: {base_name}_raw.pt")

        # 2. 保存为 JSON 格式（详细信息）
        json_data = {
            'graph_metadata': {
                'num_nodes': self.adjacency_info['num_nodes'],
                'num_edges': self.adjacency_info['num_edges'],
                'node_types': self.adjacency_info['node_types']
            },
            'relation_statistics': self.adjacency_info['relation_statistics'],
            'node_degree_statistics': self.adjacency_info['node_degree_statistics'],
            'relation_details': self.adjacency_info['relation_details'][:100],  # 只保存前100个关系详情
            'edge_list': self.adjacency_info['edge_index_coo'].tolist(),
            'edge_types': self.adjacency_info['edge_type'].tolist()
        }
        
        with open(f"{base_name}_info.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ JSON信息文件: {base_name}_info.json")

        # 3. 保存为 NumPy 格式
        np.savez(f"{base_name}_matrices.npz",
                adjacency_matrix=self.adjacency_info['adjacency_matrix_dense'],
                relation_matrix=self.adjacency_info['relation_matrix'],
                edge_index=self.adjacency_info['edge_index_coo'],
                edge_type=self.adjacency_info['edge_type'])
        print(f"  ✓ NumPy格式: {base_name}_matrices.npz")
        
        print(f"原始邻接矩阵保存完成 - 共 {self.adjacency_info['num_edges']} 条边")


    # def _are_geographically_close(self, node1: Node, node2: Node, threshold: float = 1000.0) -> bool:
    #     """
    #     判断两个节点在地理位置上是否接近
    #
    #     Args:
    #         node1, node2: 两个节点
    #         threshold: 距离阈值（公里）
    #
    #     Returns:
    #         是否地理位置接近
    #     """
    #     try:
    #         # 修正字段名：使用实际的字段名 NetworkTopology.GeoLocation
    #         geo1 = node1.NetworkTopology.GeoLocation
    #         geo2 = node2.NetworkTopology.GeoLocation
    #
    #         # 简化的地理位置判断：基于Region比较
    #         region1 = getattr(geo1, 'Region', '')
    #         region2 = getattr(geo2, 'Region', '')
    #
    #         # 如果在同一地区，认为地理位置接近
    #         return region1 == region2 and region1 != ''
    #
    #     except:
    #         return False
    #
    # def _have_communication_link(self, node1: Node, node2: Node) -> bool:
    #     """
    #     判断两个节点是否有通信链接
    #
    #     Args:
    #         node1, node2: 两个节点
    #
    #     Returns:
    #         是否有通信链接
    #     """
    #     try:
    #         # 修正字段名：使用实际的字段名 NetworkTopology.Connections
    #         connections1 = node1.NetworkTopology.Connections
    #         connections2 = node2.NetworkTopology.Connections
    #
    #         # 基于连接数量判断：如果两个节点的连接数都比较高，认为它们可能有通信链接
    #         active_conn1 = getattr(connections1, 'ActiveConn', 0)
    #         active_conn2 = getattr(connections2, 'ActiveConn', 0)
    #
    #         # 简单的启发式：活跃连接数都大于某个阈值
    #         return active_conn1 > 5 and active_conn2 > 5
    #
    #     except:
    #         return False

    # def extract_neighbor_features(self, neighbors_info: Dict[str, Any], max_neighbors: int) -> np.ndarray:
    #     """
    #     提取邻居节点特征
    #
    #     Args:
    #         neighbors_info: 邻居信息字典
    #         max_neighbors: 最大邻居数量
    #
    #     Returns:
    #         邻居特征矩阵 [max_neighbors, feature_dim]
    #     """
    #     neighbor_features = []
    #     neighbor_list = list(neighbors_info.items()) if neighbors_info else []
    #
    #     for i in range(max_neighbors):
    #         if i < len(neighbor_list):
    #             neighbor_id, neighbor_info = neighbor_list[i]
    #             features = self._extract_single_neighbor_features(neighbor_info)
    #         else:
    #             features = [0.0] * 10  # 零填充
    #
    #         neighbor_features.append(features)
    #
    #     return np.array(neighbor_features)

    # def _extract_single_neighbor_features(self, neighbor_info: Any) -> List[float]:
    #     """
    #     提取单个邻居的特征
    #
    #     Args:
    #         neighbor_info: 邻居信息
    #
    #     Returns:
    #         特征列表
    #     """
    #     features = [0.0] * 10
    #
    #     if isinstance(neighbor_info, dict):
    #         # 提取权重信息
    #         if 'weight' in neighbor_info:
    #             features[0] = float(neighbor_info['weight'])
    #         if 'strength' in neighbor_info:
    #             features[1] = float(neighbor_info['strength'])
    #         if 'distance' in neighbor_info:
    #             features[2] = float(neighbor_info['distance'])
    #         if 'bandwidth' in neighbor_info:
    #             features[3] = float(neighbor_info['bandwidth'])
    #         # 可以继续添加更多特征
    #
    #     return features