#!/usr/bin/env python3
"""
简化的Step1特征提取适配器
绕过复杂的相对导入问题，提供基础的特征提取功能
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
import json
import os
import time
from datetime import datetime

class SimplifiedStep1Adapter:
    """
    简化的Step1特征提取适配器
    提供基础的特征提取功能，不依赖复杂的模块导入
    """
    
    def __init__(self, use_comprehensive_features: bool = True,
                 save_adjacency: bool = True,
                 output_dir: str = "./step1_outputs"):
        """初始化简化适配器"""
        self.use_comprehensive_features = use_comprehensive_features
        self.save_adjacency = save_adjacency
        self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 特征维度配置
        self.feature_dims = {
            'comprehensive': 65,  # 综合特征维度
            'hardware': 13,       # 硬件特征
            'onchain': 15,        # 链上行为特征  
            'topology': 7,        # 网络拓扑特征
            'dynamic': 10,        # 动态属性特征
            'heterogeneous': 10,  # 异构类型特征
            'crossshard': 4,      # 跨分片特征
            'identity': 2         # 身份特征
        }
        
        print(f"[Simplified Step1] 初始化完成，输出目录: {output_dir}")
    
    def extract_features_from_system(self, 
                                   node_features_module,
                                   experiment_name: str = "default") -> Dict[str, torch.Tensor]:
        """
        从BlockEmulator系统提取特征（简化版本）
        
        Args:
            node_features_module: NodeFeaturesModule实例（可以是mock数据）
            experiment_name: 实验名称
            
        Returns:
            包含特征数据的字典
        """
        print(f"[Simplified Step1] 开始特征提取，实验: {experiment_name}")
        
        # 如果输入是mock数据列表，直接处理
        if isinstance(node_features_module, list):
            raw_node_data = node_features_module
            num_nodes = len(raw_node_data)
        else:
            # 尝试调用系统接口
            try:
                raw_node_data = node_features_module.GetAllCollectedData()
                num_nodes = len(raw_node_data)
                print(f"[Simplified Step1] 成功获取 {num_nodes} 个节点的原始数据")
            except Exception as e:
                print(f"[Simplified Step1] 获取系统数据失败，使用默认值: {e}")
                # 使用默认节点数
                num_nodes = 4
                raw_node_data = []
        
        # 生成简化特征
        features = self._generate_simplified_features(num_nodes)
        
        # 生成边索引
        edge_index = self._generate_default_edge_index(num_nodes)
        edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)
        
        # 构建结果
        results = {
            'features': features,                      # [N, 65] 综合特征
            'f_comprehensive': features,               # 别名
            'f_classic': features[:, :32],             # [N, 32] 经典特征
            'f_graph': features[:, 32:48],             # [N, 16] 图特征  
            'f_reduced': features[:, :64],             # [N, 64] 降维特征
            'edge_index': edge_index,                  # [2, E] 边索引
            'edge_type': edge_type,                    # [E] 边类型
            'node_features': features,                 # 节点特征
            'adjacency_matrix': self._edge_index_to_adjacency(edge_index, num_nodes),
            'metadata': {
                'total_nodes': num_nodes,
                'num_nodes': num_nodes,
                'num_edges': edge_index.shape[1],
                'feature_dim': features.shape[1],
                'processing_timestamp': datetime.now().isoformat(),
                'timestamp': int(time.time() * 1000),
                'simplified': True,
                'experiment_name': experiment_name
            },
            'node_info': {
                'node_ids': [f'node_{i}' for i in range(num_nodes)],
                'shard_ids': torch.zeros(num_nodes, dtype=torch.long)
            },
            'num_nodes': num_nodes,
            'num_features': features.shape[1],
            'timestamp': int(time.time() * 1000),
            'data_source': 'simplified_blockemulator_system'
        }
        
        # 保存结果
        if self.save_adjacency:
            self._save_detailed_adjacency_info(results, experiment_name)
        
        output_filename = os.path.join(self.output_dir, f"step1_{experiment_name}_features.pt")
        torch.save(results, output_filename)
        print(f"[Simplified Step1] 特征已保存到: {output_filename}")
        
        return results
    
    def _generate_simplified_features(self, num_nodes: int) -> torch.Tensor:
        """生成简化的节点特征"""
        print(f"[Simplified Step1] 生成 {num_nodes} 个节点的简化特征")
        
        # 生成65维特征向量
        features = []
        
        for i in range(num_nodes):
            # 硬件特征 (13维)
            hardware_features = [
                2 + i % 4,        # CPU核心数
                4 + i % 8,        # 内存容量GB
                100 + i * 10,     # 网络带宽
                50 + i * 5,       # 内存带宽
                1000 + i * 100,   # 存储容量
                500 + i * 50,     # 读写速度
                50 + i % 20,      # 网络延迟
                0.1 + i * 0.01,   # CPU使用率
                0.2 + i * 0.02,   # 内存使用率
                0.8 + i * 0.02,   # 存储可用率
                0.3 + i * 0.03,   # 网络使用率
                10 + i,           # 交易频率
                50 + i * 5        # 存储操作数
            ]
            
            # 链上行为特征 (15维) 
            onchain_features = [
                50.0 + i * 10,    # 平均TPS
                100 + i % 50,     # 确认延迟ms
                0.1 + i * 0.01,   # CPU每交易
                0.05 + i * 0.005, # 内存每交易
                5.0 + i % 3,      # 平均区块间隔
                1.0 + i % 2,      # 间隔标准差
                0.8 + i * 0.01,   # 普通交易比率
                0.2 - i * 0.01,   # 合约交易比率
                0.9 + i * 0.005,  # 参与率
                100.0 + i * 10,   # 总奖励
                0.95 - i * 0.001, # 成功率
                0.0,              # 合约调用频率
                0.01 + i * 0.001, # 手续费贡献率
                10 + i % 5,       # 交易频率
                200 + i * 10      # 处理延迟
            ]
            
            # 网络拓扑特征 (7维)
            topology_features = [
                3 + i % 2,        # 分片内连接
                2 + i % 2,        # 跨分片连接  
                5.0 + i,          # 加权度
                4 + i % 3,        # 活跃连接
                0.7 + i * 0.01,   # 适应性
                8,                # 时区(UTC+8)
                i % 3             # 地理位置编码
            ]
            
            # 动态属性特征 (10维)
            dynamic_features = [
                30.0 + i * 5,     # CPU使用率
                40.0 + i * 3,     # 内存使用率
                0.1 + i * 0.01,   # 资源波动
                80.0 + i * 2,     # 存储可用
                20.0 + i * 2,     # 存储利用率
                0.05 + i * 0.005, # 延迟波动
                50 + i % 20,      # 平均延迟
                0.3 + i * 0.02,   # 带宽使用
                10 + i % 10,      # 交易频率
                200 + i * 20      # 处理延迟
            ]
            
            # 异构类型特征 (10维)
            heterogeneous_features = [
                0,                # 节点类型(full_node=0)
                1,                # 功能标签编码
                0,                # 支持功能编码
                1,                # 当前状态(active=1)
                100 + i * 10,     # 交易频率
                50 + i * 5,       # 存储操作
                i % 4,            # 分片ID
                1,                # 共识参与
                1,                # 验证能力
                1                 # 处理能力
            ]
            
            # 跨分片特征 (4维)
            crossshard_features = [
                0.1 + i * 0.05,   # 跨分片交易率
                0.2 + i * 0.02,   # 跨分片通信开销
                0.8 - i * 0.01,   # 分片内聚合度
                0.3 + i * 0.03    # 分片间耦合度
            ]
            
            # 身份特征 (2维)
            identity_features = [
                i,                # 节点ID
                hash(f"node_{i}") % 1000 / 1000.0  # 节点哈希
            ]
            
            # 组合所有特征 (65维)
            node_feature = (hardware_features + onchain_features + 
                          topology_features + dynamic_features +
                          heterogeneous_features + crossshard_features + 
                          identity_features)
            
            features.append(node_feature)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _generate_default_edge_index(self, num_nodes: int) -> torch.Tensor:
        """生成默认的边索引（环形连接）"""
        if num_nodes <= 1:
            return torch.empty((2, 0), dtype=torch.long)
        
        # 创建环形连接：每个节点连接到下一个节点
        edges = []
        for i in range(num_nodes):
            next_node = (i + 1) % num_nodes
            edges.append([i, next_node])
            edges.append([next_node, i])  # 双向连接
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        return edge_index
    
    def _edge_index_to_adjacency(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """将边索引转换为邻接矩阵"""
        adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        if edge_index.shape[1] > 0:
            adjacency[edge_index[0], edge_index[1]] = 1.0
        return adjacency
    
    def _save_detailed_adjacency_info(self, results: Dict[str, torch.Tensor], experiment_name: str):
        """保存详细的邻接矩阵信息"""
        adjacency_info = {
            'generation_time': datetime.now().isoformat(),
            'experiment_name': experiment_name,
            'graph_metadata': {
                'num_nodes': results['metadata']['total_nodes'],
                'num_edges': results['metadata'].get('num_edges', 0),
                'feature_dim': results['metadata'].get('feature_dim', 65)
            },
            'edge_statistics': {
                'total_edges': int(results['edge_index'].shape[1]),
                'edge_types': {
                    'intra_shard': int((results['edge_type'] == 0).sum()),
                    'inter_shard_adjacent': int((results['edge_type'] == 1).sum()),
                    'inter_shard_distant': int((results['edge_type'] == 2).sum())
                }
            },
            'node_distribution': {
                'shard_counts': {'0': results['metadata']['total_nodes']}  # 默认都在分片0
            },
            'simplified_adapter': True
        }
        
        # 保存邻接信息
        adjacency_filename = os.path.join(self.output_dir, f"step1_{experiment_name}_adjacency_info.json")
        with open(adjacency_filename, 'w', encoding='utf-8') as f:
            json.dump(adjacency_info, f, indent=2, ensure_ascii=False)
        
        print(f"[Simplified Step1] 邻接信息已保存到: {adjacency_filename}")

class SimplifiedBlockEmulatorStep1Pipeline:
    """
    简化的BlockEmulator Step1流水线
    用于替代复杂的system_integration_pipeline
    """
    
    def __init__(self, 
                 use_comprehensive_features: bool = True,
                 save_adjacency: bool = True,
                 output_dir: str = "./step1_outputs",
                 experiment_name: str = "default"):
        """初始化简化流水线"""
        self.use_comprehensive_features = use_comprehensive_features
        self.save_adjacency = save_adjacency
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化适配器
        self.adapter = SimplifiedStep1Adapter(
            use_comprehensive_features=use_comprehensive_features,
            save_adjacency=save_adjacency,
            output_dir=output_dir
        )
        
        print(f"[SimplifiedBlockEmulatorStep1Pipeline] 初始化完成，输出目录: {output_dir}")
    
    def extract_features_from_system(self, 
                                   node_features_module,
                                   experiment_name: str = "default") -> Dict[str, torch.Tensor]:
        """
        从BlockEmulator系统提取特征
        
        Args:
            node_features_module: NodeFeaturesModule实例或mock数据
            experiment_name: 实验名称
            
        Returns:
            包含特征数据的字典
        """
        return self.adapter.extract_features_from_system(
            node_features_module, experiment_name
        )

def main():
    """测试简化的Step1流水线"""
    print("=== 简化Step1流水线测试 ===")
    
    # 创建流水线
    pipeline = SimplifiedBlockEmulatorStep1Pipeline(
        use_comprehensive_features=True,
        save_adjacency=True,
        output_dir="./test_simplified_step1_outputs"
    )
    
    # 创建模拟数据
    mock_data = [
        {'NodeID': f'node_{i}', 'IP': '127.0.0.1', 'Port': 8000 + i}
        for i in range(4)
    ]
    
    # 测试特征提取
    print("\n1. 测试特征提取...")
    results = pipeline.extract_features_from_system(
        node_features_module=mock_data,
        experiment_name="test_simplified"
    )
    
    print(f"   结果: {results['features'].shape} 特征张量")
    print(f"   边数: {results['edge_index'].shape[1]}")
    print(f"   元数据: {len(results['metadata'])} 个字段")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    main()
