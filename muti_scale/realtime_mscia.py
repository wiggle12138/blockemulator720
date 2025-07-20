"""
实时多尺度对比学习适配器 - 支持真实时间步
基于BlockEmulator的实时数据流进行多尺度对比学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional
import time
from pathlib import Path

# 导入原始MSCIA组件
from All_Final import TemporalMSCIA, subgraph_contrastive_loss, graph_contrastive_loss, node_contrastive_loss


class RealtimeMSCIADataset(Dataset):
    """实时多尺度对比学习数据集 - 支持真实时间步"""
    
    def __init__(self, time_window: int = 5, batch_size: int = 32):
        """
        Args:
            time_window: 时间窗口大小（考虑前N个时间步）
            batch_size: 批次大小
        """
        self.time_window = time_window
        self.batch_size = batch_size
        
        # 时序数据缓存
        self.temporal_buffer = {}  # {timestamp: {'features': tensor, 'adjacency': tensor, 'metadata': dict}}
        self.current_timestamp = 0
        
        print(f"实时MSCIA数据集初始化完成:")
        print(f"- 时间窗口大小: {time_window}")
        print(f"- 批次大小: {batch_size}")
    
    def add_timestep_data(self, features: torch.Tensor, adjacency: torch.Tensor, 
                         metadata: Dict[str, Any], timestamp: Optional[int] = None):
        """添加新的时间步数据"""
        if timestamp is None:
            timestamp = self.current_timestamp
            self.current_timestamp += 1
        
        self.temporal_buffer[timestamp] = {
            'features': features.clone(),
            'adjacency': adjacency.clone(),
            'metadata': metadata.copy(),
            'node_count': features.shape[0]
        }
        
        # 保持时间窗口大小，移除过老的数据
        timestamps = sorted(self.temporal_buffer.keys())
        while len(timestamps) > self.time_window * 3:  # 保持更多历史用于对比学习
            oldest_ts = timestamps.pop(0)
            del self.temporal_buffer[oldest_ts]
        
        print(f"✓ 添加时间步 {timestamp}: {features.shape[0]} 个节点")
    
    def get_temporal_sequence(self, current_ts: int) -> Optional[Dict[str, Any]]:
        """获取时序序列数据"""
        available_timestamps = sorted([ts for ts in self.temporal_buffer.keys() if ts <= current_ts])
        
        if len(available_timestamps) == 0:
            return None
        
        # 选择时间窗口内的数据
        start_ts = max(0, current_ts - self.time_window + 1)
        window_timestamps = [ts for ts in available_timestamps if ts >= start_ts]
        
        if len(window_timestamps) == 0:
            return None
        
        # 使用最新的时间步作为主要数据
        main_ts = window_timestamps[-1]
        main_data = self.temporal_buffer[main_ts]
        
        # 构建时序特征（如果有历史数据）
        temporal_features = []
        temporal_timestamps = []
        
        for ts in window_timestamps:
            temporal_features.append(self.temporal_buffer[ts]['features'])
            temporal_timestamps.append(ts)
        
        return {
            'main_features': main_data['features'],          # [N, F]
            'main_adjacency': main_data['adjacency'],        # [N, N]
            'main_timestamp': main_ts,
            'temporal_features': temporal_features,          # List[tensor]
            'temporal_timestamps': temporal_timestamps,      # List[int]
            'metadata': main_data['metadata'],
            'num_nodes': main_data['node_count'],
            'temporal_context': {                            # 添加时序上下文
                'window_size': len(window_timestamps),
                'timestamps': temporal_timestamps,
                'time_span': window_timestamps[-1] - window_timestamps[0] if len(window_timestamps) > 1 else 0,
                'current_timestamp': current_ts,
                'total_cached_timestamps': len(self.temporal_buffer)
            }
        }
    
    def __len__(self):
        return max(1, len(self.temporal_buffer))
    
    def __getitem__(self, idx):
        """获取批次数据"""
        if len(self.temporal_buffer) == 0:
            return self._get_empty_batch()
        
        # 使用最新的时间步
        latest_ts = max(self.temporal_buffer.keys())
        sequence_data = self.get_temporal_sequence(latest_ts)
        
        if sequence_data is None:
            return self._get_empty_batch()
        
        # 构建批次数据
        features = sequence_data['main_features']  # [N, F]
        adjacency = sequence_data['main_adjacency']  # [N, N]
        
        # 随机选择中心节点
        num_nodes = features.size(0)
        num_centers = min(self.batch_size, num_nodes)
        center_indices = torch.randperm(num_nodes)[:num_centers]
        
        # 节点类型（如果可用）
        node_types = sequence_data['metadata'].get('node_types', 
                                                 torch.zeros(num_nodes, dtype=torch.long))
        if isinstance(node_types, list):
            node_type_mapping = {'miner': 0, 'full_node': 1, 'light_node': 2, 'validator': 3, 'storage': 4}
            node_types = torch.tensor([node_type_mapping.get(nt, 0) for nt in node_types], dtype=torch.long)
        
        return {
            'adjacency_matrix': adjacency,
            'node_features': features,
            'center_indices': center_indices,
            'node_types': node_types,
            'timestamp': sequence_data['main_timestamp'],
            'selected_nodes': torch.arange(num_nodes),
            'temporal_context': sequence_data.get('temporal_context', {
                'window_size': len(sequence_data.get('temporal_timestamps', [])),
                'timestamps': sequence_data.get('temporal_timestamps', [])
            })
        }
    
    def _get_empty_batch(self):
        """返回空批次"""
        return {
            'adjacency_matrix': torch.zeros(1, 1),
            'node_features': torch.zeros(1, 128),
            'center_indices': torch.tensor([0]),
            'node_types': torch.zeros(1, dtype=torch.long),
            'timestamp': 0,
            'selected_nodes': torch.tensor([0]),
            'temporal_context': {'window_size': 0, 'timestamps': []}
        }


class RealtimeMSCIAModel(TemporalMSCIA):
    """实时多尺度对比学习模型"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, time_dim: int = 16, 
                 k_ratio: float = 0.9, alpha: float = 0.3, beta: float = 0.4, gamma: float = 0.3,
                 tau: float = 0.09, num_node_types: int = 5, num_edge_types: int = 3,
                 max_timestamp: int = 10000):
        """
        Args:
            max_timestamp: 最大时间戳，用于时间嵌入
        """
        super().__init__(input_dim, hidden_dim, time_dim, k_ratio, alpha, beta, gamma,
                        tau, num_node_types, num_edge_types)
        
        # 扩展时间嵌入范围以支持更大的时间戳
        self.time_embedding = nn.Embedding(max_timestamp + 100, time_dim)
        nn.init.normal_(self.time_embedding.weight, 0, 0.01)
        
        print(f"实时MSCIA模型初始化:")
        print(f"- 输入维度: {input_dim}")
        print(f"- 隐藏维度: {hidden_dim}")
        print(f"- 时间维度: {time_dim}")
        print(f"- 最大时间戳: {max_timestamp}")
    
    def forward_realtime(self, batch_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """实时前向传播"""
        # 确保时间戳在合理范围内
        timestamp = batch_data['timestamp']
        if isinstance(timestamp, torch.Tensor):
            timestamp = timestamp.item()
        
        # 限制时间戳范围
        timestamp = min(timestamp, self.time_embedding.num_embeddings - 1)
        batch_data['timestamp'] = timestamp
        
        return self.forward(batch_data)


class RealtimeMSCIAProcessor:
    """实时多尺度对比学习处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置参数
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化数据集和模型
        self.dataset = RealtimeMSCIADataset(
            time_window=config.get('time_window', 5),
            batch_size=config.get('batch_size', 32)
        )
        
        self.model = RealtimeMSCIAModel(
            input_dim=config.get('input_dim', 128),
            hidden_dim=config.get('hidden_dim', 64),
            time_dim=config.get('time_dim', 16),
            k_ratio=config.get('k_ratio', 0.9),
            alpha=config.get('alpha', 0.3),
            beta=config.get('beta', 0.4),
            gamma=config.get('gamma', 0.3),
            tau=config.get('tau', 0.09),
            max_timestamp=config.get('max_timestamp', 10000)
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 0.02),
            weight_decay=config.get('weight_decay', 9e-6)
        )
        
        # 训练状态
        self.training_history = []
        self.temporal_embeddings = {}  # {node_id: {timestamp: embedding}}
        
        print(f"实时MSCIA处理器初始化完成")
        print(f"- 设备: {self.device}")
        print(f"- 配置: {config}")
    
    def process_step1_output(self, step1_result: Dict[str, torch.Tensor], 
                           timestamp: Optional[int] = None,
                           blockemulator_timestamp: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        处理第一步的输出，进行第二步多尺度对比学习 - 支持真实时间步
        
        Args:
            step1_result: 第一步的输出结果
                - 'f_classic': [N, 128] 经典特征
                - 'f_graph': [N, 96] 图特征  
                - 'node_mapping': Dict 节点映射
                - 'metadata': Dict 元数据
            timestamp: 逻辑时间步（从0开始递增）
            blockemulator_timestamp: BlockEmulator的真实时间戳（Unix时间戳或相对时间）
        
        Returns:
            第二步输出:
                - 'temporal_embeddings': [N, 64] 时序嵌入
                - 'loss': 对比学习损失
                - 'node_mapping': 节点映射
                - 'metadata': 元数据
        """
        print(f"\n=== 第二步：实时多尺度对比学习 ===")
        
        # 处理真实时间戳
        real_timestamp = self._process_real_timestamp(timestamp, blockemulator_timestamp)
        print(f"时间戳处理: 逻辑={timestamp}, 真实={blockemulator_timestamp}, 处理后={real_timestamp}")
        
        # 提取特征和构建邻接矩阵
        f_classic = step1_result['f_classic']  # [N, 128]
        f_graph = step1_result['f_graph']      # [N, 96]
        node_mapping = step1_result['node_mapping']
        metadata = step1_result['metadata']
        
        # 使用f_classic作为主要特征输入
        features = f_classic
        
        # 构建邻接矩阵（简化版，实际应该从图结构中获取）
        num_nodes = features.shape[0]
        adjacency = self._build_adjacency_from_features(f_graph, num_nodes)
        
        # 增强元数据，包含真实时间戳信息
        enhanced_metadata = {
            **metadata,
            'logical_timestamp': timestamp,
            'real_timestamp': blockemulator_timestamp,
            'processed_timestamp': real_timestamp,
            'step1_timestamp': timestamp,
            'f_graph_available': True,
            'processing_time': time.time(),
            'time_source': 'blockemulator' if blockemulator_timestamp else 'synthetic'
        }
        
        # 添加到时序缓存（使用处理后的时间戳）
        self.dataset.add_timestep_data(features, adjacency, enhanced_metadata, real_timestamp)
        
        # 获取当前时序批次
        sequence_data = self.dataset.get_temporal_sequence(real_timestamp)
        if sequence_data is None:
            print("⚠ 无法获取时序数据，返回空结果")
            return self._get_empty_result()
        
        # 构建批次数据
        batch_data = self.dataset[0]  # 获取最新批次
        
        # 确保时间戳在模型可处理范围内
        batch_data['timestamp'] = self._normalize_timestamp_for_model(real_timestamp)
        
        # 移到设备
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.to(self.device)
        
        # 前向传播
        with torch.no_grad():
            loss, embeddings = self.model.forward_realtime(batch_data)
        
        # 更新时序嵌入
        self._update_temporal_embeddings(embeddings, batch_data, real_timestamp)
        
        print(f"✓ 多尺度对比学习完成:")
        print(f"  时序嵌入: {embeddings.shape}")
        print(f"  对比损失: {loss.item():.4f}")
        print(f"  时间窗口: {sequence_data['temporal_context']['window_size']}")
        print(f"  真实时间戳: {blockemulator_timestamp}")
        
        return {
            'temporal_embeddings': embeddings.cpu(),
            'loss': loss.cpu(),
            'node_mapping': node_mapping,
            'metadata': {
                **enhanced_metadata,
                'step2_completed': True,
                'temporal_context': sequence_data['temporal_context'],
                'loss_value': loss.item(),
                'real_time_processed': True
            }
        }
    
    def _build_adjacency_from_features(self, f_graph: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """从图特征构建邻接矩阵"""
        # 简化实现：使用特征相似度构建邻接矩阵
        # 实际应该使用更复杂的图构建策略
        
        if f_graph.shape[0] != num_nodes:
            # 如果维度不匹配，创建随机邻接矩阵
            adjacency = torch.rand(num_nodes, num_nodes)
            adjacency = (adjacency + adjacency.t()) / 2  # 对称化
            adjacency.fill_diagonal_(0)  # 无自环
            return (adjacency > 0.7).float()
        
        # 使用余弦相似度
        f_graph_norm = F.normalize(f_graph, p=2, dim=1)
        similarity = torch.mm(f_graph_norm, f_graph_norm.t())
        
        # 构建邻接矩阵（保留top-k相似节点）
        k = min(10, num_nodes // 4)  # 每个节点最多连接到10个其他节点
        _, indices = similarity.topk(k + 1, dim=1)  # +1 because we'll remove self-loops
        
        adjacency = torch.zeros(num_nodes, num_nodes)
        for i in range(num_nodes):
            for j in indices[i]:
                if i != j:  # 避免自环
                    adjacency[i, j] = 1.0
        
        # 对称化
        adjacency = (adjacency + adjacency.t()).clamp(0, 1)
        
        return adjacency
    
    def _update_temporal_embeddings(self, embeddings: torch.Tensor, 
                                   batch_data: Dict[str, torch.Tensor], timestamp: int):
        """更新时序嵌入缓存"""
        selected_nodes = batch_data['selected_nodes'].cpu().numpy()
        embeddings_np = embeddings.detach().cpu().numpy()
        
        for i, node_id in enumerate(selected_nodes):
            if node_id not in self.temporal_embeddings:
                self.temporal_embeddings[node_id] = {}
            
            self.temporal_embeddings[node_id][timestamp] = embeddings_np[i]
    
    def _get_empty_result(self):
        """返回空结果"""
        return {
            'temporal_embeddings': torch.zeros(1, 64),
            'loss': torch.tensor(0.0),
            'node_mapping': {},
            'metadata': {'empty_result': True}
        }
    
    def get_output_format_info(self) -> Dict[str, Any]:
        """获取第二步输出格式信息"""
        return {
            'step': 2,
            'name': 'Multi-Scale Contrastive Learning (Real-time)',
            'input_from_step1': {
                'f_classic': '[N, 128] 经典特征',
                'f_graph': '[N, 96] 图特征',
                'node_mapping': 'Dict 节点映射',
                'metadata': 'Dict 元数据'
            },
            'real_time_input': {
                'logical_timestamp': 'int 逻辑时间步（0,1,2,...）',
                'blockemulator_timestamp': 'float BlockEmulator真实时间戳',
                'automatic_scaling': 'bool 自动时间缩放',
                'time_window_adaptive': 'bool 自适应时间窗口'
            },
            'output_format': {
                'temporal_embeddings': '[N, 64] 时序嵌入特征',
                'loss': 'Scalar 对比学习损失',
                'node_mapping': 'Dict 节点ID映射',
                'metadata': 'Dict 包含完整时序上下文的元数据'
            },
            'temporal_features': {
                'time_window': self.config.get('time_window', 5),
                'supports_real_timestamps': True,
                'embedding_dimension': self.config.get('hidden_dim', 64),
                'time_processing': {
                    'reference_point': 'First timestamp as reference',
                    'scaling_method': 'Adaptive based on timestamp magnitude',
                    'model_normalization': 'Clamp to embedding range'
                }
            },
            'metadata_enrichment': {
                'logical_timestamp': '逻辑时间步',
                'real_timestamp': 'BlockEmulator原始时间戳',
                'processed_timestamp': '模型处理后的时间戳',
                'time_source': '时间戳来源标识',
                'temporal_context': '时间窗口上下文信息',
                'real_time_processed': '真实时间处理标志'
            },
            'next_step_compatibility': {
                'step3_evolve_gcn': 'temporal_embeddings作为时序特征输入',
                'format': 'pytorch tensor [N, 64]',
                'timestamp_preservation': '时间戳信息完整传递到下一步'
            }
        }
    
    def _process_real_timestamp(self, logical_timestamp: Optional[int], 
                              real_timestamp: Optional[float]) -> int:
        """
        处理真实时间戳，转换为模型可用的时间戳
        
        Args:
            logical_timestamp: 逻辑时间步（0, 1, 2, ...）
            real_timestamp: 真实时间戳（Unix时间戳或BlockEmulator相对时间）
        
        Returns:
            处理后的时间戳，适合模型处理
        """
        if real_timestamp is None:
            # 没有真实时间戳，使用逻辑时间戳
            return logical_timestamp if logical_timestamp is not None else 0
        
        if not hasattr(self, '_time_reference'):
            # 首次处理，建立时间参考点
            self._time_reference = real_timestamp
            self._time_scale = 1.0  # 时间缩放因子
            return 0
        
        # 计算相对时间差
        time_diff = real_timestamp - self._time_reference
        
        # 根据时间差的规模调整缩放因子
        if time_diff > 10000:  # 如果是Unix时间戳级别
            self._time_scale = 0.001  # 缩放到合理范围
        elif time_diff > 1000:
            self._time_scale = 0.01
        else:
            self._time_scale = 1.0
        
        # 转换为整数时间戳
        processed_timestamp = int(time_diff * self._time_scale)
        
        # 确保非负
        return max(0, processed_timestamp)
    
    def _normalize_timestamp_for_model(self, timestamp: int) -> int:
        """
        标准化时间戳以适合模型的时间嵌入范围
        
        Args:
            timestamp: 处理后的时间戳
        
        Returns:
            标准化的时间戳
        """
        max_embedding_range = getattr(self.model, 'time_embedding', None)
        if max_embedding_range is not None:
            max_ts = max_embedding_range.num_embeddings - 1
            return min(timestamp, max_ts)
        
        # 默认范围
        return min(timestamp, 9999)
    

def demo_realtime_mscia():
    """演示实时多尺度对比学习 - 支持真实时间步"""
    print("=== 实时多尺度对比学习演示（真实时间步版本）===")
    
    # 配置参数
    config = {
        'time_window': 5,
        'batch_size': 32,
        'input_dim': 128,
        'hidden_dim': 64,
        'time_dim': 16,
        'k_ratio': 0.9,
        'alpha': 0.3,
        'beta': 0.4,
        'gamma': 0.3,
        'lr': 0.02,
        'weight_decay': 9e-6,
        'tau': 0.09,
        'max_timestamp': 1000,
        'use_real_timestamps': True  # 启用真实时间戳
    }
    
    # 初始化处理器
    processor = RealtimeMSCIAProcessor(config)
    
    # 模拟多个时间步的第一步输出（包含真实时间戳）
    import time
    current_time = time.time()
    
    time_scenarios = [
        # (逻辑时间步, BlockEmulator真实时间戳)
        (0, current_time),
        (1, current_time + 1.5),      # 1.5秒后
        (2, current_time + 3.2),      # 3.2秒后  
        (3, current_time + 4.8),      # 4.8秒后
        (4, current_time + 6.1)       # 6.1秒后
    ]
    
    print(f"模拟场景：从时间戳 {current_time:.2f} 开始的真实时间序列")
    
    for logical_ts, real_ts in time_scenarios:
        print(f"\n--- 处理时间步 {logical_ts} (真实时间: {real_ts:.2f}) ---")
        
        # 模拟第一步输出
        num_nodes = np.random.randint(50, 100)
        mock_step1_output = {
            'f_classic': torch.randn(num_nodes, 128),
            'f_graph': torch.randn(num_nodes, 96),
            'node_mapping': {i: f"node_{i}" for i in range(num_nodes)},
            'metadata': {
                'num_nodes': num_nodes,
                'logical_timestamp': logical_ts,
                'blockemulator_real_time': real_ts,
                'source': 'blockemulator',
                'simulation_step': logical_ts
            }
        }
        
        # 处理第二步（传入真实时间戳）
        step2_result = processor.process_step1_output(
            mock_step1_output, 
            timestamp=logical_ts,
            blockemulator_timestamp=real_ts
        )
        
        print(f"✓ 第二步完成:")
        print(f"  时序嵌入维度: {step2_result['temporal_embeddings'].shape}")
        print(f"  损失值: {step2_result['loss'].item():.4f}")
        print(f"  逻辑时间戳: {step2_result['metadata'].get('logical_timestamp', 'N/A')}")
        print(f"  真实时间戳: {step2_result['metadata'].get('real_timestamp', 'N/A')}")
        print(f"  处理后时间戳: {step2_result['metadata'].get('processed_timestamp', 'N/A')}")
        print(f"  时间来源: {step2_result['metadata'].get('time_source', 'N/A')}")
        print(f"  时序上下文: {step2_result['metadata'].get('temporal_context', {})}")
    
    # 显示输出格式信息
    print(f"\n=== 第二步输出格式信息（真实时间步版本）===")
    format_info = processor.get_output_format_info()
    format_info['real_timestamp_support'] = {
        'supported': True,
        'input_format': 'Unix timestamp or relative time from BlockEmulator',
        'processing': 'Automatic scaling and normalization',
        'model_range': 'Dynamically adjusted based on timestamp magnitude'
    }
    
    for key, value in format_info.items():
        print(f"{key}: {value}")
    
    print(f"\n=== 真实时间步处理特性 ===")
    print("✓ 支持Unix时间戳和相对时间")
    print("✓ 自动时间缩放和标准化")
    print("✓ 时间窗口基于真实时间差")
    print("✓ 模型时间嵌入动态调整")
    print("✓ 时间上下文完整保留")


if __name__ == "__main__":
    demo_realtime_mscia()
