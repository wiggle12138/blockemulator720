"""
区块链时序数据集
"""
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class BlockchainDataset(Dataset):
    """区块链时序数据集 - 生成多时间步数据"""

    def __init__(self, embedding_path, edge_index_path, num_timesteps=10, noise_level=0.02):
        # 加载预训练嵌入
        with open(embedding_path, 'rb') as f:
            self.pretrained_embeddings = pickle.load(f)

        print(f"加载预训练嵌入: {len(self.pretrained_embeddings)} 个节点")

        # 获取节点ID和基础嵌入
        self.node_ids = sorted([ k for k in self.pretrained_embeddings.keys()])
        self.num_nodes = len(self.node_ids)
        self.num_timesteps = num_timesteps
        self.noise_level = noise_level

        # 处理基础嵌入
        self._process_base_embeddings()

        # 生成时序嵌入, 模拟部分时间步
        self._generate_temporal_embeddings()

        # 加载图结构
        self.edge_index = self._load_edge_index(edge_index_path)

        print(f"数据集创建完成: {self.num_nodes} 节点, {self.embedding_dim} 维, {num_timesteps} 时间步")

    def _process_base_embeddings(self):
        """处理基础嵌入数据"""
        base_embeddings = []
        for node_id in self.node_ids:
            node_data = self.pretrained_embeddings[node_id]
            if isinstance(node_data, dict):
                embedding = node_data[0] if 0 in node_data else list(node_data.values())[0]
            else:
                embedding = node_data

            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            if embedding.ndim > 1:
                embedding = embedding.flatten()

            base_embeddings.append(embedding)

        # 确保数据类型一致
        self.base_embeddings = np.stack(base_embeddings).astype(np.float32)
        self.embedding_dim = self.base_embeddings.shape[1]

    def _generate_temporal_embeddings(self):
        """生成时序嵌入数据"""
        self.temporal_embeddings = {}

        for t in range(self.num_timesteps):
            # 添加时间相关变化
            noise = np.random.normal(0, self.noise_level, self.base_embeddings.shape).astype(np.float32)

            # trend和drift是标量，需要广播到数组形状
            trend_scalar = np.sin(t * 2 * np.pi / self.num_timesteps) * 0.05
            drift_scalar = (t / self.num_timesteps) * 0.02

            # 将标量广播到与base_embeddings相同的形状
            trend = np.full_like(self.base_embeddings, trend_scalar, dtype=np.float32)
            drift = np.full_like(self.base_embeddings, drift_scalar, dtype=np.float32)

            temporal_features = self.base_embeddings + noise + trend + drift
            self.temporal_embeddings[t] = temporal_features.astype(np.float32)

    def _load_edge_index(self, file_path):
        """加载图结构"""
        try:
            data = torch.load(file_path, map_location='cpu')
            edge_index = data['edge_index']
            print(f"加载图结构: {edge_index.shape[1]} 条边")
            return edge_index
        except:
            print("未找到图结构文件，使用全连接图")
            return self._build_complete_graph()

    def _build_complete_graph(self):
        """构建全连接图"""
        nodes = torch.arange(self.num_nodes)
        src = nodes.repeat(self.num_nodes)
        dst = nodes.repeat_interleave(self.num_nodes)
        return torch.stack([src, dst], dim=0)

    def __len__(self):
        return self.num_timesteps

    def __getitem__(self, idx):
        # 确保返回数据类型一致
        node_features = torch.tensor(self.temporal_embeddings[idx], dtype=torch.float32)
        return node_features, self.edge_index.clone(), idx

    def get_info(self):
        """获取数据集信息"""
        return {
            'num_nodes': self.num_nodes,
            'embedding_dim': self.embedding_dim,
            'num_timesteps': self.num_timesteps,
            'noise_level': self.noise_level
        }