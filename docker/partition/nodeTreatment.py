import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.data import HeteroData
from torch_geometric.nn import RGCNConv, HeteroConv
from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
import numpy as np

# ================= 异构图构建 =================
class BlockchainHeteroGraph:
    def __init__(self, nodes):
        self.data = HeteroData()
        self._init_node_features(nodes)
        self._build_topology()

    def _init_node_features(self, nodes):
        """加载预处理后的节点特征"""
        # 假设nodes是包含所有处理后的特征的字典
        for node_type in ['miner', 'validator', 'full', 'light', 'storage']:
            if node_type in nodes:
                features = torch.stack([n['features'] for n in nodes[node_type]])
                self.data[node_type].x = features

    def _build_topology(self):
        """构建异构边关系"""
        # 竞争关系（矿工-矿工）
        miner_comp_edges = self._sample_competitive_edges('miner')
        self.data['miner', 'competes', 'miner'].edge_index = miner_comp_edges

        # 共识验证（验证-全节点）
        valid_edges = self._sample_validation_edges('validator', 'full')
        self.data['validator', 'validates', 'full'].edge_index = valid_edges

        # 数据服务（存储-轻节点）
        storage_edges = self._sample_storage_edges('storage', 'light')
        self.data['storage', 'serves', 'light'].edge_index = storage_edges

    def _sample_competitive_edges(self, node_type, k=5):
        """基于资源相似度采样竞争边"""
        # 实现相似度匹配逻辑
        pass

# ================= 动态特征编码 =================
class TemporalFeatureExtractor(nn.Module):
    def __init__(self, window_size=6, feature_dim=64):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3)
        )
        self.trend_layer = nn.Linear(24, feature_dim//2)

    def forward(self, x_hist):
        # x_hist: (batch_size, seq_len, features)
        stats_feat = self.conv_layers(x_hist.unsqueeze(1))  # 统计特征
        trend_feat = self.trend_layer(x_hist[:, -24:, :])   # 24小时趋势
        return torch.cat([stats_feat, trend_feat], dim=-1)

class CodeAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.complexity_net = nn.Sequential(
            nn.Conv1d(1, 16, 3),
            nn.MaxPool1d(2),
            nn.Linear(16, 8)
        )

    def forward(self, contract_code):
        # 代码语义特征
        tokens = tokenize_code(contract_code)
        sem_feat = self.bert(tokens).pooler_output

        # 结构复杂度特征
        ast = parse_to_ast(contract_code)
        struct_feat = self.complexity_net(ast)

        return torch.cat([sem_feat, struct_feat], dim=-1)

# ================= 异构GNN模型 =================
class HeteroRGCN(nn.Module):
    def __init__(self, hidden_size=256, num_relations=5):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(3):
            conv = HeteroConv({
                ('miner', 'competes', 'miner'): RGCNConv(-1, hidden_size, num_relations),
                ('validator', 'validates', 'full'): RGCNConv(-1, hidden_size),
                ('storage', 'serves', 'light'): RGCNConv(-1, hidden_size)
            })
            self.convs.append(conv)

    def forward(self, data):
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        return x_dict

# ================= 特征融合模块 =================
class FusionAttention(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(feat_dim*2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, f_classic, f_graph):
        combined = torch.cat([f_classic, f_graph], dim=-1)
        alpha = torch.sigmoid(self.attn_net(combined))
        return alpha * f_classic + (1 - alpha) * f_graph

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature

    def forward(self, z1, z2):
        sim_matrix = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1)
        pos_sim = torch.diag(sim_matrix)
        neg_sim = torch.logsumexp(sim_matrix / self.temp, dim=1)
        return -torch.mean(pos_sim - neg_sim)

# ================= 完整处理流程 =================
class BlockchainProcessor:
    def __init__(self):
        self.scalers = {
            'hardware': PowerTransformer(),
            'transaction': QuantileTransformer(),
            'network': PowerTransformer()
        }

        self.gnn = HeteroRGCN()
        self.temporal_encoder = TemporalFeatureExtractor()
        self.fusion = FusionAttention(512)
        self.loss_fn = ContrastiveLoss()

    def preprocess(self, raw_nodes):
        # 1. 特征归一化
        scaled_features = []
        for node in raw_nodes:
            feat = np.concatenate([
                self.scalers['hardware'].fit_transform(node['hardware']),
                self.scalers['transaction'].fit_transform(node['transaction']),
                self.scalers['network'].fit_transform(node['network'])
            ])
            scaled_features.append(feat)

        # 2. 动态特征编码
        temporal_feat = self.temporal_encoder(raw_nodes['history'])

        # 3. 代码分析
        code_feat = [CodeAnalyzer()(c) for c in raw_nodes['contract_code']]

        return torch.cat([scaled_features, temporal_feat, code_feat], dim=-1)

    def train_step(self, data):
        # 异构图学习
        graph_emb = self.gnn(data)

        # 特征融合
        fused_feat = self.fusion(data.x, graph_emb)

        # 对比学习
        loss = self.loss_fn(fused_feat, graph_emb)
        return loss

# ================= 示例用法 =================
if __name__ == "__main__":
    # 模拟原始节点数据
    sample_nodes = {
        'miner': [{'hardware': np.random.rand(8),
                   'transaction': np.random.rand(5),
                   'network': np.random.rand(3),
                   'history': np.random.randn(24, 6),
                   'contract_code': '0x...'} for _ in range(10)],
        'validator': [...]  # 类似结构
    }

    processor = BlockchainProcessor()

    # 1. 数据预处理
    processed_data = processor.preprocess(sample_nodes)

    # 2. 构建异构图
    graph_builder = BlockchainHeteroGraph(processed_data)
    hetero_data = graph_builder.data

    # 3. 训练循环
    optimizer = torch.optim.Adam(processor.parameters(), lr=1e-4)
    for epoch in range(100):
        optimizer.zero_grad()
        loss = processor.train_step(hetero_data)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")