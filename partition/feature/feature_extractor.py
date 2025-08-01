"""
主要的特征提取器
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F

# 修复相对导入问题 - 使用绝对导入
import sys
from pathlib import Path

# 添加当前目录到系统路径
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 直接导入模块，失败时立即报错
try:
    from nodeInitialize import Node
except ImportError as e:
    raise ImportError(f"nodeInitialize导入失败: {e}")

try:
    try:
        from .data_processor import DataProcessor
    except ImportError:
        from data_processor import DataProcessor
except ImportError as e:
    print(f"[Partition.Feature] 包加载警告: data_processor导入失败: {e}")
    # 创建基本的DataProcessor替代品
    class DataProcessor:
        def process_data(self, data):
            return data

try:
    try:
        from .graph_builder import HeterogeneousGraphBuilder
    except ImportError:
        from graph_builder import HeterogeneousGraphBuilder
except ImportError as e:
    print(f"[Partition.Feature] 包加载警告: graph_builder导入失败: {e}")
    # 这里不抛出异常，让系统继续运行
    HeterogeneousGraphBuilder = None

try:
    try:
        from .config import FeatureDimensions, RelationTypes, NodeTypes, EncodingMaps
    except ImportError:
        from config import FeatureDimensions, RelationTypes, NodeTypes, EncodingMaps
except ImportError as e:
    print(f"[Partition.Feature] 包加载警告: config导入失败: {e}")
    # 创建基本的配置替代品
    class FeatureDimensions:
        BASIC = 40
    class RelationTypes:
        COMPETE = 0
        SERVE = 1
        VALIDATE = 2
    class NodeTypes:
        TYPES = ['miner', 'validator', 'full_node', 'storage', 'light_node']
    class EncodingMaps:
        NODE_TYPE_MAP = {'miner': 0, 'validator': 1, 'full_node': 2, 'storage': 3, 'light_node': 4}

try:
    from sliding_window_extractor import EnhancedSequenceFeatureEncoder
except ImportError as e:
    raise ImportError(f"sliding_window_extractor导入失败: {e}")

class ComprehensiveFeatureExtractor:
    """全面的特征提取器 - 使用所有70+个特征"""

    def __init__(self):
        self.processor = DataProcessor()
        self.dims = FeatureDimensions()
        self.encodings = EncodingMaps()

    def extract_features(self, nodes: List[Node]) -> torch.Tensor:
        """
        提取所有节点的全面特征

        Args:
            nodes: 节点列表

        Returns:
            特征矩阵 [N, comprehensive_feature_dim]
        """
        all_features = []

        for node in nodes:
            features = self._extract_single_node_comprehensive_features(node)
            all_features.append(features)

        feature_tensor = torch.tensor(all_features, dtype=torch.float32)
        print(f"ComprehensiveFeatureExtractor输出维度: {feature_tensor.shape}")
        return feature_tensor

    def _extract_single_node_comprehensive_features(self, node: Node) -> List[float]:
        """提取单个节点的全面特征 - 基于committee_evolvegcn.go的40个字段"""
        features = []

        # 1. 硬件规格特征 (11维) - CPU(2) + Memory(3) + Storage(3) + Network(3)
        features.extend(self._extract_hardware_features(node))

        # 2. 网络拓扑特征 (5维) - intra_shard_conn + inter_shard_conn + weighted_degree + active_conn + adaptability
        features.extend(self._extract_network_topology_features(node))

        # 3. 异构类型特征 (2维) - node_type + core_eligibility
        features.extend(self._extract_heterogeneous_type_features(node))

        # 4. 链上行为特征 (15维) - transaction(2) + cross_shard(2) + block_gen(2) + tx_types(2) + consensus(3) + resource(3) + network_dynamic(3)
        features.extend(self._extract_onchain_behavior_features(node))

        # 5. 动态属性特征 (7维) - tx_processing(2) + application(3)
        features.extend(self._extract_dynamic_attributes_features(node))

        # print(f"单节点全面特征维度: {len(features)}")
        return features  # 总计40维

    def _extract_hardware_features(self, node: Node) -> List[float]:
        """提取硬件相关特征 (11维) - 基于committee_evolvegcn.go"""
        features = []

        try:
            hw = node.ResourceCapacity.Hardware

            # CPU特征 (2维) - cpu_cores, cpu_architecture
            cpu_arch = getattr(hw.CPU, 'Architecture', 'unknown')
            features.extend([
                float(getattr(hw.CPU, 'CoreCount', 0)),
                self.encodings.CPU_ARCHITECTURE.get(cpu_arch, 0.0),
            ])

            # 内存特征 (3维) - memory_gb, memory_bandwidth, memory_type
            mem_type = getattr(hw.Memory, 'Type', 'unknown')
            features.extend([
                float(getattr(hw.Memory, 'TotalCapacity', 0)),
                float(getattr(hw.Memory, 'Bandwidth', 0)),
                self.encodings.MEMORY_TYPE.get(mem_type, 0.0),
            ])

            # 存储特征 (3维) - storage_gb, storage_type, storage_rw_speed
            storage_type = getattr(hw.Storage, 'Type', 'unknown')
            features.extend([
                float(getattr(hw.Storage, 'Capacity', 0)),
                float(getattr(hw.Storage, 'ReadWriteSpeed', 0)),
                self.encodings.STORAGE_TYPE.get(storage_type, 0.0),
            ])

            # 网络特征 (3维) - network_upstream, network_downstream, network_latency
            features.extend([
                float(getattr(hw.Network, 'UpstreamBW', 0)),
                float(getattr(hw.Network, 'DownstreamBW', 0)),
                float(getattr(hw.Network, 'Latency', 0)),
            ])

        except Exception as e:
            print(f"Warning: Error extracting hardware features: {e}")
            features = [0.0] * 11

        # 确保维度正确
        while len(features) < 11:
            features.append(0.0)

        return features[:11]

    def _extract_onchain_behavior_features(self, node: Node) -> List[float]:
        """提取链上行为特征 (15维) - 基于committee_evolvegcn.go"""
        features = []

        try:
            ob = node.OnChainBehavior

            # 交易处理特征 (2维) - avg_tps, confirmation_delay
            features.extend([
                float(getattr(ob.TransactionCapability, 'AvgTPS', 0)),
                float(getattr(ob.TransactionCapability, 'ConfirmationDelay', 0)),
            ])

            # 跨分片交易特征 (2维) - inter_shard_volume, inter_node_volume
            inter_node_vol = getattr(ob.TransactionCapability.CrossShardTx, 'InterNodeVolume', {})
            inter_shard_vol = getattr(ob.TransactionCapability.CrossShardTx, 'InterShardVolume', {})
            features.extend([
                float(sum(inter_shard_vol.values()) if isinstance(inter_shard_vol, dict) else 0),
                float(sum(inter_node_vol.values()) if isinstance(inter_node_vol, dict) else 0),
            ])

            # 区块生成特征 (2维) - avg_block_interval, block_interval_stddev
            features.extend([
                float(getattr(ob.BlockGeneration, 'AvgInterval', 0)),
                float(getattr(ob.BlockGeneration, 'IntervalStdDev', 0)),
            ])

            # 交易类型特征 (2维) - normal_tx_ratio, contract_tx_ratio
            features.extend([
                float(getattr(ob.TransactionTypes, 'NormalTxRatio', 0)),
                float(getattr(ob.TransactionTypes, 'ContractTxRatio', 0)),
            ])

            # 共识参与特征 (3维) - participation_rate, total_reward, success_rate
            features.extend([
                float(getattr(ob.Consensus, 'ParticipationRate', 0)),
                float(getattr(ob.Consensus, 'TotalReward', 0)),
                float(getattr(ob.Consensus, 'SuccessRate', 0)),
            ])

            # 资源使用特征 (3维) - cpu_usage, memory_usage, resource_flux
            features.extend([
                float(getattr(ob.TransactionCapability.ResourcePerTx, 'CPUPerTx', 0)),
                float(getattr(ob.TransactionCapability.ResourcePerTx, 'MemPerTx', 0)),
                float(getattr(ob.TransactionCapability.ResourcePerTx, 'DiskPerTx', 0)),
            ])

            # 网络动态特征 (3维) - latency_flux, avg_latency, bandwidth_usage (来自动态属性)
            da = node.DynamicAttributes
            features.extend([
                float(getattr(da.Network, 'LatencyFlux', 0)),
                float(getattr(da.Network, 'AvgLatency', 0)),
                float(getattr(da.Network, 'BandwidthUsage', 0)),
            ])

        except Exception as e:
            print(f"Warning: Error extracting onchain behavior features: {e}")
            features = [0.0] * 15

        while len(features) < 15:
            features.append(0.0)

        return features[:15]


    def _extract_network_topology_features(self, node: Node) -> List[float]:
        """提取网络拓扑特征 (5维) - 基于committee_evolvegcn.go"""
        features = []

        try:
            nt = node.NetworkTopology

            # 连接特征 (5维) - intra_shard_conn + inter_shard_conn + weighted_degree + active_conn + adaptability
            features.extend([
                float(getattr(nt.Connections, 'IntraShardConn', 0)),
                float(getattr(nt.Connections, 'InterShardConn', 0)),
                float(getattr(nt.Connections, 'WeightedDegree', 0)),
                float(getattr(nt.Connections, 'ActiveConn', 0)),
                float(getattr(nt.ShardAllocation, 'Adaptability', 0)),
            ])

        except Exception as e:
            print(f"Warning: Error extracting network topology features: {e}")
            features = [0.0] * 5

        while len(features) < 5:
            features.append(0.0)

        return features[:5]

    def _extract_heterogeneous_type_features(self, node: Node) -> List[float]:
        """提取异构类型特征 (2维) - 基于committee_evolvegcn.go"""
        features = []

        try:
            ht = node.HeterogeneousType
            os_status = node.ResourceCapacity.OperationalStatus

            # 节点类型 (1维) - node_type
            node_type = getattr(ht, 'NodeType', 'unknown')
            node_type_value = self.encodings.NODE_TYPE.get(node_type, 0.0)
            features.append(node_type_value)

            # 核心资格 (1维) - core_eligibility
            core_eligibility = getattr(os_status, 'CoreEligibility', False)
            features.append(1.0 if core_eligibility else 0.0)

        except Exception as e:
            print(f"Warning: Error extracting heterogeneous type features: {e}")
            features = [0.0] * 2

        while len(features) < 2:
            features.append(0.0)

        return features[:2]

    def _extract_dynamic_attributes_features(self, node: Node) -> List[float]:
        """提取动态属性特征 (7维) - 基于committee_evolvegcn.go"""
        features = []

        try:
            da = node.DynamicAttributes
            ht = node.HeterogeneousType

            # 交易处理特征 (2维) - tx_frequency, processing_delay
            features.extend([
                float(getattr(da.Transactions, 'Frequency', 0)),
                float(getattr(da.Transactions, 'ProcessingDelay', 0)),
            ])

            # 应用状态特征 (3维) - application_state, tx_frequency_metric, storage_ops  
            app_state = getattr(ht.Application, 'CurrentState', 'unknown')
            app_state_value = self.encodings.APPLICATION_STATE.get(app_state, 0.0)
            
            features.extend([
                app_state_value,
                float(getattr(ht.Application.LoadMetrics, 'TxFrequency', 0)),
                float(getattr(ht.Application.LoadMetrics, 'StorageOps', 0)),
            ])

            # 其他动态特征 (2维) - 填充到7维
            features.extend([
                float(getattr(da.Compute, 'CPUUsage', 0)),
                float(getattr(da.Storage, 'Utilization', 0)),
            ])

        except Exception as e:
            print(f"Warning: Error extracting dynamic attributes features: {e}")
            features = [0.0] * 7

        while len(features) < 7:
            features.append(0.0)

        return features[:7]

# 保持兼容性：创建ClassicFeatureExtractor的别名
ClassicFeatureExtractor = ComprehensiveFeatureExtractor

# 更新UnifiedFeatureExtractor以使用新的全面特征提取器
class UnifiedFeatureExtractor(nn.Module):
    """统一的特征提取器 - 使用全面特征"""

    def __init__(self):
        super().__init__()
        self.dims = FeatureDimensions()

        # 各个组件
        self.comprehensive_extractor = ComprehensiveFeatureExtractor()
        self.sequence_encoder = EnhancedSequenceFeatureEncoder()
        self.graph_encoder = GraphStructureEncoder()
        self.graph_feature_extractor = GraphFeatureExtractor()

        # 特征投影层 - 处理更大的输入维度
        self.feature_projector = nn.Linear(self.dims.CLASSIC_RAW_DIM, self.dims.CLASSIC_DIM)

        print(f"UnifiedFeatureExtractor初始化 - 输入维度: {self.dims.CLASSIC_RAW_DIM}, 输出维度: {self.dims.CLASSIC_DIM}")

    def forward(self, nodes: List[Node]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取F_classic和F_graph - 优化为40维直接处理

        Args:
            nodes: 节点列表

        Returns:
            f_classic: [N, classic_dim]
            f_graph: [N, graph_output_dim]
        """
        print(f"UnifiedFeatureExtractor处理 {len(nodes)} 个节点")

        # 直接提取40维的基础特征（不再拼接额外维度）
        comprehensive_features = self.comprehensive_extractor.extract_features(nodes)  # [N, 40]
        print(f"40维基础特征维度: {comprehensive_features.shape}")

        # 如果维度不匹配，调整投影层输入维度
        if comprehensive_features.shape[1] != self.dims.CLASSIC_RAW_DIM:
            print(f"调整投影层：期望{self.dims.CLASSIC_RAW_DIM}维，实际{comprehensive_features.shape[1]}维")
            if hasattr(self, '_adjusted_projector'):
                f_classic_raw = comprehensive_features
            else:
                # 动态调整投影层
                actual_input_dim = comprehensive_features.shape[1]
                self.feature_projector = nn.Linear(actual_input_dim, self.dims.CLASSIC_DIM)
                self._adjusted_projector = True
                f_classic_raw = comprehensive_features
        else:
            f_classic_raw = comprehensive_features

        # 投影到统一维度
        f_classic = self.feature_projector(f_classic_raw)  # [N, 128]
        print(f"投影后的F_classic维度: {f_classic.shape}")

        # 使用F_classic计算图特征
        f_graph = self.graph_feature_extractor(f_classic, nodes)  # [N, 96]

        print(f"最终输出 - F_classic: {f_classic.shape}, F_graph: {f_graph.shape}")
        return f_classic, f_graph


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
        """保存RGCN中间表示"""
        self.intermediate_representations = {
            'rgcn_input': input_features.detach().cpu(),
            'rgcn_hidden': hidden_features.detach().cpu(),
            'rgcn_output': output_features.detach().cpu(),
            'edge_index': edge_index.cpu(),
            'edge_type': edge_type.cpu(),
            'layer_statistics': {
                'input_stats': self._compute_layer_stats(input_features),
                'hidden_stats': self._compute_layer_stats(hidden_features),
                'output_stats': self._compute_layer_stats(output_features)
            },
            'graph_info': {
                'num_nodes': input_features.shape[0],
                'num_edges': edge_index.shape[1] if edge_index.numel() > 0 else 0,
                'input_dim': input_features.shape[1],
                'hidden_dim': hidden_features.shape[1],
                'output_dim': output_features.shape[1]
            }
        }

    def _compute_layer_stats(self, features: torch.Tensor) -> Dict:
        """计算层特征统计信息"""
        features_np = features.detach().cpu().numpy()
        return {
            'shape': list(features.shape),
            'mean': float(np.mean(features_np)),
            'std': float(np.std(features_np)),
            'min': float(np.min(features_np)),
            'max': float(np.max(features_np)),
            'sparsity': float((features_np == 0).sum() / features_np.size),
            'l2_norm': float(np.linalg.norm(features_np)),
            'max_abs': float(np.max(np.abs(features_np)))
        }

    def save_rgcn_representations(self, base_name: str = "rgcn"):
        """保存RGCN中间表示"""
        if not hasattr(self, 'intermediate_representations') or not self.intermediate_representations:
            print("警告: 没有RGCN中间表示可保存")
            return

        print(f"保存RGCN中间表示...")
        
        # 1. 保存为 .pt 格式
        torch_data = {
            'rgcn_input': self.intermediate_representations['rgcn_input'],
            'rgcn_hidden': self.intermediate_representations['rgcn_hidden'],
            'rgcn_output': self.intermediate_representations['rgcn_output'],
            'edge_index': self.intermediate_representations['edge_index'],
            'edge_type': self.intermediate_representations['edge_type'],
            'metadata': self.intermediate_representations['graph_info']
        }
        
        torch.save(torch_data, f"{base_name}_layers.pt")
        print(f"  ✓ RGCN层表示: {base_name}_layers.pt")

        # 2. 保存统计信息
        stats_data = {
            'layer_statistics': self.intermediate_representations['layer_statistics'],
            'graph_info': self.intermediate_representations['graph_info'],
            'architecture_info': {
                'num_layers': 2,
                'layer_dims': [
                    self.intermediate_representations['graph_info']['input_dim'],
                    self.intermediate_representations['graph_info']['hidden_dim'],
                    self.intermediate_representations['graph_info']['output_dim']
                ],
                'activation': 'ReLU',
                'dropout_rate': 0.2,
                'num_relations': RelationTypes.NUM_RELATIONS,
                'config_dims': {
                    'CLASSIC_DIM': self.dims.CLASSIC_DIM,
                    'GRAPH_OUTPUT_DIM': self.dims.GRAPH_OUTPUT_DIM
                }
            }
        }
        
        import json
        with open(f"{base_name}_stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ RGCN统计信息: {base_name}_stats.json")

        print(f"RGCN中间表示保存完成")

    def get_adjacency_info(self) -> Dict:
        """获取邻接矩阵信息（通过图构建器）"""
        if hasattr(self.graph_builder, 'adjacency_info'):
            return self.graph_builder.adjacency_info
        else:
            return {}

    def get_rgcn_info(self) -> Dict:
        """获取RGCN中间表示信息"""
        return getattr(self, 'intermediate_representations', {})

    def clear_saved_info(self):
        """清除保存的信息（释放内存）"""
        if hasattr(self, 'intermediate_representations'):
            self.intermediate_representations.clear()
        if hasattr(self.graph_builder, 'adjacency_info'):
            self.graph_builder.adjacency_info.clear()
        print("已清除保存的邻接矩阵和RGCN信息")