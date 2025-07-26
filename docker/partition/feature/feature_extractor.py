"""
主要的特征提取器
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F

# 修复相对导入问题
try:
    from .nodeInitialize import Node
    from .data_processor import DataProcessor
    from .graph_builder import HeterogeneousGraphBuilder
    from .config import FeatureDimensions, RelationTypes, NodeTypes, EncodingMaps
    from .sliding_window_extractor import EnhancedSequenceFeatureEncoder
except ImportError:
    import sys
    import importlib.util
    from pathlib import Path
    
    # 使用绝对路径导入
    current_dir = Path(__file__).parent
    
    def load_module(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        return None
    
    # 加载所需模块
    node_init_module = load_module("nodeInitialize", current_dir / "nodeInitialize.py")
    Node = getattr(node_init_module, 'Node', None) if node_init_module else None
    
    data_processor_module = load_module("data_processor", current_dir / "data_processor.py") 
    DataProcessor = getattr(data_processor_module, 'DataProcessor', None) if data_processor_module else None
    
    graph_builder_module = load_module("graph_builder", current_dir / "graph_builder.py")
    HeterogeneousGraphBuilder = getattr(graph_builder_module, 'HeterogeneousGraphBuilder', None) if graph_builder_module else None
    
    config_module = load_module("config", current_dir / "config.py")
    if config_module:
        FeatureDimensions = getattr(config_module, 'FeatureDimensions', None)
        RelationTypes = getattr(config_module, 'RelationTypes', None)
        NodeTypes = getattr(config_module, 'NodeTypes', None) 
        EncodingMaps = getattr(config_module, 'EncodingMaps', None)
    else:
        FeatureDimensions = RelationTypes = NodeTypes = EncodingMaps = None
    
    sliding_window_module = load_module("sliding_window_extractor", current_dir / "sliding_window_extractor.py")
    EnhancedSequenceFeatureEncoder = getattr(sliding_window_module, 'EnhancedSequenceFeatureEncoder', None) if sliding_window_module else None

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
        print(f"ComprehensiveFeatureExtractor输出维度: {feature_tensor.shape} (使用所有特征)")
        return feature_tensor

    def _extract_single_node_comprehensive_features(self, node: Node) -> List[float]:
        """提取单个节点的全面特征"""
        features = []

        # 1. 硬件规格特征 (11维) - CPU(2) + Memory(3) + Storage(3) + Network(3)
        features.extend(self._extract_hardware_features(node))

        # 2. 网络拓扑特征 (8维) - GeoLocation(1) + Connections(4) + ShardAllocation(3)
        features.extend(self._extract_network_topology_features(node))

        # 3. 异构类型特征 (7维) - NodeType(1) + FunctionTags(1) + SupportedFuncs(2) + Application(3)
        features.extend(self._extract_heterogeneous_type_features(node))

        # 4. 链上行为特征 (16维) - TransactionCapability(7) + BlockGeneration(2) + EconomicContribution(1) + SmartContractUsage(1) + TransactionTypes(2) + Consensus(3)
        features.extend(self._extract_onchain_behavior_features(node))

        # 5. 动态属性特征 (13维) - Compute(3) + Storage(2) + Network(3) + Transactions(3) + Reputation(2)
        features.extend(self._extract_dynamic_attributes_features(node))

        # print(f"单节点全面特征维度: {len(features)}")
        return features  # 总计55维

    def _extract_hardware_features(self, node: Node) -> List[float]:
        """提取硬件相关特征 (11维)"""
        features = []

        try:
            hw = node.ResourceCapacity.Hardware

            # CPU特征 (2维)
            features.extend([
                float(getattr(hw.CPU, 'CoreCount', 0)),
                self.encodings.CPU_ARCHITECTURE.get(getattr(hw.CPU, 'Architecture', 'unknown'), 0.0),
            ])

            # 内存特征 (3维)
            features.extend([
                float(getattr(hw.Memory, 'TotalCapacity', 0)),
                float(getattr(hw.Memory, 'Bandwidth', 0)),
                self.encodings.MEMORY_TYPE.get(getattr(hw.Memory, 'Type', 'unknown'), 0.0),
            ])

            # 存储特征 (3维)
            features.extend([
                float(getattr(hw.Storage, 'Capacity', 0)),
                float(getattr(hw.Storage, 'ReadWriteSpeed', 0)),
                self.encodings.STORAGE_TYPE.get(getattr(hw.Storage, 'Type', 'unknown'), 0.0),
            ])

            # 网络特征 (3维)
            features.extend([
                float(getattr(hw.Network, 'UpstreamBW', 0)),
                float(getattr(hw.Network, 'DownstreamBW', 0)),
                self._parse_latency_to_float(getattr(hw.Network, 'Latency', '0ms')),
            ])

        except Exception as e:
            print(f"Warning: Error extracting hardware features: {e}")
            features = [0.0] * 11

        # 确保维度正确
        while len(features) < 11:
            features.append(0.0)

        return features[:11]

    def _extract_onchain_behavior_features(self, node: Node) -> List[float]:
        """提取链上行为特征 (17维)"""
        features = []

        try:
            ob = node.OnChainBehavior

            # 交易能力特征 (6维)
            features.extend([
                float(getattr(ob.TransactionCapability, 'AvgTPS', 0)),
                float(getattr(ob.TransactionCapability, 'ConfirmationDelay', 0)),
                float(getattr(ob.TransactionCapability.ResourcePerTx, 'CPUPerTx', 0)),
                float(getattr(ob.TransactionCapability.ResourcePerTx, 'MemPerTx', 0)),
                float(getattr(ob.TransactionCapability.ResourcePerTx, 'DiskPerTx', 0)),
                float(getattr(ob.TransactionCapability.ResourcePerTx, 'NetworkPerTx', 0)),
            ])

            # 跨分片交易特征 (2维)
            inter_node_vol = getattr(ob.TransactionCapability.CrossShardTx, 'InterNodeVolume', {})
            inter_shard_vol = getattr(ob.TransactionCapability.CrossShardTx, 'InterShardVolume', {})
            features.extend([
                float(sum(inter_node_vol.values()) if isinstance(inter_node_vol, dict) else 0),
                float(sum(inter_shard_vol.values()) if isinstance(inter_shard_vol, dict) else 0),
            ])

            # 区块生成特征 (2维)
            features.extend([
                float(getattr(ob.BlockGeneration, 'AvgInterval', 0)),
                float(getattr(ob.BlockGeneration, 'IntervalStdDev', 0)),
            ])

            # 经济贡献特征 (1维)
            features.extend([
                float(getattr(ob.EconomicContribution, 'FeeContributionRatio', 0)),
            ])

            # 智能合约使用特征 (1维)
            features.extend([
                float(getattr(ob.SmartContractUsage, 'InvocationFrequency', 0)),
            ])

            # 交易类型分布特征 (2维)
            features.extend([
                float(getattr(ob.TransactionTypes, 'NormalTxRatio', 0)),
                float(getattr(ob.TransactionTypes, 'ContractTxRatio', 0)),
            ])

            # 共识参与特征 (3维)
            features.extend([
                float(getattr(ob.Consensus, 'ParticipationRate', 0)),
                float(getattr(ob.Consensus, 'TotalReward', 0)),
                float(getattr(ob.Consensus, 'SuccessRate', 0)),
            ])

        except Exception as e:
            print(f"Warning: Error extracting onchain behavior features: {e}")
            features = [0.0] * 17

        while len(features) < 17:
            features.append(0.0)

        return features[:17]

    def _extract_network_topology_features(self, node: Node) -> List[float]:
        """提取网络拓扑特征 (20维)"""
        features = []

        try:
            nt = node.NetworkTopology

            # 地理位置特征 (3维)
            region = getattr(nt.GeoLocation, 'Region', 'unknown')
            timezone = getattr(nt.GeoLocation, 'Timezone', 'unknown')
            datacenter = getattr(nt.GeoLocation, 'DataCenter', '')

            features.extend([
                self.encodings.REGION.get(region, 0.0),
                self._parse_timezone_offset(timezone),
                float(len(datacenter)) if datacenter else 0.0,  # 数据中心名称长度作为特征
            ])

            # 连接特征 (4维)
            features.extend([
                float(getattr(nt.Connections, 'IntraShardConn', 0)),
                float(getattr(nt.Connections, 'InterShardConn', 0)),
                float(getattr(nt.Connections, 'WeightedDegree', 0)),
                float(getattr(nt.Connections, 'ActiveConn', 0)),
            ])

            # 网络层次特征 (2维)
            features.extend([
                float(getattr(nt.Hierarchy, 'Depth', 0)),
                float(getattr(nt.Hierarchy, 'ConnectionDensity', 0)),
            ])

            # 中心性特征 (4维)
            features.extend([
                float(getattr(nt.Centrality.IntraShard, 'Eigenvector', 0)),
                float(getattr(nt.Centrality.IntraShard, 'Closeness', 0)),
                float(getattr(nt.Centrality.InterShard, 'Betweenness', 0)),
                float(getattr(nt.Centrality.InterShard, 'Influence', 0)),
            ])

            # 分片分配特征 (2维)
            shard_pref = getattr(nt.ShardAllocation, 'ShardPreference', {})
            features.extend([
                float(getattr(nt.ShardAllocation, 'Priority', 0)),
                float(getattr(nt.ShardAllocation, 'Adaptability', 0)),
            ])

            # 分片偏好统计 (5维)
            if isinstance(shard_pref, dict) and shard_pref:
                pref_values = list(shard_pref.values())
                features.extend([
                    float(np.mean(pref_values)),
                    float(np.std(pref_values) if len(pref_values) > 1 else 0),
                    float(max(pref_values)),
                    float(min(pref_values)),
                    float(len(pref_values)),
                ])
            else:
                features.extend([0.0] * 5)

        except Exception as e:
            print(f"Warning: Error extracting network topology features: {e}")
            features = [0.0] * 20

        while len(features) < 20:
            features.append(0.0)

        return features[:20]

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
            print(f"Warning: Error extracting dynamic attributes features: {e}")
            features = [0.0] * 13

        while len(features) < 13:
            features.append(0.0)

        return features[:13]

    def _extract_heterogeneous_type_features(self, node: Node) -> List[float]:
        """提取异构类型特征 (17维)"""
        features = []

        try:
            ht = node.HeterogeneousType

            # 节点类型 (5维 one-hot)
            node_type = getattr(ht, 'NodeType', 'unknown')
            node_type_encoding = self.processor.create_one_hot(node_type, self.encodings.NODE_TYPES)
            features.extend(node_type_encoding)

            # 功能标签统计 (5维)
            function_tags = getattr(ht, 'FunctionTags', [])
            if isinstance(function_tags, list):
                features.extend([
                    float(len(function_tags)),
                    1.0 if 'storage' in function_tags else 0.0,
                    1.0 if 'compute' in function_tags else 0.0,
                    1.0 if 'validation' in function_tags else 0.0,
                    1.0 if 'mining' in function_tags else 0.0,
                ])
            else:
                features.extend([0.0] * 5)

            # 支持功能统计 (3维)
            supported_funcs = getattr(ht.SupportedFuncs, 'Functions', [])
            func_priorities = getattr(ht.SupportedFuncs, 'Priorities', {})

            if isinstance(supported_funcs, list) and isinstance(func_priorities, dict):
                features.extend([
                    float(len(supported_funcs)),
                    float(len(func_priorities)),
                    float(np.mean(list(func_priorities.values())) if func_priorities else 0),
                ])
            else:
                features.extend([0.0] * 3)

            # 应用状态和负载指标 (4维)
            current_state = getattr(ht.Application, 'CurrentState', 'unknown')
            tx_frequency = getattr(ht.Application.LoadMetrics, 'TxFrequency', 0)
            storage_ops = getattr(ht.Application.LoadMetrics, 'StorageOps', 0)

            features.extend([
                self.encodings.APPLICATION_STATE.get(current_state, 0.0),
                float(tx_frequency),
                float(storage_ops),
                float(tx_frequency + storage_ops),  # 总负载指标
            ])

        except Exception as e:
            print(f"Warning: Error extracting heterogeneous type features: {e}")
            features = [0.0] * 17

        while len(features) < 17:
            features.append(0.0)

        return features[:17]

    def _extract_categorical_features(self, node: Node) -> List[float]:
        """提取剩余的分类特征 (15维)"""
        features = []

        try:
            # 这里处理一些需要特殊编码的分类特征
            # 大部分分类特征已经在上面的函数中处理了

            # 添加一些复合特征
            hw = node.ResourceCapacity.Hardware

            # 硬件等级综合评估 (3维)
            cpu_score = float(getattr(hw.CPU, 'CoreCount', 0)) * float(getattr(hw.CPU, 'ClockFrequency', 0))
            memory_score = float(getattr(hw.Memory, 'TotalCapacity', 0)) * float(getattr(hw.Memory, 'Bandwidth', 0))
            storage_score = float(getattr(hw.Storage, 'Capacity', 0)) * float(getattr(hw.Storage, 'ReadWriteSpeed', 0))

            features.extend([cpu_score, memory_score, storage_score])

            # 性能稳定性指标 (3维)
            cpu_util = float(getattr(node.ResourceCapacity.OperationalStatus.ResourceUsage, 'CPUUtilization', 0))
            mem_util = float(getattr(node.ResourceCapacity.OperationalStatus.ResourceUsage, 'MemUtilization', 0))
            uptime = float(getattr(node.ResourceCapacity.OperationalStatus, 'Uptime24h', 0))

            features.extend([
                1.0 - abs(cpu_util - 0.7),  # 理想CPU使用率70%左右
                1.0 - abs(mem_util - 0.6),  # 理想内存使用率60%左右
                uptime / 24.0 if uptime <= 24 else 1.0,  # 在线时间比例
            ])

            # 网络质量综合指标 (3维)
            latency = float(getattr(node.ResourceCapacity.Hardware.Network, 'Latency', 0))
            upstream_bw = float(getattr(node.ResourceCapacity.Hardware.Network, 'UpstreamBW', 0))
            downstream_bw = float(getattr(node.ResourceCapacity.Hardware.Network, 'DownstreamBW', 0))

            features.extend([
                1.0 / (1.0 + latency/100.0),  # 延迟越小越好
                upstream_bw / 1000.0,         # 标准化上游带宽
                downstream_bw / 1000.0,       # 标准化下游带宽
            ])

            # 节点综合评级 (6维)
            tps = float(getattr(node.OnChainBehavior.TransactionCapability, 'AvgTPS', 0))
            consensus_success = float(getattr(node.OnChainBehavior.Consensus, 'SuccessRate', 0))
            reputation = float(getattr(node.DynamicAttributes.Reputation, 'ReputationScore', 0))
            active_conn = float(getattr(node.NetworkTopology.Connections, 'ActiveConn', 0))

            features.extend([
                tps / 1000.0,           # 标准化TPS
                consensus_success,       # 共识成功率
                reputation / 100.0,     # 标准化声誉分数
                active_conn / 100.0,    # 标准化连接数
                (tps * consensus_success * reputation) / 100000.0,  # 综合性能指标
                (uptime * reputation * consensus_success) / 2400.0, # 综合可靠性指标
            ])

        except Exception as e:
            print(f"Warning: Error extracting categorical features: {e}")
            features = [0.0] * 15

        while len(features) < 15:
            features.append(0.0)

        return features[:15]

    def _parse_timezone_offset(self, timezone: str) -> float:
        """解析时区偏移量"""
        if not timezone or timezone == 'unknown':
            return 0.0

        # 直接查找映射
        if timezone in self.encodings.TIMEZONE_OFFSET:
            return self.encodings.TIMEZONE_OFFSET[timezone]

        # 尝试解析UTC±N格式
        try:
            if 'UTC' in timezone:
                offset_str = timezone.replace('UTC', '').strip()
                if offset_str:
                    if offset_str.startswith('+'):
                        return float(offset_str[1:])
                    elif offset_str.startswith('-'):
                        return float(offset_str)
                    else:
                        return float(offset_str)
        except:
            pass

        return 0.0

    def prepare_for_adaptation(self):
        """为自适应调整做准备"""
        # 添加适应性支持
        if not hasattr(self, 'adaptation_enabled'):
            self.adaptation_enabled = True

            # 为各个特征提取方法添加权重
            self.layer_weights = {
                'hardware': 1.0,
                'onchain_behavior': 1.0,
                'network_topology': 1.0,
                'dynamic_attributes': 1.0,
                'heterogeneous_type': 1.0,
                'categorical': 1.0
            }

        print("[SUCCESS] ComprehensiveFeatureExtractor 已准备好自适应调整")

    def apply_layer_weights(self, features: torch.Tensor, layer_weights: Dict[str, float]):
        """应用层权重到特征 (新增方法)"""
        if not hasattr(self, 'layer_weights'):
            self.prepare_for_adaptation()

        # 更新权重
        for layer_name, weight in layer_weights.items():
            if layer_name in self.layer_weights:
                self.layer_weights[layer_name] = weight

        # 按层应用权重到99维特征
        weighted_features = features.clone()

        # 特征层映射 (与adaptive_feature_extractor.py中保持一致)
        layer_slices = {
            'hardware': slice(0, 11),
            'network_topology': slice(11, 16),  
            'heterogeneous_type': slice(16, 18),
            'onchain_behavior': slice(18, 33),
            'dynamic_attributes': slice(33, 40),
            'categorical': slice(84, 99)
        }

        for layer_name, slice_range in layer_slices.items():
            if layer_name in self.layer_weights:
                weight = self.layer_weights[layer_name]
                weighted_features[:, slice_range] *= weight

        return weighted_features

    def get_current_weights(self) -> Dict[str, float]:
        """获取当前层权重"""
        if hasattr(self, 'layer_weights'):
            return self.layer_weights.copy()
        else:
            return {}

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
        提取F_classic和F_graph

        Args:
            nodes: 节点列表

        Returns:
            f_classic: [N, classic_dim]
            f_graph: [N, graph_output_dim]
        """
        print(f"UnifiedFeatureEx"
              f"tractor处理 {len(nodes)} 个节点")

        # 提取全面的基础特征
        comprehensive_features = self.comprehensive_extractor.extract_features(nodes)  # [N, ~99]

        # 编码时序特征
        sequence_features = self.sequence_encoder(nodes)  # [N, 32]

        # 编码图结构特征
        graph_structure_features = self.graph_encoder(nodes)  # [N, 10]

        # 拼接所有经典特征
        f_classic_raw = torch.cat([
            comprehensive_features,
            sequence_features,
            graph_structure_features
        ], dim=1)  # [N, ~141]

        print(f"拼接后的全面特征维度: {f_classic_raw.shape}")

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
                    features = [0.0] * 10

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