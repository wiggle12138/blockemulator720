"""
BlockEmulator特征提取适配器 - 实时系统对接版本
连接BlockEmulator系统与动态分片算法的桥梁，支持实时特征提取
"""

import torch
import numpy as np
import json
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import time
from datetime import datetime
import os

# 导入原有的特征提取组件 - 使用绝对导入避免相对导入问题
import sys
from pathlib import Path

# 添加feature目录到系统路径
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 直接导入模块，失败时立即报错
try:
    from feature_extractor import UnifiedFeatureExtractor, ComprehensiveFeatureExtractor
except ImportError as e:
    raise ImportError(f"feature_extractor导入失败，必须使用真实实现: {e}")

try:
    from nodeInitialize import Node
except ImportError as e:
    raise ImportError(f"nodeInitialize导入失败，必须使用真实实现: {e}")

try:
    from data_processor import DataProcessor
except ImportError as e:
    raise ImportError(f"data_processor导入失败，必须使用真实实现: {e}")

try:
    from config import FeatureDimensions, EncodingMaps
except ImportError as e:
    raise ImportError(f"config导入失败，必须使用真实实现: {e}")

try:
    from graph_builder import HeterogeneousGraphBuilder
except ImportError as e:
    raise ImportError(f"graph_builder导入失败，必须使用真实实现: {e}")

@dataclass
class BlockEmulatorNodeData:
    """BlockEmulator节点数据结构 - 匹配Go系统格式"""
    shard_id: int
    node_id: int
    timestamp: int
    request_id: str
    
    # 静态特征 (对应NodeState.Static)
    static: Dict[str, Any]
    
    # 动态特征 (对应NodeState.Dynamic) 
    dynamic: Dict[str, Any]

class BlockEmulatorAdapter:
    """
    BlockEmulator实时特征提取适配器
    
    主要功能:
    1. 直接从GetAllCollectedData()获取节点数据
    2. 转换为内部Node对象格式
    3. 提取f_classic和f_graph特征
    4. 输出标准化的特征格式供后续步骤使用
    """
    
    def __init__(self):
        """初始化适配器"""
        # 初始化核心组件
        self.dims = FeatureDimensions()
        self.processor = DataProcessor()
        self.encodings = EncodingMaps()
        self.unified_extractor = UnifiedFeatureExtractor()
        self.comprehensive_extractor = ComprehensiveFeatureExtractor()
        
        # 输出维度规格
        self.output_dims = {
            'f_classic': self.dims.CLASSIC_DIM,      # 128维经典特征
            'f_graph': self.dims.GRAPH_OUTPUT_DIM,   # 96维图特征  
            'f_reduced': 64,                         # 64维精简特征
            'total_nodes': 0
        }
        
        # BlockEmulator特征字段映射 (基于committee_evolvegcn.go的extractRealStaticFeatures和extractRealDynamicFeatures)
        # 总计40个特征字段：18个静态特征 + 22个动态特征
        self.be_field_mapping = {
            # === 静态特征 (18个字段) ===
            # 硬件特征 (11个字段)
            "cpu_cores": "static.hardware.cpu.cores",                    # float64(static.ResourceCapacity.Hardware.CPU.CoreCount)
            "cpu_clock_freq": "static.hardware.cpu.clock_freq",          # static.ResourceCapacity.Hardware.CPU.ClockFrequency
            "memory_gb": "static.hardware.memory.capacity",              # float64(static.ResourceCapacity.Hardware.Memory.TotalCapacity)
            "memory_bandwidth": "static.hardware.memory.bandwidth",      # static.ResourceCapacity.Hardware.Memory.Bandwidth
            "memory_type": "static.hardware.memory.type",                # encodeMemoryType(static.ResourceCapacity.Hardware.Memory.Type)
            "storage_gb": "static.hardware.storage.capacity",            # float64(static.ResourceCapacity.Hardware.Storage.Capacity)
            "storage_type": "static.hardware.storage.type",              # encodeStorageType(static.ResourceCapacity.Hardware.Storage.Type)
            "storage_rw_speed": "static.hardware.storage.speed",         # static.ResourceCapacity.Hardware.Storage.ReadWriteSpeed
            "network_upstream": "static.hardware.network.upstream",      # static.ResourceCapacity.Hardware.Network.UpstreamBW
            "network_downstream": "static.hardware.network.downstream",  # static.ResourceCapacity.Hardware.Network.DownstreamBW
            "network_latency": "static.hardware.network.latency",        # parseLatency(static.ResourceCapacity.Hardware.Network.Latency)
            
            # 网络拓扑特征 (5个字段)
            "intra_shard_conn": "static.network.conn.intra_shard",       # float64(static.NetworkTopology.Connections.IntraShardConn)
            "inter_shard_conn": "static.network.conn.inter_shard",       # float64(static.NetworkTopology.Connections.InterShardConn)
            "weighted_degree": "static.network.conn.weighted_degree",    # static.NetworkTopology.Connections.WeightedDegree
            "active_conn": "static.network.conn.active",                 # float64(static.NetworkTopology.Connections.ActiveConn)
            "adaptability": "static.network.shard.adaptability",         # static.NetworkTopology.ShardAllocation.Adaptability
            
            # 异构类型特征 (2个字段)
            "node_type": "static.hetero.node_type",                      # encodeNodeType(static.HeterogeneousType.NodeType)
            "core_eligibility": "static.hetero.core_eligibility",        # 默认值1.0 (原字段不存在)
            
            # === 动态特征 (22个字段) ===
            # 交易处理能力 (2个字段)
            "avg_tps": "dynamic.onchain.tx.avg_tps",                     # dynamic.OnChainBehavior.TransactionCapability.AvgTPS
            "confirmation_delay": "dynamic.onchain.tx.confirmation_delay", # parseDelay(dynamic.OnChainBehavior.TransactionCapability.ConfirmationDelay)
            
            # 跨分片交易 (2个字段)
            "inter_shard_volume": "dynamic.onchain.cross.inter_shard",   # parseVolumeString(dynamic.OnChainBehavior.TransactionCapability.CrossShardTx.InterShardVolume)
            "inter_node_volume": "dynamic.onchain.cross.inter_node",     # parseVolumeString(dynamic.OnChainBehavior.TransactionCapability.CrossShardTx.InterNodeVolume)
            
            # 区块生成 (2个字段)
            "avg_block_interval": "dynamic.onchain.block.avg_interval",  # parseInterval(dynamic.OnChainBehavior.BlockGeneration.AvgInterval)
            "block_interval_stddev": "dynamic.onchain.block.stddev",     # parseInterval(dynamic.OnChainBehavior.BlockGeneration.IntervalStdDev)
            
            # 交易类型 (2个字段)
            "normal_tx_ratio": "dynamic.onchain.tx_type.normal_ratio",   # dynamic.OnChainBehavior.TransactionTypes.NormalTxRatio
            "contract_tx_ratio": "dynamic.onchain.tx_type.contract_ratio", # dynamic.OnChainBehavior.TransactionTypes.ContractTxRatio
            
            # 共识参与 (3个字段)
            "participation_rate": "dynamic.onchain.consensus.participation", # dynamic.OnChainBehavior.Consensus.ParticipationRate
            "total_reward": "dynamic.onchain.consensus.total_reward",    # dynamic.OnChainBehavior.Consensus.TotalReward
            "success_rate": "dynamic.onchain.consensus.success_rate",    # dynamic.OnChainBehavior.Consensus.SuccessRate
            
            # 资源使用率 (3个字段)
            "cpu_usage": "dynamic.resource.cpu_usage",                   # dynamic.DynamicAttributes.Compute.CPUUsage
            "memory_usage": "dynamic.resource.memory_usage",             # dynamic.DynamicAttributes.Compute.MemUsage
            "resource_flux": "dynamic.resource.resource_flux",           # dynamic.DynamicAttributes.Compute.ResourceFlux
            
            # 网络动态 (3个字段)
            "latency_flux": "dynamic.network.latency_flux",              # dynamic.DynamicAttributes.Network.LatencyFlux
            "avg_latency": "dynamic.network.avg_latency",                # parseLatency(dynamic.DynamicAttributes.Network.AvgLatency)
            "bandwidth_usage": "dynamic.network.bandwidth_usage",        # dynamic.DynamicAttributes.Network.BandwidthUsage
            
            # 交易处理 (2个字段)
            "tx_frequency": "dynamic.tx.frequency",                      # float64(dynamic.DynamicAttributes.Transactions.Frequency)
            "processing_delay": "dynamic.tx.processing_delay",           # parseDelay(dynamic.DynamicAttributes.Transactions.ProcessingDelay)
            
            # 应用状态 (3个字段)
            "application_state": "dynamic.app.state",                    # encodeApplicationState(nodeState.NodeState.Static.HeterogeneousType.Application.CurrentState)
            "tx_frequency_metric": "dynamic.app.tx_frequency_metric",    # float64(nodeState.NodeState.Static.HeterogeneousType.Application.LoadMetrics.TxFrequency)
            "storage_ops": "dynamic.app.storage_ops",                    # float64(nodeState.NodeState.Static.HeterogeneousType.Application.LoadMetrics.StorageOps)
        }
        
        print(f"BlockEmulatorAdapter初始化完成")
        print(f"  支持字段数量: {len(self.be_field_mapping)} (期望40个)")
        print(f"  输出特征维度: F_classic={self.output_dims['f_classic']}, F_graph={self.output_dims['f_graph']}")

    def extract_features_realtime(self, node_data_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        实时特征提取 - 主要接口方法
        
        Args:
            node_data_list: 从BlockEmulator.GetAllCollectedData()获取的节点数据
            格式: [{"ShardID": int, "NodeID": int, "NodeState": {"Static": {...}, "Dynamic": {...}}}, ...]
        
        Returns:
            特征字典: {
                'f_classic': torch.Tensor,     # [N, 128] 经典特征
                'f_graph': torch.Tensor,       # [N, 96] 图特征  
                'f_reduced': torch.Tensor,     # [N, 64] 精简特征
                'node_mapping': Dict,          # 节点映射关系
                'metadata': Dict               # 元数据信息
            }
        """
        if not node_data_list:
            return self._get_empty_result()
            
        print(f"\n=== 开始实时特征提取 ===")
        print(f"输入节点数量: {len(node_data_list)}")
        
        start_time = time.time()
        
        # 1. 数据格式转换
        converted_nodes = self._convert_be_data(node_data_list)
        print(f"✓ 数据转换完成: {len(converted_nodes)} 个节点")
        
        # 2. 构建Node对象
        node_objects = self._build_node_objects(converted_nodes)
        print(f"✓ Node对象构建完成: {len(node_objects)} 个对象")
        
        # 3. 提取特征
        f_classic, f_graph = self.unified_extractor(node_objects)
        print(f"✓ 特征提取完成: F_classic={f_classic.shape}, F_graph={f_graph.shape}")
        
        # 4. 生成精简特征
        f_reduced = self._generate_reduced_features(f_classic, f_graph)
        print(f"✓ 精简特征生成: F_reduced={f_reduced.shape}")
        
        # 5. 构建映射和元数据
        node_mapping = self._build_node_mapping(converted_nodes)
        metadata = self._generate_metadata(converted_nodes, f_classic, f_graph)
        
        processing_time = time.time() - start_time
        print(f"✓ 特征提取总耗时: {processing_time:.3f}秒")
        
        result = {
            'f_classic': f_classic,
            'f_graph': f_graph, 
            'f_reduced': f_reduced,
            'node_mapping': node_mapping,
            'metadata': metadata
        }
        
        # 更新元数据中的处理时间
        result['metadata']['processing_time'] = processing_time
        result['metadata']['nodes_per_second'] = len(node_data_list) / processing_time
        
        print(f"=== 实时特征提取完成 ===\n")
        return result

    def _convert_be_data(self, node_data_list: List[Dict[str, Any]]) -> List[BlockEmulatorNodeData]:
        """转换BlockEmulator数据格式"""
        converted = []
        
        for raw_data in node_data_list:
            try:
                # 提取基本信息
                shard_id = raw_data.get('ShardID', 0)
                node_id = raw_data.get('NodeID', 0)
                timestamp = raw_data.get('Timestamp', int(time.time() * 1000))
                request_id = raw_data.get('RequestID', f"req_{shard_id}_{node_id}")
                
                # 提取节点状态
                node_state = raw_data.get('NodeState', {})
                static_features = node_state.get('Static', {})
                dynamic_features = node_state.get('Dynamic', {})
                
                converted_node = BlockEmulatorNodeData(
                    shard_id=shard_id,
                    node_id=node_id,
                    timestamp=timestamp,
                    request_id=request_id,
                    static=static_features,
                    dynamic=dynamic_features
                )
                
                converted.append(converted_node)
                
            except Exception as e:
                print(f"警告: 转换节点数据失败 S{shard_id}N{node_id} - {e}")
                continue
                
        return converted

    def _build_node_objects(self, converted_nodes: List[BlockEmulatorNodeData]) -> List[Node]:
        """构建Node对象供特征提取器使用"""
        node_objects = []
        
        for converted_node in converted_nodes:
            try:
                node = Node()
                
                # 填充各类特征
                self._populate_hardware_features(node, converted_node)
                self._populate_onchain_features(node, converted_node) 
                self._populate_network_features(node, converted_node)
                self._populate_dynamic_features(node, converted_node)
                self._populate_heterogeneous_features(node, converted_node)
                
                node_objects.append(node)
                
            except Exception as e:
                print(f"警告: 构建Node对象失败 S{converted_node.shard_id}N{converted_node.node_id} - {e}")
                # 添加默认Node对象
                node_objects.append(Node())
                
        return node_objects

    def _populate_hardware_features(self, node: Node, converted_node: BlockEmulatorNodeData):
        """填充硬件特征到Node对象"""
        try:
            static = converted_node.static
            resource_capacity = static.get('ResourceCapacity', {})
            hardware = resource_capacity.get('Hardware', {})
            
            # CPU特征
            cpu = hardware.get('CPU', {})
            node.ResourceCapacity.Hardware.CPU.CoreCount = cpu.get('CoreCount', 4)
            node.ResourceCapacity.Hardware.CPU.Architecture = cpu.get('Architecture', 'x86_64')
            node.ResourceCapacity.Hardware.CPU.ClockFrequency = cpu.get('ClockFrequency', 2400.0)
            node.ResourceCapacity.Hardware.CPU.CacheSize = cpu.get('CacheSize', 8)
            
            # 内存特征
            memory = hardware.get('Memory', {})
            node.ResourceCapacity.Hardware.Memory.TotalCapacity = memory.get('TotalCapacity', 8192)
            node.ResourceCapacity.Hardware.Memory.Type = memory.get('Type', 'DDR4')
            node.ResourceCapacity.Hardware.Memory.Bandwidth = memory.get('Bandwidth', 25600.0)
            
            # 存储特征
            storage = hardware.get('Storage', {})
            node.ResourceCapacity.Hardware.Storage.Capacity = storage.get('Capacity', 512)
            node.ResourceCapacity.Hardware.Storage.Type = storage.get('Type', 'SSD')
            node.ResourceCapacity.Hardware.Storage.ReadWriteSpeed = storage.get('ReadWriteSpeed', 500.0)
            
            # 网络特征
            network = hardware.get('Network', {})
            node.ResourceCapacity.Hardware.Network.UpstreamBW = network.get('UpstreamBW', 1000.0)
            node.ResourceCapacity.Hardware.Network.DownstreamBW = network.get('DownstreamBW', 1000.0)
            node.ResourceCapacity.Hardware.Network.Latency = network.get('Latency', 10)
            
            # 运营状态
            operational = resource_capacity.get('OperationalStatus', {})
            node.ResourceCapacity.OperationalStatus.Uptime24h = operational.get('Uptime24h', 24.0)
            node.ResourceCapacity.OperationalStatus.CoreEligibility = operational.get('CoreEligibility', True)
            
            resource_usage = operational.get('ResourceUsage', {})
            node.ResourceCapacity.OperationalStatus.ResourceUsage.CPUUtilization = resource_usage.get('CPUUtilization', 0.7)
            node.ResourceCapacity.OperationalStatus.ResourceUsage.MemUtilization = resource_usage.get('MemUtilization', 0.6)
            
        except Exception as e:
            print(f"警告: 填充硬件特征失败 - {e}")

    def _populate_onchain_features(self, node: Node, converted_node: BlockEmulatorNodeData):
        """填充链上行为特征"""
        try:
            dynamic = converted_node.dynamic
            onchain = dynamic.get('OnChainBehavior', {})
            
            # 交易能力
            tx_capability = onchain.get('TransactionCapability', {})
            node.OnChainBehavior.TransactionCapability.AvgTPS = tx_capability.get('AvgTPS', 100.0)
            node.OnChainBehavior.TransactionCapability.ConfirmationDelay = tx_capability.get('ConfirmationDelay', 5000)
            
            # 跨分片交易
            cross_shard = tx_capability.get('CrossShardTx', {})
            node.OnChainBehavior.TransactionCapability.CrossShardTx.InterNodeVolume = cross_shard.get('InterNodeVolume', {})
            node.OnChainBehavior.TransactionCapability.CrossShardTx.InterShardVolume = cross_shard.get('InterShardVolume', {})
            
            # 每交易资源消耗
            resource_per_tx = tx_capability.get('ResourcePerTx', {})
            node.OnChainBehavior.TransactionCapability.ResourcePerTx.CPUPerTx = resource_per_tx.get('CPUPerTx', 0.1)
            node.OnChainBehavior.TransactionCapability.ResourcePerTx.MemPerTx = resource_per_tx.get('MemPerTx', 0.05)
            node.OnChainBehavior.TransactionCapability.ResourcePerTx.DiskPerTx = resource_per_tx.get('DiskPerTx', 0.01)
            node.OnChainBehavior.TransactionCapability.ResourcePerTx.NetworkPerTx = resource_per_tx.get('NetworkPerTx', 0.02)
            
            # 区块生成
            block_gen = onchain.get('BlockGeneration', {})
            node.OnChainBehavior.BlockGeneration.AvgInterval = block_gen.get('AvgInterval', 10000)
            node.OnChainBehavior.BlockGeneration.IntervalStdDev = block_gen.get('IntervalStdDev', 1000)
            
            # 经济贡献
            economic = onchain.get('EconomicContribution', {})
            node.OnChainBehavior.EconomicContribution.FeeContributionRatio = economic.get('FeeContributionRatio', 0.02)
            
            # 智能合约使用
            contract_usage = onchain.get('SmartContractUsage', {})
            node.OnChainBehavior.SmartContractUsage.InvocationFrequency = contract_usage.get('InvocationFrequency', 5)
            
            # 交易类型
            tx_types = onchain.get('TransactionTypes', {})
            node.OnChainBehavior.TransactionTypes.NormalTxRatio = tx_types.get('NormalTxRatio', 0.8)
            node.OnChainBehavior.TransactionTypes.ContractTxRatio = tx_types.get('ContractTxRatio', 0.2)
            
            # 共识参与
            consensus = onchain.get('Consensus', {})
            node.OnChainBehavior.Consensus.ParticipationRate = consensus.get('ParticipationRate', 0.95)
            node.OnChainBehavior.Consensus.TotalReward = consensus.get('TotalReward', 1000.0)
            node.OnChainBehavior.Consensus.SuccessRate = consensus.get('SuccessRate', 0.98)
            
        except Exception as e:
            print(f"警告: 填充链上行为特征失败 - {e}")

    def _populate_network_features(self, node: Node, converted_node: BlockEmulatorNodeData):
        """填充网络拓扑特征"""
        try:
            static = converted_node.static
            network_topology = static.get('NetworkTopology', {})
            
            # 地理位置
            geo_location = network_topology.get('GeoLocation', {})
            node.NetworkTopology.GeoLocation.Region = geo_location.get('Region', 'Asia')
            node.NetworkTopology.GeoLocation.Timezone = geo_location.get('Timezone', 'Asia/Shanghai')
            node.NetworkTopology.GeoLocation.DataCenter = geo_location.get('DataCenter', 'DC1')
            
            # 连接信息
            connections = network_topology.get('Connections', {})
            node.NetworkTopology.Connections.IntraShardConn = connections.get('IntraShardConn', 5)
            node.NetworkTopology.Connections.InterShardConn = connections.get('InterShardConn', 2)
            node.NetworkTopology.Connections.WeightedDegree = connections.get('WeightedDegree', 10.0)
            node.NetworkTopology.Connections.ActiveConn = connections.get('ActiveConn', 7)
            
            # 网络层次
            hierarchy = network_topology.get('Hierarchy', {})
            node.NetworkTopology.Hierarchy.Depth = hierarchy.get('Depth', 3)
            node.NetworkTopology.Hierarchy.ConnectionDensity = hierarchy.get('ConnectionDensity', 0.6)
            
            # 中心性指标
            centrality = network_topology.get('Centrality', {})
            intra_shard = centrality.get('IntraShard', {})
            node.NetworkTopology.Centrality.IntraShard.Eigenvector = intra_shard.get('Eigenvector', 0.5)
            node.NetworkTopology.Centrality.IntraShard.Closeness = intra_shard.get('Closeness', 0.6)
            
            inter_shard = centrality.get('InterShard', {})
            node.NetworkTopology.Centrality.InterShard.Betweenness = inter_shard.get('Betweenness', 0.3)
            node.NetworkTopology.Centrality.InterShard.Influence = inter_shard.get('Influence', 0.4)
            
            # 分片分配
            shard_allocation = network_topology.get('ShardAllocation', {})
            node.NetworkTopology.ShardAllocation.Priority = shard_allocation.get('Priority', 5)
            node.NetworkTopology.ShardAllocation.ShardPreference = shard_allocation.get('ShardPreference', {})
            node.NetworkTopology.ShardAllocation.Adaptability = shard_allocation.get('Adaptability', 0.8)
            
        except Exception as e:
            print(f"警告: 填充网络拓扑特征失败 - {e}")

    def _populate_dynamic_features(self, node: Node, converted_node: BlockEmulatorNodeData):
        """填充动态属性特征"""
        try:
            dynamic = converted_node.dynamic
            dynamic_attrs = dynamic.get('DynamicAttributes', {})
            
            # 计算指标
            compute = dynamic_attrs.get('Compute', {})
            node.DynamicAttributes.Compute.CPUUsage = compute.get('CPUUsage', 0.7)
            node.DynamicAttributes.Compute.MemUsage = compute.get('MemUsage', 0.6)
            node.DynamicAttributes.Compute.ResourceFlux = compute.get('ResourceFlux', 0.1)
            
            # 存储指标
            storage = dynamic_attrs.get('Storage', {})
            node.DynamicAttributes.Storage.Available = storage.get('Available', 400)
            node.DynamicAttributes.Storage.Utilization = storage.get('Utilization', 0.6)
            
            # 网络指标
            network = dynamic_attrs.get('Network', {})
            node.DynamicAttributes.Network.LatencyFlux = network.get('LatencyFlux', 5.0)
            node.DynamicAttributes.Network.AvgLatency = network.get('AvgLatency', 10)
            node.DynamicAttributes.Network.BandwidthUsage = network.get('BandwidthUsage', 0.6)
            
            # 交易指标
            transactions = dynamic_attrs.get('Transactions', {})
            node.DynamicAttributes.Transactions.Frequency = transactions.get('Frequency', 50)
            node.DynamicAttributes.Transactions.ProcessingDelay = transactions.get('ProcessingDelay', 2000)
            node.DynamicAttributes.Transactions.StakeChangeRate = transactions.get('StakeChangeRate', 0.05)
            
            # 声誉指标
            reputation = dynamic_attrs.get('Reputation', {})
            node.DynamicAttributes.Reputation.Uptime24h = reputation.get('Uptime24h', 24.0)
            node.DynamicAttributes.Reputation.ReputationScore = reputation.get('ReputationScore', 85.0)
            
        except Exception as e:
            print(f"警告: 填充动态属性特征失败 - {e}")

    def _populate_heterogeneous_features(self, node: Node, converted_node: BlockEmulatorNodeData):
        """填充异构类型特征"""
        try:
            static = converted_node.static
            heterogeneous = static.get('HeterogeneousType', {})
            
            # 节点类型
            node.HeterogeneousType.NodeType = heterogeneous.get('NodeType', 'full')
            
            # 功能标签
            function_tags = heterogeneous.get('FunctionTags', [])
            if isinstance(function_tags, str):
                node.HeterogeneousType.FunctionTags = function_tags.split(',') if function_tags else []
            else:
                node.HeterogeneousType.FunctionTags = function_tags or ['compute', 'storage']
            
            # 支持功能
            supported_funcs = heterogeneous.get('SupportedFuncs', {})
            functions = supported_funcs.get('Functions', [])
            if isinstance(functions, str):
                node.HeterogeneousType.SupportedFuncs.Functions = functions.split(',') if functions else []
            else:
                node.HeterogeneousType.SupportedFuncs.Functions = functions or ['validation', 'storage']
            
            priorities = supported_funcs.get('Priorities', {})
            node.HeterogeneousType.SupportedFuncs.Priorities = priorities if isinstance(priorities, dict) else {}
            
            # 应用状态
            application = heterogeneous.get('Application', {})
            node.HeterogeneousType.Application.CurrentState = application.get('CurrentState', 'active')
            
            # 负载指标
            load_metrics = application.get('LoadMetrics', {})
            node.HeterogeneousType.Application.LoadMetrics.TxFrequency = load_metrics.get('TxFrequency', 20)
            node.HeterogeneousType.Application.LoadMetrics.StorageOps = load_metrics.get('StorageOps', 10)
            
        except Exception as e:
            print(f"警告: 填充异构类型特征失败 - {e}")

    def _generate_reduced_features(self, f_classic: torch.Tensor, f_graph: torch.Tensor) -> torch.Tensor:
        """生成精简特征（用于轻量化部署和快速处理）"""
        try:
            # 拼接特征
            combined = torch.cat([f_classic, f_graph], dim=1)  # [N, 128+96=224]
            
            N, D = combined.shape
            target_dim = self.output_dims['f_reduced']
            
            if D <= target_dim:
                # 维度足够小，直接填充
                f_reduced = torch.zeros(N, target_dim)
                f_reduced[:, :D] = combined
            else:
                # 分组平均降维
                group_size = D // target_dim
                f_reduced = torch.zeros(N, target_dim)
                
                for i in range(target_dim):
                    start_idx = i * group_size
                    end_idx = min(start_idx + group_size, D)
                    if start_idx < D:
                        f_reduced[:, i] = combined[:, start_idx:end_idx].mean(dim=1)
            
            return f_reduced
            
        except Exception as e:
            print(f"警告: 生成精简特征失败 - {e}")
            return torch.zeros(f_classic.shape[0], self.output_dims['f_reduced'])

    def _build_node_mapping(self, converted_nodes: List[BlockEmulatorNodeData]) -> Dict[str, Dict]:
        """构建节点映射关系"""
        mapping = {}
        
        for i, node in enumerate(converted_nodes):
            node_key = f"S{node.shard_id}N{node.node_id}"
            mapping[node_key] = {
                'index': i,
                'shard_id': node.shard_id,
                'node_id': node.node_id,
                'timestamp': node.timestamp,
                'request_id': node.request_id
            }
            
        return mapping

    def _generate_metadata(self, converted_nodes: List[BlockEmulatorNodeData], 
                          f_classic: torch.Tensor, f_graph: torch.Tensor) -> Dict[str, Any]:
        """生成元数据信息"""
        
        # 分片分布统计
        shard_distribution = {}
        for node in converted_nodes:
            shard_id = node.shard_id
            shard_distribution[shard_id] = shard_distribution.get(shard_id, 0) + 1
        
        # 特征统计
        f_classic_np = f_classic.detach().cpu().numpy()
        f_graph_np = f_graph.detach().cpu().numpy()
        
        metadata = {
            'total_nodes': len(converted_nodes),
            'shard_distribution': shard_distribution,
            'num_shards': len(shard_distribution),
            'supported_fields': len(self.be_field_mapping),
            'feature_stats': {
                'f_classic': {
                    'shape': list(f_classic.shape),
                    'mean': float(np.mean(f_classic_np)),
                    'std': float(np.std(f_classic_np)),
                    'min': float(np.min(f_classic_np)),
                    'max': float(np.max(f_classic_np))
                },
                'f_graph': {
                    'shape': list(f_graph.shape),
                    'mean': float(np.mean(f_graph_np)),
                    'std': float(np.std(f_graph_np)),
                    'min': float(np.min(f_graph_np)),
                    'max': float(np.max(f_graph_np))
                }
            },
            'processing_timestamp': datetime.now().isoformat(),
            'adapter_version': '2.0.0'
        }
        
        return metadata

    def _get_empty_result(self) -> Dict[str, torch.Tensor]:
        """返回空结果"""
        return {
            'f_classic': torch.zeros(0, self.output_dims['f_classic']),
            'f_graph': torch.zeros(0, self.output_dims['f_graph']),
            'f_reduced': torch.zeros(0, self.output_dims['f_reduced']),
            'node_mapping': {},
            'metadata': {
                'total_nodes': 0,
                'shard_distribution': {},
                'num_shards': 0,
                'feature_stats': {},
                'processing_timestamp': datetime.now().isoformat(),
                'adapter_version': '2.0.0'
            }
        }

    def save_features_for_next_steps(self, features: Dict[str, torch.Tensor], 
                                   output_dir: str = "step1_output") -> Dict[str, str]:
        """
        保存特征文件供后续步骤使用
        
        Returns:
            保存的文件路径字典
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        # 保存torch tensor格式（第二步多尺度学习需要）
        for feature_name in ['f_classic', 'f_graph', 'f_reduced']:
            if feature_name in features and features[feature_name].numel() > 0:
                file_path = os.path.join(output_dir, f"{feature_name}.pt")
                torch.save(features[feature_name], file_path)
                saved_files[f"{feature_name}_pt"] = file_path
        
        # 保存CSV格式（调试和分析用）
        for feature_name in ['f_classic', 'f_graph', 'f_reduced']:
            if feature_name in features and features[feature_name].numel() > 0:
                feature_tensor = features[feature_name]
                df = pd.DataFrame(feature_tensor.detach().cpu().numpy())
                file_path = os.path.join(output_dir, f"{feature_name}.csv")
                df.to_csv(file_path, index=False)
                saved_files[f"{feature_name}_csv"] = file_path
        
        # 保存节点映射和元数据
        if 'node_mapping' in features:
            file_path = os.path.join(output_dir, "node_mapping.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(features['node_mapping'], f, indent=2, ensure_ascii=False)
            saved_files['node_mapping'] = file_path
        
        if 'metadata' in features:
            file_path = os.path.join(output_dir, "metadata.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(features['metadata'], f, indent=2, ensure_ascii=False)
            saved_files['metadata'] = file_path
        
        print(f"第一步特征文件已保存到 {output_dir}:")
        for key, path in saved_files.items():
            print(f"  - {key}: {path}")
        
        return saved_files

    def get_step1_output_info(self) -> Dict[str, Any]:
        """获取第一步输出格式信息"""
        return {
            'step': 1,
            'name': 'Feature Extraction',
            'output_dimensions': self.output_dims,
            'output_files': {
                'f_classic.pt': f'经典特征张量 [{self.output_dims["f_classic"]}维]',
                'f_graph.pt': f'图特征张量 [{self.output_dims["f_graph"]}维]',
                'f_reduced.pt': f'精简特征张量 [{self.output_dims["f_reduced"]}维]',
                'node_mapping.json': '节点ID到索引的映射关系',
                'metadata.json': '特征提取元数据信息'
            },
            'next_step_compatibility': {
                'step2_multi_scale': 'f_classic.pt和f_graph.pt作为输入',
                'step3_evolve_gcn': 'f_classic.pt作为节点嵌入',
                'step4_feedback': 'f_reduced.pt用于快速性能评估'
            },
            'real_time_processing': True,
            'batch_size_limit': None,
            'processing_speed': 'estimated ~1000 nodes/second'
        }


# 测试和演示功能
def create_mock_emulator_data(num_nodes: int = 10, num_shards: int = 4) -> List[Dict[str, Any]]:
    """创建模拟的BlockEmulator数据"""
    import random
    
    mock_data = []
    
    for i in range(num_nodes):
        shard_id = i % num_shards
        node_id = i + 1
        
        mock_node = {
            "ShardID": shard_id,
            "NodeID": node_id,
            "Timestamp": int(time.time() * 1000),
            "RequestID": f"req_{shard_id}_{node_id}",
            "NodeState": {
                "Static": {
                    "ResourceCapacity": {
                        "Hardware": {
                            "CPU": {
                                "CoreCount": random.choice([4, 8, 16]),
                                "Architecture": "x86_64",
                                "ClockFrequency": random.uniform(2000, 4000)
                            },
                            "Memory": {
                                "TotalCapacity": random.choice([8192, 16384, 32768]),
                                "Type": "DDR4",
                                "Bandwidth": random.uniform(20000, 40000)
                            },
                            "Storage": {
                                "Capacity": random.choice([512, 1024, 2048]),
                                "Type": "SSD",
                                "ReadWriteSpeed": random.uniform(400, 600)
                            },
                            "Network": {
                                "UpstreamBW": random.uniform(500, 2000),
                                "DownstreamBW": random.uniform(500, 2000),
                                "Latency": random.randint(5, 20)
                            }
                        },
                        "OperationalStatus": {
                            "Uptime24h": random.uniform(20, 24),
                            "CoreEligibility": random.choice([True, False]),
                            "ResourceUsage": {
                                "CPUUtilization": random.uniform(0.3, 0.9),
                                "MemUtilization": random.uniform(0.4, 0.8)
                            }
                        }
                    },
                    "NetworkTopology": {
                        "GeoLocation": {
                            "Region": random.choice(["Asia", "Europe", "America"]),
                            "Timezone": random.choice(["Asia/Shanghai", "Europe/London", "America/New_York"])
                        },
                        "Connections": {
                            "IntraShardConn": random.randint(3, 8),
                            "InterShardConn": random.randint(1, 4),
                            "WeightedDegree": random.uniform(5, 15),
                            "ActiveConn": random.randint(5, 10)
                        },
                        "ShardAllocation": {
                            "Priority": random.randint(1, 10),
                            "Adaptability": random.uniform(0.5, 0.9)
                        }
                    },
                    "HeterogeneousType": {
                        "NodeType": random.choice(["full", "validator", "light", "storage"]),
                        "FunctionTags": random.sample(["compute", "storage", "validation", "mining"], 2),
                        "SupportedFuncs": {
                            "Functions": random.sample(["validation", "storage", "consensus", "relay"], 2)
                        },
                        "Application": {
                            "CurrentState": random.choice(["active", "idle", "busy"]),
                            "LoadMetrics": {
                                "TxFrequency": random.randint(10, 50),
                                "StorageOps": random.randint(5, 25)
                            }
                        }
                    }
                },
                "Dynamic": {
                    "OnChainBehavior": {
                        "TransactionCapability": {
                            "AvgTPS": random.uniform(50, 300),
                            "ConfirmationDelay": random.randint(2000, 8000),
                            "CrossShardTx": {
                                "InterNodeVolume": {"node1": 10, "node2": 5},
                                "InterShardVolume": {"shard1": 15, "shard2": 8}
                            }
                        },
                        "BlockGeneration": {
                            "AvgInterval": random.randint(8000, 12000),
                            "IntervalStdDev": random.randint(500, 1500)
                        },
                        "TransactionTypes": {
                            "NormalTxRatio": random.uniform(0.6, 0.9),
                            "ContractTxRatio": random.uniform(0.1, 0.4)
                        },
                        "Consensus": {
                            "ParticipationRate": random.uniform(0.85, 0.99),
                            "TotalReward": random.uniform(500, 1500),
                            "SuccessRate": random.uniform(0.90, 0.99)
                        }
                    },
                    "DynamicAttributes": {
                        "Compute": {
                            "CPUUsage": random.uniform(0.3, 0.9),
                            "MemUsage": random.uniform(0.4, 0.8),
                            "ResourceFlux": random.uniform(0.05, 0.2)
                        },
                        "Network": {
                            "LatencyFlux": random.uniform(2, 10),
                            "AvgLatency": random.randint(5, 20),
                            "BandwidthUsage": random.uniform(0.3, 0.8)
                        },
                        "Transactions": {
                            "Frequency": random.randint(20, 80),
                            "ProcessingDelay": random.randint(1000, 5000)
                        },
                        "Reputation": {
                            "Uptime24h": random.uniform(20, 24),
                            "ReputationScore": random.uniform(70, 95)
                        }
                    }
                }
            }
        }
        
        mock_data.append(mock_node)
    
    return mock_data


def demo_realtime_feature_extraction():
    """演示实时特征提取功能"""
    print(f"=== BlockEmulator适配器演示 ===")
    
    # 1. 初始化适配器
    adapter = BlockEmulatorAdapter()
    
    # 2. 创建模拟数据（模拟从GetAllCollectedData()获取的数据）
    print(f"\n创建模拟数据...")
    mock_data = create_mock_emulator_data(num_nodes=20, num_shards=4)
    print(f"✓ 创建了 {len(mock_data)} 个节点的模拟数据")
    
    # 3. 实时特征提取
    print(f"\n开始实时特征提取...")
    start_time = time.time()
    features = adapter.extract_features_realtime(mock_data)
    extraction_time = time.time() - start_time
    
    # 4. 保存特征文件
    print(f"\n保存特征文件...")
    saved_files = adapter.save_features_for_next_steps(features, "demo_realtime_output")
    
    # 5. 显示结果摘要
    print(f"\n=== 处理结果摘要 ===")
    print(f"输入节点数量: {len(mock_data)}")
    print(f"处理时间: {extraction_time:.3f}秒")
    print(f"处理速度: {len(mock_data)/extraction_time:.1f} 节点/秒")
    print(f"")
    print(f"输出特征:")
    for key, tensor in features.items():
        if isinstance(tensor, torch.Tensor):
            print(f"  {key}: {tensor.shape}")
    print(f"")
    print(f"分片分布: {features['metadata']['shard_distribution']}")
    print(f"保存文件数量: {len(saved_files)}")
    
    # 6. 获取输出格式信息
    format_info = adapter.get_step1_output_info()
    print(f"\n第一步输出格式信息:")
    print(json.dumps(format_info, indent=2, ensure_ascii=False))
    
    return features, saved_files


if __name__ == "__main__":
    # 运行演示
    features, saved_files = demo_realtime_feature_extraction()
    print(f"\n演示完成！")
