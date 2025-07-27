"""
BlockEmulator实时特征提取适配器 - 基于message.go的精确字段映射
专注于StaticNodeFeatures和DynamicNodeFeatures (48个核心字段)
"""

import time
import logging
from typing import Dict, List, Any, Tuple, Optional
import torch
import numpy as np
from dataclasses import dataclass
import sys
from pathlib import Path

# 添加当前目录到系统路径
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from unified_feature_extractor import UnifiedFeatureExtractor
except ImportError as e:
    raise ImportError(f"unified_feature_extractor导入失败: {e}")

try:
    from node import Node
except ImportError:
    try:
        from nodeInitialize import Node
    except ImportError as e:
        raise ImportError(f"node或nodeInitialize导入失败: {e}")


@dataclass
class BlockEmulatorNodeData:
    """BlockEmulator节点数据结构"""
    shard_id: int
    node_id: int  
    timestamp: int
    request_id: str
    static: Dict[str, Any]    # StaticNodeFeatures
    dynamic: Dict[str, Any]   # DynamicNodeFeatures


class BlockEmulatorRealtimeAdapter:
    """
    BlockEmulator实时特征提取适配器
    基于message.go中StaticNodeFeatures和DynamicNodeFeatures的精确字段定义
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 初始化统一特征提取器
        self.unified_extractor = UnifiedFeatureExtractor()
        
        # 输出特征维度
        self.output_dims = {
            'f_classic': 128,    # 经典特征维度
            'f_graph': 96,       # 图特征维度
            'f_reduced': 64      # 精简特征维度
        }
        
        # BlockEmulator字段映射 (48个核心特征字段)
        self.static_field_count = 26
        self.dynamic_field_count = 22
        
        print(f"BlockEmulator实时适配器初始化完成")
        print(f"  静态特征字段: {self.static_field_count}")
        print(f"  动态特征字段: {self.dynamic_field_count}")
        print(f"  总特征字段: {self.static_field_count + self.dynamic_field_count}")
        print(f"  输出维度: F_classic={self.output_dims['f_classic']}, F_graph={self.output_dims['f_graph']}")

    def extract_features_realtime(self, node_data_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        实时特征提取主接口
        
        Args:
            node_data_list: 从BlockEmulator.GetAllCollectedData()获取的数据
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
            
        print(f"\n=== BlockEmulator实时特征提取 ===")
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
        
        return {
            'f_classic': f_classic,
            'f_graph': f_graph,
            'f_reduced': f_reduced,
            'node_mapping': node_mapping,
            'metadata': metadata
        }

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
                self.logger.warning(f"转换节点数据失败 S{shard_id}N{node_id}: {e}")
                continue
                
        return converted

    def _build_node_objects(self, converted_nodes: List[BlockEmulatorNodeData]) -> List[Node]:
        """构建Node对象供特征提取器使用"""
        node_objects = []
        
        for converted_node in converted_nodes:
            try:
                node = Node()
                
                # 设置基本属性
                node.node_id = converted_node.node_id
                node.shard_id = converted_node.shard_id
                
                # 填充静态特征 (26个字段)
                self._populate_static_features(node, converted_node.static)
                
                # 填充动态特征 (22个字段)
                self._populate_dynamic_features(node, converted_node.dynamic)
                
                node_objects.append(node)
                
            except Exception as e:
                self.logger.warning(f"构建Node对象失败 S{converted_node.shard_id}N{converted_node.node_id}: {e}")
                # 添加默认Node对象
                default_node = Node()
                default_node.node_id = converted_node.node_id
                default_node.shard_id = converted_node.shard_id
                node_objects.append(default_node)
                
        return node_objects

    def _populate_static_features(self, node: Node, static_data: Dict[str, Any]) -> None:
        """
        填充静态特征 (基于StaticNodeFeatures)
        总计26个字段：硬件资源(11) + 网络拓扑(8) + 异构类型(7)
        """
        try:
            # === 硬件资源特征 (11个字段) ===
            resource_capacity = static_data.get('ResourceCapacity', {})
            hardware = resource_capacity.get('Hardware', {})
            
            # CPU特征 (2个字段)
            cpu = hardware.get('CPU', {})
            node.hardware.cpu.cores = self._safe_float(cpu.get('CoreCount'), 4.0)
            node.hardware.cpu.architecture = str(cpu.get('Architecture', 'x86_64'))
            
            # 内存特征 (3个字段)
            memory = hardware.get('Memory', {})
            node.hardware.memory.capacity = self._safe_float(memory.get('TotalCapacity'), 8192.0)
            node.hardware.memory.type = str(memory.get('Type', 'DDR4'))
            node.hardware.memory.bandwidth = self._safe_float(memory.get('Bandwidth'), 25600.0)
            
            # 存储特征 (3个字段)
            storage = hardware.get('Storage', {})
            node.hardware.storage.capacity = self._safe_float(storage.get('Capacity'), 512.0)
            node.hardware.storage.type = str(storage.get('Type', 'SSD'))
            node.hardware.storage.speed = self._safe_float(storage.get('ReadWriteSpeed'), 500.0)
            
            # 网络特征 (3个字段)
            network = hardware.get('Network', {})
            node.hardware.network.upstream = self._safe_float(network.get('UpstreamBW'), 1000.0)
            node.hardware.network.downstream = self._safe_float(network.get('DownstreamBW'), 1000.0)
            node.hardware.network.latency = self._parse_duration(network.get('Latency', '10ms'))
            
            # === 网络拓扑特征 (8个字段) ===
            topology = static_data.get('NetworkTopology', {})
            
            # 地理位置 (1个字段)
            geo = topology.get('GeoLocation', {})
            node.network.geo_location = str(geo.get('Timezone', 'UTC'))
            
            # 连接信息 (4个字段)
            connections = topology.get('Connections', {})
            node.network.intra_shard_connections = self._safe_float(connections.get('IntraShardConn'), 5.0)
            node.network.inter_shard_connections = self._safe_float(connections.get('InterShardConn'), 3.0)
            node.network.weighted_degree = self._safe_float(connections.get('WeightedDegree'), 2.5)
            node.network.active_connections = self._safe_float(connections.get('ActiveConn'), 8.0)
            
            # 分片分配 (3个字段)
            shard_alloc = topology.get('ShardAllocation', {})
            node.network.shard_priority = self._safe_float(shard_alloc.get('Priority'), 1.0)
            node.network.shard_preference = str(shard_alloc.get('ShardPreference', 'auto'))
            node.network.adaptability = self._safe_float(shard_alloc.get('Adaptability'), 0.8)
            
            # === 异构类型特征 (7个字段) ===
            hetero = static_data.get('HeterogeneousType', {})
            
            # 节点类型和功能 (4个字段)
            node.heterogeneous.node_type = str(hetero.get('NodeType', 'standard'))
            node.heterogeneous.function_tags = str(hetero.get('FunctionTags', 'general'))
            
            supported_funcs = hetero.get('SupportedFuncs', {})
            node.heterogeneous.supported_functions = str(supported_funcs.get('Functions', 'basic'))
            node.heterogeneous.function_priorities = str(supported_funcs.get('Priorities', 'medium'))
            
            # 应用状态 (3个字段)
            application = hetero.get('Application', {})
            node.heterogeneous.current_state = str(application.get('CurrentState', 'active'))
            
            load_metrics = application.get('LoadMetrics', {})
            node.heterogeneous.tx_frequency = self._safe_float(load_metrics.get('TxFrequency'), 100.0)
            node.heterogeneous.storage_ops = self._safe_float(load_metrics.get('StorageOps'), 50.0)
            
        except Exception as e:
            self.logger.error(f"填充静态特征失败: {e}")

    def _populate_dynamic_features(self, node: Node, dynamic_data: Dict[str, Any]) -> None:
        """
        填充动态特征 (基于DynamicNodeFeatures)
        总计22个字段：链上行为(17) + 动态属性(14) - 有重叠，实际22个字段
        """
        try:
            # === 链上行为特征 (17个字段) ===
            onchain = dynamic_data.get('OnChainBehavior', {})
            
            # 交易能力 (8个字段)
            tx_capability = onchain.get('TransactionCapability', {})
            node.onchain.avg_tps = self._safe_float(tx_capability.get('AvgTPS'), 100.0)
            node.onchain.confirmation_delay = self._parse_duration(tx_capability.get('ConfirmationDelay', '5s'))
            
            cross_shard = tx_capability.get('CrossShardTx', {})
            node.onchain.inter_node_volume = self._parse_volume(cross_shard.get('InterNodeVolume', '1MB'))
            node.onchain.inter_shard_volume = self._parse_volume(cross_shard.get('InterShardVolume', '5MB'))
            
            resource_per_tx = tx_capability.get('ResourcePerTx', {})
            node.onchain.cpu_per_tx = self._safe_float(resource_per_tx.get('CPUPerTx'), 0.1)
            node.onchain.mem_per_tx = self._safe_float(resource_per_tx.get('MemPerTx'), 0.5)
            node.onchain.disk_per_tx = self._safe_float(resource_per_tx.get('DiskPerTx'), 0.01)
            node.onchain.network_per_tx = self._safe_float(resource_per_tx.get('NetworkPerTx'), 0.2)
            
            # 区块生成 (2个字段)
            block_gen = onchain.get('BlockGeneration', {})
            node.onchain.block_interval = self._parse_duration(block_gen.get('AvgInterval', '10s'))
            node.onchain.block_interval_std = self._parse_duration(block_gen.get('IntervalStdDev', '2s'))
            
            # 经济贡献 (1个字段)
            economic = onchain.get('EconomicContribution', {})
            node.onchain.fee_contribution_ratio = self._safe_float(economic.get('FeeContributionRatio'), 0.05)
            
            # 智能合约使用 (1个字段)
            contract = onchain.get('SmartContractUsage', {})
            node.onchain.contract_invocation_freq = self._safe_float(contract.get('InvocationFrequency'), 10.0)
            
            # 交易类型 (2个字段)
            tx_types = onchain.get('TransactionTypes', {})
            node.onchain.normal_tx_ratio = self._safe_float(tx_types.get('NormalTxRatio'), 0.8)
            node.onchain.contract_tx_ratio = self._safe_float(tx_types.get('ContractTxRatio'), 0.2)
            
            # 共识参与 (3个字段)
            consensus = onchain.get('Consensus', {})
            node.onchain.consensus_participation = self._safe_float(consensus.get('ParticipationRate'), 0.95)
            node.onchain.consensus_total_reward = self._safe_float(consensus.get('TotalReward'), 100.0)
            node.onchain.consensus_success_rate = self._safe_float(consensus.get('SuccessRate'), 0.98)
            
            # === 动态属性特征 (14个字段) ===
            # 注意：有些字段与链上行为重叠，需要避免重复计数
            dynamic_attrs = dynamic_data.get('DynamicAttributes', {})
            
            # 计算资源 (3个字段)
            compute = dynamic_attrs.get('Compute', {})
            node.dynamic.cpu_usage = self._safe_float(compute.get('CPUUsage'), 0.7)
            node.dynamic.mem_usage = self._safe_float(compute.get('MemUsage'), 0.6)
            node.dynamic.resource_flux = self._safe_float(compute.get('ResourceFlux'), 0.1)
            
            # 存储资源 (2个字段)
            storage = dynamic_attrs.get('Storage', {})
            node.dynamic.storage_available = self._safe_float(storage.get('Available'), 400.0)
            node.dynamic.storage_utilization = self._safe_float(storage.get('Utilization'), 0.3)
            
            # 网络状态 (3个字段)
            network = dynamic_attrs.get('Network', {})
            node.dynamic.latency_flux = self._safe_float(network.get('LatencyFlux'), 0.05)
            node.dynamic.avg_latency = self._parse_duration(network.get('AvgLatency', '8ms'))
            node.dynamic.bandwidth_usage = self._safe_float(network.get('BandwidthUsage'), 0.4)
            
            # 交易状态 (3个字段)
            transactions = dynamic_attrs.get('Transactions', {})
            node.dynamic.tx_frequency = self._safe_float(transactions.get('Frequency'), 150.0)
            node.dynamic.processing_delay = self._parse_duration(transactions.get('ProcessingDelay', '100ms'))
            node.dynamic.stake_change_rate = self._safe_float(transactions.get('StakeChangeRate'), 0.02)
            
            # 声誉指标 (2个字段)
            reputation = dynamic_attrs.get('Reputation', {})
            node.dynamic.uptime_24h = self._safe_float(reputation.get('Uptime24h'), 23.8)
            node.dynamic.reputation_score = self._safe_float(reputation.get('ReputationScore'), 85.0)
            
        except Exception as e:
            self.logger.error(f"填充动态特征失败: {e}")

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """安全转换为浮点数"""
        try:
            return float(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    def _parse_duration(self, duration_str: str, default: float = 0.0) -> float:
        """解析时间字符串为毫秒"""
        try:
            if not duration_str or not isinstance(duration_str, str):
                return default
                
            duration_str = duration_str.lower().strip()
            
            if duration_str.endswith('ms'):
                return float(duration_str[:-2])
            elif duration_str.endswith('s'):
                return float(duration_str[:-1]) * 1000
            elif duration_str.endswith('m'):
                return float(duration_str[:-1]) * 60000
            else:
                return float(duration_str)
                
        except (ValueError, AttributeError):
            return default

    def _parse_volume(self, volume_str: str, default: float = 0.0) -> float:
        """解析数据量字符串为MB"""
        try:
            if not volume_str or not isinstance(volume_str, str):
                return default
                
            volume_str = volume_str.upper().strip()
            
            if volume_str.endswith('KB'):
                return float(volume_str[:-2]) / 1024
            elif volume_str.endswith('MB'):
                return float(volume_str[:-2])
            elif volume_str.endswith('GB'):
                return float(volume_str[:-2]) * 1024
            else:
                return float(volume_str)
                
        except (ValueError, AttributeError):
            return default

    def _generate_reduced_features(self, f_classic: torch.Tensor, f_graph: torch.Tensor) -> torch.Tensor:
        """生成精简特征"""
        # 简单的特征降维：取前32维经典特征 + 前32维图特征
        f_classic_reduced = f_classic[:, :32]
        f_graph_reduced = f_graph[:, :32]
        return torch.cat([f_classic_reduced, f_graph_reduced], dim=1)

    def _build_node_mapping(self, converted_nodes: List[BlockEmulatorNodeData]) -> Dict[str, Any]:
        """构建节点映射关系"""
        return {
            'node_ids': [node.node_id for node in converted_nodes],
            'shard_ids': [node.shard_id for node in converted_nodes],
            'timestamps': [node.timestamp for node in converted_nodes],
            'request_ids': [node.request_id for node in converted_nodes]
        }

    def _generate_metadata(self, converted_nodes: List[BlockEmulatorNodeData], 
                          f_classic: torch.Tensor, f_graph: torch.Tensor) -> Dict[str, Any]:
        """生成元数据信息"""
        return {
            'extraction_time': time.time(),
            'total_nodes': len(converted_nodes),
            'static_field_count': self.static_field_count,
            'dynamic_field_count': self.dynamic_field_count,
            'feature_shapes': {
                'f_classic': list(f_classic.shape),
                'f_graph': list(f_graph.shape)
            },
            'shards': list(set(node.shard_id for node in converted_nodes))
        }

    def _get_empty_result(self) -> Dict[str, torch.Tensor]:
        """返回空结果"""
        return {
            'f_classic': torch.zeros((0, self.output_dims['f_classic'])),
            'f_graph': torch.zeros((0, self.output_dims['f_graph'])),
            'f_reduced': torch.zeros((0, self.output_dims['f_reduced'])),
            'node_mapping': {'node_ids': [], 'shard_ids': [], 'timestamps': [], 'request_ids': []},
            'metadata': {'extraction_time': time.time(), 'total_nodes': 0}
        }


# 工厂函数
def create_be_realtime_adapter() -> BlockEmulatorRealtimeAdapter:
    """创建BlockEmulator实时适配器实例"""
    return BlockEmulatorRealtimeAdapter()


if __name__ == "__main__":
    # 测试适配器
    adapter = create_be_realtime_adapter()
    print("BlockEmulator实时特征提取适配器测试完成")
