import csv
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# 资源能力层
@dataclass
class CPU:
    CoreCount: int = 0
    ClockFrequency: float = 0.0
    Architecture: str = ""
    CacheSize: int = 0

@dataclass
class Memory:
    TotalCapacity: int = 0
    Type: str = ""
    Bandwidth: float = 0.0

@dataclass
class Storage:
    Capacity: int = 0
    Type: str = ""
    ReadWriteSpeed: float = 0.0

@dataclass
class Network:
    UpstreamBW: float = 0.0
    DownstreamBW: float = 0.0
    Latency: int = 0  # 毫秒

@dataclass
class Hardware:
    CPU: CPU = field(default_factory=CPU)
    Memory: Memory = field(default_factory=Memory)
    Storage: Storage = field(default_factory=Storage)
    Network: Network = field(default_factory=Network)

@dataclass
class ResourceUsage:
    CPUUtilization: float = 0.0
    MemUtilization: float = 0.0

@dataclass
class OperationalStatus:
    Uptime24h: float = 0.0
    CoreEligibility: bool = False
    ResourceUsage: ResourceUsage = field(default_factory=ResourceUsage)

@dataclass
class ResourceCapacityLayer:
    Hardware: Hardware = field(default_factory=Hardware)
    OperationalStatus: OperationalStatus = field(default_factory=OperationalStatus)

# 链上行为层
@dataclass
class CrossShardTransaction:
    InterNodeVolume: Dict[str, int] = field(default_factory=dict)
    InterShardVolume: Dict[str, int] = field(default_factory=dict)

@dataclass
class ResourcePerTx:
    CPUPerTx: float = 0.0
    MemPerTx: float = 0.0
    DiskPerTx: float = 0.0
    NetworkPerTx: float = 0.0

@dataclass
class TransactionCapability:
    AvgTPS: float = 0.0
    CrossShardTx: CrossShardTransaction = field(default_factory=CrossShardTransaction)
    ConfirmationDelay: int = 0  # 毫秒
    ResourcePerTx: ResourcePerTx = field(default_factory=ResourcePerTx)

@dataclass
class BlockBehavior:
    AvgInterval: int = 0  # 毫秒
    IntervalStdDev: int = 0  # 毫秒

@dataclass
class EconomicMetrics:
    FeeContributionRatio: float = 0.0

@dataclass
class SmartContractMetrics:
    InvocationFrequency: int = 0

@dataclass
class TransactionTypeDistribution:
    NormalTxRatio: float = 0.0
    ContractTxRatio: float = 0.0

@dataclass
class ConsensusParticipation:
    ParticipationRate: float = 0.0
    TotalReward: float = 0.0
    SuccessRate: float = 0.0

@dataclass
class OnChainBehaviorLayer:
    TransactionCapability: TransactionCapability = field(default_factory=TransactionCapability)
    BlockGeneration: BlockBehavior = field(default_factory=BlockBehavior)
    EconomicContribution: EconomicMetrics = field(default_factory=EconomicMetrics)
    SmartContractUsage: SmartContractMetrics = field(default_factory=SmartContractMetrics)
    TransactionTypes: TransactionTypeDistribution = field(default_factory=TransactionTypeDistribution)
    Consensus: ConsensusParticipation = field(default_factory=ConsensusParticipation)

# 网络拓扑层
@dataclass
class GeoInfo:
    Timezone: str = ""
    DataCenter: str = ""
    Region: str = ""

@dataclass
class ConnectionFeatures:
    IntraShardConn: int = 0
    InterShardConn: int = 0
    WeightedDegree: float = 0.0
    ActiveConn: int = 0

@dataclass
class NetworkHierarchy:
    Depth: int = 0
    ConnectionDensity: float = 0.0

@dataclass
class IntraShardCentrality:
    Eigenvector: float = 0.0
    Closeness: float = 0.0

@dataclass
class InterShardCentrality:
    Betweenness: float = 0.0
    Influence: float = 0.0

@dataclass
class NodeCentrality:
    IntraShard: IntraShardCentrality = field(default_factory=IntraShardCentrality)
    InterShard: InterShardCentrality = field(default_factory=InterShardCentrality)

@dataclass
class ShardAllocationInfo:
    Priority: int = 0
    ShardPreference: Dict[str, float] = field(default_factory=dict)
    Adaptability: float = 0.0

@dataclass
class NetworkTopologyLayer:
    GeoLocation: GeoInfo = field(default_factory=GeoInfo)
    Connections: ConnectionFeatures = field(default_factory=ConnectionFeatures)
    Hierarchy: NetworkHierarchy = field(default_factory=NetworkHierarchy)
    Centrality: NodeCentrality = field(default_factory=NodeCentrality)
    ShardAllocation: ShardAllocationInfo = field(default_factory=ShardAllocationInfo)

# 动态属性层
@dataclass
class ComputeMetrics:
    CPUUsage: float = 0.0
    MemUsage: float = 0.0
    ResourceFlux: float = 0.0

@dataclass
class StorageMetrics:
    Available: int = 0
    Utilization: float = 0.0

@dataclass
class NetworkMetrics:
    LatencyFlux: float = 0.0
    AvgLatency: int = 0  # 毫秒
    BandwidthUsage: float = 0.0

@dataclass
class TransactionMetrics:
    Frequency: int = 0
    ProcessingDelay: int = 0  # 毫秒
    StakeChangeRate: float = 0.0

@dataclass
class ReputationMetrics:
    Uptime24h: float = 0.0
    ReputationScore: float = 0.0

@dataclass
class DynamicAttributesLayer:
    Compute: ComputeMetrics = field(default_factory=ComputeMetrics)
    Storage: StorageMetrics = field(default_factory=StorageMetrics)
    Network: NetworkMetrics = field(default_factory=NetworkMetrics)
    Transactions: TransactionMetrics = field(default_factory=TransactionMetrics)
    Reputation: ReputationMetrics = field(default_factory=ReputationMetrics)

# 异构类型层
@dataclass
class LoadMetrics:
    TxFrequency: int = 0
    StorageOps: int = 0

@dataclass
class Application:
    CurrentState: str = ""
    LoadMetrics: LoadMetrics = field(default_factory=LoadMetrics)

@dataclass
class SupportedFuncs:
    Functions: List[str] = field(default_factory=list)
    Priorities: Dict[str, int] = field(default_factory=dict)

@dataclass
class HeterogeneousTypeLayer:
    NodeType: str = ""
    FunctionTags: List[str] = field(default_factory=list)
    SupportedFuncs: SupportedFuncs = field(default_factory=SupportedFuncs)
    Application: Application = field(default_factory=Application)

# 节点主结构体
@dataclass
class Node:
    ResourceCapacity: ResourceCapacityLayer = field(default_factory=ResourceCapacityLayer)
    OnChainBehavior: OnChainBehaviorLayer = field(default_factory=OnChainBehaviorLayer)
    NetworkTopology: NetworkTopologyLayer = field(default_factory=NetworkTopologyLayer)
    DynamicAttributes: DynamicAttributesLayer = field(default_factory=DynamicAttributesLayer)
    HeterogeneousType: HeterogeneousTypeLayer = field(default_factory=HeterogeneousTypeLayer)


def parse_duration(s: str) -> int:
    """将持续时间字符串转换为毫秒整数"""
    if not s:
        return 0

    if s.endswith("ms"):
        return int(s.rstrip("ms"))
    elif s.endswith("s"):
        return int(float(s.rstrip("s")) * 1000)
    try:
        return int(s)  # 如果只是数字，假设是毫秒
    except ValueError:
        return 0


def parse_map(s: str, value_type=int) -> Dict[str, any]:
    """解析映射字符串（如 key1:val1;key2:val2）"""
    if not s:
        return {}

    result = {}
    pairs = s.split(";")
    for pair in pairs:
        if ":" in pair:
            k, v = pair.split(":", 1)
            try:
                result[k] = value_type(v)
            except ValueError:
                continue
    return result


def map_field(header: str, value: str, node: Node) -> None:
    """将CSV字段映射到Node对象的属性"""
    # 资源能力层
    if header == "ResourceCapacity.Hardware.CPU.CoreCount":
        try:
            node.ResourceCapacity.Hardware.CPU.CoreCount = int(value)
        except ValueError:
            pass
    elif header == "ResourceCapacity.Hardware.CPU.ClockFrequency":
        try:
            node.ResourceCapacity.Hardware.CPU.ClockFrequency = float(value)
        except ValueError:
            pass
    elif header == "ResourceCapacity.Hardware.CPU.Architecture":
        node.ResourceCapacity.Hardware.CPU.Architecture = value
    elif header == "ResourceCapacity.Hardware.CPU.CacheSize":
        try:
            node.ResourceCapacity.Hardware.CPU.CacheSize = int(value)
        except ValueError:
            pass
    elif header == "ResourceCapacity.Hardware.Memory.TotalCapacity":
        try:
            node.ResourceCapacity.Hardware.Memory.TotalCapacity = int(value)
        except ValueError:
            pass
    elif header == "ResourceCapacity.Hardware.Memory.Type":
        node.ResourceCapacity.Hardware.Memory.Type = value
    elif header == "ResourceCapacity.Hardware.Memory.Bandwidth":
        try:
            node.ResourceCapacity.Hardware.Memory.Bandwidth = float(value)
        except ValueError:
            pass
    elif header == "ResourceCapacity.Hardware.Storage.Capacity":
        try:
            node.ResourceCapacity.Hardware.Storage.Capacity = int(value)
        except ValueError:
            pass
    elif header == "ResourceCapacity.Hardware.Storage.Type":
        node.ResourceCapacity.Hardware.Storage.Type = value
    elif header == "ResourceCapacity.Hardware.Storage.ReadWriteSpeed":
        try:
            node.ResourceCapacity.Hardware.Storage.ReadWriteSpeed = float(value)
        except ValueError:
            pass
    elif header == "ResourceCapacity.Hardware.Network.UpstreamBW":
        try:
            node.ResourceCapacity.Hardware.Network.UpstreamBW = float(value)
        except ValueError:
            pass
    elif header == "ResourceCapacity.Hardware.Network.DownstreamBW":
        try:
            node.ResourceCapacity.Hardware.Network.DownstreamBW = float(value)
        except ValueError:
            pass
    elif header == "ResourceCapacity.Hardware.Network.Latency":
        node.ResourceCapacity.Hardware.Network.Latency = parse_duration(value)
    elif header == "ResourceCapacity.OperationalStatus.Uptime24h":
        try:
            node.ResourceCapacity.OperationalStatus.Uptime24h = float(value)
        except ValueError:
            pass
    elif header == "ResourceCapacity.OperationalStatus.CoreEligibility":
        node.ResourceCapacity.OperationalStatus.CoreEligibility = value.lower() == "true"
    elif header == "ResourceCapacity.OperationalStatus.ResourceUsage.CPUUtilization":
        try:
            node.ResourceCapacity.OperationalStatus.ResourceUsage.CPUUtilization = float(value)
        except ValueError:
            pass
    elif header == "ResourceCapacity.OperationalStatus.ResourceUsage.MemUtilization":
        try:
            node.ResourceCapacity.OperationalStatus.ResourceUsage.MemUtilization = float(value)
        except ValueError:
            pass

    # 链上行为层
    elif header == "OnChainBehavior.TransactionCapability.AvgTPS":
        try:
            node.OnChainBehavior.TransactionCapability.AvgTPS = float(value)
        except ValueError:
            pass
    elif header == "OnChainBehavior.TransactionCapability.CrossShardTx.InterNodeVolume":
        node.OnChainBehavior.TransactionCapability.CrossShardTx.InterNodeVolume = parse_map(value)
    elif header == "OnChainBehavior.TransactionCapability.CrossShardTx.InterShardVolume":
        node.OnChainBehavior.TransactionCapability.CrossShardTx.InterShardVolume = parse_map(value)
    elif header == "OnChainBehavior.TransactionCapability.ConfirmationDelay":
        node.OnChainBehavior.TransactionCapability.ConfirmationDelay = parse_duration(value)
    elif header == "OnChainBehavior.TransactionCapability.ResourcePerTx.CPUPerTx":
        try:
            node.OnChainBehavior.TransactionCapability.ResourcePerTx.CPUPerTx = float(value)
        except ValueError:
            pass
    elif header == "OnChainBehavior.TransactionCapability.ResourcePerTx.MemPerTx":
        try:
            node.OnChainBehavior.TransactionCapability.ResourcePerTx.MemPerTx = float(value)
        except ValueError:
            pass
    elif header == "OnChainBehavior.TransactionCapability.ResourcePerTx.DiskPerTx":
        try:
            node.OnChainBehavior.TransactionCapability.ResourcePerTx.DiskPerTx = float(value)
        except ValueError:
            pass
    elif header == "OnChainBehavior.TransactionCapability.ResourcePerTx.NetworkPerTx":
        try:
            node.OnChainBehavior.TransactionCapability.ResourcePerTx.NetworkPerTx = float(value)
        except ValueError:
            pass
    elif header == "OnChainBehavior.BlockGeneration.AvgInterval":
        node.OnChainBehavior.BlockGeneration.AvgInterval = parse_duration(value)
    elif header == "OnChainBehavior.BlockGeneration.IntervalStdDev":
        node.OnChainBehavior.BlockGeneration.IntervalStdDev = parse_duration(value)
    elif header == "OnChainBehavior.EconomicContribution.FeeContributionRatio":
        try:
            node.OnChainBehavior.EconomicContribution.FeeContributionRatio = float(value)
        except ValueError:
            pass
    elif header == "OnChainBehavior.SmartContractUsage.InvocationFrequency":
        try:
            node.OnChainBehavior.SmartContractUsage.InvocationFrequency = int(value)
        except ValueError:
            pass
    elif header == "OnChainBehavior.TransactionTypes.NormalTxRatio":
        try:
            node.OnChainBehavior.TransactionTypes.NormalTxRatio = float(value)
        except ValueError:
            pass
    elif header == "OnChainBehavior.TransactionTypes.ContractTxRatio":
        try:
            node.OnChainBehavior.TransactionTypes.ContractTxRatio = float(value)
        except ValueError:
            pass
    elif header == "OnChainBehavior.Consensus.ParticipationRate":
        try:
            node.OnChainBehavior.Consensus.ParticipationRate = float(value)
        except ValueError:
            pass
    elif header == "OnChainBehavior.Consensus.TotalReward":
        try:
            node.OnChainBehavior.Consensus.TotalReward = float(value)
        except ValueError:
            pass
    elif header == "OnChainBehavior.Consensus.SuccessRate":
        try:
            node.OnChainBehavior.Consensus.SuccessRate = float(value)
        except ValueError:
            pass

    # 网络拓扑层
    elif header == "NetworkTopology.GeoLocation.Timezone":
        node.NetworkTopology.GeoLocation.Timezone = value
    elif header == "NetworkTopology.GeoLocation.DataCenter":
        node.NetworkTopology.GeoLocation.DataCenter = value
    elif header == "NetworkTopology.GeoLocation.Region":
        node.NetworkTopology.GeoLocation.Region = value
    elif header == "NetworkTopology.Connections.IntraShardConn":
        try:
            node.NetworkTopology.Connections.IntraShardConn = int(value)
        except ValueError:
            pass
    elif header == "NetworkTopology.Connections.InterShardConn":
        try:
            node.NetworkTopology.Connections.InterShardConn = int(value)
        except ValueError:
            pass
    elif header == "NetworkTopology.Connections.WeightedDegree":
        try:
            node.NetworkTopology.Connections.WeightedDegree = float(value)
        except ValueError:
            pass
    elif header == "NetworkTopology.Connections.ActiveConn":
        try:
            node.NetworkTopology.Connections.ActiveConn = int(value)
        except ValueError:
            pass
    elif header == "NetworkTopology.Hierarchy.Depth":
        try:
            node.NetworkTopology.Hierarchy.Depth = int(value)
        except ValueError:
            pass
    elif header == "NetworkTopology.Hierarchy.ConnectionDensity":
        try:
            node.NetworkTopology.Hierarchy.ConnectionDensity = float(value)
        except ValueError:
            pass
    elif header == "NetworkTopology.Centrality.IntraShard.Eigenvector":
        try:
            node.NetworkTopology.Centrality.IntraShard.Eigenvector = float(value)
        except ValueError:
            pass
    elif header == "NetworkTopology.Centrality.IntraShard.Closeness":
        try:
            node.NetworkTopology.Centrality.IntraShard.Closeness = float(value)
        except ValueError:
            pass
    elif header == "NetworkTopology.Centrality.InterShard.Betweenness":
        try:
            node.NetworkTopology.Centrality.InterShard.Betweenness = float(value)
        except ValueError:
            pass
    elif header == "NetworkTopology.Centrality.InterShard.Influence":
        try:
            node.NetworkTopology.Centrality.InterShard.Influence = float(value)
        except ValueError:
            pass
    elif header == "NetworkTopology.ShardAllocation.Priority":
        try:
            node.NetworkTopology.ShardAllocation.Priority = int(value)
        except ValueError:
            pass
    elif header == "NetworkTopology.ShardAllocation.ShardPreference":
        node.NetworkTopology.ShardAllocation.ShardPreference = parse_map(value, float)
    elif header == "NetworkTopology.ShardAllocation.Adaptability":
        try:
            node.NetworkTopology.ShardAllocation.Adaptability = float(value)
        except ValueError:
            pass

    # 动态属性层
    elif header == "DynamicAttributes.Compute.CPUUsage":
        try:
            node.DynamicAttributes.Compute.CPUUsage = float(value)
        except ValueError:
            pass
    elif header == "DynamicAttributes.Compute.MemUsage":
        try:
            node.DynamicAttributes.Compute.MemUsage = float(value)
        except ValueError:
            pass
    elif header == "DynamicAttributes.Compute.ResourceFlux":
        try:
            node.DynamicAttributes.Compute.ResourceFlux = float(value)
        except ValueError:
            pass
    elif header == "DynamicAttributes.Storage.Available":
        try:
            node.DynamicAttributes.Storage.Available = int(value)
        except ValueError:
            pass
    elif header == "DynamicAttributes.Storage.Utilization":
        try:
            node.DynamicAttributes.Storage.Utilization = float(value)
        except ValueError:
            pass
    elif header == "DynamicAttributes.Network.LatencyFlux":
        try:
            node.DynamicAttributes.Network.LatencyFlux = float(value)
        except ValueError:
            pass
    elif header == "DynamicAttributes.Network.AvgLatency":
        node.DynamicAttributes.Network.AvgLatency = parse_duration(value)
    elif header == "DynamicAttributes.Network.BandwidthUsage":
        try:
            node.DynamicAttributes.Network.BandwidthUsage = float(value)
        except ValueError:
            pass
    elif header == "DynamicAttributes.Transactions.Frequency":
        try:
            node.DynamicAttributes.Transactions.Frequency = int(value)
        except ValueError:
            pass
    elif header == "DynamicAttributes.Transactions.ProcessingDelay":
        node.DynamicAttributes.Transactions.ProcessingDelay = parse_duration(value)
    elif header == "DynamicAttributes.Transactions.StakeChangeRate":
        try:
            node.DynamicAttributes.Transactions.StakeChangeRate = float(value)
        except ValueError:
            pass
    elif header == "DynamicAttributes.Reputation.Uptime24h":
        try:
            node.DynamicAttributes.Reputation.Uptime24h = float(value)
        except ValueError:
            pass
    elif header == "DynamicAttributes.Reputation.ReputationScore":
        try:
            node.DynamicAttributes.Reputation.ReputationScore = float(value)
        except ValueError:
            pass

    # 异构类型层
    elif header == "HeterogeneousType.NodeType":
        node.HeterogeneousType.NodeType = value
    elif header == "HeterogeneousType.FunctionTags":
        node.HeterogeneousType.FunctionTags = value.split(",") if value else []
    elif header == "HeterogeneousType.SupportedFuncs.Functions":
        node.HeterogeneousType.SupportedFuncs.Functions = value.split(",") if value else []
    elif header == "HeterogeneousType.SupportedFuncs.Priorities":
        node.HeterogeneousType.SupportedFuncs.Priorities = parse_map(value)
    elif header == "HeterogeneousType.Application.CurrentState":
        node.HeterogeneousType.Application.CurrentState = value
    elif header == "HeterogeneousType.Application.LoadMetrics.TxFrequency":
        try:
            node.HeterogeneousType.Application.LoadMetrics.TxFrequency = int(value)
        except ValueError:
            pass
    elif header == "HeterogeneousType.Application.LoadMetrics.StorageOps":
        try:
            node.HeterogeneousType.Application.LoadMetrics.StorageOps = int(value)
        except ValueError:
            pass


def load_nodes_from_csv(filename: str) -> List[Node]:
    """从CSV文件加载节点数据"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"找不到文件: {filename}")

    nodes = []

    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        try:
            headers = next(reader)
        except StopIteration:
            raise ValueError("CSV文件为空或格式错误")

        for row in reader:
            if len(row) != len(headers):
                continue  # 跳过格式不匹配的行

            node = Node()
            for i, header in enumerate(headers):
                if i < len(row):
                    map_field(header, row[i], node)

            nodes.append(node)

    return nodes


# def export_to_json(nodes: List[Node], filename: str) -> None:
#     """将节点数据导出为JSON文件，保留完整结构"""
#     # 将节点对象转换为字典
#     node_dicts = [asdict(node) for node in nodes]
#
#     # 写入JSON文件
#     with open(filename, 'w', encoding='utf-8') as f:
#         json.dump(node_dicts, f, indent=2, ensure_ascii=False)


def print_node_details(node: Node) -> None:
    """打印节点详细信息"""
    print("=== 节点详情 ===")
    print(f"类型: {node.HeterogeneousType.NodeType}")
    print(f"CPU: {node.ResourceCapacity.Hardware.CPU.CoreCount} 核 @ {node.ResourceCapacity.Hardware.CPU.ClockFrequency}GHz ({node.ResourceCapacity.Hardware.CPU.Architecture})")
    print(f"内存: {node.ResourceCapacity.Hardware.Memory.TotalCapacity}GB {node.ResourceCapacity.Hardware.Memory.Type}")
    print(f"存储: {node.ResourceCapacity.Hardware.Storage.Capacity}TB {node.ResourceCapacity.Hardware.Storage.Type}")
    print(f"网络: {node.ResourceCapacity.Hardware.Network.UpstreamBW}/{node.ResourceCapacity.Hardware.Network.DownstreamBW} Mbps (延迟: {node.ResourceCapacity.Hardware.Network.Latency}ms)")
    print(f"TPS能力: {node.OnChainBehavior.TransactionCapability.AvgTPS}")
    print(f"共识成功率: {node.OnChainBehavior.Consensus.SuccessRate}%")
    print(f"中心性: {node.NetworkTopology.Centrality.IntraShard.Eigenvector:.4f}")
    print(f"CPU使用率: {node.DynamicAttributes.Compute.CPUUsage}%")
    print(f"声誉分数: {node.DynamicAttributes.Reputation.ReputationScore}")
    print(f"功能标签: {', '.join(node.HeterogeneousType.FunctionTags)}")
    print(f"分片偏好: {node.NetworkTopology.ShardAllocation.ShardPreference}")


if __name__ == "__main__":
    # 解析CSV并保存为完整的JSON
    try:
        csv_file = "../../FeaturesToLevels.csv"
        nodes = load_nodes_from_csv(csv_file)
        print(f"成功加载 {len(nodes)} 个节点")

        # 打印第一个节点的详细信息
        if nodes:
            print_node_details(nodes[0])

        # 直接将完整的节点数据导出到JSON
        # json_file = "nodes_all_data.json"
        # export_to_json(nodes, json_file)
        # print(f"所有节点数据已导出到 {json_file}")

    except Exception as e:
        print(f"错误: {e}")