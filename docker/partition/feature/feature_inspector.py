"""
特征检查工具
"""
import torch
from typing import Dict, List
from nodeInitialize import load_nodes_from_csv, Node
from MainPipeline import Pipeline
import json

class FeatureInspector:
    """特征使用情况检查器"""

    def __init__(self):
        self.pipeline = Pipeline(use_fusion=False)

    def inspect_single_node_features(self, csv_file: str = "small_samples.csv", node_index: int = 0):
        """检查单个节点的特征提取情况"""
        # 加载节点
        nodes = load_nodes_from_csv(csv_file)
        if node_index >= len(nodes):
            print(f"节点索引 {node_index} 超出范围，总节点数: {len(nodes)}")
            return

        node = nodes[node_index]

        print(f"=== 节点 {node_index} 特征检查 ===")
        print(f"节点数据结构检查:")

        # 1. 检查原始节点数据
        self._print_node_structure(node)

        # 2. 检查特征提取结果
        results = self.pipeline.extract_features([node])

        print(f"\n=== 提取的特征向量 ===")
        print(f"F_classic shape: {results['f_classic'].shape}")
        print(f"F_classic 值: {results['f_classic'][0][:10]}... (前10维)")

        print(f"F_graph shape: {results['f_graph'].shape}")
        print(f"F_graph 值: {results['f_graph'][0][:10]}... (前10维)")

        # 3. 详细检查各层特征
        self._detailed_feature_analysis(node)

        return results

    def _print_node_structure(self, node: Node):
        """打印节点数据结构"""
        print(f"\n--- 资源能力层 (ResourceCapacity) ---")
        try:
            rc = node.ResourceCapacity
            print(f"CPU核心数: {getattr(rc.Hardware.CPU, 'CoreCount', 'N/A')}")
            print(f"CPU频率: {getattr(rc.Hardware.CPU, 'ClockFrequency', 'N/A')}")
            print(f"内存容量: {getattr(rc.Hardware.Memory, 'TotalCapacity', 'N/A')}")
            print(f"存储容量: {getattr(rc.Hardware.Storage, 'Capacity', 'N/A')}")
            print(f"存储类型: {getattr(rc.Hardware.Storage, 'Type', 'N/A')}")
            print(f"网络带宽: {getattr(rc.Hardware.Network, 'UpstreamBW', 'N/A')}")
            print(f"CPU使用率: {getattr(rc.OperationalStatus.ResourceUsage, 'CPUUtilization', 'N/A')}")
            print(f"内存使用率: {getattr(rc.OperationalStatus.ResourceUsage, 'MemUtilization', 'N/A')}")
            print(f"24h在线时间: {getattr(rc.OperationalStatus, 'Uptime24h', 'N/A')}")
        except Exception as e:
            print(f"资源能力层访问错误: {e}")

        print(f"\n--- 链上行为层 (OnChainBehavior) ---")
        try:
            ob = node.OnChainBehavior
            print(f"平均TPS: {getattr(ob.TransactionCapability, 'AvgTPS', 'N/A')}")
            print(f"费用贡献率: {getattr(ob.EconomicContribution, 'FeeContributionRatio', 'N/A')}")
            print(f"共识参与率: {getattr(ob.Consensus, 'ParticipationRate', 'N/A')}")
            print(f"共识成功率: {getattr(ob.Consensus, 'SuccessRate', 'N/A')}")
        except Exception as e:
            print(f"链上行为层访问错误: {e}")

        print(f"\n--- 网络拓扑层 (NetworkTopology) ---")
        try:
            nt = node.NetworkTopology
            print(f"地理区域: {getattr(nt.GeoLocation, 'Region', 'N/A')}")
            print(f"分片内连接: {getattr(nt.Connections, 'IntraShardConn', 'N/A')}")
            print(f"分片间连接: {getattr(nt.Connections, 'InterShardConn', 'N/A')}")
            print(f"活跃连接数: {getattr(nt.Connections, 'ActiveConn', 'N/A')}")
            print(f"加权度: {getattr(nt.Connections, 'WeightedDegree', 'N/A')}")
        except Exception as e:
            print(f"网络拓扑层访问错误: {e}")

        print(f"\n--- 动态属性层 (DynamicAttributes) ---")
        try:
            da = node.DynamicAttributes
            print(f"CPU使用: {getattr(da.Compute, 'CPUUsage', 'N/A')}")
            print(f"内存使用: {getattr(da.Compute, 'MemUsage', 'N/A')}")
            print(f"声誉分数: {getattr(da.Reputation, 'ReputationScore', 'N/A')}")
            print(f"平均延迟: {getattr(da.Network, 'AvgLatency', 'N/A')}")
        except Exception as e:
            print(f"动态属性层访问错误: {e}")

        print(f"\n--- 异构类型层 (HeterogeneousType) ---")
        try:
            ht = node.HeterogeneousType
            print(f"节点类型: {getattr(ht, 'NodeType', 'N/A')}")
        except Exception as e:
            print(f"异构类型层访问错误: {e}")

    def _detailed_feature_analysis(self, node: Node):
        """详细分析特征提取过程"""
        print(f"\n=== 详细特征分析 ===")

        # 分析数值特征
        from feature_extractor import ClassicFeatureExtractor
        extractor = ClassicFeatureExtractor()

        numeric_features = extractor._extract_numeric_features(node)
        print(f"\n数值特征 (应该15维): {len(numeric_features)}")
        print(f"数值特征值: {numeric_features}")

        categorical_features = extractor._extract_categorical_features(node)
        print(f"\n分类特征 (应该18维): {len(categorical_features)}")
        print(f"分类特征值: {categorical_features}")

        # 检查时序特征
        from sliding_window_extractor import EnhancedSequenceFeatureEncoder
        seq_encoder = EnhancedSequenceFeatureEncoder()
        sequences = seq_encoder._extract_sequences(node)
        print(f"\n时序数据 (应该50x5): {len(sequences)}x{len(sequences[0]) if sequences else 0}")
        if sequences:
            print(f"时序数据样例: {sequences[:3]}")  # 打印前3个时间点

    def check_feature_coverage(self, csv_file: str = "small_samples.csv"):
        """检查特征覆盖率"""
        nodes = load_nodes_from_csv(csv_file)

        print(f"=== 特征覆盖率检查 (总节点数: {len(nodes)}) ===")

        coverage_stats = {
            'cpu_cores': 0,
            'memory_capacity': 0,
            'storage_capacity': 0,
            'network_bandwidth': 0,
            'node_type': 0,
            'region': 0,
            'reputation_score': 0,
            'cpu_usage': 0,
            'tps': 0,
            'consensus_rate': 0
        }

        for node in nodes:
            try:
                # 检查关键特征的覆盖情况
                if getattr(node.ResourceCapacity.Hardware.CPU, 'CoreCount', 0) > 0:
                    coverage_stats['cpu_cores'] += 1
                if getattr(node.ResourceCapacity.Hardware.Memory, 'TotalCapacity', 0) > 0:
                    coverage_stats['memory_capacity'] += 1
                if getattr(node.ResourceCapacity.Hardware.Storage, 'Capacity', 0) > 0:
                    coverage_stats['storage_capacity'] += 1
                if getattr(node.ResourceCapacity.Hardware.Network, 'UpstreamBW', 0) > 0:
                    coverage_stats['network_bandwidth'] += 1
                if getattr(node.HeterogeneousType, 'NodeType', '') != '':
                    coverage_stats['node_type'] += 1
                if getattr(node.NetworkTopology.GeoLocation, 'Region', '') != '':
                    coverage_stats['region'] += 1
                if getattr(node.DynamicAttributes.Reputation, 'ReputationScore', 0) > 0:
                    coverage_stats['reputation_score'] += 1
                if getattr(node.DynamicAttributes.Compute, 'CPUUsage', 0) > 0:
                    coverage_stats['cpu_usage'] += 1
                if getattr(node.OnChainBehavior.TransactionCapability, 'AvgTPS', 0) > 0:
                    coverage_stats['tps'] += 1
                if getattr(node.OnChainBehavior.Consensus, 'ParticipationRate', 0) > 0:
                    coverage_stats['consensus_rate'] += 1
            except:
                continue

        print(f"\n特征覆盖率统计:")
        for feature, count in coverage_stats.items():
            percentage = (count / len(nodes)) * 100
            print(f"{feature}: {count}/{len(nodes)} ({percentage:.1f}%)")

        return coverage_stats



def run_detailed_inspection():
    inspector = FeatureInspector()

    print("=== 运行特征详细检查 ===")

    # 检查第一个节点的所有特征
    results = inspector.inspect_single_node_features("small_samples.csv", 0)

    # 检查特征覆盖率
    coverage = inspector.check_feature_coverage("small_samples.csv")

    # 输出特征使用情况报告
    print(f"\n=== 特征使用报告 ===")
    print(f"当前实现使用的特征维度:")
    print(f"- 数值特征: 15维")
    print(f"- 分类特征: 18维")
    print(f"- 时序特征: 32维 (LSTM编码后)")
    print(f"- 图结构特征: 10维")
    print(f"- 滑动窗口特征: 300维 (60*5个时序)")
    print(f"- 模式特征: 20维 (4*5个时序)")
    print(f"总计: 395维 -> 投影到64维 (F_classic)")

def main():
    """主检查函数"""
    inspector = FeatureInspector()

    print("开始特征检查...")

    # 1. 检查单个节点的详细特征
    results = inspector.inspect_single_node_features(node_index=0)

    # 2. 检查特征覆盖率
    coverage = inspector.check_feature_coverage()

    # 3. 统计未使用的特征字段
    print(f"\n=== 可能未充分利用的特征字段 ===")
    print("建议检查以下特征是否可以加入特征提取:")
    potential_features = [
        "StorageUtilization", "NetworkLatency", "GasCost",
        "SmartContractComplexity", "TransactionVolume",
        "PeerConnections", "BandwidthUsage", "SecurityScore"
    ]
    for feature in potential_features:
        print(f"- {feature}")

if __name__ == "__main__":
    # main()
    run_detailed_inspection()