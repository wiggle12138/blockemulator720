import csv
import random
import numpy as np
from datetime import datetime, timedelta
import json

class BlockchainNodeSampleGenerator:
    """区块链节点样本数据生成器"""

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        # 节点类型配置 - 五种类型
        self.node_types = {
            'full_node': {
                'cpu_range': (8, 32),
                'memory_range': (64, 256),
                'storage_range': (2.0, 16.0),
                'bandwidth_range': (1000, 4000),
                'tps_range': (500.0, 1500.0),
                'weight': 0.3
            },
            'light_node': {
                'cpu_range': (2, 8),
                'memory_range': (8, 32),
                'storage_range': (0.5, 2.0),
                'bandwidth_range': (100, 1000),
                'tps_range': (50.0, 300.0),
                'weight': 0.25
            },
            'miner': {
                'cpu_range': (16, 64),
                'memory_range': (128, 512),
                'storage_range': (4.0, 32.0),
                'bandwidth_range': (1500, 5000),
                'tps_range': (200.0, 800.0),
                'weight': 0.2
            },
            'validator': {
                'cpu_range': (8, 24),
                'memory_range': (64, 192),
                'storage_range': (2.0, 12.0),
                'bandwidth_range': (800, 3000),
                'tps_range': (400.0, 1200.0),
                'weight': 0.15
            },
            'storage': {
                'cpu_range': (4, 16),
                'memory_range': (32, 128),
                'storage_range': (8.0, 64.0),  # 存储节点需要更大的存储空间
                'bandwidth_range': (500, 2500),
                'tps_range': (100.0, 600.0),
                'weight': 0.1
            }
        }

        # 地理位置配置
        self.regions = [
            ('UTC+8', 'AWS', 'ap-southeast-1'),
            ('UTC-5', 'Google Cloud', 'us-east-1'),
            ('UTC+1', 'Azure', 'europe-west'),
            ('UTC-8', 'AWS', 'us-west-2'),
            ('UTC+9', 'Alibaba Cloud', 'ap-northeast-1'),
            ('UTC+0', 'Digital Ocean', 'europe-central'),
            ('UTC+3', 'Yandex Cloud', 'ru-central1'),
            ('UTC-3', 'AWS', 'sa-east-1')
        ]

    def generate_samples(self, num_samples=50, output_file='generated_samples.csv'):
        """生成指定数量的样本"""
        samples = []

        print(f"开始生成 {num_samples} 个样本...")

        for i in range(num_samples):
            # 根据权重选择节点类型
            node_type = self._select_node_type()
            sample = self._generate_single_sample(node_type, i)
            samples.append(sample)

            if (i + 1) % 10 == 0:
                print(f"已生成 {i + 1} 个样本")

        # 保存到CSV文件
        self._save_to_csv(samples, output_file)
        print(f"样本已保存到 {output_file}")

        # 显示统计信息
        self._show_statistics(samples)

        return samples

    def _select_node_type(self):
        """根据权重随机选择节点类型"""
        types = list(self.node_types.keys())
        weights = [self.node_types[t]['weight'] for t in types]
        return random.choices(types, weights=weights)[0]

    def _generate_single_sample(self, node_type, index):
        """生成单个节点样本"""
        config = self.node_types[node_type]

        # 基础硬件配置
        cpu_cores = random.randint(*config['cpu_range'])
        cpu_freq = round(random.uniform(2.0, 4.5), 1)
        cpu_cache = cpu_cores * random.randint(2, 8)
        memory_total = random.randint(*config['memory_range'])
        memory_bandwidth = memory_total * random.uniform(0.3, 0.8)

        # 存储容量 - 存储节点需要更大容量
        storage_capacity = round(random.uniform(*config['storage_range']), 1)
        if node_type == 'storage':
            storage_speed = random.randint(400, 1200)  # 存储节点需要更快的读写速度
        else:
            storage_speed = random.randint(120, 750)

        upstream_bw = random.randint(*config['bandwidth_range'])
        downstream_bw = upstream_bw * random.randint(2, 20)

        # 运行状态 - 根据节点类型调整
        if node_type in ['miner', 'validator']:
            uptime = round(random.uniform(0.98, 0.999), 3)  # 矿工和验证者需要更高在线时间
            core_eligible = True
        elif node_type == 'full_node':
            uptime = round(random.uniform(0.95, 0.998), 3)
            core_eligible = True
        else:
            uptime = round(random.uniform(0.85, 0.95), 3)
            core_eligible = False

        cpu_util = round(random.uniform(20, 90), 1)
        mem_util = round(random.uniform(15, 85), 1)

        # 交易能力 - 根据节点类型调整
        avg_tps = round(random.uniform(*config['tps_range']), 1)

        if node_type == 'miner':
            confirmation_delay = random.randint(500, 2000)  # 矿工确认延迟较高
        elif node_type == 'validator':
            confirmation_delay = random.randint(100, 800)   # 验证者确认较快
        else:
            confirmation_delay = random.randint(200, 1500)

        cpu_per_tx = round(random.uniform(0.05, 0.3), 2)
        mem_per_tx = round(random.uniform(0.02, 0.15), 2)
        disk_per_tx = round(random.uniform(0.005, 0.05), 3)
        network_per_tx = round(random.uniform(0.01, 0.1), 2)

        # 区块生成 - 矿工和验证者参与更多
        if node_type in ['miner', 'validator']:
            block_interval = random.randint(5, 15)
        else:
            block_interval = random.randint(15, 30)
        interval_stddev = round(block_interval * random.uniform(0.05, 0.3), 1)

        # 经济指标
        fee_ratio = round(random.uniform(0.1, 0.4), 2)
        invocation_freq = random.randint(100, 5000)
        normal_tx_ratio = round(random.uniform(0.4, 0.8), 2)
        contract_tx_ratio = round(1.0 - normal_tx_ratio, 2)

        # 共识参与 - 根据节点类型差异化
        if node_type == 'validator':
            participation_rate = round(random.uniform(0.9, 0.99), 2)
            success_rate = round(random.uniform(0.92, 0.99), 2)
            total_reward = round(random.uniform(10, 100), 1)
        elif node_type == 'miner':
            participation_rate = round(random.uniform(0.8, 0.95), 2)
            success_rate = round(random.uniform(0.85, 0.95), 2)
            total_reward = round(random.uniform(5, 80), 1)
        elif node_type == 'full_node':
            participation_rate = round(random.uniform(0.7, 0.9), 2)
            success_rate = round(random.uniform(0.8, 0.95), 2)
            total_reward = round(random.uniform(1, 30), 1)
        else:  # light_node, storage
            participation_rate = round(random.uniform(0.1, 0.6), 2)
            success_rate = round(random.uniform(0.7, 0.9), 2)
            total_reward = round(random.uniform(0.1, 10), 1)

        # 地理位置
        timezone, datacenter, region = random.choice(self.regions)

        # 网络拓扑 - 根据节点类型调整连接数
        if node_type == 'full_node':
            intra_shard_conn = random.randint(30, 80)
            inter_shard_conn = random.randint(15, 40)
        elif node_type == 'miner':
            intra_shard_conn = random.randint(20, 60)
            inter_shard_conn = random.randint(10, 30)
        elif node_type == 'validator':
            intra_shard_conn = random.randint(25, 70)
            inter_shard_conn = random.randint(12, 35)
        elif node_type == 'storage':
            intra_shard_conn = random.randint(15, 50)
            inter_shard_conn = random.randint(5, 20)
        else:  # light_node
            intra_shard_conn = random.randint(5, 25)
            inter_shard_conn = random.randint(2, 10)

        weighted_degree = intra_shard_conn + inter_shard_conn * 1.5
        active_conn = random.randint(intra_shard_conn // 2, intra_shard_conn)
        depth = random.randint(2, 5)
        conn_density = round(random.uniform(0.3, 0.95), 2)

        # 中心性指标 - 核心节点有更高的中心性
        if node_type in ['full_node', 'validator']:
            eigenvector = round(random.uniform(0.4, 0.95), 2)
            closeness = round(random.uniform(0.5, 0.98), 2)
            betweenness = round(random.uniform(0.3, 0.9), 2)
            influence = round(random.uniform(0.4, 0.8), 2)
        else:
            eigenvector = round(random.uniform(0.1, 0.6), 2)
            closeness = round(random.uniform(0.2, 0.7), 2)
            betweenness = round(random.uniform(0.1, 0.5), 2)
            influence = round(random.uniform(0.2, 0.6), 2)

        # 分片分配
        priority = random.randint(1, 5)
        adaptability = round(random.uniform(0.5, 0.98), 2)

        # 动态属性
        cpu_usage = round(random.uniform(30, 95), 1)
        mem_usage = round(random.uniform(25, 90), 1)
        resource_flux = round(random.uniform(0.05, 0.4), 2)
        storage_available = round(storage_capacity * random.uniform(0.2, 0.8), 1)
        storage_utilization = round(random.uniform(0.3, 0.9), 2)
        latency_flux = round(random.uniform(1, 15), 1)
        avg_latency = random.randint(20, 300)
        bandwidth_usage = round(random.uniform(0.3, 0.95), 2)

        # 交易动态
        tx_frequency = random.randint(50, 2000)
        processing_delay = random.randint(50, 800)
        stake_change_rate = round(random.uniform(0.01, 0.3), 2)

        # 声誉 - 核心节点有更高声誉
        if node_type in ['validator', 'full_node']:
            reputation_uptime = round(random.uniform(0.98, 0.999), 3)
            reputation_score = round(random.uniform(80, 98), 1)
        elif node_type == 'miner':
            reputation_uptime = round(random.uniform(0.95, 0.998), 3)
            reputation_score = round(random.uniform(70, 95), 1)
        else:
            reputation_uptime = round(random.uniform(0.85, 0.95), 3)
            reputation_score = round(random.uniform(60, 85), 1)

        # 应用负载
        app_tx_freq = random.randint(100, 3000)
        app_storage_ops = random.randint(50, 2000)

        # 生成映射特征
        inter_node_volume = self._generate_inter_node_volume()
        inter_shard_volume = self._generate_inter_shard_volume()
        shard_preference = self._generate_shard_preference()

        # 功能标签和优先级
        function_tags, supported_functions, function_priorities = self._generate_functions(node_type)

        sample = {
            # 硬件资源
            'ResourceCapacity.Hardware.CPU.CoreCount': cpu_cores,
            'ResourceCapacity.Hardware.CPU.ClockFrequency': cpu_freq,
            'ResourceCapacity.Hardware.CPU.Architecture': random.choice(['x86', 'ARM']),
            'ResourceCapacity.Hardware.CPU.CacheSize': cpu_cache,
            'ResourceCapacity.Hardware.Memory.TotalCapacity': memory_total,
            'ResourceCapacity.Hardware.Memory.Type': random.choice(['DDR4', 'DDR5']),
            'ResourceCapacity.Hardware.Memory.Bandwidth': round(memory_bandwidth, 1),
            'ResourceCapacity.Hardware.Storage.Capacity': storage_capacity,
            'ResourceCapacity.Hardware.Storage.Type': random.choice(['SSD', 'HDD', 'NVMe']),
            'ResourceCapacity.Hardware.Storage.ReadWriteSpeed': storage_speed,
            'ResourceCapacity.Hardware.Network.UpstreamBW': upstream_bw,
            'ResourceCapacity.Hardware.Network.DownstreamBW': downstream_bw,
            'ResourceCapacity.Hardware.Network.Latency': f"{random.randint(10, 200)}ms",

            # 运行状态
            'ResourceCapacity.OperationalStatus.Uptime24h': uptime,
            'ResourceCapacity.OperationalStatus.CoreEligibility': str(core_eligible).lower(),
            'ResourceCapacity.OperationalStatus.ResourceUsage.CPUUtilization': cpu_util,
            'ResourceCapacity.OperationalStatus.ResourceUsage.MemUtilization': mem_util,

            # 交易能力
            'OnChainBehavior.TransactionCapability.AvgTPS': avg_tps,
            'OnChainBehavior.TransactionCapability.CrossShardTx.InterNodeVolume': inter_node_volume,
            'OnChainBehavior.TransactionCapability.CrossShardTx.InterShardVolume': inter_shard_volume,
            'OnChainBehavior.TransactionCapability.ConfirmationDelay': f"{confirmation_delay}ms",
            'OnChainBehavior.TransactionCapability.ResourcePerTx.CPUPerTx': cpu_per_tx,
            'OnChainBehavior.TransactionCapability.ResourcePerTx.MemPerTx': mem_per_tx,
            'OnChainBehavior.TransactionCapability.ResourcePerTx.DiskPerTx': disk_per_tx,
            'OnChainBehavior.TransactionCapability.ResourcePerTx.NetworkPerTx': network_per_tx,

            # 区块生成
            'OnChainBehavior.BlockGeneration.AvgInterval': f"{block_interval}s",
            'OnChainBehavior.BlockGeneration.IntervalStdDev': f"{interval_stddev}s",

            # 经济贡献
            'OnChainBehavior.EconomicContribution.FeeContributionRatio': fee_ratio,
            'OnChainBehavior.SmartContractUsage.InvocationFrequency': invocation_freq,
            'OnChainBehavior.TransactionTypes.NormalTxRatio': normal_tx_ratio,
            'OnChainBehavior.TransactionTypes.ContractTxRatio': contract_tx_ratio,

            # 共识参与
            'OnChainBehavior.Consensus.ParticipationRate': participation_rate,
            'OnChainBehavior.Consensus.TotalReward': total_reward,
            'OnChainBehavior.Consensus.SuccessRate': success_rate,

            # 网络拓扑
            'NetworkTopology.GeoLocation.Timezone': timezone,
            'NetworkTopology.GeoLocation.DataCenter': datacenter,
            'NetworkTopology.GeoLocation.Region': region,
            'NetworkTopology.Connections.IntraShardConn': intra_shard_conn,
            'NetworkTopology.Connections.InterShardConn': inter_shard_conn,
            'NetworkTopology.Connections.WeightedDegree': round(weighted_degree, 1),
            'NetworkTopology.Connections.ActiveConn': active_conn,
            'NetworkTopology.Hierarchy.Depth': depth,
            'NetworkTopology.Hierarchy.ConnectionDensity': conn_density,
            'NetworkTopology.Centrality.IntraShard.Eigenvector': eigenvector,
            'NetworkTopology.Centrality.IntraShard.Closeness': closeness,
            'NetworkTopology.Centrality.InterShard.Betweenness': betweenness,
            'NetworkTopology.Centrality.InterShard.Influence': influence,
            'NetworkTopology.ShardAllocation.Priority': priority,
            'NetworkTopology.ShardAllocation.ShardPreference': shard_preference,
            'NetworkTopology.ShardAllocation.Adaptability': adaptability,

            # 动态属性
            'DynamicAttributes.Compute.CPUUsage': cpu_usage,
            'DynamicAttributes.Compute.MemUsage': mem_usage,
            'DynamicAttributes.Compute.ResourceFlux': resource_flux,
            'DynamicAttributes.Storage.Available': storage_available,
            'DynamicAttributes.Storage.Utilization': storage_utilization,
            'DynamicAttributes.Network.LatencyFlux': latency_flux,
            'DynamicAttributes.Network.AvgLatency': f"{avg_latency}ms",
            'DynamicAttributes.Network.BandwidthUsage': bandwidth_usage,
            'DynamicAttributes.Transactions.Frequency': tx_frequency,
            'DynamicAttributes.Transactions.ProcessingDelay': f"{processing_delay}ms",
            'DynamicAttributes.Transactions.StakeChangeRate': stake_change_rate,
            'DynamicAttributes.Reputation.Uptime24h': reputation_uptime,
            'DynamicAttributes.Reputation.ReputationScore': reputation_score,

            # 异构类型
            'HeterogeneousType.NodeType': node_type,
            'HeterogeneousType.FunctionTags': function_tags,
            'HeterogeneousType.SupportedFuncs.Functions': supported_functions,
            'HeterogeneousType.SupportedFuncs.Priorities': function_priorities,
            'HeterogeneousType.Application.CurrentState': random.choice(['active', 'idle', 'high_load', 'maintenance']),
            'HeterogeneousType.Application.LoadMetrics.TxFrequency': app_tx_freq,
            'HeterogeneousType.Application.LoadMetrics.StorageOps': app_storage_ops
        }

        return sample

    def _generate_inter_node_volume(self):
        """生成节点间交易量映射"""
        num_connections = random.randint(2, 5)
        volumes = {}
        for i in range(num_connections):
            node_id = f"node{chr(65 + i)}"
            volume = random.randint(20, 300)
            volumes[node_id] = volume
        return ";".join([f"{k}:{v}" for k, v in volumes.items()])

    def _generate_inter_shard_volume(self):
        """生成分片间交易量映射"""
        num_shards = random.randint(2, 4)
        volumes = {}
        for i in range(num_shards):
            shard_id = f"shard{i+1}"
            volume = random.randint(50, 400)
            volumes[shard_id] = volume
        return ";".join([f"{k}:{v}" for k, v in volumes.items()])

    def _generate_shard_preference(self):
        """生成分片偏好映射"""
        num_shards = random.randint(2, 3)
        total_pref = 1.0
        preferences = {}
        for i in range(num_shards - 1):
            shard_id = f"shard{i+1}"
            pref = round(random.uniform(0.1, total_pref - 0.1), 1)
            preferences[shard_id] = pref
            total_pref -= pref
        preferences[f"shard{num_shards}"] = round(total_pref, 1)
        return ";".join([f"{k}:{v}" for k, v in preferences.items()])

    def _generate_functions(self, node_type):
        """根据节点类型生成功能标签和优先级"""
        function_configs = {
            'full_node': {
                'tags': ['consensus', 'validation', 'storage', 'relay'],
                'functions': ['tx_processing', 'smart_contract_exec', 'data_storage', 'block_validation'],
                'priorities': [1, 2, 3, 2]
            },
            'light_node': {
                'tags': ['validation', 'relay'],
                'functions': ['data_verification', 'tx_relay'],
                'priorities': [1, 2]
            },
            'miner': {
                'tags': ['mining', 'consensus', 'pow'],
                'functions': ['block_mining', 'pow_computation', 'block_production'],
                'priorities': [1, 1, 1]
            },
            'validator': {
                'tags': ['validation', 'consensus', 'pos'],
                'functions': ['block_validation', 'consensus_participation', 'stake_management'],
                'priorities': [1, 1, 2]
            },
            'storage': {
                'tags': ['storage', 'data_service', 'archive'],
                'functions': ['data_storage', 'archive_service', 'data_retrieval'],
                'priorities': [1, 2, 2]
            }
        }

        config = function_configs[node_type]
        tags = ",".join(config['tags'])
        functions = ",".join(config['functions'])
        priorities = ";".join([f"{f}:{p}" for f, p in zip(config['functions'], config['priorities'])])

        return tags, functions, priorities

    def _save_to_csv(self, samples, filename):
        """保存样本到CSV文件"""
        if not samples:
            return

        # 获取所有字段名
        fieldnames = list(samples[0].keys())

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)

    def _show_statistics(self, samples):
        """显示样本统计信息"""
        node_type_counts = {}
        for sample in samples:
            node_type = sample['HeterogeneousType.NodeType']
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

        print(f"\n=== 样本统计信息 ===")
        print(f"总样本数: {len(samples)}")
        print("节点类型分布:")
        for node_type, count in sorted(node_type_counts.items()):
            percentage = count / len(samples) * 100
            print(f"  - {node_type}: {count} 个 ({percentage:.1f}%)")

        # 显示各节点类型的特征概览
        print("\n节点类型特征概览:")
        for node_type in ['full_node', 'light_node', 'miner', 'validator', 'storage']:
            if node_type in node_type_counts:
                type_samples = [s for s in samples if s['HeterogeneousType.NodeType'] == node_type]
                if type_samples:
                    avg_tps = np.mean([s['OnChainBehavior.TransactionCapability.AvgTPS'] for s in type_samples])
                    avg_cpu = np.mean([s['ResourceCapacity.Hardware.CPU.CoreCount'] for s in type_samples])
                    avg_memory = np.mean([s['ResourceCapacity.Hardware.Memory.TotalCapacity'] for s in type_samples])
                    avg_storage = np.mean([s['ResourceCapacity.Hardware.Storage.Capacity'] for s in type_samples])
                    print(f"  {node_type}:")
                    print(f"    平均TPS: {avg_tps:.1f}")
                    print(f"    平均CPU核心: {avg_cpu:.1f}")
                    print(f"    平均内存: {avg_memory:.1f}GB")
                    print(f"    平均存储: {avg_storage:.1f}TB")

def generate_debug_samples():
    """生成调试用的样本数据"""
    generator = BlockchainNodeSampleGenerator(seed=42)

    # 生成不同规模的数据集
    datasets = [
        (200, 'large_samples.csv')      # 大数据集：性能测试
    ]

    for num_samples, filename in datasets:
        print(f"\n生成 {filename}...")
        generator.generate_samples(num_samples, filename)
        print(f"完成生成 {filename}")

if __name__ == "__main__":
    generate_debug_samples()