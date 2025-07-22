# BlockEmulator 数据字段分析报告

## 实际数据结构分析

基于 `message.go` 中的 `StaticNodeFeatures` 和 `DynamicNodeFeatures` 结构体，BlockEmulator能够提供的实际字段数如下：

### 静态特征字段 (StaticNodeFeatures)
#### 硬件资源容量 (ResourceCapacity.Hardware)
1. **CPU** (2字段)
   - CoreCount: int - CPU核心数
   - Architecture: string - CPU架构

2. **Memory** (3字段)
   - TotalCapacity: int - 内存总容量(GB)
   - Type: string - 内存类型
   - Bandwidth: float64 - 内存带宽(GB/s)

3. **Storage** (3字段)
   - Capacity: int - 存储容量(TB)
   - Type: string - 存储类型
   - ReadWriteSpeed: float64 - 存储读写速度(MB/s)

4. **Network** (3字段)
   - UpstreamBW: float64 - 上行带宽(Mbps)
   - DownstreamBW: float64 - 下行带宽(Mbps)
   - Latency: string - 网络延迟

#### 网络拓扑 (NetworkTopology)
5. **GeoLocation** (1字段)
   - Timezone: string - 时区

6. **Connections** (4字段)
   - IntraShardConn: int - 分片内连接数
   - InterShardConn: int - 分片间连接数
   - WeightedDegree: float64 - 加权连接度
   - ActiveConn: int - 活跃连接数

7. **ShardAllocation** (3字段)
   - Priority: int - 分配优先级
   - ShardPreference: string - 分片偏好
   - Adaptability: float64 - 适应性

#### 异构类型 (HeterogeneousType)
8. **基本信息** (2字段)
   - NodeType: string - 节点类型
   - FunctionTags: string - 功能标签

9. **SupportedFuncs** (2字段)
   - Functions: string - 支持的功能
   - Priorities: string - 功能优先级

10. **Application** (1字段)
    - CurrentState: string - 当前状态

11. **LoadMetrics** (2字段)
    - TxFrequency: int - 交易频率
    - StorageOps: int - 存储操作

**静态特征总计：26个字段**

### 动态特征字段 (DynamicNodeFeatures)

#### 链上行为 (OnChainBehavior)
1. **TransactionCapability** (7字段)
   - AvgTPS: float64 - 平均TPS
   - InterNodeVolume: string - 节点间交易量
   - InterShardVolume: string - 分片间交易量
   - ConfirmationDelay: string - 确认延迟
   - CPUPerTx, MemPerTx, DiskPerTx, NetworkPerTx: float64 - 每交易资源消耗(4个)

2. **BlockGeneration** (2字段)
   - AvgInterval: string - 平均区块间隔
   - IntervalStdDev: string - 区块间隔标准差

3. **EconomicContribution** (1字段)
   - FeeContributionRatio: float64 - 手续费贡献比

4. **SmartContractUsage** (1字段)
   - InvocationFrequency: int - 智能合约调用频率

5. **TransactionTypes** (2字段)
   - NormalTxRatio: float64 - 普通交易比例
   - ContractTxRatio: float64 - 合约交易比例

6. **Consensus** (3字段)
   - ParticipationRate: float64 - 共识参与率
   - TotalReward: float64 - 总奖励
   - SuccessRate: float64 - 成功率

#### 动态属性 (DynamicAttributes)
7. **Compute** (3字段)
   - CPUUsage: float64 - CPU使用率
   - MemUsage: float64 - 内存使用率
   - ResourceFlux: float64 - 资源波动

8. **Storage** (2字段)
   - Available: float64 - 可用存储
   - Utilization: float64 - 存储利用率

9. **Network** (3字段)
   - LatencyFlux: float64 - 延迟波动
   - AvgLatency: string - 平均延迟
   - BandwidthUsage: float64 - 带宽使用率

10. **Transactions** (3字段)
    - Frequency: int - 交易频率
    - ProcessingDelay: string - 处理延迟
    - StakeChangeRate: float64 - 质押变化率

11. **Reputation** (2字段)
    - Uptime24h: float64 - 24小时运行时间
    - ReputationScore: float64 - 声誉分数

**动态特征总计：29个字段**

## 总结

### 实际字段数量
- **静态特征**: 26个字段
- **动态特征**: 29个字段  
- **总计**: 55个字段

### 当前硬编码问题分析

1. **第一步特征提取中的硬编码维度**:
   - `comprehensive: 65` - 但实际只有55个字段
   - `f_classic: 128` - 远超实际字段数
   - `f_graph: 96` - 没有对应的图结构特征

2. **维度不匹配问题**:
   - 设想的65维 vs 实际的55个字段
   - 强制padding到128维导致冗余
   - 后续步骤假设128维输入但实际意义不明

3. **特征分组问题**:
   当前硬编码的分组与实际数据结构不匹配：
   ```python
   # 当前硬编码 (总计65维)
   'hardware': 13,       # 实际: 11个字段
   'onchain': 15,        # 实际: 16个字段  
   'topology': 7,        # 实际: 8个字段
   'dynamic': 10,        # 实际: 13个字段
   'heterogeneous': 10,  # 实际: 7个字段
   'crossshard': 4,      # 实际: 跨分片信息分散在多个字段中
   'identity': 2         # 实际: 不在NodeState中
   ```

## 建议的修正方案

### 1. 调整特征维度配置
```python
# 基于实际字段的新配置
feature_dims = {
    'static_total': 26,      # 静态特征总数
    'dynamic_total': 29,     # 动态特征总数
    'comprehensive': 55,     # 综合特征总数(26+29)
    
    # 细分组织 
    'hardware': 11,          # CPU(2) + Memory(3) + Storage(3) + Network(3)
    'network_topology': 8,   # GeoLocation(1) + Connections(4) + ShardAllocation(3)
    'heterogeneous': 7,      # NodeType(1) + FunctionTags(1) + SupportedFuncs(2) + Application(1) + LoadMetrics(2)
    'onchain_behavior': 16,  # TransactionCapability(7) + BlockGeneration(2) + EconomicContribution(1) + SmartContractUsage(1) + TransactionTypes(2) + Consensus(3)
    'dynamic_attributes': 13 # Compute(3) + Storage(2) + Network(3) + Transactions(3) + Reputation(2)
}
```

### 2. 重新设计第一步输出格式
```python
# 新的输出规格
step1_output = {
    'f_static': torch.Tensor,      # [N, 26] 静态特征
    'f_dynamic': torch.Tensor,     # [N, 29] 动态特征  
    'f_combined': torch.Tensor,    # [N, 55] 组合特征
    'f_normalized': torch.Tensor,  # [N, 64] 标准化特征(用于后续步骤)
    'adjacency_matrix': torch.Tensor,  # [N, N] 邻接矩阵
    'edge_index': torch.Tensor,    # [2, E] 图边索引
    'node_metadata': Dict          # 节点元信息
}
```

### 3. 后续步骤适配
- **第二步**: 接受64维标准化特征而非128维
- **第三步**: 调整EvolveGCN输入维度
- **第四步**: 基于55个实际字段计算反馈指标

这样可以确保整个流水线基于真实的数据结构，避免硬编码维度带来的问题。
