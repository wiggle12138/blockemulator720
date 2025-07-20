特征类别​	​特征名称​	​特征描述​	​示例值​
​1. 硬件资源​			
CPU	ResourceCapacity.Hardware.CPU.CoreCount	CPU核心数量	16
	ResourceCapacity.Hardware.CPU.Architecture	CPU架构	x86/ARM
内存	ResourceCapacity.Hardware.Memory.TotalCapacity	内存总容量(GB)	128
	ResourceCapacity.Hardware.Memory.Type	内存类型	DDR4/DDR5 
	ResourceCapacity.Hardware.Memory.Bandwidth	内存带宽(GB/s)	85.4
存储	ResourceCapacity.Hardware.Storage.Capacity	存储容量(TB)	8
	ResourceCapacity.Hardware.Storage.Type	存储类型	SSD/HDD/NVMe
	ResourceCapacity.Hardware.Storage.ReadWriteSpeed	存储读写速度(MB/s)	1200
网络	ResourceCapacity.Hardware.Network.UpstreamBW	上行带宽(Mbps)	1500
	ResourceCapacity.Hardware.Network.DownstreamBW	下行带宽(Mbps)	15000
	ResourceCapacity.Hardware.Network.Latency	网络延迟	"50ms"
​2. 运行状态​			
可用性	
	ResourceCapacity.OperationalStatus.CoreEligibility	核心节点资格	"true"/"false"，用cpu核心数>=4就是true，反之是false
资源使用	ResourceCapacity.OperationalStatus.ResourceUsage.CPUUtilization	CPU利用率(%)	65.3
	ResourceCapacity.OperationalStatus.ResourceUsage.MemUtilization	内存利用率(%)	45.2
​3. 交易能力​			
吞吐量	OnChainBehavior.TransactionCapability.AvgTPS	平均每秒交易数(TPS)	650.5
跨分片交易	OnChainBehavior.TransactionCapability.CrossShardTx.InterNodeVolume	节点间交易量映射	"nodeA:120;nodeB:80"
	OnChainBehavior.TransactionCapability.CrossShardTx.InterShardVolume	分片间交易量映射	"shard1:200;shard2:150"
延迟	OnChainBehavior.TransactionCapability.ConfirmationDelay	交易确认延迟	"800ms"

​4. 区块生成​			
	OnChainBehavior.BlockGeneration.AvgInterval	平均区块间隔	"15s"
	OnChainBehavior.BlockGeneration.IntervalStdDev	区块间隔标准差	"1.5s"
​5. 经济贡献​			
交易类型	OnChainBehavior.TransactionTypes.NormalTxRatio	普通交易比例	0.65
	OnChainBehavior.TransactionTypes.ContractTxRatio	合约交易比例	0.35
​6. 共识参与​			
	OnChainBehavior.Consensus.ParticipationRate	参与率	0.92
	OnChainBehavior.Consensus.TotalReward	总奖励	45.8
	OnChainBehavior.Consensus.SuccessRate	成功率	0.95
​7. 网络拓扑​			
地理位置	NetworkTopology.GeoLocation.Timezone	时区	UTC+8
连接	NetworkTopology.Connections.IntraShardConn	分片内连接数	45
	NetworkTopology.Connections.InterShardConn	分片间连接数	25
	NetworkTopology.Connections.WeightedDegree	加权连接度	67.5
	NetworkTopology.Connections.ActiveConn	活跃连接数	35

分片分配	
	NetworkTopology.ShardAllocation.Adaptability	适应性	0.88    这个按照链上指标判断，参与度高适应性强
​8. 动态属性​			
	
网络	DynamicAttributes.Network.LatencyFlux	延迟波动	5.3
	DynamicAttributes.Network.AvgLatency	平均延迟	"150ms"
	DynamicAttributes.Network.BandwidthUsage	带宽利用率	0.85
交易	DynamicAttributes.Transactions.Frequency	交易频率	1200
	DynamicAttributes.Transactions.ProcessingDelay	处理延迟	"200ms"

​9. 异构类型​			
节点类型	HeterogeneousType.NodeType	节点类型	full_node
功能	HeterogeneousType.FunctionTags	功能标签	"consensus,validation"
	HeterogeneousType.SupportedFuncs.Functions	支持的功能	"tx_processing,data_verification"
应用	HeterogeneousType.Application.CurrentState	当前状态	active/idle/high_load
	HeterogeneousType.Application.LoadMetrics.TxFrequency	交易频率	2500
	HeterogeneousType.Application.LoadMetrics.StorageOps	存储操作	800



// 数据传递格式
ReplyNodeStateMsg {
    ShardID   uint64
    NodeID    uint64  
    Timestamp int64
    NodeState: {
        Static: StaticNodeFeatures {     // 静态特征
            ResourceCapacity    // 硬件资源
            NetworkTopology     // 网络拓扑
            HeterogeneousType   // 异构类型
        }
        Dynamic: DynamicNodeFeatures {   // 动态特征
            OnChainBehavior     // 链上行为
            DynamicAttributes   // 动态属性
        }
    }
}

// 44个字段完整覆盖
├── NodeID                                    // 节点标识
├── ResourceCapacity.Hardware.*              // 硬件资源 (12字段)
├── ResourceCapacity.OperationalStatus.*     // 操作状态 (3字段) 
├── OnChainBehavior.*                         // 链上行为 (11字段)
├── NetworkTopology.*                         // 网络拓扑 (6字段)
├── DynamicAttributes.*                       // 动态属性 (5字段)
└── HeterogeneousType.*                       // 异构类型 (6字段)
