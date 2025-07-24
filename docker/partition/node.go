package partition

import (
	"time"
)

// 节点主结构体
type Node struct {
	ResourceCapacity  ResourceCapacityLayer
	OnChainBehavior   OnChainBehaviorLayer
	NetworkTopology   NetworkTopologyLayer
	DynamicAttributes DynamicAttributesLayer
	HeterogeneousType HeterogeneousTypeLayer
}

// 第一层：资源能力层
type ResourceCapacityLayer struct {
	Hardware          Hardware
	OperationalStatus OperationalStatus
}

type Hardware struct {
	CPU struct {
		CoreCount      uint    // 核数
		ClockFrequency float64 // GHz
		Architecture   string  // x86/ARM
		CacheSize      uint    // MB
	}
	Memory struct {
		TotalCapacity uint    // GB
		Type          string  // DDR4/DDR5
		Bandwidth     float64 // GB/s
	}
	Storage struct {
		Capacity       uint    // TB
		Type           string  // SSD/HDD
		ReadWriteSpeed float64 // MB/s
	}
	Network struct {
		UpstreamBW   float64 // Mbps/Gbps
		DownstreamBW float64 // Mbps/Gbps
		Latency      time.Duration
	}
}

type OperationalStatus struct {
	Uptime24h       float32 // 24小时在线率
	CoreEligibility bool    // 是否满足核心分片要求(99.9%)
	ResourceUsage   struct {
		CPUUtilization float32 // %
		MemUtilization float32 // %
	}
}

// 第二层：链上行为层
type OnChainBehaviorLayer struct {
	TransactionCapability TransactionCapability
	BlockGeneration       BlockBehavior
	EconomicContribution  EconomicMetrics
	SmartContractUsage    SmartContractMetrics
	TransactionTypes      TransactionTypeDistribution
	Consensus             ConsensusParticipation
}

type TransactionCapability struct {
	AvgTPS            float64 // 平均TPS
	CrossShardTx      CrossShardTransaction
	ConfirmationDelay time.Duration // 确认延迟
	ResourcePerTx     struct {
		CPUPerTx     float64 // CPT
		MemPerTx     float64 // MSPT
		DiskPerTx    float64 // DIOPT
		NetworkPerTx float64 // NDPT
	}
}

type CrossShardTransaction struct {
	InterNodeVolume  map[string]uint // 节点间跨片交易量 [nodeID]count
	InterShardVolume map[string]uint // 分片间跨片交易量 [shardID]count
}

type BlockBehavior struct {
	AvgInterval    time.Duration // 平均出块间隔
	IntervalStdDev time.Duration // 出块间隔标准差
}

type EconomicMetrics struct {
	FeeContributionRatio float32 // 交易费贡献比例
}

type SmartContractMetrics struct {
	InvocationFrequency uint // 合约调用次数
}

type TransactionTypeDistribution struct {
	NormalTxRatio   float32 // 普通交易比例
	ContractTxRatio float32 // 合约交易比例
}

type ConsensusParticipation struct {
	ParticipationRate float32 // 参与率
	TotalReward       float64 // 总奖励
	SuccessRate       float32 // 成功率
}

// 第三层：网络拓扑层
type NetworkTopologyLayer struct {
	GeoLocation     GeoInfo
	Connections     ConnectionFeatures
	Hierarchy       NetworkHierarchy
	Centrality      NodeCentrality
	ShardAllocation ShardAllocationInfo
}

type GeoInfo struct {
	Timezone   string
	DataCenter string // AWS/Google Cloud等
	Region     string
}

type ConnectionFeatures struct {
	IntraShardConn uint    // 分片内连接数
	InterShardConn uint    // 跨分片连接数
	WeightedDegree float64 // 加权节点度
	ActiveConn     uint    // 活跃连接数
}

type NetworkHierarchy struct {
	Depth             uint    // 层级深度
	ConnectionDensity float64 // 连接密度
}

type NodeCentrality struct {
	IntraShard struct {
		Eigenvector float64
		Closeness   float64
	}
	InterShard struct {
		Betweenness float64
		Influence   float64
	}
}

type ShardAllocationInfo struct {
	Priority        uint
	ShardPreference map[string]float64 // [shardID]preference
	Adaptability    float32
}

// 第四层：动态属性层
type DynamicAttributesLayer struct {
	Compute struct {
		CPUUsage     float32
		MemUsage     float32
		ResourceFlux float32 // 资源波动系数
	}
	Storage struct {
		Available   uint    // 剩余空间
		Utilization float32 // 使用率
	}
	Network struct {
		LatencyFlux    float32 // 延迟波动
		AvgLatency     time.Duration
		BandwidthUsage float32
	}
	Transactions struct {
		Frequency       uint // 交易频率
		ProcessingDelay time.Duration
		StakeChangeRate float32 // 质押金额变化率
	}
	Reputation struct {
		Uptime24h       float32
		ReputationScore float32
	}
}

// 第五层：异构类型标识层
type HeterogeneousTypeLayer struct {
	NodeType       string
	FunctionTags   []string
	SupportedFuncs struct {
		Functions  []string
		Priorities map[string]uint // [function]priority
	}
	Application struct {
		CurrentState string
		LoadMetrics  struct {
			TxFrequency uint
			StorageOps  uint
		}
	}
}
