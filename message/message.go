package message

import (
	"blockEmulator/core"
	"blockEmulator/shard"
	"time"
)

var prefixMSGtypeLen = 30

type MessageType string
type RequestType string

const (
	CPrePrepare          MessageType = "preprepare"
	CPrepare             MessageType = "prepare"
	CCommit              MessageType = "commit"
	CRequestOldrequest   MessageType = "requestOldrequest"
	CSendOldrequest      MessageType = "sendOldrequest"
	CStop                MessageType = "stop"
	CStopAndCollect      MessageType = "StopAndCollect"
	CBatchReplyNodeState MessageType = "BatchReplyNodeState"

	CRelay          MessageType = "relay"
	CRelayWithProof MessageType = "CRelay&Proof"
	CInject         MessageType = "inject"

	CBlockInfo MessageType = "BlockInfo"
	CSeqIDinfo MessageType = "SequenceID"
	// 新增消息类型
	CRequestNodeState MessageType = "RequestNodeState"
	CReplyNodeState   MessageType = "ReplyNodeState"
)

var (
	BlockRequest RequestType = "Block"
	// add more types
	// ...
)

type RawMessage struct {
	Content []byte // the content of raw message, txs and blocks (most cases) included
}

type Request struct {
	RequestType RequestType
	Msg         RawMessage // request message
	ReqTime     time.Time  // request time
}

type PrePrepare struct {
	RequestMsg *Request // the request message should be pre-prepared
	Digest     []byte   // the digest of this request, which is the only identifier
	SeqID      uint64
}

type Prepare struct {
	Digest     []byte // To identify which request is prepared by this node
	SeqID      uint64
	SenderNode *shard.Node // To identify who send this message
}

type Commit struct {
	Digest     []byte // To identify which request is prepared by this node
	SeqID      uint64
	SenderNode *shard.Node // To identify who send this message
}

type Reply struct {
	MessageID  uint64
	SenderNode *shard.Node
	Result     bool
}

type RequestOldMessage struct {
	SeqStartHeight uint64
	SeqEndHeight   uint64
	ServerNode     *shard.Node // send this request to the server node
	SenderNode     *shard.Node
}

type SendOldMessage struct {
	SeqStartHeight uint64
	SeqEndHeight   uint64
	OldRequest     []*Request
	SenderNode     *shard.Node
}

type InjectTxs struct {
	Txs       []*core.Transaction
	ToShardID uint64
}

// data sent to the supervisor
type BlockInfoMsg struct {
	BlockBodyLength int
	InnerShardTxs   []*core.Transaction // txs which are innerShard
	Epoch           int

	ProposeTime   time.Time // record the propose time of this block (txs)
	CommitTime    time.Time // record the commit time of this block (txs)
	SenderShardID uint64

	// for transaction relay
	Relay1Txs []*core.Transaction // relay1 transactions in chain first time
	Relay2Txs []*core.Transaction // relay2 transactions in chain second time

	// for broker
	Broker1Txs []*core.Transaction // cross transactions at first time by broker
	Broker2Txs []*core.Transaction // cross transactions at second time by broker
}

type SeqIDinfo struct {
	SenderShardID uint64
	SenderSeq     uint64
}

// 节点状态请求消息
type RequestNodeStateMsg struct {
	Timestamp int64
	RequestID string
}

// 节点状态回复消息
type ReplyNodeStateMsg struct {
	ShardID   uint64
	NodeID    uint64
	Timestamp int64
	RequestID string
	NodeState NodeState
}

// 批量节点状态上报消息
// 用于节点实验结束时一次性上报所有快照
type BatchReplyNodeStateMsg struct {
	ShardID uint64
	NodeID  uint64
	States  []ReplyNodeStateMsg
}

// 静态特征结构体
// 只包含实验期间不会变化且可真实采集/计算的字段
// 对齐 wrs.md
type StaticNodeFeatures struct {
	ResourceCapacity struct {
		Hardware struct {
			CPU struct {
				CoreCount    int    `json:"core_count"`   // CPU核心数
				Architecture string `json:"architecture"` // CPU架构
			} `json:"cpu"`
			Memory struct {
				TotalCapacity int     `json:"total_capacity"` // 内存总容量(GB)
				Type          string  `json:"type"`           // 内存类型
				Bandwidth     float64 `json:"bandwidth"`      // 内存带宽(GB/s)
			} `json:"memory"`
			Storage struct {
				Capacity       int     `json:"capacity"`         // 存储容量(TB)
				Type           string  `json:"type"`             // 存储类型
				ReadWriteSpeed float64 `json:"read_write_speed"` // 存储读写速度(MB/s)
			} `json:"storage"`
			Network struct {
				UpstreamBW   float64 `json:"upstream_bw"`   // 上行带宽(Mbps)
				DownstreamBW float64 `json:"downstream_bw"` // 下行带宽(Mbps)
				Latency      string  `json:"latency"`       // 网络延迟
			} `json:"network"`
		} `json:"hardware"`
	} `json:"resource_capacity"`
	NetworkTopology struct {
		GeoLocation struct {
			Timezone string `json:"timezone"` // 时区
		} `json:"geo_location"`
		Connections struct {
			IntraShardConn int     `json:"intra_shard_conn"` // 分片内连接数
			InterShardConn int     `json:"inter_shard_conn"` // 分片间连接数
			WeightedDegree float64 `json:"weighted_degree"`  // 加权连接度
			ActiveConn     int     `json:"active_conn"`      // 活跃连接数
		} `json:"connections"`
		ShardAllocation struct {
			Priority        int     `json:"priority"`         // 分配优先级
			ShardPreference string  `json:"shard_preference"` // 分片偏好
			Adaptability    float64 `json:"adaptability"`     // 适应性
		} `json:"shard_allocation"`
	} `json:"network_topology"`
	HeterogeneousType struct {
		NodeType       string `json:"node_type"`     // 节点类型
		FunctionTags   string `json:"function_tags"` // 功能标签
		SupportedFuncs struct {
			Functions  string `json:"functions"`  // 支持的功能
			Priorities string `json:"priorities"` // 功能优先级
		} `json:"supported_funcs"`
		Application struct {
			CurrentState string `json:"current_state"` // 当前状态
			LoadMetrics  struct {
				TxFrequency int `json:"tx_frequency"` // 交易频率
				StorageOps  int `json:"storage_ops"`  // 存储操作
			} `json:"load_metrics"`
		} `json:"application"`
	} `json:"heterogeneous_type"`
}

// 动态特征结构体
// 只包含实验期间会变化的字段
type DynamicNodeFeatures struct {
	OnChainBehavior struct {
		TransactionCapability struct {
			AvgTPS       float64 `json:"avg_tps"`
			CrossShardTx struct {
				InterNodeVolume  string `json:"inter_node_volume"`
				InterShardVolume string `json:"inter_shard_volume"`
			} `json:"cross_shard_tx"`
			ConfirmationDelay string `json:"confirmation_delay"`
			ResourcePerTx     struct {
				CPUPerTx     float64 `json:"cpu_per_tx"`
				MemPerTx     float64 `json:"mem_per_tx"`
				DiskPerTx    float64 `json:"disk_per_tx"`
				NetworkPerTx float64 `json:"network_per_tx"`
			} `json:"resource_per_tx"`
		} `json:"transaction_capability"`
		BlockGeneration struct {
			AvgInterval    string `json:"avg_interval"`
			IntervalStdDev string `json:"interval_std_dev"`
		} `json:"block_generation"`
		EconomicContribution struct {
			FeeContributionRatio float64 `json:"fee_contribution_ratio"`
		} `json:"economic_contribution"`
		SmartContractUsage struct {
			InvocationFrequency int `json:"invocation_frequency"`
		} `json:"smart_contract_usage"`
		TransactionTypes struct {
			NormalTxRatio   float64 `json:"normal_tx_ratio"`
			ContractTxRatio float64 `json:"contract_tx_ratio"`
		} `json:"transaction_types"`
		Consensus struct {
			ParticipationRate float64 `json:"participation_rate"`
			TotalReward       float64 `json:"total_reward"`
			SuccessRate       float64 `json:"success_rate"`
		} `json:"consensus"`
	} `json:"on_chain_behavior"`
	DynamicAttributes struct {
		Compute struct {
			CPUUsage     float64 `json:"cpu_usage"`
			MemUsage     float64 `json:"mem_usage"`
			ResourceFlux float64 `json:"resource_flux"`
		} `json:"compute"`
		Storage struct {
			Available   float64 `json:"available"`
			Utilization float64 `json:"utilization"`
		} `json:"storage"`
		Network struct {
			LatencyFlux    float64 `json:"latency_flux"`
			AvgLatency     string  `json:"avg_latency"`
			BandwidthUsage float64 `json:"bandwidth_usage"`
		} `json:"network"`
		Transactions struct {
			Frequency       int     `json:"frequency"`
			ProcessingDelay string  `json:"processing_delay"`
			StakeChangeRate float64 `json:"stake_change_rate"`
		} `json:"transactions"`
		Reputation struct {
			Uptime24h       float64 `json:"uptime_24h"`
			ReputationScore float64 `json:"reputation_score"`
		} `json:"reputation"`
	} `json:"dynamic_attributes"`
}

// 新NodeState结构体
// 只包含静态和动态特征
type NodeState struct {
	Static  StaticNodeFeatures  `json:"static"`
	Dynamic DynamicNodeFeatures `json:"dynamic"`
}

// 新增：包含epoch信息的分区消息
type PartitionModifiedMapWithEpoch struct {
	PartitionModified map[string]uint64 `json:"partitionModified"`
	EpochID           int32             `json:"epochID"`
	Timestamp         int64             `json:"timestamp"`
}

func MergeMessage(msgType MessageType, content []byte) []byte {
	b := make([]byte, prefixMSGtypeLen)
	for i, v := range []byte(msgType) {
		b[i] = v
	}
	merge := append(b, content...)
	return merge
}

func SplitMessage(message []byte) (MessageType, []byte) {
	msgTypeBytes := message[:prefixMSGtypeLen]
	msgType_pruned := make([]byte, 0)
	for _, v := range msgTypeBytes {
		if v != byte(0) {
			msgType_pruned = append(msgType_pruned, v)
		}
	}
	msgType := string(msgType_pruned)
	content := message[prefixMSGtypeLen:]
	return MessageType(msgType), content
}
