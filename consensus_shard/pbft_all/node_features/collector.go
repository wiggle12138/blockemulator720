package node_features

import (
	"blockEmulator/core"
	"blockEmulator/message"
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// NodeFeatureCollector 节点特征收集器
type NodeFeatureCollector struct {
	// 基本信息
	shardID uint64
	nodeID  uint64

	// 配置信息
	nodeNums  uint64
	shardNums uint64

	// 状态锁
	stateLock sync.RWMutex

	// 窗口化计算相关
	windowStartTime    time.Time
	windowTxCount      int
	windowBlockCount   int
	windowPrepareCount int
	windowCommitCount  int

	// 队列状态
	txPoolSize      int
	requestPoolSize int

	// 真实区块时间统计
	blockTimestamps []time.Time
	systemStartTime time.Time

	// 真实交易统计
	crossShardTxStats map[string]int // "S{shardID}N{nodeID}" -> 交易数量
	interNodeTxStats  map[string]int // "S{shardID}N{nodeID}" -> 交易数量
	shardTxStats      map[uint64]int // shardID -> 交易数量（用于分片间统计）
	intraShardTxCount int            // 分片内交易总数

	// 第一个要求：优化网络延迟波动计算
	latencyHistory []float64 // 延迟历史记录

	// 资源使用统计
	cpuUsageHistory []float64
	memUsageHistory []float64

	// 第二个要求：处理延迟计算相关
	blockProcessingTimes []time.Duration // 区块处理时间记录
	txCountInBlocks      []int           // 每个区块的交易数量

	// 静态数据缓存（初始化时收集一次）
	staticData            message.NodeState
	staticDataInitialized bool

	// 异步采集相关
	stateCollectCh  chan struct{}
	collectorStopCh chan struct{}
	collectedStates []message.ReplyNodeStateMsg

	// 交易处理频率历史记录
	txProcessingHistory []int // 交易处理频率历史记录
}

// NewNodeFeatureCollector 创建新的节点特征收集器
func NewNodeFeatureCollector(shardID, nodeID, nodeNums, shardNums uint64) *NodeFeatureCollector {
	nfc := &NodeFeatureCollector{
		shardID:   shardID,
		nodeID:    nodeID,
		nodeNums:  nodeNums,
		shardNums: shardNums,

		windowStartTime: time.Now(),
		systemStartTime: time.Now(),

		crossShardTxStats: make(map[string]int),
		interNodeTxStats:  make(map[string]int),
		shardTxStats:      make(map[uint64]int),

		cpuUsageHistory: make([]float64, 0, 100),
		memUsageHistory: make([]float64, 0, 100),

		stateCollectCh:  make(chan struct{}, 100),
		collectorStopCh: make(chan struct{}),
		collectedStates: make([]message.ReplyNodeStateMsg, 0),
	}

	// 初始化统计映射
	nfc.initializeStatsMaps()

	// 初始化新增的历史记录
	nfc.latencyHistory = make([]float64, 0, 50)
	nfc.blockProcessingTimes = make([]time.Duration, 0, 50)
	nfc.txCountInBlocks = make([]int, 0, 50)

	// 初始化静态数据
	nfc.initializeStaticData()

	// 启动异步采集器
	go nfc.runStateCollector()

	return nfc
}

// 初始化统计映射
func (nfc *NodeFeatureCollector) initializeStatsMaps() {
	// 初始化跨分片交易统计
	for sid := uint64(0); sid < nfc.shardNums; sid++ {
		nfc.shardTxStats[sid] = 0
		for nid := uint64(0); nid < nfc.nodeNums; nid++ {
			nodeKey := fmt.Sprintf("S%dN%d", sid, nid)
			nfc.crossShardTxStats[nodeKey] = 0
			nfc.interNodeTxStats[nodeKey] = 0
		}
	}
}

// 初始化静态数据
func (nfc *NodeFeatureCollector) initializeStaticData() {
	nfc.stateLock.Lock()
	defer nfc.stateLock.Unlock()
	if nfc.staticDataInitialized {
		return
	}
	static := message.StaticNodeFeatures{}

	// 1. 硬件资源
	cpuLimit := nfc.getCpuLimit()
	if cpuLimit > 0 {
		static.ResourceCapacity.Hardware.CPU.CoreCount = cpuLimit
	} else {
		// 环境变量或宿主机
		if env := os.Getenv("CPU_LIMIT"); env != "" {
			if f, err := strconv.ParseFloat(env, 64); err == nil {
				static.ResourceCapacity.Hardware.CPU.CoreCount = int(f + 0.5)
			}
		} else {
			static.ResourceCapacity.Hardware.CPU.CoreCount = runtime.NumCPU()
		}
	}
	static.ResourceCapacity.Hardware.CPU.Architecture = runtime.GOARCH

	memLimit := nfc.getMemoryLimitGB()
	if memLimit > 0 {
		static.ResourceCapacity.Hardware.Memory.TotalCapacity = memLimit
	} else {
		memEnv := os.Getenv("MEM_LIMIT_GB")
		if memEnv == "" {
			memEnv = os.Getenv("MEM_LIMIT")
		}
		static.ResourceCapacity.Hardware.Memory.TotalCapacity = nfc.parseMemLimitEnv(memEnv)
	}
	static.ResourceCapacity.Hardware.Memory.Type = "DDR4"

	// 直接从环境变量读取MEMORY_BANDWIDTH
	static.ResourceCapacity.Hardware.Memory.Bandwidth = nfc.getEnvOrDefaultFloat("MEMORY_BANDWIDTH", 1600.0)

	// 直接从环境变量读取存储信息
	static.ResourceCapacity.Hardware.Storage.Capacity = nfc.getEnvOrDefaultInt("STORAGE_CAPACITY", 512)
	static.ResourceCapacity.Hardware.Storage.Type = "SSD"
	static.ResourceCapacity.Hardware.Storage.ReadWriteSpeed = nfc.getEnvOrDefaultFloat("STORAGE_READ_WRITE_SPEED", 100.0)

	// 直接从环境变量读取网络带宽
	static.ResourceCapacity.Hardware.Network.UpstreamBW = nfc.getEnvOrDefaultFloat("NETWORK_UPSTREAM_BW", 50.0)
	static.ResourceCapacity.Hardware.Network.DownstreamBW = nfc.getEnvOrDefaultFloat("NETWORK_DOWNSTREAM_BW", 50.0)

	// 延长ping超时时间，尝试真实ping
	static.ResourceCapacity.Hardware.Network.Latency = nfc.getNetworkLatencyWithTimeout()

	// 2. 网络拓扑 - 第四个要求：基于交易关系计算连接
	static.NetworkTopology.GeoLocation.Timezone = nfc.getEnvOrDefaultString("TIMEZONE", "UTC+8")
	static.NetworkTopology.Connections.IntraShardConn = nfc.calculateIntraShardConnections()
	static.NetworkTopology.Connections.InterShardConn = nfc.calculateInterShardConnections()
	static.NetworkTopology.Connections.WeightedDegree = nfc.calculateWeightedDegree()
	static.NetworkTopology.Connections.ActiveConn = nfc.calculateActiveConnections()
	// 第五个要求：移除ShardAllocation.Priority
	static.NetworkTopology.ShardAllocation.Adaptability = nfc.calculateAdaptability()

	// 3. 异构类型 - 第六个要求：降低节点类型分配标准
	nodeType := nfc.determineNodeTypeRelaxed(static.ResourceCapacity.Hardware.CPU.CoreCount,
		static.ResourceCapacity.Hardware.Storage.Capacity, static.ResourceCapacity.Hardware.Memory.TotalCapacity)
	static.HeterogeneousType.NodeType = nodeType
	// 第三个要求：使用优化的FunctionTags选择逻辑
	static.HeterogeneousType.FunctionTags = nfc.determineFunctionTagsOptimized(nodeType)
	// 第四个要求：使用优化的SupportedFuncs.Functions选择逻辑
	static.HeterogeneousType.SupportedFuncs.Functions = nfc.determineSupportedFunctionsOptimized(
		static.ResourceCapacity.Hardware.CPU.CoreCount,
		static.ResourceCapacity.Hardware.Memory.TotalCapacity,
		static.ResourceCapacity.Hardware.Network.UpstreamBW)
	// 移除SupportedFuncs.Priorities

	// 第八个要求：应用状态初始化
	static.HeterogeneousType.Application.CurrentState = nfc.determineApplicationState()
	// 第五个要求：使用优化的负载指标计算
	static.HeterogeneousType.Application.LoadMetrics.TxFrequency = nfc.calculateRealTxFrequency()
	static.HeterogeneousType.Application.LoadMetrics.StorageOps = nfc.calculateStorageOpsMetricOptimized()

	nfc.staticData.Static = static
	nfc.staticDataInitialized = true
}

// 第四个要求：基于节点间交易计算分片内连接数
func (nfc *NodeFeatureCollector) calculateIntraShardConnections() int {
	baseConnections := int(nfc.nodeNums - 1) // 基础连接数

	// 基于交易活跃度调整
	totalIntraShardTx := nfc.intraShardTxCount
	if totalIntraShardTx > 100 {
		return baseConnections // 高活跃度，全连接
	} else if totalIntraShardTx > 20 {
		return baseConnections - 1 // 中等活跃度
	} else {
		return int(float64(baseConnections) * 0.6) // 低活跃度，部分连接
	}
}

// 第四个要求：基于跨分片交易计算分片间连接数
func (nfc *NodeFeatureCollector) calculateInterShardConnections() int {
	baseConnections := int(nfc.shardNums - 1)

	// 统计本节点参与的跨分片交易数
	crossShardTxCount := 0
	for _, count := range nfc.shardTxStats {
		crossShardTxCount += count
	}

	if crossShardTxCount > 50 {
		return baseConnections // 高跨分片活跃度
	} else if crossShardTxCount > 10 {
		return baseConnections - 1 // 中等跨分片活跃度
	} else {
		return int(float64(baseConnections) * 0.5) // 低跨分片活跃度
	}
}

// 第四个要求：基于交易权重计算加权连接度
func (nfc *NodeFeatureCollector) calculateWeightedDegree() float64 {
	// 基于节点硬件配置和交易处理能力
	cpuCores := nfc.staticData.Static.ResourceCapacity.Hardware.CPU.CoreCount
	memCapacity := nfc.staticData.Static.ResourceCapacity.Hardware.Memory.TotalCapacity

	baseWeight := float64(cpuCores)*10.0 + float64(memCapacity)*0.1

	// 基于交易活跃度调整
	txActivityBonus := float64(nfc.windowTxCount) * 0.5
	if txActivityBonus > 30.0 {
		txActivityBonus = 30.0
	}

	totalWeight := baseWeight + txActivityBonus
	if totalWeight > 100.0 {
		totalWeight = 100.0
	}

	return totalWeight
}

// 第四个要求：基于实际网络活动计算活跃连接数
func (nfc *NodeFeatureCollector) calculateActiveConnections() int {
	// 基于共识消息活动
	consensusActivity := nfc.windowPrepareCount + nfc.windowCommitCount

	// 基于交易处理活动
	txActivity := nfc.windowTxCount

	// 基础活跃连接数
	baseActive := int(nfc.nodeNums/2) + int(nfc.shardNums/2)

	// 根据活动度调整
	if consensusActivity > 20 || txActivity > 50 {
		return baseActive + 2 // 高活跃度
	} else if consensusActivity > 5 || txActivity > 10 {
		return baseActive // 中等活跃度
	} else {
		return int(float64(baseActive) * 0.7) // 低活跃度
	}
}

// 计算适应性：基于链上参与度
func (nfc *NodeFeatureCollector) calculateAdaptability() float64 {
	// 基于共识参与度
	totalMsgs := float64(nfc.windowPrepareCount + nfc.windowCommitCount)
	expectedMsgs := float64(nfc.windowBlockCount) * 2.0

	var adaptabilityScore float64 = 0.5 // 基础适应性

	if expectedMsgs > 0 {
		participationRatio := totalMsgs / expectedMsgs
		adaptabilityScore = 0.3 + participationRatio*0.6 // 参与度高的节点适应性强
	}

	// 基于交易处理能力
	if nfc.windowTxCount > 30 {
		adaptabilityScore += 0.1
	}

	if adaptabilityScore > 1.0 {
		adaptabilityScore = 1.0
	}

	return adaptabilityScore
}

// 第六个要求：降低节点类型分配标准
func (nfc *NodeFeatureCollector) determineNodeTypeRelaxed(cpuCores, storageGB, memoryGB int) string {
	// 如果是分片内0号节点，优先考虑为full_node
	if nfc.nodeID == 0 {
		if cpuCores >= 1 && memoryGB >= 1 {
			return "full_node"
		}
	}

	// 挖矿节点：降低标准
	if cpuCores >= 2 && memoryGB >= 2 && (storageGB >= 1000 || cpuCores >= 3) {
		return "miner_node"
	}

	// 存储节点：降低标准
	if storageGB >= 1000 || (storageGB >= 500 && cpuCores >= 2) {
		return "storage_node"
	}

	// 验证节点：降低标准
	if cpuCores >= 1 && memoryGB >= 1 {
		return "validate_node"
	}

	// 轻量节点
	return "light_node"
}

// 第八个要求：基于交易量和资源利用率综合判断应用状态
func (nfc *NodeFeatureCollector) determineApplicationState() string {
	// 获取当前资源利用率
	cpuUsage := nfc.getRealCPUUsage()
	memUsage := nfc.getRealMemUsage()

	// 计算窗口期间的实际TPS
	windowDuration := time.Since(nfc.windowStartTime).Seconds()
	if windowDuration <= 0 {
		windowDuration = 1
	}
	currentTPS := float64(nfc.windowTxCount) / windowDuration

	// 基于交易处理情况判断
	hasTransactionActivity := nfc.windowTxCount > 0 || nfc.txPoolSize > 0

	// 判断高负载状态 - 基于资源利用率和交易量
	if cpuUsage >= 70.0 || memUsage >= 80.0 || currentTPS >= 20.0 || nfc.txPoolSize >= 50 {
		return "high_load"
	}

	// 判断活跃状态 - 只要有交易处理就是活跃
	if hasTransactionActivity || cpuUsage >= 30.0 || memUsage >= 40.0 {
		return "active"
	}

	// 空闲状态 - 没有交易处理且资源利用率低
	return "idle"
}

// 读取cgroup限制的CPU核数（支持v2和v1），失败则返回-1
func (nfc *NodeFeatureCollector) getCpuLimit() int {
	// cgroup v2: /sys/fs/cgroup/cpu.max
	if data, err := os.ReadFile("/sys/fs/cgroup/cpu.max"); err == nil {
		parts := strings.Fields(string(data))
		if len(parts) >= 2 && parts[0] != "max" {
			quota, _ := strconv.ParseFloat(parts[0], 64)
			period, _ := strconv.ParseFloat(parts[1], 64)
			if period > 0 {
				return int(quota/period + 0.5)
			}
		}
	}
	// cgroup v1: /sys/fs/cgroup/cpu/cpu.cfs_quota_us, cpu.cfs_period_us
	if quotaData, err1 := os.ReadFile("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"); err1 == nil {
		if periodData, err2 := os.ReadFile("/sys/fs/cgroup/cpu/cpu.cfs_period_us"); err2 == nil {
			quota, _ := strconv.ParseFloat(strings.TrimSpace(string(quotaData)), 64)
			period, _ := strconv.ParseFloat(strings.TrimSpace(string(periodData)), 64)
			if quota > 0 && period > 0 {
				return int(quota/period + 0.5)
			}
		}
	}
	return -1
}

// 读取cgroup限制的内存（GB），失败则返回-1
func (nfc *NodeFeatureCollector) getMemoryLimitGB() int {
	// cgroup v2: /sys/fs/cgroup/memory.max
	if data, err := os.ReadFile("/sys/fs/cgroup/memory.max"); err == nil {
		val := strings.TrimSpace(string(data))
		if val != "max" {
			if bytes, err := strconv.ParseInt(val, 10, 64); err == nil {
				return int(bytes / (1024 * 1024 * 1024))
			}
		}
	}
	// cgroup v1: /sys/fs/cgroup/memory/memory.limit_in_bytes
	if data, err := os.ReadFile("/sys/fs/cgroup/memory/memory.limit_in_bytes"); err == nil {
		if bytes, err := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64); err == nil {
			return int(bytes / (1024 * 1024 * 1024))
		}
	}
	return -1
}

// 辅助函数：解析带单位的内存环境变量
func (nfc *NodeFeatureCollector) parseMemLimitEnv(val string) int {
	if val == "" {
		return 0
	}
	v := strings.ToLower(strings.TrimSpace(val))
	if strings.HasSuffix(v, "g") {
		n, err := strconv.Atoi(strings.TrimSuffix(v, "g"))
		if err == nil {
			return n
		}
	}
	if strings.HasSuffix(v, "m") {
		n, err := strconv.Atoi(strings.TrimSuffix(v, "m"))
		if err == nil {
			return n / 1024
		}
	}
	if n, err := strconv.Atoi(v); err == nil {
		return n
	}
	return 0
}

// 获取环境变量或默认值
func (nfc *NodeFeatureCollector) getEnvOrDefaultInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return def
}

func (nfc *NodeFeatureCollector) getEnvOrDefaultFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

func (nfc *NodeFeatureCollector) getEnvOrDefaultString(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// 增加延长超时的ping函数
func (nfc *NodeFeatureCollector) getNetworkLatencyWithTimeout() string {
	// 使用更长的超时时间进行ping测试
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "ping", "-c", "3", "-W", "3000", "8.8.8.8")
	out, err := cmd.Output()
	if err == nil {
		lines := strings.Split(string(out), "\n")
		for _, line := range lines {
			if strings.Contains(line, "rtt min/avg/max") || strings.Contains(line, "round-trip min/avg/max") {
				parts := strings.Split(line, "=")
				if len(parts) == 2 {
					stats := strings.Split(parts[1], "/")
					if len(stats) >= 2 {
						avg := strings.TrimSpace(stats[1])
						if strings.HasSuffix(avg, " ms") {
							avg = strings.TrimSuffix(avg, " ms")
						}
						return fmt.Sprintf("%sms", avg)
					}
				}
			}
		}
	}
	// 如果ping失败，返回基于分片的模拟延迟
	baseLatency := 50 + int(nfc.shardID)*15 + int(nfc.nodeID)*5
	return fmt.Sprintf("%dms", baseLatency)
}

// 收集动态特征
func (nfc *NodeFeatureCollector) collectDynamicFeatures() message.DynamicNodeFeatures {
	var dyn message.DynamicNodeFeatures
	windowDuration := time.Since(nfc.windowStartTime).Seconds()
	if windowDuration <= 0 {
		windowDuration = 1
	}

	// === 交易能力 ===
	dyn.OnChainBehavior.TransactionCapability.AvgTPS = float64(nfc.windowTxCount) / windowDuration

	// InterShardVolume为分片间交易量格式（shard0:1000;shard1:2000）
	dyn.OnChainBehavior.TransactionCapability.CrossShardTx.InterShardVolume = nfc.generateInterShardVolumeString()
	dyn.OnChainBehavior.TransactionCapability.CrossShardTx.InterNodeVolume = nfc.generateInterNodeVolumeString()
	dyn.OnChainBehavior.TransactionCapability.ConfirmationDelay = nfc.calculateProcessingDelay()

	// === 区块生成 ===
	// 使用真实的区块间隔统计
	avgInterval, stdDev := nfc.calculateRealBlockIntervals()
	dyn.OnChainBehavior.BlockGeneration.AvgInterval = fmt.Sprintf("%.2fs", avgInterval)
	dyn.OnChainBehavior.BlockGeneration.IntervalStdDev = fmt.Sprintf("%.2fs", stdDev)

	// === 交易类型 ===
	// 第一个要求：确保NormalTxRatio和ContractTxRatio相加为1，基于交易量判断
	normalRatio := nfc.calculateNormalTxRatio()
	dyn.OnChainBehavior.TransactionTypes.NormalTxRatio = normalRatio
	dyn.OnChainBehavior.TransactionTypes.ContractTxRatio = 1.0 - normalRatio // 确保相加为1

	// === 共识参与 ===
	// 第二个要求：ParticipationRate结合交易数量评分，最低0.1
	participationRate := nfc.calculateParticipationRate()
	dyn.OnChainBehavior.Consensus.ParticipationRate = participationRate

	// 奖励基于参与度和交易量计算
	baseReward := 10.0
	dyn.OnChainBehavior.Consensus.TotalReward = baseReward * participationRate * float64(nfc.windowBlockCount+nfc.windowTxCount/10)

	// 第三个要求：SuccessRate应该比较高，基于共识活动
	successRate := nfc.calculateSuccessRate()
	dyn.OnChainBehavior.Consensus.SuccessRate = successRate

	// === 动态属性 ===
	dyn.DynamicAttributes.Compute.CPUUsage = nfc.simulateCPUUsage()
	dyn.DynamicAttributes.Compute.MemUsage = nfc.simulateMemUsage()
	// 第一个要求：使用优化的网络延迟波动计算
	dyn.DynamicAttributes.Network.LatencyFlux = nfc.calculateNetworkLatencyFluxOptimized()
	dyn.DynamicAttributes.Network.AvgLatency = nfc.calculateNetworkLatency()
	dyn.DynamicAttributes.Network.BandwidthUsage = nfc.simulateBandwidthUsage()

	// 第五个要求：使用优化的交易频率计算
	frequency := nfc.calculateTxFrequencyMetricOptimized()
	dyn.DynamicAttributes.Transactions.Frequency = frequency
	// 第二个要求：使用优化的处理延迟计算
	dyn.DynamicAttributes.Transactions.ProcessingDelay = nfc.calculateTransactionProcessingDelayOptimized()

	return dyn
}

// 第一个要求：根据节点负载和类型计算普通交易比例
func (nfc *NodeFeatureCollector) calculateNormalTxRatio() float64 {
	// 基础比例：70-90%为普通交易
	baseRatio := 0.8

	// 根据节点硬件配置调整
	cpuCores := nfc.staticData.Static.ResourceCapacity.Hardware.CPU.CoreCount
	if cpuCores >= 3 {
		// 高配置节点处理更多合约交易
		baseRatio = 0.65
	} else if cpuCores <= 1 {
		// 低配置节点主要处理普通交易
		baseRatio = 0.9
	}

	// 根据交易池大小微调
	if nfc.txPoolSize > 50 {
		baseRatio -= 0.1 // 高负载时合约交易增加
	} else if nfc.txPoolSize < 10 {
		baseRatio += 0.05 // 低负载时普通交易比例稍高
	}

	// 确保范围在0.5-0.95之间
	if baseRatio < 0.5 {
		baseRatio = 0.5
	} else if baseRatio > 0.95 {
		baseRatio = 0.95
	}

	return baseRatio
}

// 第二个要求：基于交易数量的参与率评分，最低0.1
func (nfc *NodeFeatureCollector) calculateParticipationRate() float64 {
	// 基础参与率
	baseRate := 0.1

	// 基于共识消息活动度
	totalMsgs := float64(nfc.windowPrepareCount + nfc.windowCommitCount)
	expectedMsgs := float64(nfc.windowBlockCount) * 2.0

	var consensusRate float64
	if expectedMsgs > 0 {
		consensusRate = totalMsgs / expectedMsgs
		if consensusRate > 1.0 {
			consensusRate = 1.0
		}
	} else {
		consensusRate = 0.3
	}

	// 基于交易处理量评分
	txScore := float64(nfc.windowTxCount) / 100.0 // 每100笔交易增加参与度
	if txScore > 0.4 {
		txScore = 0.4
	}

	// 基于节点配置评分
	cpuCores := nfc.staticData.Static.ResourceCapacity.Hardware.CPU.CoreCount
	configScore := float64(cpuCores) * 0.1
	if configScore > 0.3 {
		configScore = 0.3
	}

	// 综合评分
	finalRate := baseRate + consensusRate*0.4 + txScore + configScore
	if finalRate > 1.0 {
		finalRate = 1.0
	}

	return finalRate
}

// 第三个要求：高共识成功率计算
func (nfc *NodeFeatureCollector) calculateSuccessRate() float64 {
	// 基础成功率设为较高值
	baseRate := 0.85

	// 基于区块提交成功率
	if nfc.windowBlockCount > 0 {
		// 如果有区块提交，说明共识运行良好
		blockSuccessBonus := float64(nfc.windowBlockCount) * 0.02
		if blockSuccessBonus > 0.1 {
			blockSuccessBonus = 0.1
		}
		baseRate += blockSuccessBonus
	}

	// 基于消息处理效率
	totalMsgs := float64(nfc.windowPrepareCount + nfc.windowCommitCount)
	if totalMsgs > 0 {
		// 消息越多说明参与度越高，成功率也应该更高
		msgBonus := math.Min(totalMsgs/50.0, 0.1) // 最多增加10%
		baseRate += msgBonus
	}

	// 基于交易池状态（稳定的交易池表明系统运行良好）
	if nfc.txPoolSize > 0 && nfc.txPoolSize < 100 {
		baseRate += 0.02 // 适中的交易池大小
	}

	// 确保不超过1.0
	if baseRate > 0.98 {
		baseRate = 0.98
	}

	return baseRate
}

// 其余原有方法保持不变
// 更新共识统计
func (nfc *NodeFeatureCollector) UpdateConsensusStats(msgType message.MessageType) {
	nfc.stateLock.Lock()
	defer nfc.stateLock.Unlock()

	switch msgType {
	case message.CPrepare:
		nfc.windowPrepareCount++
	case message.CCommit:
		nfc.windowCommitCount++
	case message.CPrePrepare:
		// PrePrepare消息通常表示新区块提议
		nfc.windowBlockCount++
	}
}

// 记录区块提交
func (nfc *NodeFeatureCollector) RecordBlockCommit(block *core.Block) {
	nfc.stateLock.Lock()
	defer nfc.stateLock.Unlock()

	// 记录区块时间戳用于计算真实的区块间隔
	now := time.Now()
	nfc.blockTimestamps = append(nfc.blockTimestamps, now)

	// 统计区块中的交易并分类到不同节点
	if block != nil && len(block.Body) > 0 {
		for _, tx := range block.Body {
			nfc.analyzeAndRecordTransaction(tx)
		}
	}
}

// 分析交易并记录到相应的节点统计中
func (nfc *NodeFeatureCollector) analyzeAndRecordTransaction(tx *core.Transaction) {
	// 使用正确的字段名：Sender和Recipient
	fromShard := nfc.getShardByAddress([]byte(tx.Sender))
	toShard := nfc.getShardByAddress([]byte(tx.Recipient))

	// 判断是否为跨分片交易
	if fromShard != toShard {
		// 跨分片交易统计
		targetNodeKey := fmt.Sprintf("S%dN%d", toShard, nfc.getNodeByAddress([]byte(tx.Recipient), toShard))
		if _, exists := nfc.crossShardTxStats[targetNodeKey]; exists {
			nfc.crossShardTxStats[targetNodeKey]++
		}
	} else if fromShard == nfc.shardID {
		// 分片内不同节点间的交易统计
		if toShard == nfc.shardID {
			targetNodeID := nfc.getNodeByAddress([]byte(tx.Recipient), toShard)
			if targetNodeID != nfc.nodeID {
				targetNodeKey := fmt.Sprintf("S%dN%d", toShard, targetNodeID)
				if _, exists := nfc.interNodeTxStats[targetNodeKey]; exists {
					nfc.interNodeTxStats[targetNodeKey]++
				}
			}
		}
	}

	// 更新分片间和分片内交易统计
	if fromShard != toShard {
		// 跨分片交易，更新分片间统计
		if _, exists := nfc.shardTxStats[toShard]; exists {
			nfc.shardTxStats[toShard]++
		}
	} else {
		// 分片内交易，更新分片内交易计数
		nfc.intraShardTxCount++
	}
}

func (nfc *NodeFeatureCollector) getShardByAddress(addr []byte) uint64 {
	if len(addr) == 0 {
		return 0
	}
	// 使用地址的最后一个字节来确定分片
	return uint64(addr[len(addr)-1]) % nfc.shardNums
}

func (nfc *NodeFeatureCollector) getNodeByAddress(addr []byte, _ uint64) uint64 {
	if len(addr) == 0 {
		return 0
	}
	// 使用地址的倒数第二个字节来确定节点
	if len(addr) >= 2 {
		return uint64(addr[len(addr)-2]) % nfc.nodeNums
	}
	return 0
}

// 更新队列大小
func (nfc *NodeFeatureCollector) UpdatePoolSizes(txPoolSize, requestPoolSize int) {
	nfc.stateLock.Lock()
	defer nfc.stateLock.Unlock()
	nfc.txPoolSize = txPoolSize
	nfc.requestPoolSize = requestPoolSize
}

// 处理节点状态请求 - 简化接口，内部完成所有收集工作
func (nfc *NodeFeatureCollector) HandleRequestNodeState() {
	// 简洁接口：一次调用完成所有数据收集
	snap := nfc.collectCompleteNodeState()

	// 异步存储，不阻塞主流程
	go func() {
		nfc.stateLock.Lock()
		nfc.collectedStates = append(nfc.collectedStates, snap)
		nfc.stateLock.Unlock()
		fmt.Printf("S%dN%d: Node state collected and stored\n", nfc.shardID, nfc.nodeID)
	}()
}

// 内部工具函数：完整的节点状态收集
func (nfc *NodeFeatureCollector) collectCompleteNodeState() message.ReplyNodeStateMsg {
	now := time.Now()

	// 调用内部细小工具函数
	staticFeatures := nfc.collectStaticFeatures()
	dynamicFeatures := nfc.collectDynamicFeatures()

	snap := message.ReplyNodeStateMsg{
		ShardID:   nfc.shardID,
		NodeID:    nfc.nodeID,
		Timestamp: now.UnixMilli(),
		RequestID: fmt.Sprintf("req_%d", now.UnixNano()),
		NodeState: message.NodeState{
			Static:  staticFeatures,
			Dynamic: dynamicFeatures,
		},
	}

	// 重置窗口计数器
	nfc.resetWindowCounters(now)
	return snap
}

// 内部工具函数：收集静态特征
func (nfc *NodeFeatureCollector) collectStaticFeatures() message.StaticNodeFeatures {
	nfc.stateLock.RLock()
	defer nfc.stateLock.RUnlock()

	if !nfc.staticDataInitialized {
		nfc.initializeStaticData()
	}
	return nfc.staticData.Static
}

// 内部工具函数：重置计数器
func (nfc *NodeFeatureCollector) resetWindowCounters(now time.Time) {
	nfc.stateLock.Lock()
	defer nfc.stateLock.Unlock()

	nfc.windowTxCount = 0
	nfc.windowPrepareCount = 0
	nfc.windowCommitCount = 0
	nfc.windowBlockCount = 0
	nfc.windowStartTime = now
}

// 异步采集器goroutine
func (nfc *NodeFeatureCollector) runStateCollector() {
	for {
		select {
		case <-nfc.stateCollectCh:
			snap := nfc.fastCollectNodeState()
			nfc.stateLock.Lock()
			nfc.collectedStates = append(nfc.collectedStates, snap)
			nfc.stateLock.Unlock()
		case <-nfc.collectorStopCh:
			return
		}
	}
}

// 快速采集节点状态快照
func (nfc *NodeFeatureCollector) fastCollectNodeState() message.ReplyNodeStateMsg {
	nfc.stateLock.RLock()
	now := time.Now()
	dyn := nfc.collectDynamicFeatures()
	snap := message.ReplyNodeStateMsg{
		ShardID:   nfc.shardID,
		NodeID:    nfc.nodeID,
		Timestamp: now.UnixMilli(),
		RequestID: fmt.Sprintf("batch_%d", now.UnixNano()),
		NodeState: message.NodeState{
			Static:  nfc.staticData.Static,
			Dynamic: dyn,
		},
	}
	nfc.stateLock.RUnlock()
	// 采集后重置窗口计数器和起始时间
	nfc.stateLock.Lock()
	nfc.windowTxCount = 0
	nfc.windowPrepareCount = 0
	nfc.windowCommitCount = 0
	nfc.windowBlockCount = 0
	nfc.windowStartTime = now
	nfc.stateLock.Unlock()
	return snap
}

// 模拟CPU使用率
func (nfc *NodeFeatureCollector) simulateCPUUsage() float64 {
	// 获取节点的CPU核心数限制
	cpuLimit := nfc.staticData.Static.ResourceCapacity.Hardware.CPU.CoreCount
	if cpuLimit == 0 {
		cpuLimit = 1 // 避免除以零
	}

	baseUsage := 20.0 // 降低基础使用率，让负载的影响更明显

	// 基于交易队列大小调整，并考虑CPU性能
	// 核心思想：CPU越弱(limit越小)，处理相同数量的交易，模拟出的使用率越高
	queueFactor := (float64(nfc.txPoolSize) * 2.0) / float64(cpuLimit)
	if queueFactor > 50 {
		queueFactor = 50
	}

	// 基于共识活动调整，并考虑CPU性能
	consensusFactor := (float64(nfc.windowPrepareCount+nfc.windowCommitCount) * 1.5) / float64(cpuLimit)
	if consensusFactor > 25 {
		consensusFactor = 25
	}

	totalUsage := baseUsage + queueFactor + consensusFactor
	if totalUsage > 95 {
		totalUsage = 95
	}

	return totalUsage
}

// 模拟内存使用率
func (nfc *NodeFeatureCollector) simulateMemUsage() float64 {
	baseUsage := 25.0

	// 基于请求池大小调整
	poolFactor := float64(nfc.requestPoolSize) * 0.5
	if poolFactor > 30 {
		poolFactor = 30
	}

	// 基于区块链高度调整（这里需要从外部传入）
	heightFactor := 10.0 // 暂时使用固定值

	totalUsage := baseUsage + poolFactor + heightFactor
	if totalUsage > 90 {
		totalUsage = 90
	}

	return totalUsage
}

// 模拟带宽使用率
func (nfc *NodeFeatureCollector) simulateBandwidthUsage() float64 {
	// 基于交易处理量估算
	baseUsage := 0.3

	txFactor := float64(nfc.windowTxCount) * 0.01
	if txFactor > 0.4 {
		txFactor = 0.4
	}

	consensusFactor := float64(nfc.windowPrepareCount+nfc.windowCommitCount) * 0.005
	if consensusFactor > 0.2 {
		consensusFactor = 0.2
	}

	totalUsage := baseUsage + txFactor + consensusFactor
	if totalUsage > 0.95 {
		totalUsage = 0.95
	}

	return totalUsage
}

// 计算真实的区块间隔统计
func (nfc *NodeFeatureCollector) calculateRealBlockIntervals() (avgInterval float64, stdDev float64) {
	if len(nfc.blockTimestamps) < 2 {
		return 5.0, 0.5 // 默认值
	}

	// 计算所有区块间隔
	intervals := make([]float64, 0, len(nfc.blockTimestamps)-1)
	for i := 1; i < len(nfc.blockTimestamps); i++ {
		interval := nfc.blockTimestamps[i].Sub(nfc.blockTimestamps[i-1]).Seconds()
		intervals = append(intervals, interval)
	}

	// 计算平均值
	var sum float64
	for _, interval := range intervals {
		sum += interval
	}
	avgInterval = sum / float64(len(intervals))

	// 计算标准差
	var variance float64
	for _, interval := range intervals {
		variance += (interval - avgInterval) * (interval - avgInterval)
	}
	variance /= float64(len(intervals))
	stdDev = math.Sqrt(variance)

	return avgInterval, stdDev
}

// 生成分片间交易量统计字符串
func (nfc *NodeFeatureCollector) generateInterShardVolumeString() string {
	result := ""
	// 确保显示所有分片，按照shard0:1000;shard1:2000格式
	for sid := uint64(0); sid < nfc.shardNums; sid++ {
		if sid != nfc.shardID { // 只显示其他分片
			if result != "" {
				result += ";"
			}
			count := nfc.shardTxStats[sid] // 如果不存在，默认为0
			result += fmt.Sprintf("shard%d:%d", sid, count)
		}
	}
	return result
}

// 生成节点间交易统计字符串
func (nfc *NodeFeatureCollector) generateInterNodeVolumeString() string {
	result := ""
	// 确保显示所有分片，按照S{shardID}N{nodeID}格式
	for sid := uint64(0); sid < nfc.shardNums; sid++ {
		if sid != nfc.shardID { // 只显示其他分片
			for nid := uint64(0); nid < nfc.nodeNums; nid++ {
				if result != "" {
					result += ";"
				}
				nodeKey := fmt.Sprintf("S%dN%d", sid, nid)
				count := nfc.crossShardTxStats[nodeKey] // 如果不存在，默认为0
				result += fmt.Sprintf("%s:%d", nodeKey, count)
			}
		}
	}
	return result
}

// 计算处理延迟
func (nfc *NodeFeatureCollector) calculateProcessingDelay() string {
	if nfc.txPoolSize == 0 {
		return "0ms"
	}
	// 基于队列大小估算延迟
	estimatedDelay := nfc.txPoolSize * 10 // 每笔交易10ms
	if estimatedDelay > 1000 {
		return "1000ms"
	}
	return fmt.Sprintf("%dms", estimatedDelay)
}

// 计算网络延迟
func (nfc *NodeFeatureCollector) calculateNetworkLatency() string {
	// 基于分片内连接数估算
	baseLatency := 50
	shardFactor := int(nfc.shardID) * 10
	totalLatency := baseLatency + shardFactor

	if totalLatency > 200 {
		totalLatency = 200
	}

	return fmt.Sprintf("%dms", totalLatency)
}

// 第一个要求：优化网络延迟波动计算
func (nfc *NodeFeatureCollector) calculateNetworkLatencyFluxOptimized() float64 {
	// 更新延迟历史记录
	nfc.updateLatencyHistory()

	// 如果历史记录不足，返回基础波动值
	if len(nfc.latencyHistory) < 3 {
		// 基于节点活动度的基础波动
		baseFlux := 0.1 + float64(nfc.windowTxCount)*0.001
		if baseFlux > 0.5 {
			baseFlux = 0.5
		}
		return baseFlux
	}

	// 计算延迟变化的标准差作为波动率
	var sum, mean, variance float64
	for _, latency := range nfc.latencyHistory {
		sum += latency
	}
	mean = sum / float64(len(nfc.latencyHistory))

	for _, latency := range nfc.latencyHistory {
		variance += (latency - mean) * (latency - mean)
	}
	variance /= float64(len(nfc.latencyHistory))
	stdDev := math.Sqrt(variance)

	// 将标准差转换为0-1范围的波动率
	flux := stdDev / 100.0 // 假设延迟在0-100ms范围内
	if flux > 1.0 {
		flux = 1.0
	}

	// 基于网络活动调整
	networkActivity := float64(nfc.windowPrepareCount + nfc.windowCommitCount)
	activityBonus := networkActivity * 0.01
	if activityBonus > 0.3 {
		activityBonus = 0.3
	}

	finalFlux := flux + activityBonus
	if finalFlux > 1.0 {
		finalFlux = 1.0
	}

	return finalFlux
}

// 更新延迟历史记录
func (nfc *NodeFeatureCollector) updateLatencyHistory() {
	// 基于当前网络状况计算延迟
	baseLatency := 50.0 + float64(nfc.shardID)*10.0

	// 基于交易池大小调整延迟
	queueLatency := float64(nfc.txPoolSize) * 2.0
	if queueLatency > 50.0 {
		queueLatency = 50.0
	}

	// 基于共识活动调整延迟
	consensusLatency := float64(nfc.windowPrepareCount+nfc.windowCommitCount) * 1.0
	if consensusLatency > 30.0 {
		consensusLatency = 30.0
	}

	currentLatency := baseLatency + queueLatency + consensusLatency

	// 添加到历史记录，保持固定长度
	nfc.latencyHistory = append(nfc.latencyHistory, currentLatency)
	if len(nfc.latencyHistory) > 50 {
		nfc.latencyHistory = nfc.latencyHistory[1:]
	}
}

// 第二个要求：基于区块处理时间计算交易处理延迟
func (nfc *NodeFeatureCollector) calculateTransactionProcessingDelayOptimized() string {
	// 如果没有区块处理记录，使用队列估算
	if len(nfc.blockProcessingTimes) == 0 || len(nfc.txCountInBlocks) == 0 {
		if nfc.txPoolSize == 0 {
			return "0ms"
		}
		// 基于队列大小和硬件配置估算
		cpuCores := nfc.staticData.Static.ResourceCapacity.Hardware.CPU.CoreCount
		baseDelay := float64(nfc.txPoolSize) * 10.0 / float64(cpuCores) // CPU核数影响处理能力
		if baseDelay > 1000 {
			baseDelay = 1000
		}
		return fmt.Sprintf("%.0fms", baseDelay)
	}

	// 计算平均每笔交易的处理时间
	totalProcessingTime := time.Duration(0)
	totalTxCount := 0

	// 使用最近的区块处理数据
	recentBlocks := 5 // 使用最近5个区块的数据
	startIdx := len(nfc.blockProcessingTimes) - recentBlocks
	if startIdx < 0 {
		startIdx = 0
	}

	for i := startIdx; i < len(nfc.blockProcessingTimes); i++ {
		totalProcessingTime += nfc.blockProcessingTimes[i]
		totalTxCount += nfc.txCountInBlocks[i]
	}

	if totalTxCount == 0 {
		return "0ms"
	}

	// 平均每笔交易处理时间
	avgTxProcessingTime := totalProcessingTime / time.Duration(totalTxCount)

	// 考虑当前队列等待时间
	queueWaitTime := time.Duration(nfc.txPoolSize) * avgTxProcessingTime / 10

	totalDelay := avgTxProcessingTime + queueWaitTime

	// 限制最大延迟
	if totalDelay > 2*time.Second {
		totalDelay = 2 * time.Second
	}

	return fmt.Sprintf("%.0fms", float64(totalDelay.Nanoseconds())/1e6)
}

// 记录区块处理时间（需要在区块处理开始和结束时调用）
func (nfc *NodeFeatureCollector) RecordBlockProcessingStart() time.Time {
	return time.Now()
}

func (nfc *NodeFeatureCollector) RecordBlockProcessingEnd(startTime time.Time, txCount int) {
	nfc.stateLock.Lock()
	defer nfc.stateLock.Unlock()

	processingTime := time.Since(startTime)

	// 记录处理时间和交易数量
	nfc.blockProcessingTimes = append(nfc.blockProcessingTimes, processingTime)
	nfc.txCountInBlocks = append(nfc.txCountInBlocks, txCount)

	// 保持固定长度的历史记录
	if len(nfc.blockProcessingTimes) > 50 {
		nfc.blockProcessingTimes = nfc.blockProcessingTimes[1:]
		nfc.txCountInBlocks = nfc.txCountInBlocks[1:]
	}
}

// 第三个要求：FunctionTags选择逻辑优化
func (nfc *NodeFeatureCollector) determineFunctionTagsOptimized(nodeType string) string {
	cpuCores := nfc.staticData.Static.ResourceCapacity.Hardware.CPU.CoreCount
	memoryGB := nfc.staticData.Static.ResourceCapacity.Hardware.Memory.TotalCapacity

	var tags []string

	// consensus：基于节点配置和参与度判断
	if cpuCores >= 2 && memoryGB >= 2 {
		// 高配置节点支持共识
		tags = append(tags, "consensus")
	} else if nfc.windowPrepareCount+nfc.windowCommitCount > 5 {
		// 即使配置较低，但共识参与度高的节点也支持共识
		tags = append(tags, "consensus")
	}

	// validation：基于交易处理能力判断
	if cpuCores >= 1 && memoryGB >= 1 {
		// 基础配置即可支持验证
		tags = append(tags, "validation")
	} else if nfc.windowTxCount > 10 {
		// 交易处理活跃的节点也支持验证
		tags = append(tags, "validation")
	}

	// 特殊节点类型的额外标签
	switch nodeType {
	case "miner_node":
		if !contains(tags, "consensus") {
			tags = append(tags, "consensus")
		}
		tags = append(tags, "mining")
	case "storage_node":
		tags = append(tags, "storage", "data_persistence")
	case "full_node":
		if !contains(tags, "consensus") {
			tags = append(tags, "consensus")
		}
		if !contains(tags, "validation") {
			tags = append(tags, "validation")
		}
		tags = append(tags, "full_service")
	}

	// 如果没有任何标签，返回基础标签
	if len(tags) == 0 {
		return "basic"
	}

	// 用逗号连接多个标签
	result := ""
	for i, tag := range tags {
		if i > 0 {
			result += ","
		}
		result += tag
	}
	return result
}

// 辅助函数：检查切片是否包含指定元素
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// SupportedFuncs.Functions选择逻辑优化 - 基于硬件信息筛选 717
func (nfc *NodeFeatureCollector) determineSupportedFunctionsOptimized(cpuCores, memoryGB int, networkBW float64) string {
	var functions []string

	// tx_processing：基本交易处理能力要求
	// 条件：CPU>=1核 && 内存>=1GB
	if cpuCores >= 1 && memoryGB >= 1 {
		functions = append(functions, "tx_processing")
	}

	// data_verification：数据验证能力要求
	// 条件：CPU>=2核 && 内存>=2GB && 网络带宽>=50Mbps
	if cpuCores >= 2 && memoryGB >= 2 && networkBW >= 50.0 {
		functions = append(functions, "data_verification")
	}

	// 如果没有满足任何功能要求，至少返回tx_processing（最基本功能）
	if len(functions) == 0 {
		return "tx_processing"
	}

	// 用逗号连接多个功能，可能的结果：
	// - "tx_processing" (仅满足基本要求)
	// - "tx_processing,data_verification" (满足两个要求)
	result := ""
	for i, fn := range functions {
		if i > 0 {
			result += ","
		}
		result += fn
	}
	return result
}

// 第五个要求：基于硬件指标和运行数据生成TxFrequency
func (nfc *NodeFeatureCollector) calculateTxFrequencyMetricOptimized() int {
	cpuCores := nfc.staticData.Static.ResourceCapacity.Hardware.CPU.CoreCount
	memoryGB := nfc.staticData.Static.ResourceCapacity.Hardware.Memory.TotalCapacity
	networkBW := nfc.staticData.Static.ResourceCapacity.Hardware.Network.UpstreamBW

	// 硬件基础频率计算
	cpuScore := cpuCores * 50      // 每核心50笔/分钟
	memScore := memoryGB * 15      // 每GB内存15笔/分钟
	networkScore := int(networkBW) // 网络带宽直接影响

	hardwareFrequency := cpuScore + memScore + networkScore

	// 基于实际运行数据调整
	windowDuration := time.Since(nfc.windowStartTime).Seconds()
	if windowDuration <= 0 {
		windowDuration = 1
	}

	// 当前实际TPS转换为每分钟频率
	actualFrequency := int(float64(nfc.windowTxCount) / windowDuration * 60)

	// 基于交易池状态调整
	poolFactor := 1.0
	if nfc.txPoolSize > 50 {
		poolFactor = 1.2 // 高负载时频率提升
	} else if nfc.txPoolSize < 5 {
		poolFactor = 0.8 // 低负载时频率降低
	}

	// 基于共识参与度调整
	consensusFactor := 1.0
	if nfc.windowPrepareCount+nfc.windowCommitCount > 20 {
		consensusFactor = 1.15 // 高共识参与度提升频率
	} else if nfc.windowPrepareCount+nfc.windowCommitCount < 5 {
		consensusFactor = 0.9 // 低共识参与度降低频率
	}

	// 综合计算最终频率
	finalFrequency := int(float64(hardwareFrequency*2+actualFrequency) / 3.0 * poolFactor * consensusFactor)

	// 确保合理范围
	if finalFrequency < 20 {
		finalFrequency = 20
	} else if finalFrequency > 1000 {
		finalFrequency = 1000
	}

	return finalFrequency
}

// 第五个要求：基于硬件指标和运行数据生成StorageOps
func (nfc *NodeFeatureCollector) calculateStorageOpsMetricOptimized() int {
	storageGB := nfc.staticData.Static.ResourceCapacity.Hardware.Storage.Capacity
	readWriteSpeed := nfc.staticData.Static.ResourceCapacity.Hardware.Storage.ReadWriteSpeed
	cpuCores := nfc.staticData.Static.ResourceCapacity.Hardware.CPU.CoreCount

	// 硬件基础操作数计算
	storageScore := int(float64(storageGB) * 0.2) // 存储容量影响
	speedScore := int(readWriteSpeed * 2.0)       // 读写速度直接影响
	cpuScore := cpuCores * 30                     // CPU影响存储操作处理

	hardwareOps := storageScore + speedScore + cpuScore

	// 基于实际区块活动调整
	blockActivity := nfc.windowBlockCount * 50 // 每个区块增加50个存储操作

	// 基于交易量调整（交易需要存储）
	txActivity := nfc.windowTxCount * 2 // 每笔交易2个存储操作

	// 基于节点类型调整
	nodeTypeFactor := 1.0
	nodeType := nfc.staticData.Static.HeterogeneousType.NodeType
	switch nodeType {
	case "storage_node":
		nodeTypeFactor = 1.5 // 存储节点存储操作更多
	case "full_node":
		nodeTypeFactor = 1.3 // 全节点存储操作较多
	case "light_node":
		nodeTypeFactor = 0.7 // 轻量节点存储操作较少
	}

	// 综合计算最终存储操作数
	finalOps := int(float64(hardwareOps+blockActivity+txActivity) * nodeTypeFactor)

	// 确保合理范围
	if finalOps < 50 {
		finalOps = 50
	} else if finalOps > 2000 {
		finalOps = 2000
	}

	return finalOps
}

// 获取真实CPU使用率（从cgroup读取）
func (nfc *NodeFeatureCollector) getRealCPUUsage() float64 {
	// 尝试读取cgroup v2的CPU使用统计
	if data, err := os.ReadFile("/sys/fs/cgroup/cpu.stat"); err == nil {
		lines := strings.Split(string(data), "\n")
		for _, line := range lines {
			if strings.HasPrefix(line, "usage_usec ") {
				parts := strings.Fields(line)
				if len(parts) >= 2 {
					usageUsec, _ := strconv.ParseInt(parts[1], 10, 64)
					// 计算CPU使用率需要时间间隔，这里简化处理
					// 实际实现需要维护上次采样时间和使用量
					return float64(usageUsec) / 1000000.0 // 转换为秒，这里简化
				}
			}
		}
	}

	// 如果无法获取真实数据，回退到模拟方式
	return nfc.simulateCPUUsage()
}

// 获取真实内存使用率（从cgroup读取）
func (nfc *NodeFeatureCollector) getRealMemUsage() float64 {
	// 获取当前内存使用量
	var currentUsage int64
	if data, err := os.ReadFile("/sys/fs/cgroup/memory.current"); err == nil {
		currentUsage, _ = strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
	} else if data, err := os.ReadFile("/sys/fs/cgroup/memory/memory.usage_in_bytes"); err == nil {
		currentUsage, _ = strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
	}

	// 获取内存限制
	var memLimit int64
	if data, err := os.ReadFile("/sys/fs/cgroup/memory.max"); err == nil {
		val := strings.TrimSpace(string(data))
		if val != "max" {
			memLimit, _ = strconv.ParseInt(val, 10, 64)
		}
	} else if data, err := os.ReadFile("/sys/fs/cgroup/memory/memory.limit_in_bytes"); err == nil {
		memLimit, _ = strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
	}

	if memLimit > 0 && currentUsage > 0 {
		usage := float64(currentUsage) / float64(memLimit) * 100.0
		if usage > 100.0 {
			usage = 100.0
		}
		return usage
	}

	// 如果无法获取真实数据，回退到模拟方式
	return nfc.simulateMemUsage()
}

// 基于真实交易数据计算TxFrequency
func (nfc *NodeFeatureCollector) calculateRealTxFrequency() int {
	windowDuration := time.Since(nfc.windowStartTime).Seconds()
	if windowDuration <= 0 {
		windowDuration = 1
	}

	// 计算真实的每分钟交易处理频率
	realFrequency := int(float64(nfc.windowTxCount) / windowDuration * 60)

	// 考虑历史处理能力的移动平均
	if len(nfc.txProcessingHistory) > 0 {
		var historySum int
		for _, freq := range nfc.txProcessingHistory {
			historySum += freq
		}
		avgHistory := historySum / len(nfc.txProcessingHistory)

		// 真实频率与历史平均的加权平均
		realFrequency = (realFrequency*3 + avgHistory*2) / 5
	}

	// 根据当前负载状态调整
	if nfc.txPoolSize > 50 {
		// 高负载时可能处理能力下降
		realFrequency = int(float64(realFrequency) * 0.9)
	} else if nfc.txPoolSize == 0 && nfc.windowTxCount == 0 {
		// 空闲时使用最小基础值
		realFrequency = 10
	}

	// 确保合理范围
	if realFrequency < 5 {
		realFrequency = 5
	} else if realFrequency > 2000 {
		realFrequency = 2000
	}

	// 更新历史记录
	nfc.updateTxProcessingHistory(realFrequency)

	return realFrequency
}

// 更新交易处理历史记录
func (nfc *NodeFeatureCollector) updateTxProcessingHistory(frequency int) {
	if nfc.txProcessingHistory == nil {
		nfc.txProcessingHistory = make([]int, 0, 20)
	}

	nfc.txProcessingHistory = append(nfc.txProcessingHistory, frequency)

	// 保持最近20个记录
	if len(nfc.txProcessingHistory) > 20 {
		nfc.txProcessingHistory = nfc.txProcessingHistory[1:]
	}
}

// 停止收集器
func (nfc *NodeFeatureCollector) StopCollector() {
	close(nfc.collectorStopCh)
}

// 获取收集到的状态
func (nfc *NodeFeatureCollector) GetCollectedStates() []message.ReplyNodeStateMsg {
	nfc.stateLock.RLock()
	defer nfc.stateLock.RUnlock()
	return nfc.collectedStates
}
