package measure

import (
	"blockEmulator/message"
	"blockEmulator/params"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"sync"
	"time"
)

// 节点特征收集模块 - 简化统一版本
type NodeFeaturesModule struct {
	mu sync.RWMutex

	// 统一的数据存储 - 按时间戳索引
	collectedData map[int64][]message.ReplyNodeStateMsg // timestamp -> data

	// 收集模式标识
	collectMode string // "epoch" 或 "final"

	// 全局统计指标（仅用于补充，优先使用节点本地数据）
	globalMetrics map[string]string

	// 新增：按epoch收集的数据
	epochData map[int][]message.ReplyNodeStateMsg // epoch -> data

	// 新增：收集控制
	epochCollectionEnabled bool
}

func NewNodeFeaturesModule() *NodeFeaturesModule {
	return &NodeFeaturesModule{
		collectedData:          make(map[int64][]message.ReplyNodeStateMsg),
		collectMode:            "final", // 默认终态收集
		epochData:              make(map[int][]message.ReplyNodeStateMsg),
		epochCollectionEnabled: true, // 默认启用epoch收集
	}
}

func (nfm *NodeFeaturesModule) OutputMetricName() string {
	return "Node_Features"
}

// 设置收集模式
func (nfm *NodeFeaturesModule) SetCollectMode(mode string) {
	nfm.mu.Lock()
	defer nfm.mu.Unlock()
	nfm.collectMode = mode
}

// === 新增：设置全局统计指标 ===
func (nfm *NodeFeaturesModule) SetGlobalMetrics(metrics map[string]string) {
	nfm.mu.Lock()
	defer nfm.mu.Unlock()
	nfm.globalMetrics = metrics
}

// 简化的UpdateMeasureRecord - 仅用于epoch模式的触发检测
func (nfm *NodeFeaturesModule) UpdateMeasureRecord(b *message.BlockInfoMsg) {
	if nfm.collectMode != "epoch" {
		return // 非epoch模式不处理
	}

	// epoch模式：检测epoch变化时触发收集
	// 具体的收集触发由Supervisor统一管理
}

// 统一的数据接收处理
func (nfm *NodeFeaturesModule) HandleExtraMessage(msg []byte) {
	msgType, content := message.SplitMessage(msg)
	timestamp := time.Now().UnixMilli()

	switch msgType {
	case message.CReplyNodeState:
		var replyMsg message.ReplyNodeStateMsg
		if err := json.Unmarshal(content, &replyMsg); err == nil {
			nfm.storeNodeData(timestamp, replyMsg)
		}
	case message.CBatchReplyNodeState:
		var batch message.BatchReplyNodeStateMsg
		if err := json.Unmarshal(content, &batch); err == nil {
			nfm.storeNodeDataBatch(timestamp, batch.States)
		}
	}
}

// 内部统一存储方法
func (nfm *NodeFeaturesModule) storeNodeData(timestamp int64, data message.ReplyNodeStateMsg) {
	nfm.mu.Lock()
	defer nfm.mu.Unlock()

	if nfm.collectedData[timestamp] == nil {
		nfm.collectedData[timestamp] = make([]message.ReplyNodeStateMsg, 0)
	}
	nfm.collectedData[timestamp] = append(nfm.collectedData[timestamp], data)
}

func (nfm *NodeFeaturesModule) storeNodeDataBatch(timestamp int64, data []message.ReplyNodeStateMsg) {
	nfm.mu.Lock()
	defer nfm.mu.Unlock()

	if nfm.collectedData[timestamp] == nil {
		nfm.collectedData[timestamp] = make([]message.ReplyNodeStateMsg, 0)
	}
	nfm.collectedData[timestamp] = append(nfm.collectedData[timestamp], data...)
}

// 统一的输出方法 - 根据模式决定文件名
func (nfm *NodeFeaturesModule) OutputRecord() ([]float64, float64) {
	nfm.mu.RLock()
	defer nfm.mu.RUnlock()

	// 获取所有数据进行输出
	allData := nfm.getAllCollectedData()

	// 根据收集模式生成不同的文件名
	var fileName string
	if nfm.collectMode == "epoch" {
		fileName = "node_features_by_epoch.csv"
	} else {
		fileName = "node_features_final.csv"
	}

	nfm.writeToCSV(fileName, allData)

	// 返回统计信息
	totalNodes := len(allData)
	if totalNodes == 0 {
		return []float64{0}, 0
	}

	// 计算平均特征值（简化）
	avgFeatures := nfm.calculateAverageFeatures(allData)
	return avgFeatures, float64(totalNodes)
}

// 获取所有收集的数据
func (nfm *NodeFeaturesModule) getAllCollectedData() []message.ReplyNodeStateMsg {
	var allData []message.ReplyNodeStateMsg
	for _, dataList := range nfm.collectedData {
		allData = append(allData, dataList...)
	}
	return allData
}

// 写入CSV文件 - 使用节点实际上报的数据
func (nfm *NodeFeaturesModule) writeToCSV(fileName string, data []message.ReplyNodeStateMsg) {
	filePath := fmt.Sprintf("%s/%s", params.ExpDataRootDir, fileName)

	// 确保目录存在
	os.MkdirAll(params.ExpDataRootDir, 0755)

	file, err := os.Create(filePath)
	if err != nil {
		fmt.Printf("Error creating CSV file: %v\n", err)
		return
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// wrs.md字段顺序 - 添加NodeID作为第一列，移除指定字段
	headers := []string{
		"NodeID",
		"ResourceCapacity.Hardware.CPU.CoreCount",
		"ResourceCapacity.Hardware.CPU.Architecture",
		"ResourceCapacity.Hardware.Memory.TotalCapacity",
		"ResourceCapacity.Hardware.Memory.Type",
		"ResourceCapacity.Hardware.Memory.Bandwidth",
		"ResourceCapacity.Hardware.Storage.Capacity",
		"ResourceCapacity.Hardware.Storage.Type",
		"ResourceCapacity.Hardware.Storage.ReadWriteSpeed",
		"ResourceCapacity.Hardware.Network.UpstreamBW",
		"ResourceCapacity.Hardware.Network.DownstreamBW",
		"ResourceCapacity.Hardware.Network.Latency",
		"ResourceCapacity.OperationalStatus.CoreEligibility",
		"ResourceCapacity.OperationalStatus.ResourceUsage.CPUUtilization",
		"ResourceCapacity.OperationalStatus.ResourceUsage.MemUtilization",
		"OnChainBehavior.TransactionCapability.AvgTPS",
		"OnChainBehavior.TransactionCapability.CrossShardTx.InterNodeVolume",
		"OnChainBehavior.TransactionCapability.CrossShardTx.InterShardVolume",
		"OnChainBehavior.TransactionCapability.ConfirmationDelay",
		"OnChainBehavior.BlockGeneration.AvgInterval",
		"OnChainBehavior.BlockGeneration.IntervalStdDev",
		"OnChainBehavior.TransactionTypes.NormalTxRatio",
		"OnChainBehavior.TransactionTypes.ContractTxRatio",
		"OnChainBehavior.Consensus.ParticipationRate",
		"OnChainBehavior.Consensus.TotalReward",
		"OnChainBehavior.Consensus.SuccessRate",
		"NetworkTopology.GeoLocation.Timezone",
		"NetworkTopology.Connections.IntraShardConn",
		"NetworkTopology.Connections.InterShardConn",
		"NetworkTopology.Connections.WeightedDegree",
		"NetworkTopology.Connections.ActiveConn",
		// 移除 "NetworkTopology.ShardAllocation.Priority",
		"NetworkTopology.ShardAllocation.Adaptability",
		// 移除重复的 "DynamicAttributes.Compute.CPUUsage",
		// 移除重复的 "DynamicAttributes.Compute.MemUsage",
		"DynamicAttributes.Network.LatencyFlux",
		"DynamicAttributes.Network.AvgLatency",
		"DynamicAttributes.Network.BandwidthUsage",
		"DynamicAttributes.Transactions.Frequency",
		"DynamicAttributes.Transactions.ProcessingDelay",
		"HeterogeneousType.NodeType",
		"HeterogeneousType.FunctionTags",
		"HeterogeneousType.SupportedFuncs.Functions",
		// 移除 "HeterogeneousType.SupportedFuncs.Priorities",
		"HeterogeneousType.Application.CurrentState",
		"HeterogeneousType.Application.LoadMetrics.TxFrequency",
		"HeterogeneousType.Application.LoadMetrics.StorageOps",
	}

	if err := writer.Write(headers); err != nil {
		fmt.Printf("Error writing headers: %v\n", err)
		return
	}

	// 写入每个节点的数据
	for _, state := range data {
		row := make([]string, len(headers))
		for i, field := range headers {
			// 优先使用全局注入值（仅限于实际存在的全局指标）
			if nfm.globalMetrics != nil {
				if v, ok := nfm.globalMetrics[field]; ok {
					row[i] = v
					continue
				}
			}

			// 使用节点实际上报的数据
			switch field {
			// === NodeID ===
			case "NodeID":
				row[i] = fmt.Sprintf("S%dN%d", state.ShardID, state.NodeID)

				// === 静态硬件资源 ===
			case "ResourceCapacity.Hardware.CPU.CoreCount":
				row[i] = strconv.Itoa(state.NodeState.Static.ResourceCapacity.Hardware.CPU.CoreCount)
			case "ResourceCapacity.Hardware.CPU.Architecture":
				row[i] = state.NodeState.Static.ResourceCapacity.Hardware.CPU.Architecture
			case "ResourceCapacity.Hardware.Memory.TotalCapacity":
				row[i] = strconv.Itoa(state.NodeState.Static.ResourceCapacity.Hardware.Memory.TotalCapacity)
			case "ResourceCapacity.Hardware.Memory.Type":
				row[i] = state.NodeState.Static.ResourceCapacity.Hardware.Memory.Type
			case "ResourceCapacity.Hardware.Memory.Bandwidth":
				row[i] = strconv.FormatFloat(state.NodeState.Static.ResourceCapacity.Hardware.Memory.Bandwidth, 'f', 2, 64)
			case "ResourceCapacity.Hardware.Storage.Capacity":
				row[i] = strconv.Itoa(state.NodeState.Static.ResourceCapacity.Hardware.Storage.Capacity)
			case "ResourceCapacity.Hardware.Storage.Type":
				row[i] = state.NodeState.Static.ResourceCapacity.Hardware.Storage.Type
			case "ResourceCapacity.Hardware.Storage.ReadWriteSpeed":
				row[i] = strconv.FormatFloat(state.NodeState.Static.ResourceCapacity.Hardware.Storage.ReadWriteSpeed, 'f', 2, 64)
			case "ResourceCapacity.Hardware.Network.UpstreamBW":
				row[i] = strconv.FormatFloat(state.NodeState.Static.ResourceCapacity.Hardware.Network.UpstreamBW, 'f', 2, 64)
			case "ResourceCapacity.Hardware.Network.DownstreamBW":
				row[i] = strconv.FormatFloat(state.NodeState.Static.ResourceCapacity.Hardware.Network.DownstreamBW, 'f', 2, 64)
			case "ResourceCapacity.Hardware.Network.Latency":
				row[i] = state.NodeState.Static.ResourceCapacity.Hardware.Network.Latency

				// === 操作状态 ===
			case "ResourceCapacity.OperationalStatus.CoreEligibility":
				// 修改逻辑：分片0号节点或CPU核心数>=2的节点都为true（降低门槛）
				if state.NodeState.Static.ResourceCapacity.Hardware.CPU.CoreCount >= 2 || (state.NodeID == 0 && state.ShardID == 0) {
					row[i] = "true"
				} else {
					row[i] = "false"
				}
			case "ResourceCapacity.OperationalStatus.ResourceUsage.CPUUtilization":
				// 这里使用动态数据中的CPU使用率
				row[i] = strconv.FormatFloat(state.NodeState.Dynamic.DynamicAttributes.Compute.CPUUsage, 'f', 2, 64)
			case "ResourceCapacity.OperationalStatus.ResourceUsage.MemUtilization":
				// 这里使用动态数据中的内存使用率
				row[i] = strconv.FormatFloat(state.NodeState.Dynamic.DynamicAttributes.Compute.MemUsage, 'f', 2, 64)

				// === 链上行为 - 交易能力 ===
			case "OnChainBehavior.TransactionCapability.AvgTPS":
				row[i] = strconv.FormatFloat(state.NodeState.Dynamic.OnChainBehavior.TransactionCapability.AvgTPS, 'f', 2, 64)
			case "OnChainBehavior.TransactionCapability.CrossShardTx.InterNodeVolume":
				row[i] = state.NodeState.Dynamic.OnChainBehavior.TransactionCapability.CrossShardTx.InterNodeVolume
			case "OnChainBehavior.TransactionCapability.CrossShardTx.InterShardVolume":
				row[i] = state.NodeState.Dynamic.OnChainBehavior.TransactionCapability.CrossShardTx.InterShardVolume
			case "OnChainBehavior.TransactionCapability.ConfirmationDelay":
				row[i] = state.NodeState.Dynamic.OnChainBehavior.TransactionCapability.ConfirmationDelay

				// === 链上行为 - 区块生成 ===
			case "OnChainBehavior.BlockGeneration.AvgInterval":
				row[i] = state.NodeState.Dynamic.OnChainBehavior.BlockGeneration.AvgInterval
			case "OnChainBehavior.BlockGeneration.IntervalStdDev":
				row[i] = state.NodeState.Dynamic.OnChainBehavior.BlockGeneration.IntervalStdDev

				// === 链上行为 - 交易类型 ===
			case "OnChainBehavior.TransactionTypes.NormalTxRatio":
				row[i] = strconv.FormatFloat(state.NodeState.Dynamic.OnChainBehavior.TransactionTypes.NormalTxRatio, 'f', 3, 64)
			case "OnChainBehavior.TransactionTypes.ContractTxRatio":
				row[i] = strconv.FormatFloat(state.NodeState.Dynamic.OnChainBehavior.TransactionTypes.ContractTxRatio, 'f', 3, 64)

				// === 链上行为 - 共识参与 ===
			case "OnChainBehavior.Consensus.ParticipationRate":
				row[i] = strconv.FormatFloat(state.NodeState.Dynamic.OnChainBehavior.Consensus.ParticipationRate, 'f', 2, 64)
			case "OnChainBehavior.Consensus.TotalReward":
				row[i] = strconv.FormatFloat(state.NodeState.Dynamic.OnChainBehavior.Consensus.TotalReward, 'f', 2, 64)
			case "OnChainBehavior.Consensus.SuccessRate":
				row[i] = strconv.FormatFloat(state.NodeState.Dynamic.OnChainBehavior.Consensus.SuccessRate, 'f', 2, 64)

				// === 网络拓扑 ===
			case "NetworkTopology.GeoLocation.Timezone":
				row[i] = state.NodeState.Static.NetworkTopology.GeoLocation.Timezone
			case "NetworkTopology.Connections.IntraShardConn":
				row[i] = strconv.Itoa(state.NodeState.Static.NetworkTopology.Connections.IntraShardConn)
			case "NetworkTopology.Connections.InterShardConn":
				row[i] = strconv.Itoa(state.NodeState.Static.NetworkTopology.Connections.InterShardConn)
			case "NetworkTopology.Connections.WeightedDegree":
				row[i] = strconv.FormatFloat(state.NodeState.Static.NetworkTopology.Connections.WeightedDegree, 'f', 2, 64)
			case "NetworkTopology.Connections.ActiveConn":
				row[i] = strconv.Itoa(state.NodeState.Static.NetworkTopology.Connections.ActiveConn)
			case "NetworkTopology.ShardAllocation.Adaptability":
				row[i] = strconv.FormatFloat(state.NodeState.Static.NetworkTopology.ShardAllocation.Adaptability, 'f', 3, 64)

				// === 动态属性 - 移除了重复的CPUUsage和MemUsage ===
			case "DynamicAttributes.Network.LatencyFlux":
				row[i] = strconv.FormatFloat(state.NodeState.Dynamic.DynamicAttributes.Network.LatencyFlux, 'f', 2, 64)
			case "DynamicAttributes.Network.AvgLatency":
				row[i] = state.NodeState.Dynamic.DynamicAttributes.Network.AvgLatency
			case "DynamicAttributes.Network.BandwidthUsage":
				row[i] = strconv.FormatFloat(state.NodeState.Dynamic.DynamicAttributes.Network.BandwidthUsage, 'f', 3, 64)
			case "DynamicAttributes.Transactions.Frequency":
				row[i] = strconv.Itoa(state.NodeState.Dynamic.DynamicAttributes.Transactions.Frequency)
			case "DynamicAttributes.Transactions.ProcessingDelay":
				row[i] = state.NodeState.Dynamic.DynamicAttributes.Transactions.ProcessingDelay

				// === 异构类型 ===
			case "HeterogeneousType.NodeType":
				// 直接使用节点端上报的数据，移除重复的判断逻辑
				row[i] = state.NodeState.Static.HeterogeneousType.NodeType
			case "HeterogeneousType.FunctionTags":
				row[i] = state.NodeState.Static.HeterogeneousType.FunctionTags
			case "HeterogeneousType.SupportedFuncs.Functions":
				// 直接使用节点端上报的数据，移除重复的判断逻辑
				row[i] = state.NodeState.Static.HeterogeneousType.SupportedFuncs.Functions
			case "HeterogeneousType.Application.CurrentState":
				row[i] = state.NodeState.Static.HeterogeneousType.Application.CurrentState
			case "HeterogeneousType.Application.LoadMetrics.TxFrequency":
				row[i] = strconv.Itoa(state.NodeState.Static.HeterogeneousType.Application.LoadMetrics.TxFrequency)
			case "HeterogeneousType.Application.LoadMetrics.StorageOps":
				row[i] = strconv.Itoa(state.NodeState.Static.HeterogeneousType.Application.LoadMetrics.StorageOps)

			// === 未匹配字段 ===
			default:
				row[i] = ""
			}
		}

		if err := writer.Write(row); err != nil {
			fmt.Printf("Error writing row: %v\n", err)
			continue
		}
	}

	fmt.Printf("Node features data written to %s (Total %d records)\n", filePath, len(data))
}

// 计算平均特征值（简化）
func (nfm *NodeFeaturesModule) calculateAverageFeatures(data []message.ReplyNodeStateMsg) []float64 {
	if len(data) == 0 {
		return []float64{0}
	}

	// 计算一些关键指标的平均值
	var totalTPS, totalCPU, totalMem float64
	for _, state := range data {
		totalTPS += state.NodeState.Dynamic.OnChainBehavior.TransactionCapability.AvgTPS
		totalCPU += state.NodeState.Dynamic.DynamicAttributes.Compute.CPUUsage
		totalMem += state.NodeState.Dynamic.DynamicAttributes.Compute.MemUsage
	}

	count := float64(len(data))
	return []float64{
		totalTPS / count,
		totalCPU / count,
		totalMem / count,
	}
}

// 获取收集统计信息
func (nfm *NodeFeaturesModule) GetCollectionStats() (int, int, time.Time) {
	nfm.mu.RLock()
	defer nfm.mu.RUnlock()
	return len(nfm.collectedData), 0, time.Now() // 删除 blockInfoCounter, collectInterval 字段和相关逻辑
}

// 清空收集的数据（用于重置）
func (nfm *NodeFeaturesModule) ClearData() {
	nfm.mu.Lock()
	defer nfm.mu.Unlock()
	nfm.collectedData = make(map[int64][]message.ReplyNodeStateMsg)
	// 删除 blockInfoCounter, collectInterval 字段和相关逻辑
	// UpdateMeasureRecord 只做数据收集
}

// === 新增：触发epoch周期节点状态收集（异步） ===
func (nfm *NodeFeaturesModule) triggerEpochNodeStateCollection(epoch int) {
	// TODO: 实现向所有节点发送状态收集请求
	// 这里需要与supervisor中的节点通信逻辑集成
	fmt.Printf("Triggering node state collection for epoch %d\n", epoch)

	// 预留接口：可以在这里触发向所有节点发送CRequestNodeState消息
	// 收到的回复会通过HandleExtraMessage处理并存储到epochData[epoch]中
}

// === 新增：获取指定epoch的节点数据 ===
func (nfm *NodeFeaturesModule) GetEpochData(epoch int) ([]message.ReplyNodeStateMsg, bool) {
	nfm.mu.RLock()
	defer nfm.mu.RUnlock()

	data, exists := nfm.epochData[epoch]
	return data, exists
}

// === 新增：获取所有epoch数据的统计摘要 ===
func (nfm *NodeFeaturesModule) GetEpochSummary() map[int]map[string]float64 {
	nfm.mu.RLock()
	defer nfm.mu.RUnlock()

	summary := make(map[int]map[string]float64)

	for epoch, nodeStates := range nfm.epochData {
		if len(nodeStates) == 0 {
			continue
		}

		avgTPS := 0.0
		avgCPUUsage := 0.0
		avgMemUsage := 0.0

		for _, state := range nodeStates {
			avgTPS += state.NodeState.Dynamic.OnChainBehavior.TransactionCapability.AvgTPS
			avgCPUUsage += state.NodeState.Dynamic.DynamicAttributes.Compute.CPUUsage
			avgMemUsage += state.NodeState.Dynamic.DynamicAttributes.Compute.MemUsage
		}

		totalNodes := float64(len(nodeStates))
		summary[epoch] = map[string]float64{
			"avgTPS":      avgTPS / totalNodes,
			"avgCPUUsage": avgCPUUsage / totalNodes,
			"avgMemUsage": avgMemUsage / totalNodes,
			"nodeCount":   totalNodes,
		}
	}

	return summary
}

// === 新增：输出按epoch分组的CSV文件 ===
func (nfm *NodeFeaturesModule) WriteEpochCSV() {
	nfm.mu.RLock()
	defer nfm.mu.RUnlock()

	if !nfm.epochCollectionEnabled {
		fmt.Println("Epoch collection not enabled, skipping epoch CSV output")
		return
	}

	for epoch, nodeStates := range nfm.epochData {
		if len(nodeStates) == 0 {
			continue
		}

		fileName := fmt.Sprintf("node_features_epoch_%d.csv", epoch)
		filePath := fmt.Sprintf("%s/%s", params.ExpDataRootDir, fileName)

		// 确保目录存在
		os.MkdirAll(params.ExpDataRootDir, 0755)

		file, err := os.Create(filePath)
		if err != nil {
			fmt.Printf("Error creating epoch CSV file %s: %v\n", fileName, err)
			continue
		}

		writer := csv.NewWriter(file)

		// 使用相同的headers
		headers := []string{
			"NodeID", "ResourceCapacity.Hardware.CPU.CoreCount",
			"ResourceCapacity.Hardware.CPU.Architecture",
			// ... 其他所有headers保持一致
		}

		writer.Write(headers)

		// 写入该epoch的所有节点数据
		for _, state := range nodeStates {
			row := nfm.buildRowFromState(state, headers)
			writer.Write(row)
		}

		writer.Flush()
		file.Close()

		fmt.Printf("Epoch %d node features written to %s (%d records)\n",
			epoch, fileName, len(nodeStates))
	}
}

// === 新增：从节点状态构建CSV行数据的辅助方法 ===
func (nfm *NodeFeaturesModule) buildRowFromState(state message.ReplyNodeStateMsg, headers []string) []string {
	row := make([]string, len(headers))
	for i, field := range headers {
		// 复用现有的switch逻辑
		switch field {
		case "NodeID":
			row[i] = fmt.Sprintf("S%dN%d", state.ShardID, state.NodeID)
		case "ResourceCapacity.Hardware.CPU.CoreCount":
			row[i] = strconv.Itoa(state.NodeState.Static.ResourceCapacity.Hardware.CPU.CoreCount)
		case "ResourceCapacity.Hardware.CPU.Architecture":
			row[i] = state.NodeState.Static.ResourceCapacity.Hardware.CPU.Architecture
		// ... 可以继续添加其他字段，这里简化处理
		default:
			row[i] = "" // 暂时为空，可以后续完善
		}
	}
	return row
}

// 新增：获取当前数据而不触发CSV写入
func (nfm *NodeFeaturesModule) GetCurrentData() ([]float64, float64) {
	// NodeFeaturesModule不需要返回数值数据，返回默认值
	return []float64{}, 0.0
}
