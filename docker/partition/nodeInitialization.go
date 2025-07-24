package partition

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

func loadNodesFromCSV(filename string) ([]Node, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1 // Allow variable fields

	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	if len(records) < 2 {
		return nil, fmt.Errorf("insufficient records")
	}

	headers := records[0]
	var nodes []Node

	for _, row := range records[1:] {
		node := Node{}
		for i, header := range headers {
			if i >= len(row) {
				continue
			}
			value := row[i]
			if err := mapField(header, value, &node); err != nil {
				return nil, fmt.Errorf("error mapping %s: %v", header, err)
			}
		}
		nodes = append(nodes, node)
	}

	return nodes, nil
}

func mapField(header, value string, node *Node) error {
	switch header {
	// 资源能力层
	case "ResourceCapacity.Hardware.CPU.CoreCount":
		v, _ := strconv.ParseUint(value, 10, 32)
		node.ResourceCapacity.Hardware.CPU.CoreCount = uint(v)
	case "ResourceCapacity.Hardware.CPU.ClockFrequency":
		v, _ := strconv.ParseFloat(value, 64)
		node.ResourceCapacity.Hardware.CPU.ClockFrequency = v
	case "ResourceCapacity.Hardware.CPU.Architecture":
		node.ResourceCapacity.Hardware.CPU.Architecture = value
	case "ResourceCapacity.Hardware.CPU.CacheSize":
		v, _ := strconv.ParseUint(value, 10, 32)
		node.ResourceCapacity.Hardware.CPU.CacheSize = uint(v)
	case "ResourceCapacity.Hardware.Memory.TotalCapacity":
		v, _ := strconv.ParseUint(value, 10, 32)
		node.ResourceCapacity.Hardware.Memory.TotalCapacity = uint(v)
	case "ResourceCapacity.Hardware.Memory.Type":
		node.ResourceCapacity.Hardware.Memory.Type = value
	case "ResourceCapacity.Hardware.Memory.Bandwidth":
		v, _ := strconv.ParseFloat(value, 64)
		node.ResourceCapacity.Hardware.Memory.Bandwidth = v
	case "ResourceCapacity.Hardware.Storage.Capacity":
		v, _ := strconv.ParseUint(value, 10, 32)
		node.ResourceCapacity.Hardware.Storage.Capacity = uint(v)
	case "ResourceCapacity.Hardware.Storage.Type":
		node.ResourceCapacity.Hardware.Storage.Type = value
	case "ResourceCapacity.Hardware.Storage.ReadWriteSpeed":
		v, _ := strconv.ParseFloat(value, 64)
		node.ResourceCapacity.Hardware.Storage.ReadWriteSpeed = v
	case "ResourceCapacity.Hardware.Network.UpstreamBW":
		v, _ := strconv.ParseFloat(value, 64)
		node.ResourceCapacity.Hardware.Network.UpstreamBW = v
	case "ResourceCapacity.Hardware.Network.DownstreamBW":
		v, _ := strconv.ParseFloat(value, 64)
		node.ResourceCapacity.Hardware.Network.DownstreamBW = v
	case "ResourceCapacity.Hardware.Network.Latency":
		d, _ := parseDuration(value)
		node.ResourceCapacity.Hardware.Network.Latency = d

	// 运行状态
	case "ResourceCapacity.OperationalStatus.Uptime24h":
		v, _ := strconv.ParseFloat(value, 32)
		node.ResourceCapacity.OperationalStatus.Uptime24h = float32(v)
	case "ResourceCapacity.OperationalStatus.CoreEligibility":
		node.ResourceCapacity.OperationalStatus.CoreEligibility = strings.ToLower(value) == "true"
	case "ResourceCapacity.OperationalStatus.ResourceUsage.CPUUtilization":
		v, _ := strconv.ParseFloat(value, 32)
		node.ResourceCapacity.OperationalStatus.ResourceUsage.CPUUtilization = float32(v)
	case "ResourceCapacity.OperationalStatus.ResourceUsage.MemUtilization":
		v, _ := strconv.ParseFloat(value, 32)
		node.ResourceCapacity.OperationalStatus.ResourceUsage.MemUtilization = float32(v)

	// 链上行为层
	case "OnChainBehavior.TransactionCapability.AvgTPS":
		v, _ := strconv.ParseFloat(value, 64)
		node.OnChainBehavior.TransactionCapability.AvgTPS = v
	case "OnChainBehavior.TransactionCapability.CrossShardTx.InterNodeVolume":
		node.OnChainBehavior.TransactionCapability.CrossShardTx.InterNodeVolume = parseNodeVolume(value)
	case "OnChainBehavior.TransactionCapability.CrossShardTx.InterShardVolume":
		node.OnChainBehavior.TransactionCapability.CrossShardTx.InterShardVolume = parseShardVolume(value)
	case "OnChainBehavior.TransactionCapability.ConfirmationDelay":
		d, _ := parseDuration(value)
		node.OnChainBehavior.TransactionCapability.ConfirmationDelay = d
	case "OnChainBehavior.TransactionCapability.ResourcePerTx.CPUPerTx":
		v, _ := strconv.ParseFloat(value, 64)
		node.OnChainBehavior.TransactionCapability.ResourcePerTx.CPUPerTx = v
	case "OnChainBehavior.TransactionCapability.ResourcePerTx.MemPerTx":
		v, _ := strconv.ParseFloat(value, 64)
		node.OnChainBehavior.TransactionCapability.ResourcePerTx.MemPerTx = v
	case "OnChainBehavior.TransactionCapability.ResourcePerTx.DiskPerTx":
		v, _ := strconv.ParseFloat(value, 64)
		node.OnChainBehavior.TransactionCapability.ResourcePerTx.DiskPerTx = v
	case "OnChainBehavior.TransactionCapability.ResourcePerTx.NetworkPerTx":
		v, _ := strconv.ParseFloat(value, 64)
		node.OnChainBehavior.TransactionCapability.ResourcePerTx.NetworkPerTx = v
	case "OnChainBehavior.BlockGeneration.AvgInterval":
		d, _ := parseDuration(value)
		node.OnChainBehavior.BlockGeneration.AvgInterval = d
	case "OnChainBehavior.BlockGeneration.IntervalStdDev":
		d, _ := parseDuration(value)
		node.OnChainBehavior.BlockGeneration.IntervalStdDev = d
	case "OnChainBehavior.EconomicContribution.FeeContributionRatio":
		v, _ := strconv.ParseFloat(value, 32)
		node.OnChainBehavior.EconomicContribution.FeeContributionRatio = float32(v)
	case "OnChainBehavior.SmartContractUsage.InvocationFrequency":
		v, _ := strconv.ParseUint(value, 10, 32)
		node.OnChainBehavior.SmartContractUsage.InvocationFrequency = uint(v)
	case "OnChainBehavior.TransactionTypes.NormalTxRatio":
		v, _ := strconv.ParseFloat(value, 32)
		node.OnChainBehavior.TransactionTypes.NormalTxRatio = float32(v)
	case "OnChainBehavior.TransactionTypes.ContractTxRatio":
		v, _ := strconv.ParseFloat(value, 32)
		node.OnChainBehavior.TransactionTypes.ContractTxRatio = float32(v)
	case "OnChainBehavior.Consensus.ParticipationRate":
		v, _ := strconv.ParseFloat(value, 32)
		node.OnChainBehavior.Consensus.ParticipationRate = float32(v)
	case "OnChainBehavior.Consensus.TotalReward":
		v, _ := strconv.ParseFloat(value, 64)
		node.OnChainBehavior.Consensus.TotalReward = v
	case "OnChainBehavior.Consensus.SuccessRate":
		v, _ := strconv.ParseFloat(value, 32)
		node.OnChainBehavior.Consensus.SuccessRate = float32(v)

	// 网络拓扑层
	case "NetworkTopology.GeoLocation.Timezone":
		node.NetworkTopology.GeoLocation.Timezone = value
	case "NetworkTopology.GeoLocation.DataCenter":
		node.NetworkTopology.GeoLocation.DataCenter = value
	case "NetworkTopology.GeoLocation.Region":
		node.NetworkTopology.GeoLocation.Region = value
	case "NetworkTopology.Connections.IntraShardConn":
		v, _ := strconv.ParseUint(value, 10, 32)
		node.NetworkTopology.Connections.IntraShardConn = uint(v)
	case "NetworkTopology.Connections.InterShardConn":
		v, _ := strconv.ParseUint(value, 10, 32)
		node.NetworkTopology.Connections.InterShardConn = uint(v)
	case "NetworkTopology.Connections.WeightedDegree":
		v, _ := strconv.ParseFloat(value, 64)
		node.NetworkTopology.Connections.WeightedDegree = v
	case "NetworkTopology.Connections.ActiveConn":
		v, _ := strconv.ParseUint(value, 10, 32)
		node.NetworkTopology.Connections.ActiveConn = uint(v)
	case "NetworkTopology.Hierarchy.Depth":
		v, _ := strconv.ParseUint(value, 10, 32)
		node.NetworkTopology.Hierarchy.Depth = uint(v)
	case "NetworkTopology.Hierarchy.ConnectionDensity":
		v, _ := strconv.ParseFloat(value, 64)
		node.NetworkTopology.Hierarchy.ConnectionDensity = v
	case "NetworkTopology.Centrality.IntraShard.Eigenvector":
		v, _ := strconv.ParseFloat(value, 64)
		node.NetworkTopology.Centrality.IntraShard.Eigenvector = v
	case "NetworkTopology.Centrality.IntraShard.Closeness":
		v, _ := strconv.ParseFloat(value, 64)
		node.NetworkTopology.Centrality.IntraShard.Closeness = v
	case "NetworkTopology.Centrality.InterShard.Betweenness":
		v, _ := strconv.ParseFloat(value, 64)
		node.NetworkTopology.Centrality.InterShard.Betweenness = v
	case "NetworkTopology.Centrality.InterShard.Influence":
		v, _ := strconv.ParseFloat(value, 64)
		node.NetworkTopology.Centrality.InterShard.Influence = v
	case "NetworkTopology.ShardAllocation.Priority":
		v, _ := strconv.ParseUint(value, 10, 32)
		node.NetworkTopology.ShardAllocation.Priority = uint(v)
	case "NetworkTopology.ShardAllocation.ShardPreference":
		node.NetworkTopology.ShardAllocation.ShardPreference = parseShardPreference(value)
	case "NetworkTopology.ShardAllocation.Adaptability":
		v, _ := strconv.ParseFloat(value, 32)
		node.NetworkTopology.ShardAllocation.Adaptability = float32(v)

	// 动态属性层
	case "DynamicAttributes.Compute.CPUUsage":
		v, _ := strconv.ParseFloat(value, 32)
		node.DynamicAttributes.Compute.CPUUsage = float32(v)
	case "DynamicAttributes.Compute.MemUsage":
		v, _ := strconv.ParseFloat(value, 32)
		node.DynamicAttributes.Compute.MemUsage = float32(v)
	case "DynamicAttributes.Compute.ResourceFlux":
		v, _ := strconv.ParseFloat(value, 32)
		node.DynamicAttributes.Compute.ResourceFlux = float32(v)
	case "DynamicAttributes.Storage.Available":
		v, _ := strconv.ParseUint(value, 10, 32)
		node.DynamicAttributes.Storage.Available = uint(v)
	case "DynamicAttributes.Storage.Utilization":
		v, _ := strconv.ParseFloat(value, 32)
		node.DynamicAttributes.Storage.Utilization = float32(v)
	case "DynamicAttributes.Network.LatencyFlux":
		v, _ := strconv.ParseFloat(value, 32)
		node.DynamicAttributes.Network.LatencyFlux = float32(v)
	case "DynamicAttributes.Network.AvgLatency":
		d, _ := parseDuration(value)
		node.DynamicAttributes.Network.AvgLatency = d
	case "DynamicAttributes.Network.BandwidthUsage":
		v, _ := strconv.ParseFloat(value, 32)
		node.DynamicAttributes.Network.BandwidthUsage = float32(v)
	case "DynamicAttributes.Transactions.Frequency":
		v, _ := strconv.ParseUint(value, 10, 32)
		node.DynamicAttributes.Transactions.Frequency = uint(v)
	case "DynamicAttributes.Transactions.ProcessingDelay":
		d, _ := parseDuration(value)
		node.DynamicAttributes.Transactions.ProcessingDelay = d
	case "DynamicAttributes.Transactions.StakeChangeRate":
		v, _ := strconv.ParseFloat(value, 32)
		node.DynamicAttributes.Transactions.StakeChangeRate = float32(v)
	case "DynamicAttributes.Reputation.Uptime24h":
		v, _ := strconv.ParseFloat(value, 32)
		node.DynamicAttributes.Reputation.Uptime24h = float32(v)
	case "DynamicAttributes.Reputation.ReputationScore":
		v, _ := strconv.ParseFloat(value, 32)
		node.DynamicAttributes.Reputation.ReputationScore = float32(v)

	// 异构类型层
	case "HeterogeneousType.NodeType":
		node.HeterogeneousType.NodeType = value
	case "HeterogeneousType.FunctionTags":
		node.HeterogeneousType.FunctionTags = strings.Split(value, ",")
	case "HeterogeneousType.SupportedFuncs.Functions":
		node.HeterogeneousType.SupportedFuncs.Functions = strings.Split(value, ",")
	case "HeterogeneousType.SupportedFuncs.Priorities":
		node.HeterogeneousType.SupportedFuncs.Priorities = parsePriorities(value)
	case "HeterogeneousType.Application.CurrentState":
		node.HeterogeneousType.Application.CurrentState = value
	case "HeterogeneousType.Application.LoadMetrics.TxFrequency":
		v, _ := strconv.ParseUint(value, 10, 32)
		node.HeterogeneousType.Application.LoadMetrics.TxFrequency = uint(v)
	case "HeterogeneousType.Application.LoadMetrics.StorageOps":
		v, _ := strconv.ParseUint(value, 10, 32)
		node.HeterogeneousType.Application.LoadMetrics.StorageOps = uint(v)

	default:
		return fmt.Errorf("unexpected header field: %s", header)
	}
	return nil
}

// 辅助函数
func parseDuration(s string) (time.Duration, error) {
	if strings.HasSuffix(s, "ms") {
		ms, _ := strconv.Atoi(strings.TrimSuffix(s, "ms"))
		return time.Duration(ms) * time.Millisecond, nil
	}
	if strings.HasSuffix(s, "s") {
		sec, _ := strconv.Atoi(strings.TrimSuffix(s, "s"))
		return time.Duration(sec) * time.Second, nil
	}
	return 0, fmt.Errorf("unknown duration format: %s", s)
}

func parseNodeVolume(s string) map[string]uint {
	result := make(map[string]uint)
	if s == "" {
		return result
	}
	pairs := strings.Split(s, ";")
	for _, p := range pairs {
		kv := strings.Split(p, ":")
		if len(kv) == 2 {
			v, _ := strconv.ParseUint(kv[1], 10, 32)
			result[kv[0]] = uint(v)
		}
	}
	return result
}

func parseShardVolume(s string) map[string]uint {
	return parseNodeVolume(s) // 复用相同逻辑
}

func parseShardPreference(s string) map[string]float64 {
	result := make(map[string]float64)
	if s == "" {
		return result
	}
	pairs := strings.Split(s, ";")
	for _, p := range pairs {
		kv := strings.Split(p, ":")
		if len(kv) == 2 {
			v, _ := strconv.ParseFloat(kv[1], 64)
			result[kv[0]] = v
		}
	}
	return result
}

func parsePriorities(s string) map[string]uint {
	result := make(map[string]uint)
	if s == "" {
		return result
	}
	pairs := strings.Split(s, ";")
	for _, p := range pairs {
		kv := strings.Split(p, ":")
		if len(kv) == 2 {
			v, _ := strconv.ParseUint(kv[1], 10, 32)
			result[kv[0]] = uint(v)
		}
	}
	return result
}

// 打印节点信息
func printNodeDetails(n Node) {
	fmt.Println("=== Node Details ===")
	fmt.Printf("CPU: %d cores @ %.1fGHz (%s)\n",
		n.ResourceCapacity.Hardware.CPU.CoreCount,
		n.ResourceCapacity.Hardware.CPU.ClockFrequency,
		n.ResourceCapacity.Hardware.CPU.Architecture)

	fmt.Printf("Memory: %dGB %s (%.1fGB/s)\n",
		n.ResourceCapacity.Hardware.Memory.TotalCapacity,
		n.ResourceCapacity.Hardware.Memory.Type,
		n.ResourceCapacity.Hardware.Memory.Bandwidth)

	fmt.Printf("Network: %.0f/%.0f Mbps (Latency: %v)\n",
		n.ResourceCapacity.Hardware.Network.UpstreamBW,
		n.ResourceCapacity.Hardware.Network.DownstreamBW,
		n.ResourceCapacity.Hardware.Network.Latency)

	fmt.Printf("Consensus Success Rate: %.1f%%\n",
		n.OnChainBehavior.Consensus.SuccessRate)

	fmt.Printf("Shard Preferences: %v\n",
		n.NetworkTopology.ShardAllocation.ShardPreference)

	fmt.Printf("Current Load: %d TPS | Storage: %d ops\n",
		n.HeterogeneousType.Application.LoadMetrics.TxFrequency,
		n.HeterogeneousType.Application.LoadMetrics.StorageOps)
}
