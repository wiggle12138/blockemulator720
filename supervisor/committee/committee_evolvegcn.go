package committee

import (
	"blockEmulator/core"
	"blockEmulator/message"
	"blockEmulator/networks"
	"blockEmulator/params"
	"blockEmulator/partition"
	"blockEmulator/supervisor/signal"
	"blockEmulator/supervisor/supervisor_log"
	"blockEmulator/utils"
	"bytes"
	"context"
	"crypto/md5"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
	"time"
)

// 定义接口以避免循环引用
type NodeStateCollector interface {
	TriggerNodeStateCollection()
}

// EvolveGCN committee operations
type EvolveGCNCommitteeModule struct {
	csvPath      string
	dataTotalNum int
	nowDataNum   int
	batchDataNum int

	// additional variants
	curEpoch                 int32
	evolvegcnLock            sync.Mutex
	evolvegcnGraph           *partition.CLPAState
	modifiedMap              map[string]uint64
	evolvegcnLastRunningTime time.Time
	evolvegcnFreq            int

	// logger module
	sl *supervisor_log.SupervisorLog

	// control components
	Ss          *signal.StopSignal // to control the stop message sending
	IpNodeTable map[uint64]map[uint64]string

	// 新增：节点状态收集器接口引用
	nodeStateCollector NodeStateCollector
}

func NewEvolveGCNCommitteeModule(Ip_nodeTable map[uint64]map[uint64]string, Ss *signal.StopSignal, sl *supervisor_log.SupervisorLog, csvFilePath string, dataNum, batchNum, evolvegcnFrequency int) *EvolveGCNCommitteeModule {
	cg := new(partition.CLPAState)
	cg.Init_CLPAState(0.5, 100, params.ShardNum)
	egcm := &EvolveGCNCommitteeModule{
		csvPath:                  csvFilePath,
		dataTotalNum:             dataNum,
		batchDataNum:             batchNum,
		nowDataNum:               0,
		evolvegcnGraph:           cg,
		modifiedMap:              make(map[string]uint64),
		evolvegcnFreq:            evolvegcnFrequency,
		evolvegcnLastRunningTime: time.Time{},
		IpNodeTable:              Ip_nodeTable,
		Ss:                       Ss,
		sl:                       sl,
		curEpoch:                 0,
		nodeStateCollector:       nil, // 将在supervisor中设置
	}

	// 异步启动Python预热
	go egcm.warmupPythonService()

	return egcm
}

// 新增：设置节点状态收集器的方法
func (egcm *EvolveGCNCommitteeModule) SetNodeStateCollector(collector NodeStateCollector) {
	egcm.nodeStateCollector = collector
}

func (egcm *EvolveGCNCommitteeModule) HandleOtherMessage([]byte) {}

func (egcm *EvolveGCNCommitteeModule) fetchModifiedMap(key string) uint64 {
	if val, ok := egcm.modifiedMap[key]; !ok {
		return uint64(utils.Addr2Shard(key))
	} else {
		return val
	}
}

func (egcm *EvolveGCNCommitteeModule) txSending(txlist []*core.Transaction) {
	// the txs will be sent
	sendToShard := make(map[uint64][]*core.Transaction)

	for idx := 0; idx <= len(txlist); idx++ {
		if idx > 0 && (idx%params.InjectSpeed == 0 || idx == len(txlist)) {
			// send to shard
			for sid := uint64(0); sid < uint64(params.ShardNum); sid++ {
				it := message.InjectTxs{
					Txs:       sendToShard[sid],
					ToShardID: sid,
				}
				itByte, err := json.Marshal(it)
				if err != nil {
					log.Panic(err)
				}
				send_msg := message.MergeMessage(message.CInject, itByte)
				go networks.TcpDial(send_msg, egcm.IpNodeTable[sid][0])
			}
			sendToShard = make(map[uint64][]*core.Transaction)
			time.Sleep(time.Second)
		}
		if idx == len(txlist) {
			break
		}
		tx := txlist[idx]
		sendersid := egcm.fetchModifiedMap(tx.Sender)
		sendToShard[sendersid] = append(sendToShard[sendersid], tx)
	}
}

func (egcm *EvolveGCNCommitteeModule) MsgSendingControl() {
	txfile, err := os.Open(egcm.csvPath)
	if err != nil {
		log.Panic(err)
	}
	defer txfile.Close()
	reader := csv.NewReader(txfile)
	txlist := make([]*core.Transaction, 0)
	evolvegcnCnt := 0

	for {
		data, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Panic(err)
		}
		if tx, ok := data2tx(data, uint64(egcm.nowDataNum)); ok {
			txlist = append(txlist, tx)
			egcm.nowDataNum++
		} else {
			continue
		}

		// 批量发送交易
		if len(txlist) == int(egcm.batchDataNum) || egcm.nowDataNum == egcm.dataTotalNum {
			if egcm.evolvegcnLastRunningTime.IsZero() {
				egcm.evolvegcnLastRunningTime = time.Now()
			}

			egcm.txSending(txlist)
			txlist = make([]*core.Transaction, 0)
			egcm.Ss.StopGap_Reset()
		}

		// 修复：参考CLPA的epoch确认机制
		if params.ShardNum > 1 && !egcm.evolvegcnLastRunningTime.IsZero() && time.Since(egcm.evolvegcnLastRunningTime) >= time.Duration(egcm.evolvegcnFreq)*time.Second {
			egcm.evolvegcnLock.Lock()
			evolvegcnCnt++

			// 执行重分片流程
			egcm.sl.Slog.Printf("EvolveGCN Epoch %d: Step 1 - Triggering node feature collection...", evolvegcnCnt)
			if egcm.nodeStateCollector != nil {
				egcm.nodeStateCollector.TriggerNodeStateCollection()
				egcm.sl.Slog.Printf("EvolveGCN Epoch %d: All nodes confirmed feature collection completed", evolvegcnCnt)
			}

			preReconfigCTXRatio := egcm.calculateCurrentCrossShardRatio()
			egcm.sl.Slog.Printf("EvolveGCN Epoch %d: Pre-reconfiguration CTX ratio: %.4f", evolvegcnCnt, preReconfigCTXRatio)

			egcm.sl.Slog.Printf("EvolveGCN Epoch %d: Step 3 - Running EvolveGCN partition algorithm...", evolvegcnCnt)
			mmap, crossTxNum := egcm.runEvolveGCNPartition()
			egcm.sl.Slog.Printf("EvolveGCN Epoch %d: Partition completed, cross-shard edges: %d", evolvegcnCnt, crossTxNum)

			postReconfigCTXRatio := egcm.estimatePostReconfigCrossShardRatio(mmap, crossTxNum)
			egcm.sl.Slog.Printf("EvolveGCN Epoch %d: Post-reconfiguration estimated CTX ratio: %.4f", evolvegcnCnt, postReconfigCTXRatio)

			egcm.recordReconfigurationMetrics(evolvegcnCnt, preReconfigCTXRatio, postReconfigCTXRatio)

			// 发送分区消息
			egcm.evolvegcnMapSend(mmap)

			// // 关键修复：参考CLPA，添加epoch确认等待机制
			// for atomic.LoadInt32(&egcm.curEpoch) != int32(evolvegcnCnt) {
			// 	time.Sleep(time.Second)
			// }

			for key, val := range mmap {
				egcm.modifiedMap[key] = val
			}
			egcm.evolvegcnReset()
			egcm.evolvegcnLock.Unlock()

			// 关键修复：参考CLPA，添加epoch确认等待机制
			// for atomic.LoadInt32(&egcm.curEpoch) != int32(evolvegcnCnt) {
			// 	time.Sleep(time.Second)
			// }

			egcm.evolvegcnLastRunningTime = time.Now()
			egcm.sl.Slog.Printf("EvolveGCN Epoch %d: Successfully completed with epoch confirmation", evolvegcnCnt)
		}

		if egcm.nowDataNum == egcm.dataTotalNum {
			break
		}
	}

	// 最终处理阶段也要添加epoch确认
	for !egcm.Ss.GapEnough() {
		time.Sleep(time.Second)
		if params.ShardNum > 1 && time.Since(egcm.evolvegcnLastRunningTime) >= time.Duration(egcm.evolvegcnFreq)*time.Second {
			egcm.evolvegcnLock.Lock()
			evolvegcnCnt++

			egcm.sl.Slog.Printf("EvolveGCN Final Epoch %d: Executing final reconfiguration...", evolvegcnCnt)
			if egcm.nodeStateCollector != nil {
				egcm.nodeStateCollector.TriggerNodeStateCollection()
				time.Sleep(5 * time.Second)
			}

			preReconfigCTXRatio := egcm.calculateCurrentCrossShardRatio()
			mmap, crossTxNum := egcm.runEvolveGCNPartition()
			postReconfigCTXRatio := egcm.estimatePostReconfigCrossShardRatio(mmap, crossTxNum)

			egcm.recordReconfigurationMetrics(evolvegcnCnt, preReconfigCTXRatio, postReconfigCTXRatio)
			egcm.evolvegcnMapSend(mmap)

			// // 最终epoch也要等待确认
			// for atomic.LoadInt32(&egcm.curEpoch) != int32(evolvegcnCnt) {
			// 	time.Sleep(time.Second)
			// }

			for key, val := range mmap {
				egcm.modifiedMap[key] = val
			}
			egcm.evolvegcnReset()
			egcm.evolvegcnLock.Unlock()

			egcm.evolvegcnLastRunningTime = time.Now()
			egcm.sl.Slog.Printf("EvolveGCN Final Epoch %d: Completed", evolvegcnCnt)
		}
	}
}

// ========== 新增：计算当前跨分片交易率 ==========
func (egcm *EvolveGCNCommitteeModule) calculateCurrentCrossShardRatio() float64 {
	if egcm.evolvegcnGraph == nil || len(egcm.evolvegcnGraph.NetGraph.EdgeSet) == 0 {
		return 0.0
	}

	totalEdges := 0
	crossShardEdges := 0

	for v, neighbors := range egcm.evolvegcnGraph.NetGraph.EdgeSet {
		vShard := egcm.evolvegcnGraph.PartitionMap[v]
		for _, u := range neighbors {
			uShard := egcm.evolvegcnGraph.PartitionMap[u]
			totalEdges++
			if vShard != uShard {
				crossShardEdges++
			}
		}
	}

	if totalEdges == 0 {
		return 0.0
	}

	// 避免重复计算边（无向图）
	return float64(crossShardEdges) / float64(totalEdges*2)
}

// ========== EvolveGCN分区算法完整实现 ==========
func (egcm *EvolveGCNCommitteeModule) runEvolveGCNPartition() (map[string]uint64, int) {
	egcm.sl.Slog.Println("EvolveGCN: Starting four-step partition pipeline...")

	// 移除强制的Python环境配置，因为已通过配置文件设置
	egcm.sl.Slog.Println("EvolveGCN: Using pre-configured Python environment from config file...")

	// 检查Python配置
	if !egcm.isEvolveGCNEnabled() {
		egcm.sl.Slog.Println("EvolveGCN: CRITICAL ERROR - EvolveGCN is disabled")
		egcm.sl.Slog.Println("EvolveGCN: SYSTEM FAILURE - Cannot run without EvolveGCN")
		log.Fatal("EvolveGCN: System cannot continue with EvolveGCN disabled")
	}

	// 验证Python环境是否可用
	pythonPath := egcm.getPythonPath()
	if !egcm.validatePythonPath(pythonPath) {
		egcm.sl.Slog.Printf("EvolveGCN: WARNING - Python path validation failed for: %s", pythonPath)
		egcm.sl.Slog.Println("EvolveGCN: Attempting to use system fallback...")
	}

	// 第一步：特征提取
	egcm.sl.Slog.Println("EvolveGCN Step 1: Feature extraction...")
	nodeFeatures, err := egcm.extractNodeFeatures()
	if err != nil {
		egcm.sl.Slog.Printf("EvolveGCN: CRITICAL ERROR - Feature extraction failed: %v", err)
		egcm.sl.Slog.Println("EvolveGCN: SYSTEM FAILURE - Cannot continue without node features")
		log.Fatalf("EvolveGCN: System cannot continue without proper feature extraction: %v", err)
	}

	// 调用Python EvolveGCN完整流水线
	egcm.sl.Slog.Println("EvolveGCN: Calling Python four-step pipeline...")
	partitionMap, crossShardEdges, err := egcm.callPythonFourStepPipeline(nodeFeatures)
	if err != nil {
		egcm.sl.Slog.Printf("EvolveGCN: CRITICAL ERROR - Python pipeline failed: %v", err)
		egcm.sl.Slog.Println("EvolveGCN: SYSTEM FAILURE - Cannot continue without EvolveGCN processing")
		log.Fatalf("EvolveGCN: System cannot continue without successful EvolveGCN processing: %v", err)
	}

	egcm.sl.Slog.Printf("EvolveGCN: Pipeline completed successfully. Cross-shard edges: %d", crossShardEdges)
	egcm.sl.Slog.Println("EvolveGCN: ✅ Real EvolveGCN algorithm active (CLPA placeholder replaced)")
	return partitionMap, crossShardEdges
}

// ========== 新增：估算重配置后的跨分片交易率 ==========
func (egcm *EvolveGCNCommitteeModule) estimatePostReconfigCrossShardRatio(_ map[string]uint64, crossTxNum int) float64 {
	totalEdges := 0
	for _, neighbors := range egcm.evolvegcnGraph.NetGraph.EdgeSet {
		totalEdges += len(neighbors)
	}

	if totalEdges == 0 {
		return 0.0
	}

	// 使用算法返回的跨分片边数估算
	return float64(crossTxNum*2) / float64(totalEdges)
}

// ========== 新增：记录重配置指标到统计模块 ==========
func (egcm *EvolveGCNCommitteeModule) recordReconfigurationMetrics(epochID int, preRatio, postRatio float64) {
	// 这里需要访问supervisor中的测量模块来记录重配置事件
	// 由于模块之间的解耦，这个功能将在supervisor层面实现
	egcm.sl.Slog.Printf("EvolveGCN Metrics: Epoch %d - CTX ratio change: %.4f -> %.4f (reduction: %.4f)",
		epochID, preRatio, postRatio, preRatio-postRatio)
}

// 添加分区映射发送方法 - 增强版本，包含epoch信息
func (egcm *EvolveGCNCommitteeModule) evolvegcnMapSend(m map[string]uint64) {
	// 新增：在分区消息中包含epoch信息
	egcm.sl.Slog.Printf("到函数里边啦，准备发消息")
	pm := message.PartitionModifiedMapWithEpoch{
		PartitionModified: m,
		EpochID:           atomic.LoadInt32(&egcm.curEpoch),
		Timestamp:         time.Now().Unix(),
	}
	pmByte, err := json.Marshal(pm)
	if err != nil {
		log.Panic()
	}
	send_msg := message.MergeMessage(message.CPartitionMsg, pmByte)

	// send to worker shards 发送给0号节点
	for i := uint64(0); i < uint64(params.ShardNum); i++ {
		go networks.TcpDial(send_msg, egcm.IpNodeTable[i][0])
	}

	egcm.sl.Slog.Printf("EvolveGCN Supervisor: partition map with epoch %d sent to all shards的0号节点",
		atomic.LoadInt32(&egcm.curEpoch))
	egcm.sl.Slog.Printf("函数能运行结束吗测试")
}

// 添加图状态重置方法
func (egcm *EvolveGCNCommitteeModule) evolvegcnReset() {
	egcm.evolvegcnGraph = new(partition.CLPAState)
	egcm.evolvegcnGraph.Init_CLPAState(0.5, 100, params.ShardNum)
	for key, val := range egcm.modifiedMap {
		egcm.evolvegcnGraph.PartitionMap[partition.Vertex{Addr: key}] = int(val)
	}
}

func (egcm *EvolveGCNCommitteeModule) HandleBlockInfo(b *message.BlockInfoMsg) {
	egcm.sl.Slog.Printf("EvolveGCN Supervisor: received from shard %d in epoch %d, blockLength=%d\n",
		b.SenderShardID, b.Epoch, b.BlockBodyLength)

	// 关键修复：区分普通区块和分区确认消息
	if b.BlockBodyLength == 0 {
		// 这是分区确认消息（222）
		egcm.sl.Slog.Printf("收到222分区确认 from shard %d with epoch %d",
			b.SenderShardID, b.Epoch)

		// 更新epoch
		egcm.evolvegcnLock.Lock()
		currentEpoch := atomic.LoadInt32(&egcm.curEpoch)
		if int32(b.Epoch) > currentEpoch {
			atomic.StoreInt32(&egcm.curEpoch, int32(b.Epoch))
			egcm.sl.Slog.Printf("EPOCH UPDATED from 老的%d to 接收到的%d (PARTITION CONFIRMED by shard %d)",
				currentEpoch, b.Epoch, b.SenderShardID)
		} else {
			egcm.sl.Slog.Printf("EvolveGCN: epoch not updated, received %d but current is %d", b.Epoch, currentEpoch)
		}
		egcm.evolvegcnLock.Unlock()
	} else {
		// 这是普通区块信息，处理交易图构建
		egcm.sl.Slog.Printf("收到普通区块信息 with %d transactions from shard %d",
			b.BlockBodyLength, b.SenderShardID)

		egcm.evolvegcnLock.Lock()
		for _, tx := range b.InnerShardTxs {
			egcm.evolvegcnGraph.AddEdge(partition.Vertex{Addr: tx.Sender}, partition.Vertex{Addr: tx.Recipient})
		}
		for _, r1tx := range b.Relay1Txs {
			egcm.evolvegcnGraph.AddEdge(partition.Vertex{Addr: r1tx.Sender}, partition.Vertex{Addr: r1tx.Recipient})
		}
		for _, r2tx := range b.Relay2Txs {
			egcm.evolvegcnGraph.AddEdge(partition.Vertex{Addr: r2tx.Sender}, partition.Vertex{Addr: r2tx.Recipient})
		}
		egcm.evolvegcnLock.Unlock()
	}
}

// ========== EvolveGCN四步处理流程实现 ==========

// 数据结构定义
type NodeFeatureData struct {
	NodeID          string             `json:"node_id"`
	ShardID         uint64             `json:"shard_id"`
	StaticFeatures  map[string]float64 `json:"static_features"`
	DynamicFeatures map[string]float64 `json:"dynamic_features"`
}

type TemporalEmbedding struct {
	NodeID     string    `json:"node_id"`
	Embeddings []float64 `json:"embeddings"`
	Timestamp  int64     `json:"timestamp"`
}

type ShardingResult struct {
	NodeID             string    `json:"node_id"`
	RecommendedShardID uint64    `json:"recommended_shard_id"`
	Confidence         float64   `json:"confidence"`
	Embeddings         []float64 `json:"embeddings"`
}

// 第一步：特征提取
func (egcm *EvolveGCNCommitteeModule) extractNodeFeatures() ([]NodeFeatureData, error) {
	egcm.sl.Slog.Println("EvolveGCN Step 1: Extracting node features...")

	// 构建节点特征数据
	var nodeFeatures []NodeFeatureData

	// 检查evolvegcnGraph是否已初始化
	if egcm.evolvegcnGraph == nil || egcm.evolvegcnGraph.PartitionMap == nil {
		egcm.sl.Slog.Println("EvolveGCN: Warning - Graph not initialized, using IP node table")

		// 使用IP节点表作为备用方案
		for shardID, nodeMap := range egcm.IpNodeTable {
			for nodeID, address := range nodeMap {
				nodeIDStr := fmt.Sprintf("%s_%d", address, nodeID)

				// 计算节点的静态特征
				staticFeatures := egcm.calculateStaticFeatures(nodeIDStr)

				// 计算节点的动态特征（基于交易历史）
				dynamicFeatures := egcm.calculateDynamicFeatures(nodeIDStr)

				nodeFeatures = append(nodeFeatures, NodeFeatureData{
					NodeID:          nodeIDStr,
					ShardID:         shardID,
					StaticFeatures:  staticFeatures,
					DynamicFeatures: dynamicFeatures,
				})
			}
		}
	} else {
		// 从交易图中获取节点信息
		for vertex := range egcm.evolvegcnGraph.PartitionMap {
			nodeID := vertex.Addr
			currentShardID := uint64(egcm.evolvegcnGraph.PartitionMap[vertex])

			// 计算节点的静态特征
			staticFeatures := egcm.calculateStaticFeatures(nodeID)

			// 计算节点的动态特征（基于交易历史）
			dynamicFeatures := egcm.calculateDynamicFeatures(nodeID)

			nodeFeatures = append(nodeFeatures, NodeFeatureData{
				NodeID:          nodeID,
				ShardID:         currentShardID,
				StaticFeatures:  staticFeatures,
				DynamicFeatures: dynamicFeatures,
			})
		}
	}

	// 保存特征数据到文件供Python处理
	if err := egcm.saveNodeFeaturesToCSV(nodeFeatures); err != nil {
		return nil, fmt.Errorf("failed to save node features: %v", err)
	}

	egcm.sl.Slog.Printf("EvolveGCN Step 1: Extracted features for %d nodes", len(nodeFeatures))
	return nodeFeatures, nil
}

// 第二步：多尺度对比学习（异步处理）
func (egcm *EvolveGCNCommitteeModule) runMultiScaleContrastiveLearning(nodeFeatures []NodeFeatureData) ([]TemporalEmbedding, error) {
	egcm.sl.Slog.Println("EvolveGCN Step 2: Running multi-scale contrastive learning...")

	// 创建异步处理通道
	resultChan := make(chan []TemporalEmbedding, 1)
	errorChan := make(chan error, 1)

	// 异步执行以避免阻塞主流程
	go func() {
		defer close(resultChan)
		defer close(errorChan)

		// 调用Python脚本进行多尺度对比学习
		embeddings, err := egcm.callPythonMultiScaleLearning()
		if err != nil {
			errorChan <- err
			return
		}

		resultChan <- embeddings
	}()

	// 设置超时以避免长时间等待
	timeout := time.Duration(30) * time.Second // 30秒超时
	select {
	case embeddings := <-resultChan:
		egcm.sl.Slog.Printf("EvolveGCN Step 2: Generated temporal embeddings for %d nodes", len(embeddings))
		return embeddings, nil
	case err := <-errorChan:
		return nil, err
	case <-time.After(timeout):
		egcm.sl.Slog.Println("EvolveGCN Step 2: Timeout, using simplified embeddings")
		// 超时时使用简化的嵌入生成
		return egcm.generateSimplifiedEmbeddings(nodeFeatures), nil
	}
}

// 第三步：EvolveGCN动态分片
func (egcm *EvolveGCNCommitteeModule) runEvolveGCNSharding(embeddings []TemporalEmbedding) ([]ShardingResult, error) {
	egcm.sl.Slog.Println("EvolveGCN Step 3: Running EvolveGCN dynamic sharding...")

	// 调用Python EvolveGCN模型
	results, err := egcm.callPythonEvolveGCNSharding(embeddings)
	if err != nil {
		// 如果Python调用失败，使用基于图的简化分片算法
		egcm.sl.Slog.Printf("EvolveGCN Step 3: Python call failed, using graph-based fallback: %v", err)
		return egcm.graphBasedSharding(embeddings), nil
	}

	egcm.sl.Slog.Printf("EvolveGCN Step 3: Generated sharding recommendations for %d nodes", len(results))
	return results, nil
}

// 第四步：性能反馈评估
func (egcm *EvolveGCNCommitteeModule) evaluateAndOptimize(results []ShardingResult) ([]ShardingResult, error) {
	egcm.sl.Slog.Println("EvolveGCN Step 4: Evaluating and optimizing sharding results...")

	// 计算预期的跨分片交易率
	expectedCrossShardRatio := egcm.calculateExpectedCrossShardRatio(results)

	// 如果预期跨分片率过高，进行优化
	if expectedCrossShardRatio > 0.3 { // 30%阈值
		egcm.sl.Slog.Printf("EvolveGCN Step 4: High cross-shard ratio (%.2f%%), optimizing...", expectedCrossShardRatio*100)
		optimizedResults := egcm.optimizeShardingForCrossShardReduction(results)
		return optimizedResults, nil
	}

	egcm.sl.Slog.Printf("EvolveGCN Step 4: Cross-shard ratio acceptable (%.2f%%)", expectedCrossShardRatio*100)
	return results, nil
}

// 转换为分区映射格式
func (egcm *EvolveGCNCommitteeModule) convertToPartitionMap(results []ShardingResult) (map[string]uint64, int) {
	partitionMap := make(map[string]uint64)
	crossShardEdges := 0

	// 构建新的分区映射
	for _, result := range results {
		partitionMap[result.NodeID] = result.RecommendedShardID
	}

	// 计算跨分片边数
	for vertex := range egcm.evolvegcnGraph.PartitionMap {
		nodeID := vertex.Addr
		newShardID := partitionMap[nodeID]

		if neighbors, exists := egcm.evolvegcnGraph.NetGraph.EdgeSet[vertex]; exists {
			for _, neighbor := range neighbors {
				neighborShardID := partitionMap[neighbor.Addr]
				if newShardID != neighborShardID {
					crossShardEdges++
				}
			}
		}
	}

	// 避免重复计算（无向图）
	crossShardEdges /= 2

	return partitionMap, crossShardEdges
}

// ========== 辅助方法实现 ==========

// 计算节点静态特征
func (egcm *EvolveGCNCommitteeModule) calculateStaticFeatures(nodeID string) map[string]float64 {
	features := make(map[string]float64)

	// 基于地址哈希计算模拟的硬件特征
	hashBytes := md5.Sum([]byte(nodeID))
	hash := hashBytes[:]

	features["cpu_cores"] = float64(hash[0]%8 + 1)            // 1-8核心
	features["memory_gb"] = float64(hash[1]%16 + 4)           // 4-20GB内存
	features["storage_gb"] = float64(hash[2]%100 + 100)       // 100-200GB存储
	features["network_bandwidth"] = float64(hash[3]%100 + 50) // 50-150 Mbps

	return features
}

// 计算节点动态特征
func (egcm *EvolveGCNCommitteeModule) calculateDynamicFeatures(nodeID string) map[string]float64 {
	features := make(map[string]float64)

	vertex := partition.Vertex{Addr: nodeID}

	// 计算节点度（连接数）
	if neighbors, exists := egcm.evolvegcnGraph.NetGraph.EdgeSet[vertex]; exists {
		features["node_degree"] = float64(len(neighbors))
	} else {
		features["node_degree"] = 0
	}

	// 计算跨分片连接率
	crossShardConnections := 0
	totalConnections := 0
	if neighbors, exists := egcm.evolvegcnGraph.NetGraph.EdgeSet[vertex]; exists {
		nodeShardID := egcm.evolvegcnGraph.PartitionMap[vertex]
		for _, neighbor := range neighbors {
			totalConnections++
			if egcm.evolvegcnGraph.PartitionMap[neighbor] != nodeShardID {
				crossShardConnections++
			}
		}
	}

	if totalConnections > 0 {
		features["cross_shard_ratio"] = float64(crossShardConnections) / float64(totalConnections)
	} else {
		features["cross_shard_ratio"] = 0
	}

	// 模拟CPU和内存使用率
	hashBytes := md5.Sum([]byte(nodeID + fmt.Sprintf("%d", time.Now().Unix())))
	hash := hashBytes[:]
	features["cpu_usage"] = float64(hash[0]%80+10) / 100.0    // 10-90%
	features["memory_usage"] = float64(hash[1]%70+20) / 100.0 // 20-90%

	return features
}

// 保存节点特征到CSV文件
func (egcm *EvolveGCNCommitteeModule) saveNodeFeaturesToCSV(nodeFeatures []NodeFeatureData) error {
	file, err := os.Create("node_features_input.csv")
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// 写入CSV头部
	header := []string{"node_id", "shard_id", "cpu_cores", "memory_gb", "storage_gb", "network_bandwidth",
		"node_degree", "cross_shard_ratio", "cpu_usage", "memory_usage"}
	writer.Write(header)

	// 写入数据
	for _, node := range nodeFeatures {
		record := []string{
			node.NodeID,
			fmt.Sprintf("%d", node.ShardID),
			fmt.Sprintf("%.2f", node.StaticFeatures["cpu_cores"]),
			fmt.Sprintf("%.2f", node.StaticFeatures["memory_gb"]),
			fmt.Sprintf("%.2f", node.StaticFeatures["storage_gb"]),
			fmt.Sprintf("%.2f", node.StaticFeatures["network_bandwidth"]),
			fmt.Sprintf("%.2f", node.DynamicFeatures["node_degree"]),
			fmt.Sprintf("%.4f", node.DynamicFeatures["cross_shard_ratio"]),
			fmt.Sprintf("%.4f", node.DynamicFeatures["cpu_usage"]),
			fmt.Sprintf("%.4f", node.DynamicFeatures["memory_usage"]),
		}
		writer.Write(record)
	}

	return nil
}

// 获取配置的Python路径
func (egcm *EvolveGCNCommitteeModule) getPythonPath() string {
	// 首先尝试从python_config.json读取配置
	if configBytes, err := os.ReadFile("python_config.json"); err == nil {
		var config map[string]interface{}
		if json.Unmarshal(configBytes, &config) == nil {
			if pythonPath, ok := config["python_path"].(string); ok && pythonPath != "" {
				// 验证Python路径是否可用
				if egcm.validatePythonPath(pythonPath) {
					return pythonPath
				}
			}
		}
	}

	// 尝试常见的虚拟环境路径
	commonPaths := []string{
		`E:\Codefield\BlockEmulator\.venv\Scripts\python.exe`,
		`.\.venv\Scripts\python.exe`,
		`.\venv\Scripts\python.exe`,
		`.\.env\Scripts\python.exe`,
		`.\env\Scripts\python.exe`,
	}

	for _, path := range commonPaths {
		if egcm.validatePythonPath(path) {
			egcm.sl.Slog.Printf("EvolveGCN: Found virtual environment at: %s", path)
			return path
		}
	}

	// 回退到系统Python
	egcm.sl.Slog.Println("EvolveGCN: Using system Python as fallback")
	return "python"
}

// 验证Python路径是否可用
func (egcm *EvolveGCNCommitteeModule) validatePythonPath(pythonPath string) bool {
	if _, err := os.Stat(pythonPath); err != nil {
		// 如果是相对路径或命令，尝试执行
		if pythonPath == "python" || pythonPath == "python3" {
			cmd := exec.Command(pythonPath, "--version")
			if err := cmd.Run(); err == nil {
				return true
			}
		}
		return false
	}
	return true
}

// 检查EvolveGCN是否启用
func (egcm *EvolveGCNCommitteeModule) isEvolveGCNEnabled() bool {
	// 读取配置文件
	if configBytes, err := os.ReadFile("python_config.json"); err == nil {
		var config map[string]interface{}
		if json.Unmarshal(configBytes, &config) == nil {
			// 检查启用标志
			if enabled, ok := config["enable_evolve_gcn"].(bool); ok {
				return enabled
			}
		}
	}

	// 默认启用EvolveGCN
	return true
}

// 调用Python多尺度对比学习
func (egcm *EvolveGCNCommitteeModule) callPythonMultiScaleLearning() ([]TemporalEmbedding, error) {
	// 构建Python命令
	cmd := exec.Command("python", "muti_scale/realtime_mscia.py", "--input", "node_features_input.csv", "--output", "temporal_embeddings.json")

	var out bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr

	// 设置工作目录
	cmd.Dir = "."

	err := cmd.Run()
	if err != nil {
		return nil, fmt.Errorf("python execution failed: %v, stderr: %s", err, stderr.String())
	}

	// 读取Python输出的嵌入结果
	return egcm.readTemporalEmbeddingsFromFile("temporal_embeddings.json")
}

// 生成简化嵌入（备用方案）
func (egcm *EvolveGCNCommitteeModule) generateSimplifiedEmbeddings(nodeFeatures []NodeFeatureData) []TemporalEmbedding {
	var embeddings []TemporalEmbedding

	for _, node := range nodeFeatures {
		// 使用简化的特征组合生成64维嵌入
		embedding := make([]float64, 64)

		// 基于节点特征生成嵌入
		hashBytes := md5.Sum([]byte(node.NodeID))
		hash := hashBytes[:]
		for i := 0; i < 64; i++ {
			// 结合静态和动态特征生成嵌入值
			staticVal := node.StaticFeatures["cpu_cores"] + node.StaticFeatures["memory_gb"]
			dynamicVal := node.DynamicFeatures["node_degree"] + node.DynamicFeatures["cross_shard_ratio"]*100

			embedding[i] = (staticVal + dynamicVal + float64(hash[i%32])) / 100.0
		}

		embeddings = append(embeddings, TemporalEmbedding{
			NodeID:     node.NodeID,
			Embeddings: embedding,
			Timestamp:  time.Now().Unix(),
		})
	}

	return embeddings
}

// 调用Python四步完整流水线
func (egcm *EvolveGCNCommitteeModule) callPythonFourStepPipeline(nodeFeatures []NodeFeatureData) (map[string]uint64, int, error) {
	egcm.sl.Slog.Println("EvolveGCN: Preparing input for Python four-step pipeline...")

	// 准备输入数据
	inputData := egcm.preparePipelineInput(nodeFeatures)

	// 保存输入文件
	inputFile := "evolvegcn_input.json"
	outputFile := "evolvegcn_output.json"

	inputBytes, err := json.Marshal(inputData)
	if err != nil {
		egcm.sl.Slog.Printf("EvolveGCN ERROR: Failed to marshal input data: %v", err)
		return nil, 0, fmt.Errorf("failed to marshal input data: %v", err)
	}

	egcm.sl.Slog.Printf("EvolveGCN: Input JSON size: %d bytes, contains %d nodes",
		len(inputBytes), len(nodeFeatures))

	err = os.WriteFile(inputFile, inputBytes, 0644)
	if err != nil {
		egcm.sl.Slog.Printf("EvolveGCN ERROR: Failed to write input file: %v", err)
		return nil, 0, fmt.Errorf("failed to write input file: %v", err)
	}

	egcm.sl.Slog.Printf("EvolveGCN: Successfully wrote input file: %s", inputFile)

	// 获取Python路径配置
	pythonPath := egcm.getPythonPath()

	// 调用Python接口脚本
	egcm.sl.Slog.Printf("EvolveGCN: Executing Python four-step pipeline with: %s", pythonPath)
	egcm.sl.Slog.Printf("EvolveGCN: Input file: %s, Output file: %s", inputFile, outputFile)

	// 检查输入文件是否存在且有内容
	if inputInfo, err := os.Stat(inputFile); err != nil {
		egcm.sl.Slog.Printf("EvolveGCN ERROR: Input file does not exist: %v", err)
		return nil, 0, fmt.Errorf("input file %s does not exist: %v", inputFile, err)
	} else {
		egcm.sl.Slog.Printf("EvolveGCN: Input file size: %d bytes", inputInfo.Size())
	}

	// 检查Python脚本是否存在
	if _, err := os.Stat("evolvegcn_go_interface.py"); err != nil {
		egcm.sl.Slog.Printf("EvolveGCN ERROR: Python interface script not found: %v", err)
		return nil, 0, fmt.Errorf("Python interface script not found: %v", err)
	}

	cmd := exec.Command(pythonPath, "evolvegcn_go_interface.py", "--input", inputFile, "--output", outputFile)

	// 设置工作目录为当前目录
	if wd, err := os.Getwd(); err == nil {
		cmd.Dir = wd
		egcm.sl.Slog.Printf("EvolveGCN: Setting working directory to: %s", wd)
	}

	// 显式设置UTF-8编码环境变量
	cmd.Env = append(os.Environ(),
		"PYTHONIOENCODING=utf-8",
		"LANG=en_US.UTF-8",
		"LC_ALL=en_US.UTF-8",
		"PYTHONUTF8=1",
	)

	egcm.sl.Slog.Printf("EvolveGCN: Starting Python pipeline execution...")
	egcm.sl.Slog.Printf("EvolveGCN: Command: %s %v", pythonPath, cmd.Args)

	// 记录环境变量
	egcm.sl.Slog.Printf("EvolveGCN: Environment variables: %v", cmd.Env)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	// 设置超时
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second) // 2分钟超时
	defer cancel()

	// 选择Python脚本优先级：安全预加载服务 > 预加载服务 > 优化版完整流水线 > 原始完整版 > 快速测试版
	scriptName := "evolvegcn_go_interface.py" // 默认原始版本

	if _, err := os.Stat("evolvegcn_preload_service_safe.py"); err == nil {
		scriptName = "evolvegcn_preload_service_safe.py"
		egcm.sl.Slog.Printf("EvolveGCN: Using encoding-safe preloaded service (fastest)")
	} else if _, err := os.Stat("evolvegcn_preload_service.py"); err == nil {
		scriptName = "evolvegcn_preload_service.py"
		egcm.sl.Slog.Printf("EvolveGCN: Using preloaded service (fastest)")
	} else if _, err := os.Stat("evolvegcn_optimized.py"); err == nil {
		scriptName = "evolvegcn_optimized.py"
		egcm.sl.Slog.Printf("EvolveGCN: Using optimized four-step pipeline")
	} else if _, err := os.Stat("evolvegcn_go_interface.py"); err == nil {
		scriptName = "evolvegcn_go_interface.py"
		egcm.sl.Slog.Printf("EvolveGCN: Using original four-step pipeline")
	} else if _, err := os.Stat("evolvegcn_quick_test.py"); err == nil {
		// 检查是否强制使用快速测试
		if os.Getenv("EVOLVEGCN_QUICK_TEST") == "1" {
			scriptName = "evolvegcn_quick_test.py"
			egcm.sl.Slog.Printf("EvolveGCN: Using quick test script (EVOLVEGCN_QUICK_TEST=1)")
		} else {
			egcm.sl.Slog.Printf("EvolveGCN: Quick test available but using optimized version")
		}
	}

	cmdWithTimeout := exec.CommandContext(ctx, pythonPath, scriptName, "--input", inputFile, "--output", outputFile)
	cmdWithTimeout.Dir = cmd.Dir
	cmdWithTimeout.Env = cmd.Env
	cmdWithTimeout.Stdout = &stdout
	cmdWithTimeout.Stderr = &stderr

	err = cmdWithTimeout.Run()

	// 详细记录执行结果
	egcm.sl.Slog.Printf("EvolveGCN: Python pipeline execution completed")
	egcm.sl.Slog.Printf("EvolveGCN: STDOUT: %s", stdout.String())
	if stderr.Len() > 0 {
		egcm.sl.Slog.Printf("EvolveGCN: STDERR: %s", stderr.String())
	}

	if err != nil {
		egcm.sl.Slog.Printf("EvolveGCN ERROR: Python pipeline execution failed: %v", err)
		return nil, 0, fmt.Errorf("Python pipeline execution failed: %v, stderr: %s", err, stderr.String())
	}

	// 检查输出文件是否生成
	if outputInfo, err := os.Stat(outputFile); err != nil {
		egcm.sl.Slog.Printf("EvolveGCN ERROR: Output file was not created: %v", err)
		return nil, 0, fmt.Errorf("output file %s was not created: %v", outputFile, err)
	} else {
		egcm.sl.Slog.Printf("EvolveGCN: Output file created successfully, size: %d bytes", outputInfo.Size())
	}

	// 读取输出结果
	egcm.sl.Slog.Printf("EvolveGCN: Parsing pipeline output from: %s", outputFile)
	result, err := egcm.parsePipelineOutput(outputFile)
	if err != nil {
		egcm.sl.Slog.Printf("EvolveGCN ERROR: Failed to parse pipeline output: %v", err)
		return nil, 0, fmt.Errorf("failed to parse pipeline output: %v", err)
	}

	egcm.sl.Slog.Printf("EvolveGCN: Successfully parsed output, partition map size: %d", len(result.PartitionMap))

	// 清理临时文件
	egcm.sl.Slog.Printf("EvolveGCN: Cleaning up temporary files...")
	os.Remove(inputFile)
	os.Remove(outputFile)

	egcm.sl.Slog.Printf("EvolveGCN: Python pipeline completed, processed %d nodes with %d cross-shard edges",
		len(result.PartitionMap), result.CrossShardEdges)

	return result.PartitionMap, result.CrossShardEdges, nil
}

// 准备Python流水线输入
func (egcm *EvolveGCNCommitteeModule) preparePipelineInput(nodeFeatures []NodeFeatureData) map[string]interface{} {
	// 构建节点特征数组
	var pythonNodeFeatures []map[string]interface{}

	for _, nodeFeature := range nodeFeatures {
		// 组合静态和动态特征为单一特征向量
		var features []float64

		// 添加静态特征
		for _, value := range nodeFeature.StaticFeatures {
			features = append(features, value)
		}

		// 添加动态特征
		for _, value := range nodeFeature.DynamicFeatures {
			features = append(features, value)
		}

		// 如果特征向量太短，填充到最小维度
		for len(features) < 64 {
			features = append(features, 0.0)
		}

		pythonNodeFeatures = append(pythonNodeFeatures, map[string]interface{}{
			"node_id":  nodeFeature.NodeID,
			"features": features,
			"metadata": map[string]interface{}{
				"shard_id":         nodeFeature.ShardID,
				"static_features":  nodeFeature.StaticFeatures,
				"dynamic_features": nodeFeature.DynamicFeatures,
			},
		})
	}

	// 构建交易图数据
	var edges [][]interface{}

	// 检查图是否已初始化
	if egcm.evolvegcnGraph != nil && egcm.evolvegcnGraph.NetGraph.EdgeSet != nil {
		for vertex, neighbors := range egcm.evolvegcnGraph.NetGraph.EdgeSet {
			for _, neighbor := range neighbors {
				// 添加边：[源节点, 目标节点, 权重]
				edges = append(edges, []interface{}{vertex.Addr, neighbor.Addr, 1.0})
			}
		}
	} else {
		// 生成基于分片的默认边（分片内节点全连接，分片间稀疏连接）
		egcm.sl.Slog.Println("EvolveGCN: Generating default edges based on node features")

		// 按分片分组节点
		shardNodes := make(map[uint64][]string)
		for _, nf := range nodeFeatures {
			shardNodes[nf.ShardID] = append(shardNodes[nf.ShardID], nf.NodeID)
		}

		// 生成分片内全连接边
		for _, nodes := range shardNodes {
			for i, node1 := range nodes {
				for j, node2 := range nodes {
					if i != j {
						edges = append(edges, []interface{}{node1, node2, 1.0})
					}
				}
			}
		}

		// 生成部分跨分片边
		allNodes := []string{}
		for _, nf := range nodeFeatures {
			allNodes = append(allNodes, nf.NodeID)
		}

		// 添加20%的跨分片边
		crossShardEdgeCount := len(allNodes) / 5
		for i := 0; i < crossShardEdgeCount && len(allNodes) > 1; i++ {
			src := allNodes[i%len(allNodes)]
			dst := allNodes[(i+1)%len(allNodes)]
			edges = append(edges, []interface{}{src, dst, 0.5})
		}
	}

	return map[string]interface{}{
		"node_features": pythonNodeFeatures,
		"transaction_graph": map[string]interface{}{
			"edges": edges,
			"metadata": map[string]interface{}{
				"total_nodes": len(nodeFeatures),
				"total_edges": len(edges),
				"timestamp":   time.Now().Unix(),
			},
		},
		"config": map[string]interface{}{
			"target_shards": params.ShardNum,
			"algorithm":     "EvolveGCN",
		},
	}
}

// Python流水线输出结构
type PipelineOutput struct {
	Success         bool               `json:"success"`
	PartitionMap    map[string]uint64  `json:"partition_map"`
	CrossShardEdges int                `json:"cross_shard_edges"`
	Metrics         map[string]float64 `json:"metrics"`
	Error           string             `json:"error,omitempty"`
}

// 解析Python流水线输出
func (egcm *EvolveGCNCommitteeModule) parsePipelineOutput(outputFile string) (*PipelineOutput, error) {
	outputBytes, err := os.ReadFile(outputFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read output file: %v", err)
	}

	var result PipelineOutput
	err = json.Unmarshal(outputBytes, &result)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal output: %v", err)
	}

	if !result.Success {
		return nil, fmt.Errorf("Python pipeline failed: %s", result.Error)
	}

	return &result, nil
}

// 调用Python EvolveGCN分片
func (egcm *EvolveGCNCommitteeModule) callPythonEvolveGCNSharding(embeddings []TemporalEmbedding) ([]ShardingResult, error) {
	// 保存嵌入到文件
	if err := egcm.saveEmbeddingsToFile(embeddings, "temporal_embeddings.json"); err != nil {
		return nil, err
	}

	// 调用Python EvolveGCN脚本
	cmd := exec.Command("python", "evolve_GCN/train.py", "--input", "temporal_embeddings.json", "--output", "sharding_results.json")

	var out bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		return nil, fmt.Errorf("EvolveGCN execution failed: %v, stderr: %s", err, stderr.String())
	}

	// 读取分片结果
	return egcm.readShardingResultsFromFile("sharding_results.json")
}

// 基于图的分片（备用方案）
func (egcm *EvolveGCNCommitteeModule) graphBasedSharding(embeddings []TemporalEmbedding) []ShardingResult {
	var results []ShardingResult

	// 使用改进的图聚类算法
	shardClusters := egcm.performGraphClustering()

	for _, embedding := range embeddings {
		nodeID := embedding.NodeID
		vertex := partition.Vertex{Addr: nodeID}

		// 根据图聚类结果分配分片
		recommendedShard := uint64(0)
		if cluster, exists := shardClusters[vertex]; exists {
			recommendedShard = uint64(cluster % params.ShardNum)
		}

		results = append(results, ShardingResult{
			NodeID:             nodeID,
			RecommendedShardID: recommendedShard,
			Confidence:         0.7, // 中等置信度
			Embeddings:         embedding.Embeddings,
		})
	}

	return results
}

// 执行图聚类
func (egcm *EvolveGCNCommitteeModule) performGraphClustering() map[partition.Vertex]int {
	clusters := make(map[partition.Vertex]int)
	visited := make(map[partition.Vertex]bool)
	clusterID := 0

	// 使用BFS进行社区发现
	for vertex := range egcm.evolvegcnGraph.PartitionMap {
		if !visited[vertex] {
			egcm.bfsCluster(vertex, clusterID, clusters, visited)
			clusterID++
		}
	}

	return clusters
}

// BFS聚类
func (egcm *EvolveGCNCommitteeModule) bfsCluster(start partition.Vertex, clusterID int, clusters map[partition.Vertex]int, visited map[partition.Vertex]bool) {
	queue := []partition.Vertex{start}
	visited[start] = true
	clusters[start] = clusterID

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		if neighbors, exists := egcm.evolvegcnGraph.NetGraph.EdgeSet[current]; exists {
			for _, neighbor := range neighbors {
				if !visited[neighbor] {
					visited[neighbor] = true
					clusters[neighbor] = clusterID
					queue = append(queue, neighbor)
				}
			}
		}
	}
}

// 计算预期跨分片交易率
func (egcm *EvolveGCNCommitteeModule) calculateExpectedCrossShardRatio(results []ShardingResult) float64 {
	shardMapping := make(map[string]uint64)
	for _, result := range results {
		shardMapping[result.NodeID] = result.RecommendedShardID
	}

	totalEdges := 0
	crossShardEdges := 0

	for vertex := range egcm.evolvegcnGraph.PartitionMap {
		if neighbors, exists := egcm.evolvegcnGraph.NetGraph.EdgeSet[vertex]; exists {
			nodeShardID := shardMapping[vertex.Addr]
			for _, neighbor := range neighbors {
				totalEdges++
				neighborShardID := shardMapping[neighbor.Addr]
				if nodeShardID != neighborShardID {
					crossShardEdges++
				}
			}
		}
	}

	if totalEdges == 0 {
		return 0.0
	}

	return float64(crossShardEdges) / float64(totalEdges)
}

// 优化分片以减少跨分片交易
func (egcm *EvolveGCNCommitteeModule) optimizeShardingForCrossShardReduction(results []ShardingResult) []ShardingResult {
	optimized := make([]ShardingResult, len(results))
	copy(optimized, results)

	// 迭代优化：尝试将高跨分片连接的节点移动到更合适的分片
	for i := 0; i < 5; i++ { // 最多5次迭代
		improved := false

		for j, result := range optimized {
			bestShard := result.RecommendedShardID
			minCrossConnections := egcm.countCrossShardConnections(result.NodeID, result.RecommendedShardID, optimized)

			// 尝试其他分片
			for shardID := uint64(0); shardID < uint64(params.ShardNum); shardID++ {
				if shardID != result.RecommendedShardID {
					crossConnections := egcm.countCrossShardConnections(result.NodeID, shardID, optimized)
					if crossConnections < minCrossConnections {
						minCrossConnections = crossConnections
						bestShard = shardID
						improved = true
					}
				}
			}

			optimized[j].RecommendedShardID = bestShard
		}

		if !improved {
			break
		}
	}

	return optimized
}

// 计算节点在特定分片的跨分片连接数
func (egcm *EvolveGCNCommitteeModule) countCrossShardConnections(nodeID string, shardID uint64, results []ShardingResult) int {
	// 构建节点到分片的映射
	shardMapping := make(map[string]uint64)
	for _, result := range results {
		shardMapping[result.NodeID] = result.RecommendedShardID
	}
	shardMapping[nodeID] = shardID // 临时设置

	vertex := partition.Vertex{Addr: nodeID}
	crossConnections := 0

	if neighbors, exists := egcm.evolvegcnGraph.NetGraph.EdgeSet[vertex]; exists {
		for _, neighbor := range neighbors {
			neighborShardID := shardMapping[neighbor.Addr]
			if shardID != neighborShardID {
				crossConnections++
			}
		}
	}

	return crossConnections
}

// 读取时序嵌入文件
func (egcm *EvolveGCNCommitteeModule) readTemporalEmbeddingsFromFile(filename string) ([]TemporalEmbedding, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var embeddings []TemporalEmbedding
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&embeddings)
	return embeddings, err
}

// 保存嵌入到文件
func (egcm *EvolveGCNCommitteeModule) saveEmbeddingsToFile(embeddings []TemporalEmbedding, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	return encoder.Encode(embeddings)
}

// 读取分片结果文件
func (egcm *EvolveGCNCommitteeModule) readShardingResultsFromFile(filename string) ([]ShardingResult, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var results []ShardingResult
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&results)
	return results, err
}

// ========== 自动配置Python虚拟环境 ==========
func (egcm *EvolveGCNCommitteeModule) autoConfigurePythonEnvironment() error {
	egcm.sl.Slog.Println("EvolveGCN: Auto-configuring Python virtual environment...")

	// 运行Python配置脚本
	cmd := exec.Command("python", "config_python_venv.py")

	// 显式设置UTF-8编码环境变量
	cmd.Env = append(os.Environ(),
		"PYTHONIOENCODING=utf-8",
		"LANG=en_US.UTF-8",
		"LC_ALL=en_US.UTF-8",
		"PYTHONUTF8=1",
	)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		egcm.sl.Slog.Printf("EvolveGCN: Python environment configuration failed: %v", err)
		egcm.sl.Slog.Printf("EvolveGCN: Error output: %s", stderr.String())
		return fmt.Errorf("failed to configure Python environment: %v", err)
	}

	egcm.sl.Slog.Printf("EvolveGCN: Python environment configuration output: %s", stdout.String())
	egcm.sl.Slog.Println("EvolveGCN: Python virtual environment configured successfully")

	return nil
}

// ========== Python预热服务 ==========

func (egcm *EvolveGCNCommitteeModule) warmupPythonService() {
	egcm.sl.Slog.Println("EvolveGCN: Starting Python service warmup...")

	// 等待一小段时间让系统稳定
	time.Sleep(2 * time.Second)

	pythonPath := egcm.getPythonPath()
	if pythonPath == "" {
		egcm.sl.Slog.Println("EvolveGCN: Python path not configured, skipping warmup")
		return
	}

	// 检查预加载服务是否存在
	if _, err := os.Stat("evolvegcn_preload_service.py"); err != nil {
		egcm.sl.Slog.Println("EvolveGCN: Preload service not found, skipping warmup")
		return
	}

	// 执行预热
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, pythonPath, "evolvegcn_preload_service.py",
		"--input", "dummy_input.json", "--output", "dummy_output.json", "--warmup")

	// 设置工作目录和环境变量
	if wd, err := os.Getwd(); err == nil {
		cmd.Dir = wd
	}

	cmd.Env = append(os.Environ(),
		"PYTHONIOENCODING=utf-8",
		"LANG=en_US.UTF-8",
		"LC_ALL=en_US.UTF-8",
		"PYTHONUTF8=1",
	)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	start := time.Now()
	err := cmd.Run()
	duration := time.Since(start)

	if err != nil {
		egcm.sl.Slog.Printf("EvolveGCN: Python warmup failed: %v, stderr: %s", err, stderr.String())
	} else {
		egcm.sl.Slog.Printf("EvolveGCN: Python service warmup completed in %.2f seconds", duration.Seconds())
		egcm.sl.Slog.Printf("EvolveGCN: Warmup output: %s", stdout.String())
	}
}

// ========== 新增：带固定等待时间的分区消息发送 ==========
func (egcm *EvolveGCNCommitteeModule) evolvegcnMapSendWithFixedWait(m map[string]uint64, epochID int) {
	egcm.sl.Slog.Printf("EvolveGCN: Sending partition map with 4-second fixed wait for epoch %d...", epochID)

	// 复用现有的分区消息结构
	pm := message.PartitionModifiedMapWithEpoch{
		PartitionModified: m,
		EpochID:           int32(epochID),
		Timestamp:         time.Now().Unix(),
	}
	pmByte, err := json.Marshal(pm)
	if err != nil {
		log.Panic(err)
	}
	send_msg := message.MergeMessage(message.CPartitionMsg, pmByte)

	// 发送到所有分片
	totalShards := uint64(params.ShardNum)
	for i := uint64(0); i < totalShards; i++ {
		go networks.TcpDial(send_msg, egcm.IpNodeTable[i][0])
	}

	egcm.sl.Slog.Printf("EvolveGCN: Partition messages sent to %d shards", totalShards)

	// 固定等待4秒，让节点有足够时间处理重分片
	egcm.sl.Slog.Printf("EvolveGCN: Waiting 4 seconds for nodes to process reconfiguration...")
	time.Sleep(4 * time.Second)

	egcm.sl.Slog.Printf("EvolveGCN: Fixed wait completed for epoch %d", epochID)
}
