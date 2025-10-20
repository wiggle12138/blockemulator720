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
	"runtime/debug"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// 定义接口以避免循环引用
type NodeStateCollector interface {
	TriggerNodeStateCollection()
}

// 新增：NodeFeaturesModule接口，用于获取真实收集的节点数据
type NodeFeaturesModule interface {
	GetLatestNodeStates() map[string]interface{}
	GetEpochData(epoch int) ([]message.ReplyNodeStateMsg, bool)
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

	// 新增：节点特征模块接口引用
	nodeFeatureModule NodeFeaturesModule
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
		nodeFeatureModule:        nil, // 将在supervisor中设置
	}

	// 异步启动Python预热
	go egcm.warmupPythonService()

	return egcm
}

// 新增：设置节点状态收集器的方法
func (egcm *EvolveGCNCommitteeModule) SetNodeStateCollector(collector NodeStateCollector) {
	egcm.nodeStateCollector = collector
}

// 新增：设置节点特征模块的方法
func (egcm *EvolveGCNCommitteeModule) SetNodeFeaturesModule(module NodeFeaturesModule) {
	egcm.nodeFeatureModule = module
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
			egcm.sl.Slog.Printf("读取到文件末尾EOF，结束MsgSendingControl")
			break
		}
		if err != nil {
			egcm.sl.Slog.Printf("MsgSendingControl读取文件遇到panic")
			log.Panic(err)
		}
		if tx, ok := data2tx(data, uint64(egcm.nowDataNum)); ok {
			txlist = append(txlist, tx)
			//egcm.sl.Slog.Printf("txlist增加交易")
			egcm.nowDataNum++
		} else {
			continue
		}

		// 批量发送交易
		if len(txlist) == int(egcm.batchDataNum) || egcm.nowDataNum == egcm.dataTotalNum {
			egcm.sl.Slog.Printf("到了batchDataNum或dataTotalNum, 开始发送交易")
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

			egcm.sl.Slog.Printf("===============================================")
			egcm.sl.Slog.Printf(" EvolveGCN Epoch %d 开始执行重分片算法", evolvegcnCnt)
			egcm.sl.Slog.Printf("===============================================")

			// 执行重分片流程
			egcm.sl.Slog.Printf("EvolveGCN Epoch %d: Step 1 - 触发节点特征收集...", evolvegcnCnt)
			if egcm.nodeStateCollector != nil {
				egcm.nodeStateCollector.TriggerNodeStateCollection()
				egcm.sl.Slog.Printf("EvolveGCN Epoch %d:  节点特征收集完成", evolvegcnCnt)
			}

			preReconfigCTXRatio := egcm.calculateCurrentCrossShardRatio()
			egcm.sl.Slog.Printf("EvolveGCN Epoch %d:  重分片前跨分片交易率: %.4f (%.2f%%)",
				evolvegcnCnt, preReconfigCTXRatio, preReconfigCTXRatio*100)

			egcm.sl.Slog.Printf("EvolveGCN Epoch %d: Step 3 -  执行EvolveGCN分片算法...", evolvegcnCnt)
			mmap, crossTxNum := egcm.runEvolveGCNPartition()
			egcm.sl.Slog.Printf("EvolveGCN Epoch %d:  分片算法完成，跨分片边数: %d", evolvegcnCnt, crossTxNum)

			postReconfigCTXRatio := egcm.estimatePostReconfigCrossShardRatio(mmap, crossTxNum)
			egcm.sl.Slog.Printf("EvolveGCN Epoch %d:  重分片后预期跨分片率: %.4f (%.2f%%) [改善: %.2f%%]",
				evolvegcnCnt, postReconfigCTXRatio, postReconfigCTXRatio*100,
				(preReconfigCTXRatio-postReconfigCTXRatio)*100)

			egcm.recordReconfigurationMetrics(evolvegcnCnt, preReconfigCTXRatio, postReconfigCTXRatio)

			// ========== 核心修改：直接使用转换后的mmap，移除硬编码示例 =========
			egcm.sl.Slog.Printf("EvolveGCN Epoch %d:  发送分区映射消息到所有分片...", evolvegcnCnt)
			egcm.evolvegcnMapSend(mmap) // 直接使用转换后的真实mmap
			egcm.sl.Slog.Printf("EvolveGCN Epoch %d:  所有分区映射消息已发送完成", evolvegcnCnt)

			// 更新本地分区映射
			egcm.sl.Slog.Printf("EvolveGCN Epoch %d:  更新本地分区映射开始", evolvegcnCnt)
			for key, val := range mmap {
				egcm.modifiedMap[key] = val
			}
			egcm.sl.Slog.Printf("EvolveGCN Epoch %d:  更新本地分区映射完成", evolvegcnCnt)
			egcm.evolvegcnReset()
			egcm.sl.Slog.Printf("EvolveGCN Epoch %d:  evolvegcnReset执行完成", evolvegcnCnt)
			egcm.evolvegcnLock.Unlock()

			// 等待epoch确认 - 参考CLPA的确认机制
			egcm.sl.Slog.Printf("EvolveGCN Epoch %d:  等待所有分片确认epoch更新...", evolvegcnCnt)
			//waitStart := time.Now()
			for atomic.LoadInt32(&egcm.curEpoch) != int32(evolvegcnCnt) {
				time.Sleep(time.Second)
				// if time.Since(waitStart) > 30*time.Second {
				// 	egcm.sl.Slog.Printf("EvolveGCN Epoch %d:   等待超时，强制继续", evolvegcnCnt)
				// 	atomic.StoreInt32(&egcm.curEpoch, int32(evolvegcnCnt))
				// 	break
				// }
			}

			egcm.evolvegcnLastRunningTime = time.Now()
			egcm.sl.Slog.Printf("===============================================")
			egcm.sl.Slog.Printf(" EvolveGCN Epoch %d 重分片完成! 下一轮开始准备...", evolvegcnCnt)
			egcm.sl.Slog.Printf("===============================================")
		}

		if egcm.nowDataNum == egcm.dataTotalNum {
			egcm.sl.Slog.Printf("交易数量达到dataTotalNum，结束MsgSendingControl")
			break
		}
	}

	// all transactions are sent. keep sending partition message...
	for !egcm.Ss.GapEnough() { // wait all txs to be handled
		time.Sleep(time.Second)
		if params.ShardNum > 1 && time.Since(egcm.evolvegcnLastRunningTime) >= time.Duration(egcm.evolvegcnFreq)*time.Second {
			egcm.sl.Slog.Printf("到GapEnough中了,且触发重分片")
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

			for key, val := range mmap {
				egcm.modifiedMap[key] = val
			}
			egcm.evolvegcnReset()
			egcm.evolvegcnLock.Unlock()

			egcm.evolvegcnLastRunningTime = time.Now()
			egcm.sl.Slog.Printf("EvolveGCN Final Epoch %d: Completed", evolvegcnCnt)
		}
	}

	egcm.sl.Slog.Printf("从MsgSendingControl出去了")
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
	egcm.sl.Slog.Println("EvolveGCN Step 1: 特征提取...")
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

	// ========== 新增：节点映射转换为账户映射 ==========
	if len(partitionMap) > 0 {
		// 检查返回的是节点格式还是账户格式
		firstKey := ""
		for k := range partitionMap {
			firstKey = k
			break
		}

		if strings.HasPrefix(firstKey, "S") && strings.Contains(firstKey, "N") {
			// 这是节点-分片映射，需要转换为账户-分片映射
			egcm.sl.Slog.Printf("EvolveGCN: 检测到节点格式映射，开始转换为账户映射...")
			egcm.sl.Slog.Printf("EvolveGCN: 原始节点映射数量: %d", len(partitionMap))

			// ========== 新增：打印partitionMap重映射内容 ==========
			egcm.sl.Slog.Printf("========== partitionMap重映射详细内容 ==========")

			if len(partitionMap) > 0 {
				count := 0
				for key, value := range partitionMap {
					egcm.sl.Slog.Printf("partitionMap[%d] - 键: '%s', 值: %d", count, key, value)
					count++
					if count >= 8 { // 只打印前8条
						egcm.sl.Slog.Printf("... (还有%d条记录)", len(partitionMap)-8)
						break
					}
				}
			}

			// 转换节点映射为账户映射
			accountMapping := egcm.convertNodeMappingToAccountMapping(partitionMap)

			if len(accountMapping) > 0 {
				egcm.sl.Slog.Printf("EvolveGCN: 成功转换为账户映射，数量: %d", len(accountMapping))
				egcm.sl.Slog.Printf("EvolveGCN: ✅ 节点级重分片转换完成")
				return accountMapping, crossShardEdges
			} else {
				egcm.sl.Slog.Println("EvolveGCN: 转换结果为空，使用空映射")
				return make(map[string]uint64), crossShardEdges
			}
		} else {
			// 已经是账户格式，直接返回
			egcm.sl.Slog.Printf("EvolveGCN: 检测到账户格式映射，直接使用")
			return partitionMap, crossShardEdges
		}
	}

	egcm.sl.Slog.Printf("EvolveGCN: Pipeline completed successfully. Cross-shard edges: %d", crossShardEdges)
	egcm.sl.Slog.Println("Real EvolveGCN algorithm active ")
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

	egcm.sl.Slog.Printf("EvolveGCNMapSend函数发送CPartitionMsg完毕 %d 广播到0号节点",
		atomic.LoadInt32(&egcm.curEpoch))
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
	// egcm.sl.Slog.Printf("EvolveGCN Supervisor: received from shard %d in epoch %d, blockLength=%d\n",
	// b.SenderShardID, b.Epoch, b.BlockBodyLength)

	// 关键修复：区分普通区块和分区确认消息
	if b.BlockBodyLength == -1 {
		// 这是分区确认消息（222）
		egcm.sl.Slog.Printf("收到-1长度的重分片确认区块 from shard %d epoch %d",
			b.SenderShardID, b.Epoch)

		// 更新epoch
		egcm.evolvegcnLock.Lock()
		currentEpoch := atomic.LoadInt32(&egcm.curEpoch)
		if int32(b.Epoch) > currentEpoch {
			atomic.StoreInt32(&egcm.curEpoch, int32(b.Epoch))
			egcm.sl.Slog.Printf("EPOCH UPDATED from 老的%d to 接收到的%d (PARTITION CONFIRMED by shard %d)",
				currentEpoch, b.Epoch, b.SenderShardID)
		} else {
			egcm.sl.Slog.Printf("epoch received %d and current is %d 无需更新", b.Epoch, currentEpoch)
		}
		egcm.evolvegcnLock.Unlock()
	} else {
		// 这是普通区块信息，处理交易图构建
		egcm.sl.Slog.Printf("收到普通区块信息 with %d transactions from shard %d epoch %d",
			b.BlockBodyLength, b.SenderShardID, b.Epoch)

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

// 第一步：特征提取 - 修改为使用真实收集的节点数据
func (egcm *EvolveGCNCommitteeModule) extractNodeFeatures() ([]NodeFeatureData, error) {
	egcm.sl.Slog.Println("EvolveGCN Step 1: 提取真实特征开始...")

	var nodeFeatures []NodeFeatureData

	// 优先尝试从NodeFeaturesModule获取真实收集的数据
	if egcm.nodeFeatureModule != nil {
		egcm.sl.Slog.Println("真实特征存在 NodeFeaturesModule")

		// 获取最新收集的节点状态数据
		egcm.sl.Slog.Println("尝试执行GetLatestNodeStates")

		// 添加完整的错误捕获和调试逻辑
		var latestNodeStates map[string]interface{}
		var getStatesError error

		func() {
			defer func() {
				if r := recover(); r != nil {
					getStatesError = fmt.Errorf("GetLatestNodeStates panic: %v", r)
					egcm.sl.Slog.Printf("[PANIC捕获] GetLatestNodeStates发生panic: %v", r)
					egcm.sl.Slog.Printf("[PANIC捕获] 调用栈信息: %s", debug.Stack())
				}
			}()

			egcm.sl.Slog.Println("[调试] 开始调用GetLatestNodeStates方法")

			// 检查nodeFeatureModule是否为nil
			if egcm.nodeFeatureModule == nil {
				getStatesError = fmt.Errorf("nodeFeatureModule is nil")
				egcm.sl.Slog.Println("[错误] nodeFeatureModule为nil")
				return
			}

			egcm.sl.Slog.Println("[调试] nodeFeatureModule不为nil，开始调用GetLatestNodeStates")

			// 调用GetLatestNodeStates方法
			latestNodeStates = egcm.nodeFeatureModule.GetLatestNodeStates()

			egcm.sl.Slog.Printf("[调试] GetLatestNodeStates调用成功，返回数据量: %d", len(latestNodeStates))
		}()

		// 检查是否发生了错误
		if getStatesError != nil {
			egcm.sl.Slog.Printf("[错误处理] GetLatestNodeStates调用失败: %v", getStatesError)
			egcm.sl.Slog.Println("[错误处理] 跳过真实数据处理，准备使用模拟数据")
		} else {
			egcm.sl.Slog.Printf("GetLatestNodeStates执行成功，返回%d个节点数据", len(latestNodeStates))
		}

		if len(latestNodeStates) > 0 {
			egcm.sl.Slog.Printf("找到 %d 个真实节点 NodeFeaturesModule", len(latestNodeStates))

			// 转换真实节点状态数据为EvolveGCN需要的格式
			for nodeKey, stateInterface := range latestNodeStates {
				if nodeState, ok := stateInterface.(message.ReplyNodeStateMsg); ok {
					// 从真实数据中提取静态特征
					staticFeatures := egcm.extractRealStaticFeatures(nodeState)

					// 从真实数据中提取动态特征
					dynamicFeatures := egcm.extractRealDynamicFeatures(nodeState)

					nodeFeatures = append(nodeFeatures, NodeFeatureData{
						NodeID:          nodeKey, // 使用节点键值 "S0N0" 格式
						ShardID:         nodeState.ShardID,
						StaticFeatures:  staticFeatures,
						DynamicFeatures: dynamicFeatures,
					})
				}
			}

			if len(nodeFeatures) > 0 {
				egcm.sl.Slog.Printf("成功提取节点特征 for %d nodes", len(nodeFeatures))

				// 保存真实特征数据到文件供Python处理
				if err := egcm.saveNodeFeaturesToCSV(nodeFeatures); err != nil {
					return nil, fmt.Errorf("failed to save real node features: %v", err)
				}

				return nodeFeatures, nil
			}
		}

		egcm.sl.Slog.Println("没有真实node states found, 回退到模拟数据提取")
	} else {
		egcm.sl.Slog.Println("NodeFeaturesModule 是nil，回退到模拟数据提取")
	}

	// 备用方案：使用模拟数据（保持原有逻辑作为fallback）
	egcm.sl.Slog.Println("回退到模拟数据...")

	// 检查evolvegcnGraph是否已初始化
	if egcm.evolvegcnGraph == nil || egcm.evolvegcnGraph.PartitionMap == nil {
		egcm.sl.Slog.Println("EvolveGCN: Warning - Graph not initialized, using IP node table")

		// 使用IP节点表作为备用方案
		for shardID, nodeMap := range egcm.IpNodeTable {
			for nodeID := range nodeMap {
				nodeIDStr := fmt.Sprintf("S%dN%d", shardID, nodeID)

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

	egcm.sl.Slog.Printf("EvolveGCN Step 1: Extracted simulated features for %d nodes", len(nodeFeatures))
	return nodeFeatures, nil
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

// 调用Python四步完整流水线
func (egcm *EvolveGCNCommitteeModule) callPythonFourStepPipeline(nodeFeatures []NodeFeatureData) (map[string]uint64, int, error) {
	egcm.sl.Slog.Println("EvolveGCN: 准备输入 for Python four-step pipeline...")

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
	// egcm.sl.Slog.Printf("EvolveGCN: Environment variables: %v", cmd.Env)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	// 设置超时
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second) // 2分钟超时
	defer cancel()

	// 选择Python脚本优先级：安全预加载服务 > 预加载服务 > 优化版完整流水线 > 原始完整版 > 快速测试版
	scriptName := "evolvegcn_go_interface.py" // 723优化版本

	egcm.sl.Slog.Printf("EvolveGCN: Using script: %s", scriptName)

	cmdWithTimeout := exec.CommandContext(ctx, pythonPath, scriptName, "--input", inputFile, "--output", outputFile)
	cmdWithTimeout.Dir = cmd.Dir
	cmdWithTimeout.Env = cmd.Env
	cmdWithTimeout.Stdout = &stdout
	cmdWithTimeout.Stderr = &stderr

	err = cmdWithTimeout.Run()

	// 详细记录执行结果
	egcm.sl.Slog.Printf("EvolveGCN: Python pipeline 执行结束")
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

// ========== 改进：从真实节点状态提取静态特征 ==========
func (egcm *EvolveGCNCommitteeModule) extractRealStaticFeatures(nodeState message.ReplyNodeStateMsg) map[string]float64 {
	features := make(map[string]float64)

	// 从真实收集的静态数据中提取硬件特征
	static := nodeState.NodeState.Static

	// CPU特征
	features["cpu_cores"] = float64(static.ResourceCapacity.Hardware.CPU.CoreCount)
	features["cpu_architecture"] = egcm.encodeArchitecture(static.ResourceCapacity.Hardware.CPU.Architecture)

	// 内存特征
	features["memory_gb"] = float64(static.ResourceCapacity.Hardware.Memory.TotalCapacity)
	features["memory_bandwidth"] = static.ResourceCapacity.Hardware.Memory.Bandwidth
	features["memory_type"] = egcm.encodeMemoryType(static.ResourceCapacity.Hardware.Memory.Type)

	// 存储特征
	features["storage_gb"] = float64(static.ResourceCapacity.Hardware.Storage.Capacity)
	features["storage_type"] = egcm.encodeStorageType(static.ResourceCapacity.Hardware.Storage.Type)
	features["storage_rw_speed"] = static.ResourceCapacity.Hardware.Storage.ReadWriteSpeed

	// 网络特征
	features["network_upstream"] = static.ResourceCapacity.Hardware.Network.UpstreamBW
	features["network_downstream"] = static.ResourceCapacity.Hardware.Network.DownstreamBW
	features["network_latency"] = egcm.parseLatency(static.ResourceCapacity.Hardware.Network.Latency)

	// 网络拓扑特征
	features["intra_shard_conn"] = float64(static.NetworkTopology.Connections.IntraShardConn)
	features["inter_shard_conn"] = float64(static.NetworkTopology.Connections.InterShardConn)
	features["weighted_degree"] = static.NetworkTopology.Connections.WeightedDegree
	features["active_conn"] = float64(static.NetworkTopology.Connections.ActiveConn)
	features["adaptability"] = static.NetworkTopology.ShardAllocation.Adaptability

	// 异构类型特征
	features["node_type"] = egcm.encodeNodeType(static.HeterogeneousType.NodeType)
	// 注释掉不存在的字段，使用默认值
	// features["core_eligibility"] = egcm.encodeBool(static.ResourceCapacity.OperationalStatus.CoreEligibility)
	features["core_eligibility"] = 1.0 // 默认符合条件

	egcm.sl.Slog.Printf("EvolveGCN: Extracted %d real static features for node %d", len(features), nodeState.NodeID)

	return features
}

// ========== 改进：从真实节点状态提取动态特征 ==========
func (egcm *EvolveGCNCommitteeModule) extractRealDynamicFeatures(nodeState message.ReplyNodeStateMsg) map[string]float64 {
	features := make(map[string]float64)

	// 从真实收集的动态数据中提取运行时特征
	dynamic := nodeState.NodeState.Dynamic

	// 交易处理能力特征
	features["avg_tps"] = dynamic.OnChainBehavior.TransactionCapability.AvgTPS
	features["confirmation_delay"] = egcm.parseDelay(dynamic.OnChainBehavior.TransactionCapability.ConfirmationDelay)

	// 跨分片交易特征
	features["inter_shard_volume"] = egcm.parseVolumeString(dynamic.OnChainBehavior.TransactionCapability.CrossShardTx.InterShardVolume)
	features["inter_node_volume"] = egcm.parseVolumeString(dynamic.OnChainBehavior.TransactionCapability.CrossShardTx.InterNodeVolume)

	// 区块生成特征
	features["avg_block_interval"] = egcm.parseInterval(dynamic.OnChainBehavior.BlockGeneration.AvgInterval)
	features["block_interval_stddev"] = egcm.parseInterval(dynamic.OnChainBehavior.BlockGeneration.IntervalStdDev)

	// 交易类型特征
	features["normal_tx_ratio"] = dynamic.OnChainBehavior.TransactionTypes.NormalTxRatio
	features["contract_tx_ratio"] = dynamic.OnChainBehavior.TransactionTypes.ContractTxRatio

	// 共识参与特征
	features["participation_rate"] = dynamic.OnChainBehavior.Consensus.ParticipationRate
	features["total_reward"] = dynamic.OnChainBehavior.Consensus.TotalReward
	features["success_rate"] = dynamic.OnChainBehavior.Consensus.SuccessRate

	// 资源使用率特征
	features["cpu_usage"] = dynamic.DynamicAttributes.Compute.CPUUsage
	features["memory_usage"] = dynamic.DynamicAttributes.Compute.MemUsage
	features["resource_flux"] = dynamic.DynamicAttributes.Compute.ResourceFlux

	// 网络动态特征
	features["latency_flux"] = dynamic.DynamicAttributes.Network.LatencyFlux
	features["avg_latency"] = egcm.parseLatency(dynamic.DynamicAttributes.Network.AvgLatency)
	features["bandwidth_usage"] = dynamic.DynamicAttributes.Network.BandwidthUsage

	// 交易处理特征
	features["tx_frequency"] = float64(dynamic.DynamicAttributes.Transactions.Frequency)
	features["processing_delay"] = egcm.parseDelay(dynamic.DynamicAttributes.Transactions.ProcessingDelay)

	// 应用状态特征
	features["application_state"] = egcm.encodeApplicationState(nodeState.NodeState.Static.HeterogeneousType.Application.CurrentState)
	features["tx_frequency_metric"] = float64(nodeState.NodeState.Static.HeterogeneousType.Application.LoadMetrics.TxFrequency)
	features["storage_ops"] = float64(nodeState.NodeState.Static.HeterogeneousType.Application.LoadMetrics.StorageOps)

	egcm.sl.Slog.Printf("EvolveGCN: Extracted %d real dynamic features for node %d", len(features), nodeState.NodeID)

	return features
}

// ========== 辅助编码方法 ==========

// 编码CPU架构
func (egcm *EvolveGCNCommitteeModule) encodeArchitecture(arch string) float64 {
	switch arch {
	case "x86_64", "amd64":
		return 1.0
	case "arm64", "aarch64":
		return 2.0
	case "x86", "i386":
		return 3.0
	default:
		return 0.0
	}
}

// 编码内存类型
func (egcm *EvolveGCNCommitteeModule) encodeMemoryType(memType string) float64 {
	switch memType {
	case "DDR4":
		return 4.0
	case "DDR5":
		return 5.0
	case "DDR3":
		return 3.0
	default:
		return 4.0 // 默认DDR4
	}
}

// 编码存储类型
func (egcm *EvolveGCNCommitteeModule) encodeStorageType(storageType string) float64 {
	switch storageType {
	case "SSD":
		return 2.0
	case "NVMe":
		return 3.0
	case "HDD":
		return 1.0
	default:
		return 2.0 // 默认SSD

	}
}

// 编码节点类型
func (egcm *EvolveGCNCommitteeModule) encodeNodeType(nodeType string) float64 {
	switch nodeType {
	case "full_node":
		return 4.0
	case "miner_node":
		return 3.0
	case "storage_node":
		return 2.0
	case "validate_node":
		return 1.5
	case "light_node":
		return 1.0
	default:
		return 2.0 // 默认值
	}
}

// 编码应用状态
func (egcm *EvolveGCNCommitteeModule) encodeApplicationState(state string) float64 {
	switch state {
	case "active":
		return 3.0
	case "high_load":
		return 4.0
	case "idle":
		return 1.0
	default:
		return 2.0 // 默认值
	}
}

// 解析延迟字符串 (如 "50ms")
func (egcm *EvolveGCNCommitteeModule) parseLatency(latencyStr string) float64 {
	if latencyStr == "" {
		return 50.0 // 默认50ms
	}

	// 简单解析，去掉"ms"后缀
	if len(latencyStr) > 2 && latencyStr[len(latencyStr)-2:] == "ms" {
		if val, err := fmt.Sscanf(latencyStr[:len(latencyStr)-2], "%f"); err == nil && val > 0 {
			return float64(val)
		}
	}

	return 50.0 // 解析失败时的默认值
}

// 解析延迟字符串 (如 "200ms")
func (egcm *EvolveGCNCommitteeModule) parseDelay(delayStr string) float64 {
	return egcm.parseLatency(delayStr) // 复用延迟解析逻辑
}

// 解析时间间隔字符串 (如 "5.0s")
func (egcm *EvolveGCNCommitteeModule) parseInterval(intervalStr string) float64 {
	if intervalStr == "" {
		return 5.0 // 默认5秒
	}

	// 简单解析，去掉"s"后缀
	if len(intervalStr) > 1 && intervalStr[len(intervalStr)-1:] == "s" {
		if val, err := fmt.Sscanf(intervalStr[:len(intervalStr)-1], "%f"); err == nil && val > 0 {
			return float64(val)
		}
	}

	return 5.0 // 解析失败时的默认值
}

// 解析交易量字符串 (如 "shard0:1000;shard1:2000")
func (egcm *EvolveGCNCommitteeModule) parseVolumeString(volumeStr string) float64 {
	if volumeStr == "" {
		return 0.0
	}

	// 简单统计：计算总交易量
	var total float64 = 0.0

	// 按分号分割
	parts := strings.Split(volumeStr, ";")
	for _, part := range parts {
		// 按冒号分割
		if kvPair := strings.Split(part, ":"); len(kvPair) == 2 {
			if val, err := fmt.Sscanf(kvPair[1], "%f"); err == nil && val > 0 {
				total += float64(val)
			}
		}
	}

	return total
}

// ========== 节点映射转换核心功能 ==========

// 将节点-分片映射转换为账户-分片映射的核心函数
func (egcm *EvolveGCNCommitteeModule) convertNodeMappingToAccountMapping(nodeMappings map[string]uint64) map[string]uint64 {
	egcm.sl.Slog.Printf("EvolveGCN: 开始转换节点映射到账户映射，输入节点数: %d", len(nodeMappings))

	// 分析节点投票，决定分片级迁移
	shardMigrations := egcm.analyzeShardMigrations(nodeMappings)

	if len(shardMigrations) == 0 {
		egcm.sl.Slog.Println("EvolveGCN: 没有分片需要迁移，返回空映射")
		return make(map[string]uint64)
	}

	// 将分片级迁移转换为账户级映射
	accountMappings := egcm.generateAccountMappingsFromShardMigrations(shardMigrations)

	egcm.sl.Slog.Printf("EvolveGCN: 转换完成，生成账户映射数: %d", len(accountMappings))
	return accountMappings
}

// 分析节点投票，决定哪些分片需要迁移
func (egcm *EvolveGCNCommitteeModule) analyzeShardMigrations(nodeMappings map[string]uint64) map[uint64]uint64 {
	egcm.sl.Slog.Println("EvolveGCN: 开始分析节点投票...")

	// 统计每个分片的节点投票情况: [原分片][目标分片] = 票数
	shardVotes := make(map[uint64]map[uint64]int)
	shardNodeCounts := make(map[uint64]int) // [原分片] = 总节点数

	for nodeID, targetShard := range nodeMappings {
		if originalShard := egcm.parseShardFromNodeID(nodeID); originalShard != nil {
			if shardVotes[*originalShard] == nil {
				shardVotes[*originalShard] = make(map[uint64]int)
			}
			shardVotes[*originalShard][targetShard]++
			shardNodeCounts[*originalShard]++
		}
	}

	// 决策：超过60%阈值则整个分片迁移
	shardMigrations := make(map[uint64]uint64) // [原分片] -> [目标分片]
	threshold := 0.6                           // 60%阈值

	for originalShard, votes := range shardVotes {
		totalNodes := shardNodeCounts[originalShard]
		if totalNodes == 0 {
			continue
		}

		maxVotes := 0
		var targetShard uint64

		for shard, count := range votes {
			if count > maxVotes {
				maxVotes = count
				targetShard = shard
			}
		}

		// 如果超过阈值且不是迁移到自己，则执行分片迁移
		consensusRatio := float64(maxVotes) / float64(totalNodes)
		if consensusRatio >= threshold && targetShard != originalShard {
			shardMigrations[originalShard] = targetShard
			egcm.sl.Slog.Printf("EvolveGCN: 分片 %d 将迁移到分片 %d (共识度: %.1f%%)",
				originalShard, targetShard, consensusRatio*100)
		}
	}

	egcm.sl.Slog.Printf("EvolveGCN: 分析完成，共有 %d 个分片需要迁移", len(shardMigrations))
	return shardMigrations
}

// 将分片级迁移转换为账户级映射
func (egcm *EvolveGCNCommitteeModule) generateAccountMappingsFromShardMigrations(shardMigrations map[uint64]uint64) map[string]uint64 {
	egcm.sl.Slog.Println("EvolveGCN: 开始生成账户映射...")

	accountMappings := make(map[string]uint64)

	// 为每个需要迁移的分片生成所有账户的重映射
	for originalShard, targetShard := range shardMigrations {
		// 获取该分片的所有账户
		shardAccounts := egcm.getAccountsInShard(originalShard)

		// 生成账户映射
		for _, account := range shardAccounts {
			accountMappings[account] = targetShard
		}

		egcm.sl.Slog.Printf("EvolveGCN: 分片 %d 的 %d 个账户将迁移到分片 %d",
			originalShard, len(shardAccounts), targetShard)
	}

	return accountMappings
}

// 解析节点ID，提取分片信息 (解析 "S0N0" 格式)
func (egcm *EvolveGCNCommitteeModule) parseShardFromNodeID(nodeID string) *uint64 {
	if !strings.HasPrefix(nodeID, "S") || !strings.Contains(nodeID, "N") {
		return nil
	}

	// 提取S和N之间的数字
	parts := strings.Split(nodeID, "N")
	if len(parts) != 2 {
		return nil
	}

	shardStr := strings.TrimPrefix(parts[0], "S")
	if shardID, err := fmt.Sscanf(shardStr, "%d"); err == nil && shardID >= 0 {
		result := uint64(shardID)
		return &result
	}

	return nil
}

// 获取分片中的所有账户
func (egcm *EvolveGCNCommitteeModule) getAccountsInShard(shardID uint64) []string {
	var accounts []string

	// 从现有的evolvegcnGraph获取分片中的账户
	if egcm.evolvegcnGraph != nil && egcm.evolvegcnGraph.PartitionMap != nil {
		for vertex, currentShardID := range egcm.evolvegcnGraph.PartitionMap {
			if uint64(currentShardID) == shardID {
				accounts = append(accounts, vertex.Addr)
			}
		}
	}

	// 如果没有找到账户，生成一些示例账户以确保系统正常运行
	if len(accounts) == 0 {
		egcm.sl.Slog.Printf("EvolveGCN: 分片 %d 没有找到账户，生成示例账户", shardID)
		// 为每个分片生成2个示例账户
		for i := 0; i < 2; i++ {
			account := fmt.Sprintf("0x%016x%016x%08x",
				uint64(shardID)*1000+uint64(i),
				uint64(shardID)*2000+uint64(i),
				uint64(shardID)*100+uint64(i))
			accounts = append(accounts, account)
		}
	}

	return accounts
}
