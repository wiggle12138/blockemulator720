// Supervisor is an abstract role in this simulator that may read txs, generate partition infos,
// and handle history data.

package supervisor

import (
	"blockEmulator/message"
	"blockEmulator/networks"
	"blockEmulator/params"
	"blockEmulator/supervisor/committee"
	"blockEmulator/supervisor/measure"
	"blockEmulator/supervisor/signal"
	"blockEmulator/supervisor/supervisor_log"
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"strconv"
	"sync"
	"time"
)

type Supervisor struct {
	// basic infos
	IPaddr       string // ip address of this Supervisor
	ChainConfig  *params.ChainConfig
	Ip_nodeTable map[uint64]map[uint64]string

	// tcp control
	listenStop bool
	tcpLn      net.Listener
	tcpLock    sync.Mutex
	// logger module
	sl *supervisor_log.SupervisorLog

	// control components
	Ss *signal.StopSignal // to control the stop message sending

	// supervisor and committee components
	comMod committee.CommitteeModule

	// measure components
	testMeasureMods []measure.MeasureModule

	// 节点特征收集相关
	nodeFeaturesMod    *measure.NodeFeaturesModule
	collectionCounter  int
	collectionInterval int // 每收到多少个区块信息后触发一次收集

	// 新增：节点数据收集确认
	reportWg           sync.WaitGroup
	reportReceived     map[string]bool // "shardID-nodeID" -> true
	reportReceivedLock sync.Mutex
	allReportTimeout   int64 // 超时秒数

	// diy, add more structures or classes here ...
}

func (d *Supervisor) NewSupervisor(ip string, pcc *params.ChainConfig, committeeMethod string, measureModNames ...string) {
	d.IPaddr = ip
	d.ChainConfig = pcc
	d.Ip_nodeTable = params.IPmap_nodeTable

	d.sl = supervisor_log.NewSupervisorLog()

	d.Ss = signal.NewStopSignal(3 * int(pcc.ShardNums))

	switch committeeMethod {
	case "CLPA_Broker":
		d.comMod = committee.NewCLPACommitteeMod_Broker(d.Ip_nodeTable, d.Ss, d.sl, params.DatasetFile, params.TotalDataSize, params.TxBatchSize, params.ReconfigTimeGap)
	case "CLPA":
		d.comMod = committee.NewCLPACommitteeModule(d.Ip_nodeTable, d.Ss, d.sl, params.DatasetFile, params.TotalDataSize, params.TxBatchSize, params.ReconfigTimeGap)
	case "EvolveGCN":
		d.comMod = committee.NewEvolveGCNCommitteeModule(d.Ip_nodeTable, d.Ss, d.sl, params.DatasetFile, params.TotalDataSize, params.TxBatchSize, params.ReconfigTimeGap)
		// 新增：设置节点状态收集器引用到EvolveGCN模块
		if evolveGCNMod, ok := d.comMod.(*committee.EvolveGCNCommitteeModule); ok {
			evolveGCNMod.SetNodeStateCollector(d)
		}
	case "Broker":
		d.comMod = committee.NewBrokerCommitteeMod(d.Ip_nodeTable, d.Ss, d.sl, params.DatasetFile, params.TotalDataSize, params.TxBatchSize)
	default:
		d.comMod = committee.NewRelayCommitteeModule(d.Ip_nodeTable, d.Ss, d.sl, params.DatasetFile, params.TotalDataSize, params.TxBatchSize)
	}

	// 只创建一个NodeFeaturesModule实例
	d.nodeFeaturesMod = measure.NewNodeFeaturesModule()
	d.testMeasureMods = make([]measure.MeasureModule, 0)
	for _, mModName := range measureModNames {
		switch mModName {
		case "TPS_Relay":
			d.testMeasureMods = append(d.testMeasureMods, measure.NewTestModule_avgTPS_Relay())
		case "TPS_Broker":
			d.testMeasureMods = append(d.testMeasureMods, measure.NewTestModule_avgTPS_Broker())
		case "TPS_EvolveGCN":
			// 新增：EvolveGCN使用标准TPS测量模块
			d.testMeasureMods = append(d.testMeasureMods, measure.NewTestModule_avgTPS_EvolveGCN())
		case "TCL_Relay":
			d.testMeasureMods = append(d.testMeasureMods, measure.NewTestModule_TCL_Relay())
		case "TCL_Broker":
			d.testMeasureMods = append(d.testMeasureMods, measure.NewTestModule_TCL_Broker())
		case "CrossTxRate_Relay":
			d.testMeasureMods = append(d.testMeasureMods, measure.NewTestCrossTxRate_Relay())
		case "CrossTxRate_Broker":
			d.testMeasureMods = append(d.testMeasureMods, measure.NewTestCrossTxRate_Broker())
		case "CrossTxRate_EvolveGCN":
			// 新增：EvolveGCN使用标准跨分片交易率测量模块
			d.testMeasureMods = append(d.testMeasureMods, measure.NewTestCrossTxRate_EvolveGCN())
		case "TxNumberCount_Relay":
			d.testMeasureMods = append(d.testMeasureMods, measure.NewTestTxNumCount_Relay())
		case "TxNumberCount_Broker":
			d.testMeasureMods = append(d.testMeasureMods, measure.NewTestTxNumCount_Broker())
		case "Tx_Details":
			d.testMeasureMods = append(d.testMeasureMods, measure.NewTestTxDetail())
		case "Node_Features":
			// 只将唯一实例加入testMeasureMods
			d.testMeasureMods = append(d.testMeasureMods, d.nodeFeaturesMod)
		default:
		}
	}

	// 删除多余的 d.nodeFeaturesMod = measure.NewNodeFeaturesModule()

	d.collectionCounter = 0
	d.collectionInterval = 10

	d.reportReceived = make(map[string]bool)
	d.allReportTimeout = 30
	totalNodes := int(pcc.ShardNums * pcc.Nodes_perShard)
	d.reportWg = sync.WaitGroup{}
	d.reportWg.Add(totalNodes)
}

// Supervisor received the block information from the leaders, and handle these
// message to measure the performances.
func (d *Supervisor) handleBlockInfos(content []byte) {
	bim := new(message.BlockInfoMsg)
	err := json.Unmarshal(content, bim)
	if err != nil {
		log.Panic()
	}
	// StopSignal check
	if bim.BlockBodyLength == 0 {
		d.Ss.StopGap_Inc()
	} else {
		d.Ss.StopGap_Reset()
	}

	d.comMod.HandleBlockInfo(bim)

	// measure update
	for _, measureMod := range d.testMeasureMods {
		measureMod.UpdateMeasureRecord(bim)
	}

	// 更新节点特征收集模块
	d.nodeFeaturesMod.UpdateMeasureRecord(bim)

	// // 检查是否需要触发节点状态收集
	// d.collectionCounter++
	// if d.collectionCounter >= d.collectionInterval {
	// 	d.triggerNodeStateCollection()
	// 	//d.collectionCounter = 0 // 重置计数器
	// }
}

// read transactions from dataFile. When the number of data is enough,
// the Supervisor will do re-partition and send partitionMSG and txs to leaders.
func (d *Supervisor) SupervisorTxHandling() {
	d.comMod.MsgSendingControl()
	// TxHandling is end
	for !d.Ss.GapEnough() { // wait all txs to be handled
		time.Sleep(time.Second)
	}

	// ========== 新增：实验终态前主动触发一次节点特征收集 ==========
	d.sl.Slog.Println("Supervisor: triggering node state collection before stop&collect...")
	d.TriggerNodeStateCollection()
	// 可选：等待节点响应（如1~2秒，或更长，视网络情况）
	time.Sleep(2 * time.Second)

	// send stop and collect message
	stopmsg := message.MergeMessage(message.CStopAndCollect, []byte("please collect and report all states"))
	d.sl.Slog.Println("Supervisor: now sending cstop_and_collect message to all nodes")
	for sid := uint64(0); sid < d.ChainConfig.ShardNums; sid++ {
		for nid := uint64(0); nid < d.ChainConfig.Nodes_perShard; nid++ {
			networks.TcpDial(stopmsg, d.Ip_nodeTable[sid][nid])
		}
	}
	// 修复：延长等待时间，确保节点有足够时间进行ping等真实数据收集
	time.Sleep(time.Duration(params.Delay+params.JitterRange+10) * time.Second)

	totalNodes := int(d.ChainConfig.ShardNums * d.ChainConfig.Nodes_perShard)
	d.sl.Slog.Printf("Supervisor: waiting for %d nodes to report features...", totalNodes)
	// 阻塞等待所有节点上报或超时
	d.waitAllNodeReports()

	// ========== 新增：收集全局统计指标并注入到nodeFeaturesMod ==========
	globalMetrics := make(map[string]string)
	for _, mod := range d.testMeasureMods {
		name := mod.OutputMetricName()
		// 这里只处理wrs.md中需要的全局指标，复用measure模块服务于我们的异构特征数据
		if name == "Average_TPS" {
			_, totalTPS := mod.OutputRecord()
			globalMetrics["OnChainBehavior.TransactionCapability.AvgTPS"] = strconv.FormatFloat(totalTPS, 'f', 2, 64)
		}
		if name == "CrossTransaction_ratio" {
			_, totCTXratio := mod.OutputRecord()
			globalMetrics["OnChainBehavior.TransactionTypes.NormalTxRatio"] = strconv.FormatFloat(totCTXratio, 'f', 3, 64)
		}
		if name == "Transaction_Confirm_Latency" {
			_, totLatency := mod.OutputRecord()
			globalMetrics["OnChainBehavior.TransactionCapability.ConfirmationDelay"] = strconv.FormatFloat(totLatency, 'f', 2, 64) + "s"
		}
		// 你可以在这里继续添加其他measure模块与wrs.md字段的映射
	}
	d.nodeFeaturesMod.SetGlobalMetrics(globalMetrics)
	// ========== 生成node_features.csv ==========
	d.nodeFeaturesMod.OutputRecord()

	d.sl.Slog.Println("Supervisor: now Closing")
	d.listenStop = true
	d.CloseSupervisor()
}

// 新增：等待所有节点上报或超时
func (d *Supervisor) waitAllNodeReports() {
	timeout := time.After(time.Duration(d.allReportTimeout) * time.Second)
	ch := make(chan struct{})
	go func() {
		d.reportWg.Wait()
		close(ch)
	}()
	select {
	case <-ch:
		d.sl.Slog.Println("All node reports received.")
		// 打印所有收到的节点
		d.reportReceivedLock.Lock()
		var received []string
		for k := range d.reportReceived {
			received = append(received, k)
		}
		d.sl.Slog.Printf("Received reports from nodes: %v", received)
		d.reportReceivedLock.Unlock()
	case <-timeout:
		d.sl.Slog.Println("Timeout waiting for all node reports!")
		// 打印已收到和未收到的节点
		d.reportReceivedLock.Lock()
		var received, missing []string
		for sid := uint64(0); sid < d.ChainConfig.ShardNums; sid++ {
			for nid := uint64(0); nid < d.ChainConfig.Nodes_perShard; nid++ {
				key := fmt.Sprintf("%d-%d", sid, nid)
				if d.reportReceived[key] {
					received = append(received, key)
				} else {
					missing = append(missing, key)
				}
			}
		}
		d.sl.Slog.Printf("Received reports from nodes: %v", received)
		d.sl.Slog.Printf("Missing reports from nodes: %v", missing)
		d.reportReceivedLock.Unlock()
	}
}

// 触发节点状态收集 - 使用确认机制，替代固定等待时间
func (d *Supervisor) TriggerNodeStateCollection() {
	d.sl.Slog.Println("Triggering node state collection with confirmation mechanism...")

	// 重置确认状态
	d.reportReceivedLock.Lock()
	d.reportReceived = make(map[string]bool)
	totalNodes := int(d.ChainConfig.ShardNums * d.ChainConfig.Nodes_perShard)
	d.reportWg = sync.WaitGroup{}
	d.reportWg.Add(totalNodes)
	d.reportReceivedLock.Unlock()

	// 发送收集请求到所有节点
	d.sendStateRequestsToAllNodes()

	// 等待所有节点确认完成
	d.waitForCollectionConfirmation()

	d.sl.Slog.Println("All nodes confirmed state collection completed")
}

// 等待收集确认 - 替代固定等待时间
func (d *Supervisor) waitForCollectionConfirmation() {
	d.sl.Slog.Println("Waiting for collection confirmation from all nodes...")

	timeout := time.After(time.Duration(d.allReportTimeout) * time.Second)
	ch := make(chan struct{})

	go func() {
		d.reportWg.Wait()
		close(ch)
	}()

	select {
	case <-ch:
		d.sl.Slog.Println("All nodes confirmed collection completion")
	case <-timeout:
		d.sl.Slog.Printf("Collection confirmation timeout after %d seconds", d.allReportTimeout)
		// 打印已确认和未确认的节点
		d.reportReceivedLock.Lock()
		var confirmed, missing []string
		for sid := uint64(0); sid < d.ChainConfig.ShardNums; sid++ {
			for nid := uint64(0); nid < d.ChainConfig.Nodes_perShard; nid++ {
				key := fmt.Sprintf("%d-%d", sid, nid)
				if d.reportReceived[key] {
					confirmed = append(confirmed, key)
				} else {
					missing = append(missing, key)
				}
			}
		}
		d.sl.Slog.Printf("Confirmed nodes: %v", confirmed)
		d.sl.Slog.Printf("Missing confirmations: %v", missing)
		d.reportReceivedLock.Unlock()
	}
}

// 内部工具函数：发送请求到所有节点
func (d *Supervisor) sendStateRequestsToAllNodes() {
	// 创建请求消息
	requestMsg := message.RequestNodeStateMsg{
		Timestamp: time.Now().UnixMilli(),
		RequestID: fmt.Sprintf("req_%d", time.Now().UnixNano()),
	}

	requestBytes, err := json.Marshal(requestMsg)
	if err != nil {
		d.sl.Slog.Printf("Error marshaling request message: %v\n", err)
		return
	}

	// 向所有节点发送状态请求
	msg := message.MergeMessage(message.CRequestNodeState, requestBytes)
	totalNodes := d.ChainConfig.ShardNums * d.ChainConfig.Nodes_perShard
	sentCount := 0

	for shardID := uint64(0); shardID < d.ChainConfig.ShardNums; shardID++ {
		for nodeID := uint64(0); nodeID < d.ChainConfig.Nodes_perShard; nodeID++ {
			nodeIP := d.Ip_nodeTable[shardID][nodeID]
			networks.TcpDial(msg, nodeIP)
			sentCount++
		}
	}

	d.sl.Slog.Printf("Sent node state requests to %d/%d nodes", sentCount, totalNodes)
}

// 内部工具函数：等待节点响应
func (d *Supervisor) waitForNodeResponses() {
	d.sl.Slog.Println("Waiting for nodes to process state collection requests...")
	time.Sleep(5 * time.Second) // 给节点足够时间收集数据
}

// handle message. only one message to be handled now
func (d *Supervisor) handleMessage(msg []byte) {
	msgType, content := message.SplitMessage(msg)
	switch msgType {
	case message.CBlockInfo:
		d.handleBlockInfos(content)
	case message.CReplyNodeState:
		// 处理节点状态回复
		d.nodeFeaturesMod.HandleExtraMessage(msg)
	case message.CBatchReplyNodeState:
		// 处理批量节点状态回复
		d.nodeFeaturesMod.HandleExtraMessage(msg)
		// 新增：确认节点上报
		var batch message.BatchReplyNodeStateMsg
		if err := json.Unmarshal(content, &batch); err == nil {
			key := fmt.Sprintf("%d-%d", batch.ShardID, batch.NodeID)
			d.reportReceivedLock.Lock()
			if !d.reportReceived[key] {
				d.reportReceived[key] = true
				d.sl.Slog.Printf("Received node report from %s", key)
				d.reportWg.Done()
			}
			d.reportReceivedLock.Unlock()
		}
	default:
		d.comMod.HandleOtherMessage(msg)
		for _, mm := range d.testMeasureMods {
			mm.HandleExtraMessage(msg)
		}
	}
}

func (d *Supervisor) handleClientRequest(con net.Conn) {
	defer con.Close()
	clientReader := bufio.NewReader(con)
	for {
		clientRequest, err := clientReader.ReadBytes('\n')
		switch err {
		case nil:
			d.tcpLock.Lock()
			d.handleMessage(clientRequest)
			d.tcpLock.Unlock()
		case io.EOF:
			log.Println("client closed the connection by terminating the process")
			return
		default:
			log.Printf("error: %v\n", err)
			return
		}
	}
}

func (d *Supervisor) TcpListen() {
	ln, err := net.Listen("tcp", d.IPaddr)
	if err != nil {
		log.Panic(err)
	}
	d.tcpLn = ln
	for {
		conn, err := d.tcpLn.Accept()
		if err != nil {
			return
		}
		go d.handleClientRequest(conn)
	}
}

// close Supervisor, and record the data in .csv file
func (d *Supervisor) CloseSupervisor() {
	d.sl.Slog.Println("Closing...")
	for _, measureMod := range d.testMeasureMods {
		d.sl.Slog.Println(measureMod.OutputMetricName())
		d.sl.Slog.Println(measureMod.OutputRecord())
		println()
	}

	// 输出节点特征数据
	d.sl.Slog.Println("Outputting node features data...")
	d.sl.Slog.Println(d.nodeFeaturesMod.OutputMetricName())
	d.sl.Slog.Println(d.nodeFeaturesMod.OutputRecord())

	networks.CloseAllConnInPool()
	d.tcpLn.Close()
}
