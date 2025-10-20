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

	// 修复：提前初始化NodeFeaturesModule，确保在设置到EvolveGCN前已初始化
	d.nodeFeaturesMod = measure.NewNodeFeaturesModule()
	d.collectionCounter = 0
	d.collectionInterval = 10
	d.reportReceived = make(map[string]bool)
	d.allReportTimeout = 30
	totalNodes := int(pcc.ShardNums * pcc.Nodes_perShard)
	d.reportWg = sync.WaitGroup{}
	d.reportWg.Add(totalNodes)

	switch committeeMethod {
	case "CLPA_Broker":
		d.comMod = committee.NewCLPACommitteeMod_Broker(d.Ip_nodeTable, d.Ss, d.sl, params.DatasetFile, params.TotalDataSize, params.TxBatchSize, params.ReconfigTimeGap)
	case "CLPA":
		d.comMod = committee.NewCLPACommitteeModule(d.Ip_nodeTable, d.Ss, d.sl, params.DatasetFile, params.TotalDataSize, params.TxBatchSize, params.ReconfigTimeGap)
	case "EvolveGCN":
		d.comMod = committee.NewEvolveGCNCommitteeModule(d.Ip_nodeTable, d.Ss, d.sl, params.DatasetFile, params.TotalDataSize, params.TxBatchSize, params.ReconfigTimeGap)
		// 修复：现在d.nodeFeaturesMod已经初始化，可以安全设置
		if evolveGCNMod, ok := d.comMod.(*committee.EvolveGCNCommitteeModule); ok {
			evolveGCNMod.SetNodeStateCollector(d)
			evolveGCNMod.SetNodeFeaturesModule(d.nodeFeaturesMod)
			d.sl.Slog.Println("EvolveGCN: Successfully configured with real node feature data access")
		}
	case "Broker":
		d.comMod = committee.NewBrokerCommitteeMod(d.Ip_nodeTable, d.Ss, d.sl, params.DatasetFile, params.TotalDataSize, params.TxBatchSize)
	default:
		d.comMod = committee.NewRelayCommitteeModule(d.Ip_nodeTable, d.Ss, d.sl, params.DatasetFile, params.TotalDataSize, params.TxBatchSize)
	}

	// 初始化测量模块列表
	d.testMeasureMods = make([]measure.MeasureModule, 0)
	for _, mModName := range measureModNames {
		switch mModName {
		case "TPS_Relay":
			d.testMeasureMods = append(d.testMeasureMods, measure.NewTestModule_avgTPS_Relay())
		case "TPS_Broker":
			d.testMeasureMods = append(d.testMeasureMods, measure.NewTestModule_avgTPS_Broker())
		case "TPS_EvolveGCN":
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
			d.testMeasureMods = append(d.testMeasureMods, measure.NewTestCrossTxRate_EvolveGCN())
		case "TxNumberCount_Relay":
			d.testMeasureMods = append(d.testMeasureMods, measure.NewTestTxNumCount_Relay())
		case "TxNumberCount_Broker":
			d.testMeasureMods = append(d.testMeasureMods, measure.NewTestTxNumCount_Broker())
		case "Tx_Details":
			d.testMeasureMods = append(d.testMeasureMods, measure.NewTestTxDetail())
		case "Node_Features":
			// 将已初始化的nodeFeaturesMod加入testMeasureMods
			d.testMeasureMods = append(d.testMeasureMods, d.nodeFeaturesMod)
		default:
		}
	}
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
	d.sl.Slog.Println("MsgSendingControl运行结束")
	// TxHandling is end
	for !d.Ss.GapEnough() { // wait all txs to be handled
		time.Sleep(time.Second)
	}

	// send stop message
	stopmsg := message.MergeMessage(message.CStop, []byte("stop"))
	d.sl.Slog.Println("Supervisor: 开始发送CStop关闭系统")
	for sid := uint64(0); sid < d.ChainConfig.ShardNums; sid++ {
		for nid := uint64(0); nid < d.ChainConfig.Nodes_perShard; nid++ {
			networks.TcpDial(stopmsg, d.Ip_nodeTable[sid][nid])
		}
	}
	time.Sleep(time.Duration(params.Delay+params.JitterRange+2) * time.Second)

	// ========== 生成最终测量结果 ==========
	var finalResults []string
	for _, mod := range d.testMeasureMods {
		name := mod.OutputMetricName()
		perEpochData, totalData := mod.OutputRecord()
		finalResults = append(finalResults, fmt.Sprintf("%s: %v, Total: %v", name, perEpochData, totalData))
	}

	d.sl.Slog.Println("Supervisor: 要关了哦")
	d.listenStop = true
	d.CloseSupervisor(finalResults)
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
	d.sl.Slog.Println("开始触发节点状态收集（使用确认机制）...")

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

	d.sl.Slog.Println("所有节点已确认状态收集完成")
}

// 等待收集确认 - 替代固定等待时间
func (d *Supervisor) waitForCollectionConfirmation() {
	d.sl.Slog.Println("等待所有节点确认收集完成...")

	timeout := time.After(time.Duration(d.allReportTimeout) * time.Second)
	ch := make(chan struct{})

	go func() {
		d.reportWg.Wait()
		close(ch)
	}()

	select {
	case <-ch:
		d.sl.Slog.Println("所有节点已确认收集完成")
	case <-timeout:
		d.sl.Slog.Printf("收集确认超时（%d秒后）", d.allReportTimeout)
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
		d.sl.Slog.Printf("已确认节点: %v", confirmed)
		d.sl.Slog.Printf("缺失确认节点: %v", missing)
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
		d.sl.Slog.Printf("序列化请求消息失败: %v\n", err)
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

	d.sl.Slog.Printf("已向 %d/%d 个节点发送状态请求", sentCount, totalNodes)
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
		// 解析消息内容进行基本验证
		var reply message.ReplyNodeStateMsg
		if err := json.Unmarshal(content, &reply); err != nil {
			d.sl.Slog.Printf("[DEBUG] Supervisor: 解析节点状态消息失败: %v\n", err)
			return
		}

		d.sl.Slog.Printf("[DEBUG] Supervisor: 收到节点S%dN%d状态数据，开始传递给NodeFeaturesModule\n",
			reply.ShardID, reply.NodeID)

		// 检查nodeFeaturesMod是否为nil
		if d.nodeFeaturesMod == nil {
			d.sl.Slog.Printf("[ERROR] Supervisor: nodeFeaturesMod为nil，无法处理节点状态数据\n")
			return
		}

		d.sl.Slog.Printf("[DEBUG] Supervisor: nodeFeaturesMod不为nil，开始调用HandleExtraMessage\n")

		// 处理节点状态回复 - 统一使用ReplyNodeStateMsg
		d.nodeFeaturesMod.HandleExtraMessage(msg)

		// d.sl.Slog.Printf("[DEBUG] Supervisor: 节点S%dN%d数据已传递给NodeFeaturesModule处理完成\n",
		// 	reply.ShardID, reply.NodeID)

		// 打印当前存储状态
		collectedCount, _, _ := d.nodeFeaturesMod.GetCollectionStats()
		d.sl.Slog.Printf("[DEBUG] Supervisor: 当前NodeFeaturesModule中存储的数据量: %d\n", collectedCount)

		// var reply message.ReplyNodeStateMsg
		// if err := json.Unmarshal(content, &reply); err == nil {
		// 解析消息获取节点ID并确认
		key := fmt.Sprintf("%d-%d", reply.ShardID, reply.NodeID)
		d.reportReceivedLock.Lock()
		if !d.reportReceived[key] {
			d.reportReceived[key] = true
			d.sl.Slog.Printf("收到节点%s状态上报 (轮次%d)", key, reply.Epoch)
			d.reportWg.Done()
		}
		d.reportReceivedLock.Unlock()
		//}
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
func (d *Supervisor) CloseSupervisor(finalResults []string) {
	d.sl.Slog.Println("Closing...")

	// 修复：不再重复调用OutputRecord，直接输出已计算的结果
	for _, result := range finalResults {
		d.sl.Slog.Println(result)
	}

	// 输出节点特征数据
	d.sl.Slog.Println("Outputting node features data...")
	d.sl.Slog.Println(d.nodeFeaturesMod.OutputMetricName())
	d.sl.Slog.Println(d.nodeFeaturesMod.OutputRecord())

	networks.CloseAllConnInPool()
	d.tcpLn.Close()
}
