// The pbft consensus process

package pbft_all

import (
	"blockEmulator/chain"
	"blockEmulator/consensus_shard/pbft_all/dataSupport"
	"blockEmulator/consensus_shard/pbft_all/node_features" // 节点特征收集器
	"blockEmulator/consensus_shard/pbft_all/pbft_log"
	"blockEmulator/core"
	"blockEmulator/message"
	"blockEmulator/networks"
	"blockEmulator/params"
	"blockEmulator/shard"
	"bufio"
	"encoding/json"
	"io"
	"log"
	"net"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ethereum/go-ethereum/core/rawdb"
	"github.com/ethereum/go-ethereum/ethdb"
)

type PbftConsensusNode struct {
	// the local config about pbft
	RunningNode *shard.Node // the node information

	// the global config about this pbft(read-only)
	ShardID uint64 // denote the ID of the shard (or pbft), only one pbft consensus in a shard
	NodeID  uint64 // denote the ID of the node in the pbft (shard)

	// the data structure for blockchain
	CurChain *chain.BlockChain // all node in the shard maintain the same blockchain
	db       ethdb.Database    // to save the mpt

	// the global config about pbft
	pbftChainConfig *params.ChainConfig          // the chain config in this pbft
	ip_nodeTable    map[uint64]map[uint64]string // denote the ip of the specific node
	node_nums       uint64                       // the number of nodes in this pfbt, denoted by N
	malicious_nums  uint64                       // f, 3f + 1 = N

	// view change
	view           atomic.Int32 // denote the view of this pbft, the main node can be inferred from this variant
	lastCommitTime atomic.Int64 // the time since last commit.

	viewChangeMap map[ViewChangeData]map[uint64]bool
	newViewMap    map[ViewChangeData]map[uint64]bool

	// the control message and message checking utils in pbft
	sequenceID        uint64                          // the message sequence id of the pbft
	stopSignal        atomic.Bool                     // send stop signal
	pStop             chan uint64                     // channle for stopping consensus
	requestPool       map[string]*message.Request     // RequestHash to Request
	cntPrepareConfirm map[string]map[*shard.Node]bool // count the prepare confirm message, [messageHash][Node]bool
	cntCommitConfirm  map[string]map[*shard.Node]bool // count the commit confirm message, [messageHash][Node]bool
	isCommitBordcast  map[string]bool                 // denote whether the commit is broadcast
	isReply           map[string]bool                 // denote whether the message is reply
	height2Digest     map[uint64]string               // sequence (block height) -> request, fast read

	// pbft stage wait
	pbftStage              atomic.Int32 // 1->Preprepare, 2->Prepare, 3->Commit, 4->Done
	pbftLock               sync.Mutex
	conditionalVarpbftLock sync.Cond

	// locks about pbft
	sequenceLock sync.Mutex // the lock of sequence
	lock         sync.Mutex // lock the stage
	askForLock   sync.Mutex // lock for asking for a serise of requests

	// seqID of other Shards, to synchronize
	seqIDMap   map[uint64]uint64
	seqMapLock sync.Mutex

	// logger
	pl *pbft_log.PbftLog
	// tcp control
	tcpln       net.Listener
	tcpPoolLock sync.Mutex

	// to handle the message in the pbft
	ihm ExtraOpInConsensus

	// to handle the message outside of pbft
	ohm OpInterShards

	// 节点特征收集器
	nodeFeatureCollector *node_features.NodeFeatureCollector
}

// generate a pbft consensus for a node
func NewPbftNode(shardID, nodeID uint64, pcc *params.ChainConfig, messageHandleType string) *PbftConsensusNode {
	p := new(PbftConsensusNode)
	p.ip_nodeTable = params.IPmap_nodeTable
	p.node_nums = pcc.Nodes_perShard
	p.ShardID = shardID
	p.NodeID = nodeID
	p.pbftChainConfig = pcc
	fp := params.DatabaseWrite_path + "mptDB/ldb/s" + strconv.FormatUint(shardID, 10) + "/n" + strconv.FormatUint(nodeID, 10)
	var err error
	p.db, err = rawdb.NewLevelDBDatabase(fp, 0, 1, "accountState", false)
	if err != nil {
		log.Panic(err)
	}
	p.CurChain, err = chain.NewBlockChain(pcc, p.db)
	if err != nil {
		log.Panic("cannot new a blockchain")
	}

	p.RunningNode = &shard.Node{
		NodeID:  nodeID,
		ShardID: shardID,
		IPaddr:  p.ip_nodeTable[shardID][nodeID],
	}

	p.stopSignal.Store(false)
	p.sequenceID = p.CurChain.CurrentBlock.Header.Number + 1
	p.pStop = make(chan uint64)
	p.requestPool = make(map[string]*message.Request)
	p.cntPrepareConfirm = make(map[string]map[*shard.Node]bool)
	p.cntCommitConfirm = make(map[string]map[*shard.Node]bool)
	p.isCommitBordcast = make(map[string]bool)
	p.isReply = make(map[string]bool)
	p.height2Digest = make(map[uint64]string)
	p.malicious_nums = (p.node_nums - 1) / 3

	// init view & last commit time
	p.view.Store(0)
	p.lastCommitTime.Store(time.Now().Add(time.Second * 5).UnixMilli())
	p.viewChangeMap = make(map[ViewChangeData]map[uint64]bool)
	p.newViewMap = make(map[ViewChangeData]map[uint64]bool)

	p.seqIDMap = make(map[uint64]uint64)

	p.pl = pbft_log.NewPbftLog(shardID, nodeID)

	// choose how to handle the messages in pbft or beyond pbft
	//创建内部共识模块ihm和外部共识模块ohm
	switch string(messageHandleType) {
	case "CLPA_Broker":
		ncdm := dataSupport.NewCLPADataSupport()
		p.ihm = &CLPAPbftInsideExtraHandleMod_forBroker{
			pbftNode: p,
			cdm:      ncdm,
		}
		p.ohm = &CLPABrokerOutsideModule{
			pbftNode: p,
			cdm:      ncdm,
		}
	case "CLPA":
		ncdm := dataSupport.NewCLPADataSupport()
		p.ihm = &CLPAPbftInsideExtraHandleMod{
			pbftNode: p,
			cdm:      ncdm,
		}
		p.ohm = &CLPARelayOutsideModule{
			pbftNode: p,
			cdm:      ncdm,
		}
	case "EvolveGCN":
		ncdm := dataSupport.NewCLPADataSupport()
		p.ihm = &EvolveGCNPbftInsideExtraHandleMod{
			pbftNode: p,
			cdm:      ncdm,
		}
		p.ohm = &EvolveGCNRelayOutsideModule{
			pbftNode: p,
			cdm:      ncdm,
		}
	case "Broker":
		p.ihm = &RawBrokerPbftExtraHandleMod{
			pbftNode: p,
		}
		p.ohm = &RawBrokerOutsideModule{
			pbftNode: p,
		}
	default:
		p.ihm = &RawRelayPbftExtraHandleMod{
			pbftNode: p,
		}
		p.ohm = &RawRelayOutsideModule{
			pbftNode: p,
		}
	}

	// set pbft stage now
	p.conditionalVarpbftLock = *sync.NewCond(&p.pbftLock)
	p.pbftStage.Store(1)

	// 初始化节点特征收集器
	p.nodeFeatureCollector = node_features.NewNodeFeatureCollector(
		shardID, nodeID, p.node_nums, p.pbftChainConfig.ShardNums)

	return p
}

// handle the raw message, send it to corresponded interfaces
func (p *PbftConsensusNode) handleMessage(msg []byte) {
	msgType, content := message.SplitMessage(msg)

	// 更新共识统计（在消息处理前）
	p.nodeFeatureCollector.UpdateConsensusStats(msgType)

	switch msgType {
	// pbft inside message type
	case message.CPrePrepare:
		// use "go" to start a go routine to handle this message, so that a pre-arrival message will not be aborted.
		go p.handlePrePrepare(content)
	case message.CPrepare:
		// use "go" to start a go routine to handle this message, so that a pre-arrival message will not be aborted.
		go p.handlePrepare(content)
	case message.CCommit:
		// use "go" to start a go routine to handle this message, so that a pre-arrival message will not be aborted.
		go p.handleCommit(content)

	case message.ViewChangePropose:
		p.handleViewChangeMsg(content)
	case message.NewChange:
		p.handleNewViewMsg(content)

	case message.CRequestOldrequest:
		p.handleRequestOldSeq(content)
	case message.CSendOldrequest:
		p.handleSendOldSeq(content)

	case message.CRequestNodeState:
		// 处理节点状态请求
		go p.handleRequestNodeState(content)
	case message.CStopAndCollect:
		go p.handleStopAndCollect()

	case message.CStop:
		p.WaitToStop()

	// handle the message from outside
	default:
		go p.ohm.HandleMessageOutsidePBFT(msgType, content)
	}
}

func (p *PbftConsensusNode) handleClientRequest(con net.Conn) {
	defer con.Close()
	clientReader := bufio.NewReader(con)
	for {
		clientRequest, err := clientReader.ReadBytes('\n')
		if p.stopSignal.Load() {
			return
		}
		switch err {
		case nil:
			p.tcpPoolLock.Lock()
			p.handleMessage(clientRequest)
			p.tcpPoolLock.Unlock()
		case io.EOF:
			log.Println("client closed the connection by terminating the process")
			return
		default:
			log.Printf("error: %v\n", err)
			return
		}
	}
}

// A consensus node starts tcp-listen.
func (p *PbftConsensusNode) TcpListen() {
	ln, err := net.Listen("tcp", p.RunningNode.IPaddr)
	p.tcpln = ln
	if err != nil {
		log.Panic(err)
	}
	for {
		conn, err := p.tcpln.Accept()
		if err != nil {
			return
		}
		go p.handleClientRequest(conn)
	}
}

// When receiving a stop message, this node try to stop.
func (p *PbftConsensusNode) WaitToStop() {
	p.pl.Plog.Println("handling stop message")
	p.stopSignal.Store(true)
	networks.CloseAllConnInPool()
	p.tcpln.Close()
	p.closePbft()
	p.pl.Plog.Println("handled stop message in TCPListen Routine")
	p.pStop <- 1
}

// close the pbft
func (p *PbftConsensusNode) closePbft() {
	p.CurChain.CloseBlockChain()
}

// 处理节点状态请求 - 修复：添加数据发送逻辑
func (p *PbftConsensusNode) handleRequestNodeState(_ []byte) {
	p.pl.Plog.Printf("EvolveGCN S%dN%d: received node state collection request", p.ShardID, p.NodeID)

	// 触发数据收集
	p.nodeFeatureCollector.HandleRequestNodeState()

	// 等待一小段时间确保收集完成
	time.Sleep(100 * time.Millisecond)

	// 获取收集到的状态数据
	collectedStates := p.nodeFeatureCollector.GetCollectedStates()

	p.pl.Plog.Printf("EvolveGCN S%dN%d: collected %d state snapshots for feature collection",
		p.ShardID, p.NodeID, len(collectedStates))

	// 发送批量上报消息到Supervisor
	if len(collectedStates) > 0 {
		batch := message.BatchReplyNodeStateMsg{
			ShardID: p.ShardID,
			NodeID:  p.NodeID,
			States:  collectedStates,
		}
		data, err := json.Marshal(batch)
		if err != nil {
			p.pl.Plog.Printf("EvolveGCN S%dN%d: error marshaling batch message: %v", p.ShardID, p.NodeID, err)
			return
		}

		msg := message.MergeMessage(message.CBatchReplyNodeState, data)

		// 异步发送数据到supervisor
		go func() {
			networks.TcpDial(msg, params.SupervisorAddr)
			p.pl.Plog.Printf("EvolveGCN S%dN%d: feature data sent to supervisor", p.ShardID, p.NodeID)
		}()
	} else {
		p.pl.Plog.Printf("EvolveGCN S%dN%d: no state data collected for features", p.ShardID, p.NodeID)
	}
}

func (p *PbftConsensusNode) handleStopAndCollect() {
	p.pl.Plog.Printf("S%dN%d : handling stop and collect message", p.ShardID, p.NodeID)

	// 移除重复的数据收集，因为Supervisor已经通过CRequestNodeState触发过了
	// p.nodeFeatureCollector.HandleRequestNodeState() // 删除这行

	// 等待一小段时间确保之前的采集完成
	time.Sleep(100 * time.Millisecond)

	// 停止收集器
	p.nodeFeatureCollector.StopCollector()

	// 获取所有收集到的状态数据
	collectedStates := p.nodeFeatureCollector.GetCollectedStates()

	p.pl.Plog.Printf("S%dN%d : collected %d state snapshots, preparing to send batch",
		p.ShardID, p.NodeID, len(collectedStates))

	// 如果有收集到的数据，则发送批量上报消息
	if len(collectedStates) > 0 {
		batch := message.BatchReplyNodeStateMsg{
			ShardID: p.ShardID,
			NodeID:  p.NodeID,
			States:  collectedStates,
		}
		data, err := json.Marshal(batch)
		if err != nil {
			p.pl.Plog.Printf("S%dN%d : error marshaling batch message: %v", p.ShardID, p.NodeID, err)
		} else {
			msg := message.MergeMessage(message.CBatchReplyNodeState, data)

			// 同步发送数据，确保发送完成后再关闭节点
			var wg sync.WaitGroup
			wg.Add(1)

			// 发送数据到supervisor
			go networks.TcpDialAndWait(msg, params.SupervisorAddr, &wg)

			// 等待发送完成，最多等待5秒
			done := make(chan struct{})
			go func() {
				wg.Wait()
				close(done)
			}()

			select {
			case <-done:
				p.pl.Plog.Printf("S%dN%d : batch data sent successfully", p.ShardID, p.NodeID)
			case <-time.After(5 * time.Second):
				p.pl.Plog.Printf("S%dN%d : timeout waiting for batch data to be sent", p.ShardID, p.NodeID)
			}
		}
	} else {
		p.pl.Plog.Printf("S%dN%d : no state data collected", p.ShardID, p.NodeID)
	}

	// 最后停止节点
	p.WaitToStop()
}

// 在区块提交时记录区块时间戳和统计交易
func (p *PbftConsensusNode) recordBlockCommit(block *core.Block) {
	p.nodeFeatureCollector.RecordBlockCommit(block)
}
