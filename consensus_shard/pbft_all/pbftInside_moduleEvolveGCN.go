package pbft_all

import (
	"blockEmulator/consensus_shard/pbft_all/dataSupport"
	"blockEmulator/core"
	"blockEmulator/message"
	"blockEmulator/networks"
	"blockEmulator/params"
	"encoding/json"
	"fmt"
	"log"
	"strconv"
	"time"
)

// EvolveGCN Inside Module - 基于 CLPA 但增加数据收集功能
type EvolveGCNPbftInsideExtraHandleMod struct {
	pbftNode *PbftConsensusNode
	cdm      *dataSupport.Data_supportCLPA
}

func (eihm *EvolveGCNPbftInsideExtraHandleMod) HandleinPropose() (bool, *message.Request) {
	eihm.pbftNode.pl.Plog.Println("EvolveGCN: HandleinPropose")
	if eihm.cdm.PartitionOn {
		// EvolveGCN 分区重配置逻辑（基于 CLPA）
		eihm.pbftNode.pl.Plog.Println("EvolveGCN: ready to partition")

		// 继续原有的分区逻辑
		eihm.sendPartitionReady()
		for !eihm.getPartitionReady() {
			time.Sleep(time.Second)
		}
		// send accounts and txs
		eihm.sendAccounts_and_Txs()
		// propose a partition
		for !eihm.getCollectOver() {
			time.Sleep(time.Second)
		}
		return eihm.proposePartition()
	}

	// ELSE: propose a block
	block := eihm.pbftNode.CurChain.GenerateBlock(int32(eihm.pbftNode.NodeID))
	r := &message.Request{
		RequestType: message.BlockRequest,
		ReqTime:     time.Now(),
	}
	r.Msg.Content = block.Encode()
	return true, r
}

func (eihm *EvolveGCNPbftInsideExtraHandleMod) HandleinPrePrepare(ppmsg *message.PrePrepare) bool {
	eihm.pbftNode.pl.Plog.Printf("EvolveGCN: HandleinPrePrepare")

	// 记录处理开始时间
	startTime := eihm.pbftNode.nodeFeatureCollector.RecordBlockProcessingStart()

	// judge whether it is a partitionRequest or not
	isPartitionReq := ppmsg.RequestMsg.RequestType == message.PartitionReq

	if isPartitionReq {
		// after some checking
		eihm.pbftNode.pl.Plog.Printf("S%dN%d : a partition block\n", eihm.pbftNode.ShardID, eihm.pbftNode.NodeID)
	} else {
		// the request is a block
		if eihm.pbftNode.CurChain.IsValidBlock(core.DecodeB(ppmsg.RequestMsg.Msg.Content)) != nil {
			eihm.pbftNode.pl.Plog.Printf("S%dN%d : not a valid block\n", eihm.pbftNode.ShardID, eihm.pbftNode.NodeID)
			return false
		}

		// 记录处理结束时间
		block := core.DecodeB(ppmsg.RequestMsg.Msg.Content)
		eihm.pbftNode.nodeFeatureCollector.RecordBlockProcessingEnd(startTime, len(block.Body))

		// 更新队列大小
		eihm.pbftNode.nodeFeatureCollector.UpdatePoolSizes(
			len(eihm.pbftNode.CurChain.Txpool.TxQueue),
			len(eihm.pbftNode.requestPool),
		)
	}
	eihm.pbftNode.pl.Plog.Printf("S%dN%d : the pre-prepare message is correct, putting it into the RequestPool. \n", eihm.pbftNode.ShardID, eihm.pbftNode.NodeID)
	eihm.pbftNode.requestPool[string(ppmsg.Digest)] = ppmsg.RequestMsg
	return true
}

func (eihm *EvolveGCNPbftInsideExtraHandleMod) HandleinPrepare(pmsg *message.Prepare) bool {
	fmt.Println("EvolveGCN: No operations are performed in Extra handle mod")
	return true
}

func (eihm *EvolveGCNPbftInsideExtraHandleMod) HandleinCommit(cmsg *message.Commit) bool {
	eihm.pbftNode.pl.Plog.Printf("EvolveGCN: HandleinCommit")
	r := eihm.pbftNode.requestPool[string(cmsg.Digest)]

	// requestType ...
	if r.RequestType == message.PartitionReq {
		// 执行账户迁移
		eihm.pbftNode.pl.Plog.Printf("收到PartitionReq请求，处理账户迁移")
		atm := message.DecodeAccountTransferMsg(r.Msg.Content)
		eihm.accountTransfer_do(atm)

		// 关键修复：发送正确的epoch信息
		// 使用更新后的AccountTransferRound作为epoch
		currentEpoch := int(eihm.cdm.AccountTransferRound)

		bim := message.BlockInfoMsg{
			BlockBodyLength: 0, // 分区请求没有普通交易
			InnerShardTxs:   make([]*core.Transaction, 0),
			Epoch:           currentEpoch, // 使用正确的epoch
			Relay1Txs:       make([]*core.Transaction, 0),
			Relay2Txs:       make([]*core.Transaction, 0),
			SenderShardID:   eihm.pbftNode.ShardID,
			ProposeTime:     r.ReqTime,
			CommitTime:      time.Now(),
		}

		bByte, err := json.Marshal(bim)
		if err != nil {
			log.Panic()
		}
		msg_send := message.MergeMessage(message.CBlockInfo, bByte)
		go networks.TcpDial(msg_send, eihm.pbftNode.ip_nodeTable[params.SupervisorShard][0])

		eihm.pbftNode.pl.Plog.Printf("节点重配置反馈发送 epoch %d from shard %d",
			currentEpoch, eihm.pbftNode.ShardID)

		return true
	}

	// if a block request ...
	block := core.DecodeB(r.Msg.Content)
	eihm.pbftNode.pl.Plog.Printf("S%dN%d : adding the block %d...now height = %d \n", eihm.pbftNode.ShardID, eihm.pbftNode.NodeID, block.Header.Number, eihm.pbftNode.CurChain.CurrentBlock.Header.Number)
	eihm.pbftNode.CurChain.AddBlock(block)

	// 记录区块提交时间戳和统计交易
	eihm.pbftNode.recordBlockCommit(block)

	eihm.pbftNode.pl.Plog.Printf("S%dN%d : added the block %d... \n", eihm.pbftNode.ShardID, eihm.pbftNode.NodeID, block.Header.Number)
	eihm.pbftNode.CurChain.PrintBlockChain()

	// now try to relay txs to other shards (for main nodes)
	if eihm.pbftNode.NodeID == uint64(eihm.pbftNode.view.Load()) {
		eihm.pbftNode.pl.Plog.Printf("S%dN%d : main node is trying to send relay txs at height = %d \n", eihm.pbftNode.ShardID, eihm.pbftNode.NodeID, block.Header.Number)
		// generate relay pool and collect txs excuted
		eihm.pbftNode.CurChain.Txpool.RelayPool = make(map[uint64][]*core.Transaction)
		interShardTxs := make([]*core.Transaction, 0)
		relay1Txs := make([]*core.Transaction, 0)
		relay2Txs := make([]*core.Transaction, 0)

		for _, tx := range block.Body {
			ssid := eihm.pbftNode.CurChain.Get_PartitionMap(tx.Sender)
			rsid := eihm.pbftNode.CurChain.Get_PartitionMap(tx.Recipient)
			if !tx.Relayed && ssid != eihm.pbftNode.ShardID {
				log.Panic("incorrect tx")
			}
			if tx.Relayed && rsid != eihm.pbftNode.ShardID {
				log.Panic("incorrect tx")
			}
			if rsid != eihm.pbftNode.ShardID {
				relay1Txs = append(relay1Txs, tx)
				tx.Relayed = true
				eihm.pbftNode.CurChain.Txpool.AddRelayTx(tx, rsid)
			} else {
				if tx.Relayed {
					relay2Txs = append(relay2Txs, tx)
				} else {
					interShardTxs = append(interShardTxs, tx)
				}
			}
		}

		// send relay txs
		if params.RelayWithMerkleProof == 1 {
			eihm.pbftNode.RelayWithProofSend(block)
		} else {
			eihm.pbftNode.RelayMsgSend()
		}

		// send txs excuted in this block to the listener
		// add more message to measure more metrics
		bim := message.BlockInfoMsg{
			BlockBodyLength: len(block.Body),
			InnerShardTxs:   interShardTxs,
			Epoch:           int(eihm.cdm.AccountTransferRound),

			Relay1Txs: relay1Txs,
			Relay2Txs: relay2Txs,

			SenderShardID: eihm.pbftNode.ShardID,
			ProposeTime:   r.ReqTime,
			CommitTime:    time.Now(),
		}
		bByte, err := json.Marshal(bim)
		if err != nil {
			log.Panic()
		}
		msg_send := message.MergeMessage(message.CBlockInfo, bByte)
		go networks.TcpDial(msg_send, eihm.pbftNode.ip_nodeTable[params.SupervisorShard][0])
		eihm.pbftNode.pl.Plog.Printf("S%dN%d : sended excuted txs\n", eihm.pbftNode.ShardID, eihm.pbftNode.NodeID)

		eihm.pbftNode.CurChain.Txpool.GetLocked()

		metricName := []string{
			"Block Height",
			"EpochID of this block",
			"TxPool Size",
			"# of all Txs in this block",
			"# of Relay1 Txs in this block",
			"# of Relay2 Txs in this block",
			"TimeStamp - Propose (unixMill)",
			"TimeStamp - Commit (unixMill)",

			"SUM of confirm latency (ms, All Txs)",
			"SUM of confirm latency (ms, Relay1 Txs) (Duration: Relay1 proposed -> Relay1 Commit)",
			"SUM of confirm latency (ms, Relay2 Txs) (Duration: Relay1 proposed -> Relay2 Commit)",
		}
		metricVal := []string{
			strconv.Itoa(int(block.Header.Number)),
			strconv.Itoa(bim.Epoch),
			strconv.Itoa(len(eihm.pbftNode.CurChain.Txpool.TxQueue)),
			strconv.Itoa(len(block.Body)),
			strconv.Itoa(len(relay1Txs)),
			strconv.Itoa(len(relay2Txs)),
			strconv.FormatInt(bim.ProposeTime.UnixMilli(), 10),
			strconv.FormatInt(bim.CommitTime.UnixMilli(), 10),

			strconv.FormatInt(computeTCL(block.Body, bim.CommitTime), 10),
			strconv.FormatInt(computeTCL(relay1Txs, bim.CommitTime), 10),
			strconv.FormatInt(computeTCL(relay2Txs, bim.CommitTime), 10),
		}
		eihm.pbftNode.writeCSVline(metricName, metricVal)
		eihm.pbftNode.CurChain.Txpool.GetUnlocked()
	}
	return true
}

func (eihm *EvolveGCNPbftInsideExtraHandleMod) HandleReqestforOldSeq(*message.RequestOldMessage) bool {
	fmt.Println("EvolveGCN: No operations are performed in Extra handle mod")
	return true
}

// the operation for sequential requests
func (eihm *EvolveGCNPbftInsideExtraHandleMod) HandleforSequentialRequest(som *message.SendOldMessage) bool {
	if int(som.SeqEndHeight-som.SeqStartHeight+1) != len(som.OldRequest) {
		eihm.pbftNode.pl.Plog.Printf("S%dN%d : the SendOldMessage message is not enough\n", eihm.pbftNode.ShardID, eihm.pbftNode.NodeID)
	} else { // add the block into the node pbft blockchain
		for height := som.SeqStartHeight; height <= som.SeqEndHeight; height++ {
			r := som.OldRequest[height-som.SeqStartHeight]
			if r.RequestType == message.BlockRequest {
				b := core.DecodeB(r.Msg.Content)
				eihm.pbftNode.CurChain.AddBlock(b)
			} else {
				atm := message.DecodeAccountTransferMsg(r.Msg.Content)
				eihm.accountTransfer_do(atm)
			}
		}
		eihm.pbftNode.sequenceID = som.SeqEndHeight + 1
		eihm.pbftNode.CurChain.PrintBlockChain()
	}
	return true
}

// 复用 CLPA 的分区相关方法
func (eihm *EvolveGCNPbftInsideExtraHandleMod) sendPartitionReady() {
	eihm.cdm.P_ReadyLock.Lock()
	eihm.cdm.PartitionReady[eihm.pbftNode.ShardID] = true
	eihm.cdm.P_ReadyLock.Unlock()

	pr := message.PartitionReady{
		FromShard: eihm.pbftNode.ShardID,
		NowSeqID:  eihm.pbftNode.sequenceID,
	}
	pByte, err := json.Marshal(pr)
	if err != nil {
		log.Panic()
	}
	send_msg := message.MergeMessage(message.CPartitionReady, pByte)
	for sid := 0; sid < int(eihm.pbftNode.pbftChainConfig.ShardNums); sid++ {
		if sid != int(pr.FromShard) {
			go networks.TcpDial(send_msg, eihm.pbftNode.ip_nodeTable[uint64(sid)][0])
		}
	}
	eihm.pbftNode.pl.Plog.Print("EvolveGCN: Ready for partition\n")
}

func (eihm *EvolveGCNPbftInsideExtraHandleMod) getPartitionReady() bool {
	eihm.cdm.P_ReadyLock.Lock()
	defer eihm.cdm.P_ReadyLock.Unlock()
	eihm.pbftNode.seqMapLock.Lock()
	defer eihm.pbftNode.seqMapLock.Unlock()
	eihm.cdm.ReadySeqLock.Lock()
	defer eihm.cdm.ReadySeqLock.Unlock()

	flag := true
	for sid, val := range eihm.pbftNode.seqIDMap {
		if rval, ok := eihm.cdm.ReadySeq[sid]; !ok || (rval-1 != val) {
			flag = false
		}
	}
	return len(eihm.cdm.PartitionReady) == int(eihm.pbftNode.pbftChainConfig.ShardNums) && flag
}

func (eihm *EvolveGCNPbftInsideExtraHandleMod) sendAccounts_and_Txs() {
	// generate accout transfer and txs message
	accountToFetch := make([]string, 0)
	lastMapid := len(eihm.cdm.ModifiedMap) - 1
	for key, val := range eihm.cdm.ModifiedMap[lastMapid] {
		if val != eihm.pbftNode.ShardID && eihm.pbftNode.CurChain.Get_PartitionMap(key) == eihm.pbftNode.ShardID {
			accountToFetch = append(accountToFetch, key)
		}
	}
	asFetched := eihm.pbftNode.CurChain.FetchAccounts(accountToFetch)
	// send the accounts to other shards
	eihm.pbftNode.CurChain.Txpool.GetLocked()
	eihm.pbftNode.pl.Plog.Println("EvolveGCN: The size of tx pool is: ", len(eihm.pbftNode.CurChain.Txpool.TxQueue))
	for i := uint64(0); i < eihm.pbftNode.pbftChainConfig.ShardNums; i++ {
		if i == eihm.pbftNode.ShardID {
			continue
		}
		addrSend := make([]string, 0)
		addrSet := make(map[string]bool)
		asSend := make([]*core.AccountState, 0)
		for idx, addr := range accountToFetch {
			if eihm.cdm.ModifiedMap[lastMapid][addr] == i {
				addrSend = append(addrSend, addr)
				addrSet[addr] = true
				asSend = append(asSend, asFetched[idx])
			}
		}
		// fetch transactions to it, after the transactions is fetched, delete it in the pool
		txSend := make([]*core.Transaction, 0)
		firstPtr := 0
		for secondPtr := 0; secondPtr < len(eihm.pbftNode.CurChain.Txpool.TxQueue); secondPtr++ {
			ptx := eihm.pbftNode.CurChain.Txpool.TxQueue[secondPtr]
			// if this is a normal transaction or ctx1 before re-sharding && the addr is correspond
			_, ok1 := addrSet[ptx.Sender]
			condition1 := ok1 && !ptx.Relayed
			// if this tx is ctx2
			_, ok2 := addrSet[ptx.Recipient]
			condition2 := ok2 && ptx.Relayed
			if condition1 || condition2 {
				txSend = append(txSend, ptx)
			} else {
				eihm.pbftNode.CurChain.Txpool.TxQueue[firstPtr] = ptx
				firstPtr++
			}
		}
		eihm.pbftNode.CurChain.Txpool.TxQueue = eihm.pbftNode.CurChain.Txpool.TxQueue[:firstPtr]

		eihm.pbftNode.pl.Plog.Printf("EvolveGCN: The txSend to shard %d is generated \n", i)
		ast := message.AccountStateAndTx{
			Addrs:        addrSend,
			AccountState: asSend,
			FromShard:    eihm.pbftNode.ShardID,
			Txs:          txSend,
		}
		aByte, err := json.Marshal(ast)
		if err != nil {
			log.Panic()
		}
		send_msg := message.MergeMessage(message.AccountState_and_TX, aByte)
		networks.TcpDial(send_msg, eihm.pbftNode.ip_nodeTable[i][0])
		eihm.pbftNode.pl.Plog.Printf("EvolveGCN: The message to shard %d is sent\n", i)
	}
	eihm.pbftNode.pl.Plog.Println("EvolveGCN: after sending, The size of tx pool is: ", len(eihm.pbftNode.CurChain.Txpool.TxQueue))
	eihm.pbftNode.CurChain.Txpool.GetUnlocked()
}

func (eihm *EvolveGCNPbftInsideExtraHandleMod) getCollectOver() bool {
	eihm.cdm.CollectLock.Lock()
	defer eihm.cdm.CollectLock.Unlock()
	return eihm.cdm.CollectOver
}

func (eihm *EvolveGCNPbftInsideExtraHandleMod) proposePartition() (bool, *message.Request) {
	eihm.pbftNode.pl.Plog.Printf("EvolveGCN S%dN%d : begin partition proposing\n", eihm.pbftNode.ShardID, eihm.pbftNode.NodeID)
	// add all data in pool into the set
	for _, at := range eihm.cdm.AccountStateTx {
		for i, addr := range at.Addrs {
			eihm.cdm.ReceivedNewAccountState[addr] = at.AccountState[i]
		}
		eihm.cdm.ReceivedNewTx = append(eihm.cdm.ReceivedNewTx, at.Txs...)
	}
	// propose, send all txs to other nodes in shard
	eihm.pbftNode.pl.Plog.Println("EvolveGCN: The number of ReceivedNewTx: ", len(eihm.cdm.ReceivedNewTx))
	for _, tx := range eihm.cdm.ReceivedNewTx {
		if !tx.Relayed && eihm.cdm.ModifiedMap[eihm.cdm.AccountTransferRound][tx.Sender] != eihm.pbftNode.ShardID {
			log.Panic("EvolveGCN: error tx")
		}
		if tx.Relayed && eihm.cdm.ModifiedMap[eihm.cdm.AccountTransferRound][tx.Recipient] != eihm.pbftNode.ShardID {
			log.Panic("EvolveGCN: error tx")
		}
	}
	eihm.pbftNode.CurChain.Txpool.AddTxs2Pool(eihm.cdm.ReceivedNewTx)
	eihm.pbftNode.pl.Plog.Println("EvolveGCN: The size of txpool: ", len(eihm.pbftNode.CurChain.Txpool.TxQueue))

	atmaddr := make([]string, 0)
	atmAs := make([]*core.AccountState, 0)
	for key, val := range eihm.cdm.ReceivedNewAccountState {
		atmaddr = append(atmaddr, key)
		atmAs = append(atmAs, val)
	}
	atm := message.AccountTransferMsg{
		ModifiedMap:  eihm.cdm.ModifiedMap[eihm.cdm.AccountTransferRound],
		Addrs:        atmaddr,
		AccountState: atmAs,
		ATid:         uint64(len(eihm.cdm.ModifiedMap)),
	}
	atmbyte := atm.Encode()
	r := &message.Request{
		RequestType: message.PartitionReq,
		Msg: message.RawMessage{
			Content: atmbyte,
		},
		ReqTime: time.Now(),
	}
	return true, r
}

func (eihm *EvolveGCNPbftInsideExtraHandleMod) accountTransfer_do(atm *message.AccountTransferMsg) {
	// change the partition Map
	cnt := 0
	for key, val := range atm.ModifiedMap {
		cnt++
		eihm.pbftNode.CurChain.Update_PartitionMap(key, val)
	}
	eihm.pbftNode.pl.Plog.Printf("EvolveGCN: %d key-vals are updated\n", cnt)

	// add the account into the state trie
	eihm.pbftNode.pl.Plog.Printf("EvolveGCN: %d addrs to add\n", len(atm.Addrs))
	eihm.pbftNode.pl.Plog.Printf("EvolveGCN: %d accountstates to add\n", len(atm.AccountState))
	eihm.pbftNode.CurChain.AddAccounts(atm.Addrs, atm.AccountState, eihm.pbftNode.view.Load())

	// 关键修复：确保正确更新AccountTransferRound
	previousRound := eihm.cdm.AccountTransferRound
	if uint64(len(eihm.cdm.ModifiedMap)) != atm.ATid {
		eihm.cdm.ModifiedMap = append(eihm.cdm.ModifiedMap, atm.ModifiedMap)
	}
	eihm.cdm.AccountTransferRound = atm.ATid

	eihm.pbftNode.pl.Plog.Printf("AccountTransferRound成功updated from %d to %d",
		previousRound, eihm.cdm.AccountTransferRound)

	eihm.cdm.AccountStateTx = make(map[uint64]*message.AccountStateAndTx)
	eihm.cdm.ReceivedNewAccountState = make(map[string]*core.AccountState)
	eihm.cdm.ReceivedNewTx = make([]*core.Transaction, 0)
	eihm.cdm.PartitionOn = false

	eihm.cdm.CollectLock.Lock()
	eihm.cdm.CollectOver = false
	eihm.cdm.CollectLock.Unlock()

	eihm.cdm.P_ReadyLock.Lock()
	eihm.cdm.PartitionReady = make(map[uint64]bool)
	eihm.cdm.P_ReadyLock.Unlock()

	eihm.pbftNode.CurChain.PrintBlockChain()
}
