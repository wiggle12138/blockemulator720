package measure

import (
	"blockEmulator/message"
	"strconv"
	"time"
)

// EvolveGCN使用标准的TPS统计模块，与Relay方式完全一致
type TestModule_avgTPS_EvolveGCN struct {
	epochID      int
	excutedTxNum []float64 // 每个epoch处理的交易数

	normalTxNum []int // 普通交易数
	relay1TxNum []int // relay1交易数
	relay2TxNum []int // relay2交易数

	startTime []time.Time // epoch开始时间
	endTime   []time.Time // epoch结束时间
}

func NewTestModule_avgTPS_EvolveGCN() *TestModule_avgTPS_EvolveGCN {
	return &TestModule_avgTPS_EvolveGCN{
		epochID:      -1,
		excutedTxNum: make([]float64, 0),
		startTime:    make([]time.Time, 0),
		endTime:      make([]time.Time, 0),

		normalTxNum: make([]int, 0),
		relay1TxNum: make([]int, 0),
		relay2TxNum: make([]int, 0),
	}
}

func (tat *TestModule_avgTPS_EvolveGCN) OutputMetricName() string {
	return "Average_TPS"
}

// 更新每个区块的测量记录 - 与Relay方式完全一致
func (tat *TestModule_avgTPS_EvolveGCN) UpdateMeasureRecord(b *message.BlockInfoMsg) {
	if b.BlockBodyLength == 0 { // 空区块
		return
	}

	epochid := b.Epoch
	earliestTime := b.ProposeTime
	latestTime := b.CommitTime
	r1TxNum := len(b.Relay1Txs)
	r2TxNum := len(b.Relay2Txs)

	// 动态扩展epoch数组
	for tat.epochID < epochid {
		tat.excutedTxNum = append(tat.excutedTxNum, 0)
		tat.startTime = append(tat.startTime, time.Time{})
		tat.endTime = append(tat.endTime, time.Time{})

		tat.relay1TxNum = append(tat.relay1TxNum, 0)
		tat.relay2TxNum = append(tat.relay2TxNum, 0)
		tat.normalTxNum = append(tat.normalTxNum, 0)

		tat.epochID++
	}

	// 累计当前epoch数据
	tat.excutedTxNum[epochid] += float64(r1TxNum+r2TxNum) / 2 // 保持除以2的逻辑
	tat.excutedTxNum[epochid] += float64(len(b.InnerShardTxs))

	tat.normalTxNum[epochid] += len(b.InnerShardTxs)
	tat.relay1TxNum[epochid] += r1TxNum
	tat.relay2TxNum[epochid] += r2TxNum

	// 修复：正确的时间范围记录逻辑
	if tat.startTime[epochid].IsZero() {
		tat.startTime[epochid] = earliestTime
	} else {
		// 记录最早的开始时间
		if earliestTime.Before(tat.startTime[epochid]) {
			tat.startTime[epochid] = earliestTime
		}
	}

	// 记录最晚的结束时间
	if tat.endTime[epochid].IsZero() || latestTime.After(tat.endTime[epochid]) {
		tat.endTime[epochid] = latestTime
	}
}

func (tat *TestModule_avgTPS_EvolveGCN) HandleExtraMessage([]byte) {}

// 输出结果 - 与Relay方式完全一致
func (tat *TestModule_avgTPS_EvolveGCN) OutputRecord() (perEpochTPS []float64, totalTPS float64) {
	tat.writeToCSV()

	// 计算每个epoch的TPS
	perEpochTPS = make([]float64, tat.epochID+1)
	totalTxNum := 0.0
	eTime := time.Now()
	lTime := time.Time{}

	for eid, exTxNum := range tat.excutedTxNum {
		timeGap := tat.endTime[eid].Sub(tat.startTime[eid]).Seconds()
		perEpochTPS[eid] = exTxNum / timeGap
		totalTxNum += exTxNum

		if eTime.After(tat.startTime[eid]) {
			eTime = tat.startTime[eid]
		}
		if tat.endTime[eid].After(lTime) {
			lTime = tat.endTime[eid]
		}
	}

	totalTPS = totalTxNum / (lTime.Sub(eTime).Seconds())
	return
}

func (tat *TestModule_avgTPS_EvolveGCN) writeToCSV() {
	fileName := tat.OutputMetricName()
	measureName := []string{"EpochID", "Total tx # in this epoch", "Normal tx # in this epoch", "Relay1 tx # in this epoch", "Relay2 tx # in this epoch", "Epoch start time", "Epoch end time", "Avg. TPS of this epoch"}
	measureVals := make([][]string, 0)

	for eid, exTxNum := range tat.excutedTxNum {
		timeGap := tat.endTime[eid].Sub(tat.startTime[eid]).Seconds()
		csvLine := []string{
			strconv.Itoa(eid),
			strconv.FormatFloat(exTxNum, 'f', -1, 64),
			strconv.Itoa(tat.normalTxNum[eid]),
			strconv.Itoa(tat.relay1TxNum[eid]),
			strconv.Itoa(tat.relay2TxNum[eid]),
			strconv.FormatInt(tat.startTime[eid].UnixMilli(), 10),
			strconv.FormatInt(tat.endTime[eid].UnixMilli(), 10),
			strconv.FormatFloat(exTxNum/timeGap, 'f', -1, 64),
		}
		measureVals = append(measureVals, csvLine)
	}

	WriteMetricsToCSV(fileName, measureName, measureVals)
}
