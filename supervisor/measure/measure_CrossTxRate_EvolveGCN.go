package measure

import (
	"blockEmulator/message"
	"strconv"
)

// EvolveGCN使用标准的跨分片交易率统计模块，与Relay方式完全一致
type TestCrossTxRate_EvolveGCN struct {
	epochID int

	normalTxNum []int
	relay1TxNum []int
	relay2TxNum []int

	totTxNum      []float64
	totCrossTxNum []float64
}

func NewTestCrossTxRate_EvolveGCN() *TestCrossTxRate_EvolveGCN {
	return &TestCrossTxRate_EvolveGCN{
		epochID:       -1,
		totTxNum:      make([]float64, 0),
		totCrossTxNum: make([]float64, 0),

		normalTxNum: make([]int, 0),
		relay1TxNum: make([]int, 0),
		relay2TxNum: make([]int, 0),
	}
}

func (tctr *TestCrossTxRate_EvolveGCN) OutputMetricName() string {
	return "CrossTransaction_ratio"
}

func (tctr *TestCrossTxRate_EvolveGCN) UpdateMeasureRecord(b *message.BlockInfoMsg) {
	if b.BlockBodyLength == 0 { // 空区块
		return
	}

	epochid := b.Epoch
	r1TxNum := len(b.Relay1Txs)
	r2TxNum := len(b.Relay2Txs)

	// 动态扩展epoch数组
	for tctr.epochID < epochid {
		tctr.totTxNum = append(tctr.totTxNum, 0)
		tctr.totCrossTxNum = append(tctr.totCrossTxNum, 0)

		tctr.relay1TxNum = append(tctr.relay1TxNum, 0)
		tctr.relay2TxNum = append(tctr.relay2TxNum, 0)
		tctr.normalTxNum = append(tctr.normalTxNum, 0)

		tctr.epochID++
	}

	// 累计当前epoch数据
	tctr.normalTxNum[epochid] += len(b.InnerShardTxs)
	tctr.relay1TxNum[epochid] += r1TxNum
	tctr.relay2TxNum[epochid] += r2TxNum

	// 跨分片交易计算（relay1+relay2为跨分片交易）
	tctr.totCrossTxNum[epochid] += float64(r1TxNum+r2TxNum) / 2
	tctr.totTxNum[epochid] += float64(r1TxNum+r2TxNum)/2 + float64(len(b.InnerShardTxs))
}

func (tctr *TestCrossTxRate_EvolveGCN) HandleExtraMessage([]byte) {}

// 新增：获取当前数据而不触发CSV写入
func (tctr *TestCrossTxRate_EvolveGCN) GetCurrentData() (perEpochCTXratio []float64, totCTXratio float64) {
	// 计算每个epoch的跨分片交易率，但不写入CSV
	perEpochCTXratio = make([]float64, tctr.epochID+1)
	allEpoch_ctxNum := 0.0
	allEpoch_totTxNum := 0.0

	for eid, totTxInE := range tctr.totTxNum {
		if totTxInE > 0 {
			perEpochCTXratio[eid] = tctr.totCrossTxNum[eid] / totTxInE
		} else {
			perEpochCTXratio[eid] = 0.0
		}
		allEpoch_ctxNum += tctr.totCrossTxNum[eid]
		allEpoch_totTxNum += totTxInE
	}

	if allEpoch_totTxNum > 0 {
		totCTXratio = allEpoch_ctxNum / allEpoch_totTxNum
	} else {
		totCTXratio = 0.0
	}

	return perEpochCTXratio, totCTXratio
}

func (tctr *TestCrossTxRate_EvolveGCN) OutputRecord() (perEpochCTXratio []float64, totCTXratio float64) {
	tctr.writeToCSV()

	// 计算每个epoch的跨分片交易率
	perEpochCTXratio = make([]float64, tctr.epochID+1)
	allEpoch_ctxNum := 0.0
	allEpoch_totTxNum := 0.0

	for eid, totTxInE := range tctr.totTxNum {
		if totTxInE > 0 {
			perEpochCTXratio[eid] = tctr.totCrossTxNum[eid] / totTxInE
		} else {
			perEpochCTXratio[eid] = 0.0
		}
		allEpoch_ctxNum += tctr.totCrossTxNum[eid]
		allEpoch_totTxNum += totTxInE
	}

	if allEpoch_totTxNum > 0 {
		totCTXratio = allEpoch_ctxNum / allEpoch_totTxNum
	} else {
		totCTXratio = 0.0
	}

	return perEpochCTXratio, totCTXratio
}

func (tctr *TestCrossTxRate_EvolveGCN) writeToCSV() {
	fileName := tctr.OutputMetricName()
	measureName := []string{"EpochID", "Total tx # in this epoch", "CTX # in this epoch", "Normal tx # in this epoch", "Relay1 tx # in this epoch", "Relay2 tx # in this epoch", "CTX ratio of this epoch"}
	measureVals := make([][]string, 0)

	for eid, totTxInE := range tctr.totTxNum {
		csvLine := []string{
			strconv.Itoa(eid),
			strconv.FormatFloat(totTxInE, 'f', 8, 64),
			strconv.FormatFloat(tctr.totCrossTxNum[eid], 'f', 8, 64),
			strconv.Itoa(tctr.normalTxNum[eid]),
			strconv.Itoa(tctr.relay1TxNum[eid]),
			strconv.Itoa(tctr.relay2TxNum[eid]),
			strconv.FormatFloat(tctr.totCrossTxNum[eid]/totTxInE, 'f', 8, 64),
		}
		measureVals = append(measureVals, csvLine)
	}
	WriteMetricsToCSV(fileName, measureName, measureVals)
}
