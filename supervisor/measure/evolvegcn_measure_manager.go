package measure

import (
	"blockEmulator/message"
	"log"
)

// EvolveGCN使用标准的测量模块管理，与CLPA保持一致
type EvolveGCNMeasureManager struct {
	tpsModule     *TestModule_avgTPS_EvolveGCN
	ctxRateModule *TestCrossTxRate_EvolveGCN
}

func NewEvolveGCNMeasureManager() *EvolveGCNMeasureManager {
	return &EvolveGCNMeasureManager{
		tpsModule:     NewTestModule_avgTPS_EvolveGCN(),
		ctxRateModule: NewTestCrossTxRate_EvolveGCN(),
	}
}

// 更新测量记录 - 标准接口
func (emm *EvolveGCNMeasureManager) UpdateMeasureRecord(b *message.BlockInfoMsg) {
	emm.tpsModule.UpdateMeasureRecord(b)
	emm.ctxRateModule.UpdateMeasureRecord(b)
}

// 输出最终结果 - 与标准测量模块保持一致的输出格式
func (emm *EvolveGCNMeasureManager) OutputResults() {
	// TPS结果
	perEpochTPS, totalTPS := emm.tpsModule.OutputRecord()
	log.Printf("EvolveGCN TPS Results - Total TPS: %.2f", totalTPS)
	for i, tps := range perEpochTPS {
		log.Printf("  Epoch %d TPS: %.2f", i, tps)
	}

	// 跨分片交易率结果
	perEpochCTX, totalCTX := emm.ctxRateModule.OutputRecord()
	log.Printf("EvolveGCN CTX Rate Results - Total CTX Rate: %.4f", totalCTX)
	for i, ctx := range perEpochCTX {
		log.Printf("  Epoch %d CTX Rate: %.4f", i, ctx)
	}
}
