package measure

import "blockEmulator/message"

type MeasureModule interface {
	UpdateMeasureRecord(*message.BlockInfoMsg)
	HandleExtraMessage([]byte)
	OutputMetricName() string
	OutputRecord() ([]float64, float64)
}

// 新增：可选的实时数据获取接口
type RealTimeDataProvider interface {
	GetCurrentData() ([]float64, float64)
}
