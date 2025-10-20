# BlockEmulator Ablation Study Analysis Report
**生成时间**: 2025-08-01 23:15:00

## 执行摘要 (Executive Summary)
- **完整实现平均 TPS**: 880.21
- **完整实现平均延迟**: 46732.20 ms
- **完整实现跨分片比例**: 0.772

## 变体对比分析 (Variant Comparison)
### 无对比学习 (No Contrastive Learning)
- TPS 提升: **-1.9%**
- 延迟降低: **-0.8%**
- 跨分片比例变化: **0.0%**

### 无分层处理 (No Layered Processing)
- TPS 提升: **-0.8%**
- 延迟降低: **0.5%**
- 跨分片比例变化: **-0.1%**

## 关键发现 (Key Findings)
1. **性能最佳变体**: 无对比学习 (No Contrastive Learning) (平均 TPS: 897.53)

## 技术洞察 (Technical Insights)
1. **系统负载特性**: 从跨分片交易比例可以看出系统的分片效果
2. **性能稳定性**: 从不同 epoch 的性能变化可以评估系统稳定性
3. **优化方向**: 基于 ablation 结果确定后续优化的重点组件
