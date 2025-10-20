# BlockEmulator 分片系统对接指南

## 概述

本指南说明如何将动态分片系统的第一步特征提取与BlockEmulator系统进行对接，实现从系统直接获取节点特征数据，而不是从CSV文件读取。

## 系统架构

### 原有架构
```
CSV文件 → 特征提取器 → 第一步输出 → 后续步骤
```

### 新对接架构
```
BlockEmulator系统 → GetAllCollectedData() → 适配器 → 第一步输出 → 后续步骤
```

## 核心组件

### 1. BlockEmulatorAdapter (`blockemulator_adapter.py`)

负责将BlockEmulator系统的原始数据转换为标准特征格式。

**主要功能：**
- 解析系统的`ReplyNodeStateMsg`数据结构
- 提取65维综合特征向量
- 构建节点间的邻接矩阵
- 生成标准化的输出格式

**特征分组（总计65维）：**
- 硬件资源特征：13维
- 链上行为特征：15维
- 网络拓扑特征：7维
- 动态属性特征：10维
- 异构类型特征：10维
- 跨分片交易特征：4维
- 身份特征：2维

### 2. SystemIntegrationPipeline (`system_integration_pipeline.py`)

第一步完整流水线，集成系统接口。

**主要功能：**
- 调用系统的`GetAllCollectedData()`接口
- 数据格式转换和适配
- 特征提取和处理
- 生成与后续步骤兼容的输出格式

## 使用方法

### 基本使用

```python
from partition.feature.system_integration_pipeline import BlockEmulatorStep1Pipeline

# 1. 创建流水线
pipeline = BlockEmulatorStep1Pipeline(
    use_comprehensive_features=True,
    save_adjacency=True,
    output_dir="./step1_outputs"
)

# 2. 从系统提取特征（假设你有NodeFeaturesModule实例）
results = pipeline.extract_features_from_system(
    node_features_module=supervisor.measureManager.nodeFeatureModule,
    experiment_name="experiment_1"
)

# 3. 结果包含：
# - results['features']: [N, 65] 节点特征矩阵
# - results['edge_index']: [2, E] 边索引
# - results['edge_type']: [E] 边类型
# - results['metadata']: 元数据信息
```

### 按Epoch提取

```python
# 提取特定epoch的特征
epoch_result = pipeline.extract_features_from_epoch_data(
    node_features_module=supervisor.measureManager.nodeFeatureModule,
    epoch=5,
    experiment_name="epoch_test"
)

# 批量提取多个epoch
batch_results = pipeline.batch_extract_epoch_features(
    node_features_module=supervisor.measureManager.nodeFeatureModule,
    epochs=[1, 2, 3, 4, 5],
    experiment_name="multi_epoch"
)
```

### 与后续步骤对接

```python
# 获取第一步输出用于第二步
step2_input = pipeline.get_step1_output_for_step2("experiment_1")

# step2_input 包含后续步骤需要的所有数据
if step2_input:
    # 传递给第二步（多尺度对比学习）
    pass_to_step2(step2_input)
```

## 数据映射

### 静态特征映射

| 系统字段 | 特征名称 | 维度位置 | 说明 |
|---------|---------|---------|------|
| `ResourceCapacity.Hardware.CPU.CoreCount` | CPU核心数 | 0 | 整数值 |
| `ResourceCapacity.Hardware.Memory.TotalCapacity` | 内存容量 | 1 | GB单位 |
| `ResourceCapacity.Hardware.Memory.Bandwidth` | 内存带宽 | 2 | GB/s单位 |
| `ResourceCapacity.Hardware.Storage.Capacity` | 存储容量 | 3 | TB单位 |
| `ResourceCapacity.Hardware.Storage.ReadWriteSpeed` | 存储速度 | 4 | MB/s单位 |
| `ResourceCapacity.Hardware.Network.UpstreamBW` | 上行带宽 | 5 | Mbps单位 |
| `ResourceCapacity.Hardware.Network.DownstreamBW` | 下行带宽 | 6 | Mbps单位 |
| `ResourceCapacity.Hardware.Network.Latency` | 网络延迟 | 7 | ms单位，字符串解析 |

### 动态特征映射

| 系统字段 | 特征名称 | 维度位置 | 说明 |
|---------|---------|---------|------|
| `OnChainBehavior.TransactionCapability.AvgTPS` | 平均TPS | 13 | 浮点数 |
| `OnChainBehavior.TransactionCapability.ConfirmationDelay` | 确认延迟 | 14 | ms单位，字符串解析 |
| `OnChainBehavior.TransactionCapability.ResourcePerTx.CPUPerTx` | 每笔交易CPU消耗 | 15 | 浮点数 |
| `OnChainBehavior.Consensus.ParticipationRate` | 共识参与率 | 23 | 0-1范围 |
| `DynamicAttributes.Compute.CPUUsage` | CPU使用率 | 35 | 百分比 |
| `DynamicAttributes.Compute.MemUsage` | 内存使用率 | 36 | 百分比 |

### 特殊字段处理

#### 时间字符串解析
系统中的时间字段（如延迟、间隔）以字符串形式存储：
- `"50ms"` → `50.0`（毫秒）
- `"5.0s"` → `5000.0`（转换为毫秒）
- `"100μs"` → `0.1`（转换为毫秒）

#### 交易量字符串解析
跨分片交易量以特定格式存储：
- `"shard0:1000;shard1:2000"` → 解析为字典并统计总量和平均量

#### 节点类型编码
节点类型转换为one-hot编码：
- `"full_node"` → `[1,0,0,0,0]`
- `"validator"` → `[0,1,0,0,0]`
- `"miner"` → `[0,0,1,0,0]`

## 输出格式

### 主要输出文件

1. **`step1_{experiment_name}_features.pt`**
   - 主要特征数据文件
   - 包含综合特征、图结构、元数据

2. **`step1_{experiment_name}_compatible.pt`**
   - 与后续步骤兼容的格式
   - 添加了邻接矩阵等额外信息

3. **`step1_{experiment_name}_stats.json`**
   - 可读的统计信息
   - 特征分布、节点分布等

4. **`step1_{experiment_name}_adjacency_info.json`**
   - 图结构详细信息
   - 边类型统计、节点分布等

### 数据结构

```python
{
    'features': torch.Tensor,           # [N, 65] 节点特征
    'edge_index': torch.Tensor,         # [2, E] 边索引
    'edge_type': torch.Tensor,          # [E] 边类型
    'adjacency_matrix': torch.Tensor,   # [N, N] 邻接矩阵
    'node_info': {
        'shard_ids': torch.Tensor,      # [N] 分片ID
        'node_ids': torch.Tensor,       # [N] 节点ID
        'timestamps': torch.Tensor      # [N] 时间戳
    },
    'metadata': {
        'num_nodes': int,               # 节点数量
        'num_edges': int,               # 边数量
        'feature_dim': int,             # 特征维度
        'timestamp': int,               # 生成时间戳
        'data_source': str              # 数据源标识
    }
}
```

## 集成步骤

### 1. 修改系统调用代码

在需要运行分片算法的地方：

```python
# 原来的方式
# nodes = load_nodes_from_csv("node_features.csv")
# results = classic_pipeline.extract_features(nodes)

# 新的对接方式
from partition.feature.system_integration_pipeline import BlockEmulatorStep1Pipeline

pipeline = BlockEmulatorStep1Pipeline()
results = pipeline.extract_features_from_system(
    node_features_module=supervisor.measureManager.nodeFeatureModule,
    experiment_name="dynamic_sharding"
)
```

### 2. 传递给后续步骤

```python
# 第二步：多尺度对比学习
step2_features = run_multiscale_learning(results['features'])

# 第三步：EvolveGCN
step3_results = run_evolve_gcn(
    node_features=results['features'],
    edge_index=results['edge_index'],
    adjacency_matrix=results['adjacency_matrix']
)

# 第四步：反馈优化
final_sharding = apply_feedback(step3_results, performance_metrics)
```

### 3. 适应不同运行模式

#### 实时模式
```python
# 在共识过程中实时提取特征
def trigger_dynamic_resharding():
    current_features = pipeline.extract_features_from_system(
        node_features_module=supervisor.measureManager.nodeFeatureModule,
        experiment_name=f"realtime_{current_epoch}"
    )
    new_sharding = run_sharding_algorithm(current_features)
    apply_new_sharding(new_sharding)
```

#### 批处理模式
```python
# 批量处理多个epoch的数据
epochs_to_process = [10, 20, 30, 40, 50]
batch_results = pipeline.batch_extract_epoch_features(
    node_features_module=supervisor.measureManager.nodeFeatureModule,
    epochs=epochs_to_process,
    experiment_name="batch_analysis"
)

# 分析不同epoch的分片策略
for epoch, features in batch_results.items():
    sharding_strategy = analyze_epoch_sharding(features)
    print(f"Epoch {epoch}: {sharding_strategy}")
```

## 错误处理

### 常见错误及解决方案

1. **数据获取失败**
```python
try:
    results = pipeline.extract_features_from_system(node_features_module, "test")
except Exception as e:
    print(f"特征提取失败: {e}")
    # 使用默认特征或重试
```

2. **特征维度不匹配**
```python
# 检查特征维度
if results['features'].shape[1] != 65:
    print(f"警告：特征维度异常 {results['features'].shape[1]}，期望65维")
```

3. **空数据处理**
```python
# 检查是否有有效数据
if results['metadata']['num_nodes'] == 0:
    print("警告：没有获取到节点数据")
    return None
```

## 性能优化建议

1. **缓存机制**：对于相同epoch的重复请求，可以缓存结果
2. **异步处理**：特征提取可以在后台异步进行
3. **内存管理**：及时释放不需要的中间数据
4. **批处理**：尽量批量处理多个节点的数据

## 调试和监控

### 启用详细日志
```python
import logging
logging.basicConfig(level=logging.DEBUG)

pipeline = BlockEmulatorStep1Pipeline()
# 现在会输出详细的处理日志
```

### 特征质量检查
```python
def validate_features(results):
    features = results['features']
    
    # 检查NaN和Inf
    nan_count = torch.isnan(features).sum().item()
    inf_count = torch.isinf(features).sum().item()
    
    print(f"特征质量检查:")
    print(f"  - NaN数量: {nan_count}")
    print(f"  - Inf数量: {inf_count}")
    print(f"  - 特征范围: [{features.min():.4f}, {features.max():.4f}]")
    print(f"  - 特征均值: {features.mean():.4f}")
    print(f"  - 特征标准差: {features.std():.4f}")
    
    return nan_count == 0 and inf_count == 0

# 使用
if validate_features(results):
    print("特征质量检查通过")
else:
    print("特征质量检查失败，需要进一步处理")
```

## 版本兼容性

### 与原有系统的兼容性
- 保持与第二、三、四步的接口兼容
- 输出格式与原CSV方式一致
- 支持回退到CSV模式

### 配置选项
```python
pipeline = BlockEmulatorStep1Pipeline(
    use_comprehensive_features=True,    # 使用65维全面特征
    save_adjacency=True,               # 保存邻接矩阵信息
    output_dir="./outputs",            # 输出目录
    fallback_to_csv=True               # 失败时回退到CSV模式（可选）
)
```

## 总结

通过本对接方案，动态分片系统可以：

1. **直接从BlockEmulator系统获取实时节点特征**
2. **保持与后续步骤的完全兼容性**
3. **支持多种运行模式（实时、批处理、epoch）**
4. **提供丰富的监控和调试能力**
5. **确保数据质量和系统稳定性**

这种对接方式不仅提高了系统的实时性和准确性，还为后续的优化和扩展提供了良好的基础。
