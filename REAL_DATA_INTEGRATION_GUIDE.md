# 完整集成分片系统 - 真实数据接入指南

## 概述

当前 `complete_integrated_sharding_system.py` 使用内部生成的模拟数据。本指南说明如何接入真实的BlockEmulator数据。

## 当前数据流

```
内部模拟数据生成 → DirectStep1Processor → 99维特征 → 四步流水线
```

## 真实数据源选项

### 1. BlockEmulator系统直接对接

**优点**: 最真实的运行时数据
**用法**:
```python
from blockemulator_real_data_interface import BlockEmulatorDataInterface

interface = BlockEmulatorDataInterface()
real_data = interface.trigger_node_feature_collection(
    node_count=200, 
    shard_count=4, 
    collection_timeout=30
)
```

### 2. CSV文件输入

**优点**: 静态数据，可重复测试
**用法**:
```python
import pandas as pd
node_data = pd.read_csv("node_features_input.csv")
```

### 3. 实时适配器接口

**优点**: 标准化的实时数据处理
**用法**:
```python
from partition.feature.be_realtime_adapter import BlockEmulatorRealtimeAdapter

adapter = BlockEmulatorRealtimeAdapter()
features = adapter.extract_features_realtime(node_data_list)
```

## 修改步骤

### 步骤1: 修改 DirectStep1Processor

在 `complete_integrated_sharding_system.py` 中，修改 `extract_real_features` 方法：

```python
def extract_real_features(self, node_data=None, feature_dims=None):
    """提取真实特征"""
    
    # 选项1: 使用BlockEmulator直接对接
    if node_data is None:
        from blockemulator_real_data_interface import BlockEmulatorDataInterface
        interface = BlockEmulatorDataInterface()
        real_data = interface.trigger_node_feature_collection(node_count=200, shard_count=4)
        pipeline_data = interface.convert_to_pipeline_format(real_data)
        node_data = pipeline_data['node_data']
    
    # 选项2: 使用CSV文件
    if node_data is None:
        import pandas as pd
        df = pd.read_csv("node_features_input.csv")
        node_data = df.to_dict('records')
    
    # 选项3: 使用实时适配器
    if node_data is None:
        from partition.feature.be_realtime_adapter import create_be_realtime_adapter
        adapter = create_be_realtime_adapter()
        # 需要先获取原始数据...
        
    # 处理真实数据...
    return self._process_real_node_data(node_data)
```

### 步骤2: 数据格式转换

添加真实数据处理方法：

```python
def _process_real_node_data(self, node_data):
    """处理真实节点数据"""
    features = {}
    num_nodes = len(node_data)
    
    # 从真实数据提取特征
    for feature_name, dim in self.feature_dims.items():
        feature_matrix = torch.zeros(num_nodes, dim, device=self.device)
        
        for i, node in enumerate(node_data):
            if feature_name == 'hardware':
                feature_matrix[i] = self._extract_hardware_from_real_data(node)
            elif feature_name == 'onchain_behavior':
                feature_matrix[i] = self._extract_onchain_from_real_data(node)
            # ... 其他特征类别
            
        features[feature_name] = feature_matrix
    
    return {
        'features': features,
        'edge_index': self._build_real_edge_index(node_data),
        'num_nodes': num_nodes,
        'source': 'real_blockemulator_data'
    }
```

### 步骤3: 特征映射

实现具体的特征提取方法：

```python
def _extract_hardware_from_real_data(self, node_data):
    """从真实数据提取硬件特征"""
    static_data = node_data.get('static_features', {})
    hardware = static_data.get('ResourceCapacity', {}).get('Hardware', {})
    
    features = torch.zeros(17, device=self.device)
    features[0] = hardware.get('CPUCores', 4)
    features[1] = hardware.get('MemoryGB', 8)
    features[2] = hardware.get('DiskCapacityGB', 500)
    features[3] = hardware.get('NetworkBandwidthMbps', 1000)
    # ... 其余13维
    
    return features
```

## 数据源配置

### 配置文件

在 `python_config.json` 中添加数据源配置：

```json
{
  "data_source": {
    "type": "blockemulator",  // "blockemulator", "csv", "adapter"
    "blockemulator": {
      "node_count": 200,
      "shard_count": 4,
      "timeout": 30
    },
    "csv": {
      "file_path": "node_features_input.csv"
    },
    "adapter": {
      "realtime": true
    }
  }
}
```

### 环境变量

```bash
export BLOCKEMULATOR_DATA_SOURCE=blockemulator  # 或 csv, adapter
export BLOCKEMULATOR_NODE_COUNT=200
export BLOCKEMULATOR_SHARD_COUNT=4
```

## 验证数据质量

### 数据完整性检查

```python
def validate_real_data(self, features):
    """验证真实数据质量"""
    total_dims = sum(self.real_feature_dims.values())  # 应该是99
    
    for feature_name, tensor in features.items():
        expected_dim = self.real_feature_dims[feature_name]
        actual_dim = tensor.shape[1]
        
        if actual_dim != expected_dim:
            logger.warning(f"特征维度不匹配: {feature_name} 期望{expected_dim}, 实际{actual_dim}")
            
    logger.info(f"数据验证完成: {len(features)} 个特征类别，总维度 {total_dims}")
```

## 性能对比

| 数据源 | 真实性 | 性能 | 可重复性 | 推荐场景 |
|--------|--------|------|----------|----------|
| 模拟数据 | 低 | 高 | 高 | 算法开发、调试 |
| CSV文件 | 中 | 高 | 高 | 测试、验证 |
| 实时适配器 | 高 | 中 | 中 | 生产环境 |
| BlockEmulator直接 | 最高 | 低 | 低 | 真实场景测试 |

## 故障排除

### 常见问题

1. **BlockEmulator连接失败**
   - 检查 blockemulator 可执行文件路径
   - 确认端口未被占用
   - 查看日志文件

2. **CSV数据格式错误**
   - 验证CSV列名匹配
   - 检查数据类型转换
   - 处理缺失值

3. **特征维度不匹配**
   - 确认字段映射正确
   - 检查特征维度配置
   - 验证数据预处理

### 调试命令

```bash
# 测试BlockEmulator接口
python blockemulator_real_data_interface.py

# 测试实时适配器
python partition/feature/be_realtime_adapter.py

# 验证CSV数据
python -c "import pandas as pd; print(pd.read_csv('node_features_input.csv').info())"
```

## 总结

目前系统使用模拟数据是为了确保算法的可靠性和可重复性。要切换到真实数据，需要：

1. 选择合适的数据源（BlockEmulator/CSV/适配器）
2. 修改 `DirectStep1Processor.extract_real_features` 方法
3. 实现数据格式转换
4. 添加数据验证机制
5. 配置数据源参数

建议先使用CSV文件测试，确认数据流正常后再切换到实时BlockEmulator数据。
