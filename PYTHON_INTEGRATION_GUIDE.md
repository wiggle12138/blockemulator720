# BlockEmulator Python 集成模块使用说明

## 概述

本集成方案将 EvolveGCN 和 feedback 两个反馈优化分片模块成功接入 BlockEmulator 系统中，实现了动态分片优化和性能反馈功能。

## 系统架构

```
BlockEmulator (Go)
       |
   数据交换层
       |
Python 集成模块
    |      |
EvolveGCN  Feedback
动态分片    反馈控制
```

## 安装依赖

### Python 依赖

```bash
pip install torch numpy scikit-learn matplotlib pandas tqdm
```

### Go 依赖

项目已包含必要的 Go 依赖，运行时会自动下载。

## 配置文件

### python_config.json

```json
{
  "enable_evolve_gcn": true,      // 启用 EvolveGCN 模块
  "enable_feedback": true,        // 启用反馈模块
  "python_path": "python",        // Python 解释器路径
  "module_path": "./",            // 模块路径
  "max_iterations": 10,           // 最大迭代次数
  "epochs_per_iteration": 8,      // 每轮epoch数
  "data_exchange_dir": "./data_exchange",  // 数据交换目录
  "output_interval": 30,          // 输出间隔(秒)
  "continuous_mode": true,        // 连续模式
  "log_level": "INFO"            // 日志级别
}
```

## 运行方式

### 1. 完整集成模式 (推荐)

同时启动 BlockEmulator 主系统和 Python 集成模块：

```bash
# Windows
start_with_python.bat

# Linux/Mac
chmod +x start_with_python.sh
./start_with_python.sh
```

### 2. 独立启动 Python 模块

仅启动 Python 集成模块（用于测试）：

```bash
# Windows
start_python_only.bat continuous 10 8

# Linux/Mac
chmod +x start_python_only.sh
./start_python_only.sh continuous 10 8
```

### 3. 手动启动

```bash
# 启动 BlockEmulator 主系统
go run main.go -c -N 4 -S 2 -p

# 启动 Python 集成模块
python integrated_test.py --mode continuous --max_iterations 10 --epochs_per_iteration 8
```

## 参数说明

### 命令行参数

- `-p, --python`: 启用 Python 集成模块
- `-N, --nodeNum`: 每个分片的节点数量 (默认: 4)
- `-S, --shardNum`: 分片数量 (默认: 2)
- `-c, --supervisor`: 以监督节点模式运行

### Python 模块参数

- `--mode`: 运行模式 (single/continuous)
- `--max_iterations`: 最大迭代次数
- `--epochs_per_iteration`: 每轮epoch数
- `--log_level`: 日志级别 (DEBUG/INFO/WARNING/ERROR)

## 数据流程

### 1. 数据导出 (Go → Python)

BlockEmulator 将区块链数据导出到 `data_exchange/blockchain_data_*.json`:

```json
{
  "timestamp": 1642781234,
  "node_id": 1,
  "shard_id": 0,
  "transactions": [...],
  "performance": {
    "tps": 100.0,
    "latency": 50.0,
    "cross_shard_tx": 10,
    "total_tx": 100,
    "block_time": 5.0,
    "queue_length": 5
  },
  "shard_info": {
    "shard_id": 0,
    "node_count": 4,
    "active_nodes": [1, 2, 3, 4],
    "load_balance": 0.8,
    "security_score": 0.9
  }
}
```

### 2. 数据处理 (Python)

Python 模块处理数据并运行：
- EvolveGCN 动态分片优化
- Feedback 反馈控制

### 3. 结果反馈 (Python → Go)

Python 模块将结果写入 `data_exchange/feedback_results.json`:

```json
{
  "timestamp": 1642781234,
  "shard_assignments": {
    "0": 0,
    "1": 0,
    "2": 1,
    "3": 1
  },
  "performance_score": 0.85,
  "load_balance_score": 0.78,
  "security_score": 0.92,
  "cross_shard_ratio": 0.15,
  "recommendations": [
    "建议重新平衡分片负载分布",
    "当前分片配置运行良好"
  ],
  "optimized_sharding": {
    "0": {
      "shard_id": 0,
      "node_ids": [0, 1],
      "load_score": 0.8,
      "capacity": 100
    }
  }
}
```

## 监控和调试

### 1. 状态监控

查看 Python 模块状态：

```bash
# Windows
type data_exchange\python_status.json

# Linux/Mac
cat data_exchange/python_status.json
```

### 2. 日志监控

Python 模块会在控制台输出详细日志，包括：
- 模块初始化状态
- 数据处理进度
- 性能指标
- 错误信息

### 3. 性能调优

根据实际需求调整参数：

```json
{
  "max_iterations": 15,        // 增加迭代次数提高精度
  "epochs_per_iteration": 12,  // 增加epoch数提高训练效果
  "output_interval": 60        // 调整输出间隔
}
```

## 常见问题

### 1. Python 模块启动失败

**问题**: `Error: Python not found`

**解决**: 确保 Python 已安装并添加到 PATH 环境变量中。

### 2. 依赖包缺失

**问题**: `ModuleNotFoundError: No module named 'torch'`

**解决**: 安装必要的 Python 依赖：

```bash
pip install torch numpy scikit-learn matplotlib pandas tqdm
```

### 3. 数据交换目录权限问题

**问题**: `Permission denied: data_exchange`

**解决**: 确保程序有读写权限：

```bash
chmod 755 data_exchange
```

### 4. Go 模块编译错误

**问题**: `package blockEmulator/build: cannot find package`

**解决**: 确保在正确的目录中运行：

```bash
cd /path/to/BlockEmulator
go mod tidy
go run main.go -c -N 4 -S 2 -p
```

## 进阶使用

### 1. 自定义配置

创建自定义配置文件：

```json
{
  "enable_evolve_gcn": true,
  "enable_feedback": true,
  "max_iterations": 20,
  "epochs_per_iteration": 15,
  "continuous_mode": false,
  "log_level": "DEBUG"
}
```

### 2. 单独运行模块

只运行 EvolveGCN 模块：

```bash
python integrated_test.py --mode single --max_iterations 10
```

### 3. 批量测试

创建批量测试脚本：

```bash
#!/bin/bash
for i in {5..20..5}; do
    echo "Testing with $i iterations..."
    python integrated_test.py --mode single --max_iterations $i
done
```

## 输出结果说明

### 性能指标

- **performance_score**: 综合性能分数 (0-1)
- **load_balance_score**: 负载均衡分数 (0-1)
- **security_score**: 安全性分数 (0-1)
- **cross_shard_ratio**: 跨分片交易比例 (0-1)

### 推荐建议

系统会根据性能指标自动生成优化建议：

- 负载均衡分数 < 0.7: 建议重新平衡分片负载
- 跨分片比例 > 0.3: 建议优化跨分片交易处理
- 安全分数 < 0.8: 建议增强分片安全性

## 技术支持

如遇问题，请检查：

1. Python 环境是否正确安装
2. 依赖包是否完整
3. 配置文件是否正确
4. 日志文件中的错误信息

更多技术细节请参考源代码注释和相关文档。
