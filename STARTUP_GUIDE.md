# BlockEmulator 系统启动指南 (UTF-8版本)

## 快速启动

### 方法1: 使用系统管理脚本 (推荐)
```bash
.\manage-blockemulator.bat
```
然后选择选项:
- `1` - 编译系统 (如果还没编译)
- `2` - 启动系统

### 方法2: 直接启动
```bash
# 1. 确保已编译UTF-8版本
.\compile-utf8-simple.bat

# 2. 启动系统
.\start-blockemulator-utf8.bat
```

### 方法3: 使用PowerShell (预热版本)
```powershell
.\run-blockemulator-preload-safe.ps1
```

## 系统架构

启动后将运行以下组件:
- **分片0**: 4个节点 (端口: 32216-32219)
- **分片1**: 4个节点 (端口: 32220-32223) 
- **Supervisor**: EvolveGCN控制器 (端口: 32224)

## 监控

- **Supervisor窗口**: 显示EvolveGCN动态分片算法运行状态
- **节点窗口**: 显示PBFT共识和交易处理状态
- **UTF-8日志**: 支持中文和多语言字符显示

## 停止系统

```bash
.\stop-blockemulator.bat
```

## 状态检查

```bash
.\status-blockemulator.bat
```

## 系统要求

### 必须:
- ✅ Go 1.20+ (编译需要)
- ✅ Python 3.8+ with虚拟环境
- ✅ Windows 10/11 with UTF-8支持

### 推荐:
- ✅ Windows Terminal (最佳UTF-8显示效果)
- ✅ 8GB+ RAM (多节点运行)
- ✅ CUDA GPU (EvolveGCN加速，可选)

## 端口配置

| 组件 | 端口 | 用途 |
|------|------|------|
| 分片0-节点0 | 32216 | PBFT共识 |
| 分片0-节点1 | 32217 | PBFT共识 |
| 分片0-节点2 | 32218 | PBFT共识 |
| 分片0-节点3 | 32219 | PBFT共识 |
| 分片1-节点0 | 32220 | PBFT共识 |
| 分片1-节点1 | 32221 | PBFT共识 |
| 分片1-节点2 | 32222 | PBFT共识 |
| 分片1-节点3 | 32223 | PBFT共识 |
| Supervisor | 32224 | EvolveGCN控制 |

## 故障排除

### 编译问题
```bash
# 检查Go环境
go version

# 重新编译
.\compile-utf8-simple.bat
```

### 启动失败
```bash
# 检查端口占用
netstat -an | findstr ":322"

# 检查系统状态
.\status-blockemulator.bat
```

### Python环境问题
```bash
# 激活虚拟环境
.venv\Scripts\activate

# 检查依赖
pip list | findstr torch
```

## 配置文件

- `paramsConfig.json` - 主配置文件
- `ipTable.json` - 节点IP配置
- `selectedTxs_300K.csv` - 交易数据集

## 日志输出

- **Supervisor日志**: EvolveGCN算法执行状态
- **节点日志**: PBFT共识过程和交易处理
- **Python日志**: 机器学习模型训练和推理

## 性能监控

- **TPS监控**: 实时交易处理速度
- **分片效率**: 跨分片交易比例
- **EvolveGCN性能**: 动态分片质量评估

## 高级用法

### 自定义配置
编辑 `paramsConfig.json` 修改:
- 区块间隔
- 交易批次大小
- 分片重配置间隔

### 数据集更换
替换 `selectedTxs_300K.csv` 为自定义交易数据

### Python组件开发
修改 `evolvegcn_preload_service_safe.py` 进行算法优化
