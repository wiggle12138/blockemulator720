# BlockEmulator EvolveGCN集成使用指南

## 🎯 概述

本指南介绍如何使用已集成到BlockEmulator中的EvolveGCN四步分片算法。CLPA占位算法已被真实的EvolveGCN算法替换。

## 🚀 快速启动

### 方法1: 一键启动（推荐）
```bash
# Windows
start_evolvegcn_integrated.bat

# 或者手动步骤
python config_python_venv.py
cd docker
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" setup
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" build
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" start
```

### 方法2: 手动配置
```bash
# 1. 配置Python虚拟环境
python config_python_venv.py

# 2. 测试EvolveGCN集成
cd docker
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" test-python

# 3. 构建和启动
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" build
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" start
```

## 📋 系统要求

- ✅ Python 3.8+
- ✅ Docker Desktop
- ✅ PowerShell（Windows）
- ✅ 虚拟环境（推荐）

## 🔧 配置文件

### python_config.json
```json
{
  "enable_evolve_gcn": true,      // 启用EvolveGCN
  "enable_feedback": true,        // 启用反馈机制
  "python_path": "python",        // Python路径（自动检测）
  "evolvegcn_integration": {
    "enabled": true,              // EvolveGCN集成状态
    "algorithm": "four_step_pipeline",
    "fallback_to_clpa": true      // 失败时回退到CLPA
  }
}
```

## 🎮 管理命令

### 基础操作
```bash
cd docker

# 查看帮助
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" help

# 配置环境
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" setup

# 测试Python集成
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" test-python

# 构建镜像
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" build

# 启动服务
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" start

# 查看状态
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" status

# 查看日志
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" logs

# 停止服务
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" stop

# 清理资源
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" cleanup
```

## 🔍 EvolveGCN算法工作流程

### 四步流水线
1. **特征提取**: 从区块链网络中提取节点特征
2. **多尺度学习**: 进行对比学习和时序建模
3. **EvolveGCN分片**: 使用演化图神经网络进行动态分片
4. **反馈评估**: 评估分片效果并提供反馈优化

### 替换CLPA算法
- ✅ CLPA占位算法已被真实EvolveGCN算法替换
- ✅ 支持动态分片和自适应优化
- ✅ 集成反馈机制和性能监控
- ✅ 保持CLPA作为可靠的回退选项

## 📊 监控和调试

### 查看EvolveGCN状态
```bash
# 查看Python集成状态
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" status

# 查看supervisor日志（EvolveGCN运行日志）
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" logs supervisor

# 进入supervisor容器调试
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" exec supervisor
```

### 日志关键信息
```
EvolveGCN: Starting four-step partition pipeline...
EvolveGCN Step 1: Feature extraction...
EvolveGCN: Calling Python four-step pipeline...
EvolveGCN: Pipeline completed successfully. Cross-shard edges: X
EvolveGCN: ✅ Real EvolveGCN algorithm active (CLPA placeholder replaced)
```

## 🛠️ 故障排除

### 常见问题

#### Python环境问题
```bash
# 重新配置Python环境
python config_python_venv.py

# 手动指定Python路径
# 编辑python_config.json，设置正确的python_path
```

#### EvolveGCN启动失败
```bash
# 测试Python集成
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" test-python

# 检查依赖
python -c "import torch, numpy, json"

# 查看详细日志
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" logs supervisor
```

#### 回退到CLPA
如果EvolveGCN失败，系统会自动回退到CLPA算法：
```
EvolveGCN: Python pipeline failed, falling back to CLPA: error_details
EvolveGCN: Using CLPA as fallback for reliability
```

### 强制使用CLPA
如需暂时禁用EvolveGCN：
```json
// 编辑python_config.json
{
  "enable_evolve_gcn": false
}
```

## 🎯 性能优化

### 虚拟环境配置
推荐使用虚拟环境以避免依赖冲突：
```bash
# 使用检测到的虚拟环境
python config_python_venv.py

# 手动配置虚拟环境路径
# 编辑python_config.json中的python_path
```

### 资源监控
```bash
# 实时监控容器资源使用
powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" status

# 查看Docker资源使用
docker stats
```

## 📝 开发和测试

### 测试EvolveGCN接口
```bash
# 运行集成测试
python test_evolvegcn_integration.py

# 测试Go-Python接口
python evolvegcn_go_interface.py --help

# 运行完整流水线测试
python examples/full_pipeline_demo.py
```

### 自定义配置
可以通过修改以下文件自定义EvolveGCN行为：
- `python_config.json`: 基础配置
- `integration_config.json`: 集成配置
- `evolve_gcn_feedback_config.json`: 反馈配置

## 🚀 高级用法

### 直接Python执行
```bash
# 使用配置的虚拟环境运行流水线
python integrated_four_step_pipeline.py

# 运行四步闭环迭代
python run_steps_python.py

# 运行增强型流水线
python run_enhanced_pipeline.py
```

### 自定义部署
如需修改部署配置，编辑：
- `docker/deploy_evolvegcn.ps1`: 部署脚本
- `docker/docker-compose.yml`: 容器配置
- `supervisor/committee/committee_evolvegcn.go`: Go集成代码

---

## 📞 技术支持

如遇到问题，请：
1. 查看日志文件
2. 运行诊断命令
3. 检查配置文件
4. 参考故障排除部分

**EvolveGCN四步分片算法现已完全集成到BlockEmulator中！** 🎉
