# Docker快速部署指南

本指南说明如何使用统一的 PowerShell 脚本 `deploy.ps1` 进行区块链模拟器的 Docker 部署和实验管理。

## 目录结构
- **Dockerfile** / **docker-compose.yml**：镜像构建与服务编排配置
- **deploy.ps1**：统一的部署和实验管理脚本（主要脚本）
- **Files/**：包含必要的可执行文件和配置文件
  - `blockEmulator_Linux_Precompile`：Linux 可执行文件
  - `paramsConfig.json`：参数配置文件
  - `ipTable.json`：IP 地址配置文件
  - `selectedTxs_300K.csv`：实验数据集（仅用于 supervisor 节点）

## 系统架构
- **1 个 Supervisor 节点**：负责协调整个实验系统
- **4 个工作节点**：
  - shard0-node0, shard0-node1（分片 0）
  - shard1-node0, shard1-node1（分片 1）
  后续进行节点的扩展，现在在进行16节点的实验
  对于supervisor节点的debug信息可以少打印一些，日志里看着太多了
  对于数据集转向交易，分清楚训练数据集和测试数据集


## 快速开始

### 方式一：一键运行完整实验（推荐）
```powershell
# 在 docker 目录下执行
.\deploy.ps1 run-exp
```
此命令会自动完成：构建镜像 → 启动容器 → 等待实验完成 → 收集数据 → 清理资源

### 方式二：分步骤手动控制
```powershell
# 1. 构建 Docker 镜像
.\deploy.ps1 build

# 2. 启动所有节点
.\deploy.ps1 start

# 3. 查看运行状态
.\deploy.ps1 status

# 4. 查看实时日志（可选）
.\deploy.ps1 logs                # 查看所有节点日志
.\deploy.ps1 logs supervisor     # 查看指定节点日志

# 5. 停止所有服务
.\deploy.ps1 stop
```

## 主要命令说明

| 命令 | 功能描述 |
|------|----------|
| `.\deploy.ps1 run-exp` | **一键完整实验**：自动化整个实验流程 |
| `.\deploy.ps1 build` | 构建 Docker 镜像 |
| `.\deploy.ps1 start` | 启动所有服务容器 |
| `.\deploy.ps1 stop` | 停止并移除所有容器 |
| `.\deploy.ps1 status` | 查看容器状态和资源使用 |
| `.\deploy.ps1 logs [节点名]` | 查看实时日志 |
| `.\deploy.ps1 exec [节点名]` | 进入容器的 shell 环境 |
| `.\deploy.ps1 cleanup` | 彻底清理所有 Docker 资源 |
| `.\deploy.ps1 help` | 显示帮助信息 |

## 实验数据收集

当使用 `run-exp` 命令时，系统会：

1. **自动监控**：等待所有工作节点完成实验并退出
2. **数据收集**：从每个节点容器收集实验数据文件
3. **数据合并**：将所有节点的数据合并到 `../expTest/result/large_samples.csv`
4. **自动清理**：实验结束后自动停止并清理容器

## 监控和调试

### 查看系统状态
```powershell
.\deploy.ps1 status
```

### 查看特定节点日志
```powershell
.\deploy.ps1 logs supervisor      # 查看 supervisor 节点
.\deploy.ps1 logs shard0-node0    # 查看分片0的节点0
```

### 进入容器调试
```powershell
.\deploy.ps1 exec supervisor      # 进入 supervisor 容器
```

## 故障排除

### 常见问题
1. **Docker 未启动**：确保 Docker Desktop 正在运行
2. **端口冲突**：确保相关端口未被占用
3. **文件缺失**：确保 `Files/` 目录下包含所有必要文件
4. **权限问题**：以管理员身份运行 PowerShell

### 检查步骤
```powershell
# 1. 检查 Docker 状态
docker info

# 2. 检查容器状态
.\deploy.ps1 status

# 3. 查看详细日志
.\deploy.ps1 logs

# 4. 完全重置环境
.\deploy.ps1 cleanup
```

## 注意事项

- **运行环境**：需要 Windows 系统和 PowerShell
- **Docker 要求**：需要安装 Docker Desktop 并确保其正在运行
- **系统资源**：建议至少 4GB 可用内存
- **工作目录**：所有命令都需要在 `docker` 目录下执行
- **实验时长**：系统会根据业务逻辑自动终止，无需手动指定时长
- **数据位置**：最终合并的实验数据保存在 `../expTest/result/large_samples.csv`

## 快速开始示例

```powershell
# 完整的实验流程示例
cd docker
.\deploy.ps1 run-exp
# 等待实验自动完成，数据将保存在 ../expTest/result/large_samples.csv
```

---

如需更多帮助，请使用 `.\deploy.ps1 help` 查看详细的命令说明。