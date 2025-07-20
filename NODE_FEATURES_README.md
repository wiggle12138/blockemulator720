# 节点特征收集功能说明

## 功能概述

本功能为block-emulator系统新增了节点特征收集能力，能够收集wrs.md中定义的异构特征类型，生成高质量、完整的node_features.csv数据集。

## 最新实现原理

### 1. 静态特征采集机制
- **CPU核心数、内存容量**：节点端通过读取cgroup文件系统（/sys/fs/cgroup/），自动获取Docker容器的真实资源限制，保证与docker-compose.yml中的`cpus`和`mem_limit`参数完全一致。
- **内存带宽、存储性能、网络性能等**：通过docker-compose.yml的`environment`环境变量配置，Go端用`os.Getenv`读取，模拟异构特征。
- **网络延迟**：节点端通过真实ping 8.8.8.8获取平均延迟，采集失败时默认521ms。
- **配置建议**：只用`cpus`和`mem_limit`控制真实资源，其他硬件特征用环境变量模拟，Go端自动读取。

### 2. 节点数据完整性保障
- 节点端数据上报采用同步阻塞机制（WaitGroup+TcpDialAndWait），确保Supervisor收到所有节点数据后再关闭节点，彻底避免节点丢失。
- Supervisor端采用WaitGroup机制，阻塞等待所有节点上报，超时和缺失节点均有详细日志，便于debug。
- 日志记录每个节点数据上报与收集状态，便于定位问题。

### 3. 多次终态聚合实验范式
- 推荐采用“多次独立实验终态聚合”方案：每次实验运行结束后，收集一次全局终态数据（如Average_TPS.csv、CrossTransaction_ratio.csv、node_features.csv等）。
- 通过自动化脚本（如run_experiments.py）多次运行实验，每次注入不同参数（如交易量、节点异构配置），收集每次实验的最终结果。
- 用聚合脚本（如aggregate_results.py）将所有实验终态数据整合为高质量的时间序列或参数序列数据集。
- 该方案极大简化了节点端和Supervisor端的采集逻辑，避免分布式快照、时钟同步等复杂问题，数据质量高、可控性强、易于复现。

### 4. 代码与配置要点
- 节点端pbft.go只需保证终止时能上报完整的静态+动态终态特征，无需定时快照、窗口滑动等复杂逻辑。
- Supervisor端supervisor.go采用WaitGroup机制，确保所有节点数据收集完整。
- networks/p2pNetwork.go实现了TcpDialAndWait，节点端数据上报时同步等待发送完成。
- docker-compose.yml配置示例：
  ```yaml
  services:
    shard0-node0:
      image: block-emulator:latest
      cpus: '1.0'           # 真实CPU核数限制，Go端cgroup自动读取
      mem_limit: 2g         # 真实内存限制，Go端cgroup自动读取
      environment:
        - MEM_TYPE=DDR4
        - MEM_BANDWIDTH=85.4
        - STORAGE_TYPE=SSD
        - STORAGE_CAPACITY=2
        - STORAGE_RW=1200
        - NET_UP_BW=1000
        - NET_DOWN_BW=10000
        - NET_LATENCY=50ms
        - TIMEZONE=UTC+8
        - DATACENTER=AWS
        - REGION=ap-southeast-1
  ```

### 5. 多次终态聚合实验自动化流程建议
- 编写run_experiments.py脚本，循环多次运行docker-compose，每次注入不同参数，收集每次实验的最终结果文件（如node_features.csv、Average_TPS.csv等），移动到results/run_N/目录。
- 编写aggregate_results.py脚本，遍历所有实验结果目录，提取各类最终指标，聚合为一份大表（如final_node_features.csv），每行代表一次实验的终态特征。
- 每次实验前务必彻底清理环境（docker-compose down -v），避免数据残留。
- 各实验结果目录命名清晰（如run_1, run_2），便于后续聚合。

## 数据质量保证

- **静态特征**：CPU/内存容量100%真实反映容器cgroup限制，其他硬件特征可灵活模拟。网络延迟为真实ping测量。
- **动态特征**：CPU/内存使用率等仅保留动态真实测量部分。
- **数据完整性**：Supervisor端阻塞等待所有节点上报，节点端同步发送，彻底避免节点数据丢失。
- **可复现性**：所有实验参数、节点异构配置均可通过compose和脚本自动化管理，便于复现实验。
- **日志可观测性**：Supervisor和节点端均有详细日志，记录每个节点数据上报与收集状态。

## 注意事项

1. **静态特征优先cgroup读取**，只用compose环境变量模拟无法直接配置的特征。
2. **终态聚合优先**，不再推荐高频快照或窗口化采集。
3. **自动化脚本管理实验流程**，提升效率和数据质量。
4. **实验前后清理环境，避免数据残留。**

## 扩展建议

- 支持自动化参数扫描、实验失败重试、异常日志收集。
- 支持最终聚合表自动生成可视化报告。
- 可集成Prometheus等监控工具获取更精确的系统指标。

---

如需具体自动化脚本模板或聚合逻辑示例，请联系开发者或查阅相关文档。 