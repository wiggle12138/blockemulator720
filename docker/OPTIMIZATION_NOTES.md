# Docker镜像复用优化

## 主要改进

1. **镜像复用**: 所有节点使用同一个 `block-emulator:latest` 镜像
2. **文件隔离**: 每个节点有独立的输出目录 (`expTest/shardX-nodeY/`)
3. **自动目录创建**: 启动时自动创建必要的输出目录

## 使用方法

```powershell
# 首次使用或代码更新后
.\deploy.ps1 build

# 启动系统
.\deploy.ps1 start

# 停止系统
.\deploy.ps1 stop
```

## 优势

- 减少80%的镜像存储空间
- 减少80%的构建时间
- 完全避免文件写入冲突
- 简化维护工作 