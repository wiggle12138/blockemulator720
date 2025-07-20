#!/bin/bash
echo "Starting BlockEmulator with Python Integration..."

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python."
    exit 1
fi

# 安装必要的Python依赖
echo "Installing Python dependencies..."
pip install torch numpy scikit-learn matplotlib pandas tqdm

# 创建数据交换目录
mkdir -p data_exchange

# 启动 BlockEmulator 主程序（带Python集成）
echo "Starting BlockEmulator with Python integration..."
go run main.go -c -N 4 -S 2 -p

echo "BlockEmulator with Python Integration started!"
