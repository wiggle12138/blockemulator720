#!/bin/bash
echo "Starting Python Integration Modules Only..."

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python."
    exit 1
fi

# 创建数据交换目录
mkdir -p data_exchange

# 启动Python集成模块
echo "Starting Python integration modules..."
python integrated_test.py --mode ${1:-continuous} --max_iterations ${2:-10} --epochs_per_iteration ${3:-8}
