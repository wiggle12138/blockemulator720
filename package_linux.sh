#!/bin/bash
# Linux版本的Python分片系统打包脚本

echo "🚀 开始打包EvolveGCN分片系统 (Linux版本)..."

# 检查PyInstaller
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "❌ PyInstaller未安装，正在安装..."
    python3 -m pip install pyinstaller
fi

# 打包
python3 -m PyInstaller \
    --onefile \
    --name=evolvegcn_sharding_linux \
    --distpath=./docker/Files \
    --workpath=./build_temp \
    evolvegcn_main.py

if [ -f "./docker/Files/evolvegcn_sharding_linux" ]; then
    echo "✅ Linux可执行文件生成成功"
    ls -lh ./docker/Files/evolvegcn_sharding_linux
else
    echo "❌ Linux可执行文件生成失败"
fi
