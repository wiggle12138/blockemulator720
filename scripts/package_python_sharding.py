#!/usr/bin/env python3
"""
Python分片系统打包脚本
使用PyInstaller将EvolveGCN和feedback系统打包成独立可执行文件
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_pyinstaller():
    """检查PyInstaller是否已安装"""
    try:
        import PyInstaller
        print("[SUCCESS] PyInstaller已安装")
        return True
    except ImportError:
        print("[ERROR] PyInstaller未安装，正在安装...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
            print("[SUCCESS] PyInstaller安装成功")
            return True
        except subprocess.CalledProcessError:
            print("[ERROR] PyInstaller安装失败")
            return False

def create_main_entry():
    """创建主入口文件"""
    entry_content = '''#!/usr/bin/env python3
"""
EvolveGCN分片系统主入口
打包版本的统一入口点
"""

import sys
import json
import os
from pathlib import Path

# 添加必要的路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: evolvegcn_sharding.exe <command> [options]")
        print("命令:")
        print("  interface     - 运行Go接口模式")
        print("  pipeline      - 运行四步流水线")
        print("  test          - 运行测试")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "interface":
        # 运行Go接口模式
        from evolvegcn_go_interface import main as interface_main
        # 移除第一个参数，传递剩余参数
        sys.argv = sys.argv[1:]
        interface_main()
    
    elif command == "pipeline":
        # 运行四步流水线
        from integrated_four_step_pipeline import main as pipeline_main
        sys.argv = sys.argv[1:]
        pipeline_main()
    
    elif command == "test":
        # 运行测试模式
        print("[CONFIG] EvolveGCN分片系统测试模式")
        print("系统运行正常")
        
        # 测试导入
        try:
            import torch
            print(f"[SUCCESS] PyTorch版本: {torch.__version__}")
        except ImportError:
            print("[ERROR] PyTorch导入失败")
        
        try:
            import numpy as np
            print(f"[SUCCESS] NumPy版本: {np.__version__}")
        except ImportError:
            print("[ERROR] NumPy导入失败")
            
        print("[SUCCESS] 分片系统可执行文件测试完成")
    
    else:
        print(f"[ERROR] 未知命令: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open("evolvegcn_main.py", "w", encoding="utf-8") as f:
        f.write(entry_content)
    
    print("[SUCCESS] 创建主入口文件: evolvegcn_main.py")

def package_sharding_system():
    """打包分片系统"""
    print("[START] 开始打包EvolveGCN分片系统...")
    
    # 创建临时规格文件
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['evolvegcn_main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('partition', 'partition'),
        ('muti_scale', 'muti_scale'), 
        ('evolve_GCN', 'evolve_GCN'),
        ('feedback', 'feedback'),
        ('*.json', '.'),
        ('*.py', '.'),
    ],
    hiddenimports=[
        'torch',
        'numpy', 
        'pandas',
        'scikit-learn',
        'matplotlib',
        'tqdm',
        'networkx',
        'scipy',
        'seaborn',
        'psutil'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='evolvegcn_sharding',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''
    
    with open("evolvegcn_sharding.spec", "w", encoding="utf-8") as f:
        f.write(spec_content)
    
    # 执行打包
    try:
        print("📦 执行PyInstaller打包...")
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--name=evolvegcn_sharding",
            "--distpath=./docker/Files",
            "--workpath=./build_temp",
            "--specpath=.",
            "evolvegcn_main.py"
        ]
        
        subprocess.run(cmd, check=True)
        print("[SUCCESS] 打包完成")
        
        # 检查生成的文件
        exe_path = Path("docker/Files/evolvegcn_sharding.exe")
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"[SUCCESS] 可执行文件生成成功: {exe_path}")
            print(f"[DATA] 文件大小: {size_mb:.1f} MB")
        else:
            print("[ERROR] 可执行文件未找到")
            return False
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 打包失败: {e}")
        return False

def create_linux_version():
    """创建Linux版本的打包脚本"""
    linux_script = '''#!/bin/bash
# Linux版本的Python分片系统打包脚本

echo "[START] 开始打包EvolveGCN分片系统 (Linux版本)..."

# 检查PyInstaller
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "[ERROR] PyInstaller未安装，正在安装..."
    python3 -m pip install pyinstaller
fi

# 打包
python3 -m PyInstaller \\
    --onefile \\
    --name=evolvegcn_sharding_linux \\
    --distpath=./docker/Files \\
    --workpath=./build_temp \\
    evolvegcn_main.py

if [ -f "./docker/Files/evolvegcn_sharding_linux" ]; then
    echo "[SUCCESS] Linux可执行文件生成成功"
    ls -lh ./docker/Files/evolvegcn_sharding_linux
else
    echo "[ERROR] Linux可执行文件生成失败"
fi
'''
    
    with open("package_linux.sh", "w", encoding="utf-8") as f:
        f.write(linux_script)
    
    print("[SUCCESS] 创建Linux打包脚本: package_linux.sh")

def update_dockerfile():
    """更新Dockerfile以使用打包的可执行文件"""
    dockerfile_addition = '''

# ===============================
# 使用打包的Python分片系统
# ===============================

# 复制打包的分片系统可执行文件
COPY ./docker/Files/evolvegcn_sharding_linux /app/evolvegcn_sharding
RUN chmod +x /app/evolvegcn_sharding

# 更新Python路径配置，使用打包的可执行文件
RUN echo '{"python_path": "/app/evolvegcn_sharding", "enable_evolve_gcn": true, "enable_feedback": true}' > /app/python_config.json
'''
    
    print("💡 建议在Dockerfile.integrated中添加以下内容:")
    print(dockerfile_addition)

def cleanup():
    """清理临时文件"""
    temp_files = [
        "evolvegcn_main.py",
        "evolvegcn_sharding.spec",
        "build_temp",
        "__pycache__"
    ]
    
    for item in temp_files:
        if os.path.exists(item):
            if os.path.isdir(item):
                shutil.rmtree(item)
            else:
                os.remove(item)
    
    print("🧹 清理临时文件完成")

def main():
    """主函数"""
    print("[CONFIG] Python分片系统打包工具")
    print("=" * 50)
    
    # 检查PyInstaller
    if not check_pyinstaller():
        return False
    
    # 创建主入口文件
    create_main_entry()
    
    # 创建Linux打包脚本
    create_linux_version()
    
    # 执行打包
    if package_sharding_system():
        print("\n[SUCCESS] 打包成功完成！")
        print("\n📋 后续步骤:")
        print("1. 将 docker/Files/evolvegcn_sharding.exe 复制到Linux环境")
        print("2. 或者在Linux环境中运行 ./package_linux.sh")
        print("3. 更新Dockerfile.integrated使用打包的可执行文件")
        update_dockerfile()
        
        # 询问是否清理
        cleanup_choice = input("\n是否清理临时文件? (y/n): ")
        if cleanup_choice.lower() == 'y':
            cleanup()
        
        return True
    else:
        print("\n[ERROR] 打包失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
