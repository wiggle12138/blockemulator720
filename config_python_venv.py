#!/usr/bin/env python3
"""
Python虚拟环境配置检测脚本
用于检测并配置BlockEmulator项目的Python虚拟环境
"""

import os
import sys
import json
from pathlib import Path

def detect_virtual_environment():
    """检测可用的Python虚拟环境"""
    # Docker容器内优先检查虚拟环境
    docker_venv = "/opt/venv/bin/python"
    if os.path.exists(docker_venv):
        print(f"[SUCCESS] 发现Docker虚拟环境: {docker_venv}")
        # 验证虚拟环境中的依赖
        try:
            import subprocess
            result = subprocess.run([docker_venv, "-c", "import torch, numpy"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"[SUCCESS] Docker虚拟环境依赖完整: {docker_venv}")
                return docker_venv
            else:
                print(f"[WARNING] Docker虚拟环境依赖不完整: {result.stderr.strip()}")
        except Exception as e:
            print(f"[WARNING] Docker虚拟环境测试失败: {e}")
    
    # 检测常见的虚拟环境路径
    potential_venvs = [
        # Windows虚拟环境路径
        r"E:\Codefield\BlockEmulator\.venv\Scripts\python.exe",
        r".\.venv\Scripts\python.exe",
        r".\venv\Scripts\python.exe",
        r".\.env\Scripts\python.exe",
        r".\env\Scripts\python.exe",
    ]
    
    # 检查当前是否在虚拟环境中
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        current_python = sys.executable
        print(f"[SUCCESS] 当前已在虚拟环境中: {current_python}")
        return current_python
    
    # 检测可用的虚拟环境
    for venv_path in potential_venvs:
        if os.path.exists(venv_path):
            print(f"[SUCCESS] 发现虚拟环境: {venv_path}")
            return venv_path
    
    # 回退到系统Python
    print(f"[WARNING] 未发现虚拟环境，使用系统Python: {sys.executable}")
    return sys.executable

def update_python_config(python_path):
    """更新python_config.json配置文件"""
    config_file = "python_config.json"
    
    try:
        # 读取现有配置
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
        
        # 更新Python路径
        config["python_path"] = python_path
        config["enable_evolve_gcn"] = True
        config["enable_feedback"] = True
        
        # 保存配置
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"[SUCCESS] 已更新配置文件 {config_file}")
        print(f"   Python路径: {python_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] 更新配置文件失败: {e}")
        return False

def test_python_environment(python_path):
    """测试Python环境是否可用"""
    try:
        import subprocess
        
        # 测试基础Python
        result = subprocess.run([python_path, "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print(f"[ERROR] Python环境不可用: {python_path}")
            return False
        
        print(f"[SUCCESS] Python版本: {result.stdout.strip()}")
        
        # 测试关键依赖
        test_imports = [
            "import torch",
            "import numpy",
            "import json",
            "import sys"
        ]
        
        for test_import in test_imports:
            result = subprocess.run([python_path, "-c", test_import], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print(f"[WARNING] 依赖检查失败: {test_import}")
                print(f"   错误: {result.stderr.strip()}")
            else:
                print(f"[SUCCESS] 依赖检查通过: {test_import}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 测试Python环境失败: {e}")
        return False

def main():
    """主函数"""
    print("[CONFIG] BlockEmulator Python虚拟环境配置")
    print("=" * 50)
    
    # 检测虚拟环境
    python_path = detect_virtual_environment()
    
    # 测试环境
    if test_python_environment(python_path):
        # 更新配置
        if update_python_config(python_path):
            print("\n[SUCCESS] Python虚拟环境配置完成！")
            print(f"使用的Python路径: {python_path}")
            return python_path
    
    print("\n[ERROR] Python虚拟环境配置失败！")
    return None

if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)
