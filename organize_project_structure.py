#!/usr/bin/env python3
"""
项目目录结构整理脚本
将文件按功能分类到对应目录，保持根目录简洁
"""
import os
import shutil
from pathlib import Path

def organize_project_structure():
    """整理项目目录结构"""
    print("=== BlockEmulator 项目目录整理 ===\n")
    
    # 创建目标目录
    dirs_to_create = [
        "tests",         # 测试脚本
        "scripts",       # 启动和运行脚本  
        "configs",       # 配置文件
        "data",          # 数据文件
    ]
    
    for dir_name in dirs_to_create:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 创建目录: {dir_name}/")
    
    # 定义文件移动规则
    file_moves = {
        # 测试脚本 -> tests/
        "tests/": [
            "quick_real_system_test.py",
            "quick_sharding_test.py", 
            "minimal_real_test.py",
            "simplified_sharding_test.py",
            "system_verification.py",
            "training_ablation_test.py",
            "quick_test.bat"
        ],
        
        # 运行脚本 -> scripts/
        "scripts/": [
            "run_enhanced_pipeline.py",
            "run_steps_python.py", 
            "step1_to_step2_pipeline.py",
            "integrated_four_step_pipeline.py",
            "integration_complete.py",
            "simplified_integration_fixed.py",
            "package_python_sharding.py",
            "full_test.bat",
            "full_test.sh",
            "start_evolvegcn_integrated.bat",
            "start_integration.bat",
            "start-blockemulator-utf8.ps1",
            "run-blockemulator.ps1",
            "run-blockemulator-preload.ps1", 
            "run-blockemulator-preload-safe.ps1",
            "windows_exe_run_IpAddr=127_0_0_1.bat",
            "deploy-integrated.ps1"
        ],
        
        # 配置文件 -> configs/  
        "configs/": [
            "python_config.json",
            "integration_config.json",
            "real_system_test_config.json",
            "evolve_gcn_feedback_config.json",
            "ipTable.json"
        ],
        
        # 数据文件 -> data/
        "data/": [
            "selectedTxs_100K.csv",
            "selectedTxs_300K.csv", 
            "node_features_input.csv"
        ]
    }
    
    # 根目录保留的核心文件
    keep_in_root = [
        "main.go", "go.mod", "go.sum", 
        "blockEmulator.exe", "blockEmulator_Windows_UTF8.exe",
        "paramsConfig.json",  # 主配置文件保留在根目录
        "LICENSE", "README.md", "wrs.md", "NODE_FEATURES_README.md",
        "2024Dec31-(139页)使用指南-黄华威.pdf",
        "new_workflow.md", "optimized_testing_guide.md",
        # 各种GUIDE文档
        "EVOLVEGCN_INTEGRATED_GUIDE.md", 
        "EVOLVEGCN_INTEGRATION_GUIDE.md",
        "INTEGRATION_README.md",
        "LIGHTWEIGHT_INTEGRATION_GUIDE.md", 
        "PYTHON_INTEGRATION_GUIDE.md",
        # 核心接口文件
        "blockchain_interface.py",
        "blockemulator_integration_interface.py",
        "evolvegcn_go_interface.py",
        "config_loader.py",
        "config_python_venv.py"
    ]
    
    moved_files = []
    
    # 执行文件移动
    for target_dir, files in file_moves.items():
        for file_name in files:
            src_path = Path(file_name)
            if src_path.exists():
                try:
                    dst_path = Path(target_dir) / file_name
                    shutil.move(str(src_path), str(dst_path))
                    moved_files.append(f"{file_name} -> {target_dir}")
                    print(f"📄 移动: {file_name} -> {target_dir}")
                except Exception as e:
                    print(f"❌ 移动失败: {file_name} - {e}")
    
    # 移动Docker相关文件到docker目录 (如果不存在则创建)
    docker_files = ["Dockerfile.integrated", "Dockerfile.light", "docker-compose.integrated.yml"]
    if not Path("docker").exists():
        Path("docker").mkdir()
        print("📁 创建目录: docker/")
    
    for docker_file in docker_files:
        src = Path(docker_file)
        if src.exists():
            try:
                dst = Path("docker") / docker_file
                shutil.move(str(src), str(dst))
                moved_files.append(f"{docker_file} -> docker/")
                print(f"🐳 移动: {docker_file} -> docker/")
            except Exception as e:
                print(f"❌ 移动Docker文件失败: {docker_file} - {e}")
    
    # 清理输出
    print(f"\n=== 整理完成 ===")
    print(f"移动文件数量: {len(moved_files)}")
    
    return moved_files

def create_structure_summary():
    """创建整理后的目录结构说明"""
    content = """# 整理后的项目结构

## 📂 目录说明

### 根目录 (核心文件)
- `main.go` - Go程序主入口
- `blockEmulator.exe` - 主程序可执行文件  
- `paramsConfig.json` - 主要系统配置
- `README.md`, `LICENSE` - 项目说明和许可证
- 各种集成接口文件和配置加载器

### tests/ (测试目录) 
- 所有测试脚本和验证工具
- `quick_real_system_test.py` - 真实系统集成测试
- `system_verification.py` - 系统验证脚本

### scripts/ (脚本目录)
- 启动脚本、运行脚本、集成流水线
- `start_evolvegcn_integrated.bat` - 集成启动脚本
- `run_enhanced_pipeline.py` - 增强流水线

### configs/ (配置目录) 
- 各种JSON配置文件
- `python_config.json` - Python模块配置
- `integration_config.json` - 集成配置

### data/ (数据目录)
- CSV数据文件和输入数据
- `selectedTxs_*.csv` - 交易数据
- `node_features_input.csv` - 节点特征数据

### 源码目录 (保持不变)
- `partition/` - 分片相关源码
- `evolve_GCN/` - EvolveGCN算法实现  
- `feedback/` - 反馈系统
- `muti_scale/` - 多尺度学习

## 🎯 整理效果
- 根目录更加简洁，只保留核心系统文件
- 按功能分类，便于查找和维护
- 保持了重要文件的可访问性
"""

    with open("project_structure.md", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("📋 已生成结构说明: project_structure.md")

if __name__ == "__main__":
    try:
        moved = organize_project_structure()
        create_structure_summary()
        
        print(f"\n✨ 项目结构整理完成! 现在目录结构更加清晰易懂。")
        print("💡 建议查看 project_structure.md 了解新的目录结构")
        
    except Exception as e:
        print(f"❌ 整理过程出错: {e}")
        import traceback
        traceback.print_exc()
