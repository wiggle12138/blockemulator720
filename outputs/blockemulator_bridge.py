#!/usr/bin/env python3
"""
BlockEmulator兼容桥梁
用于在现有BlockEmulator系统中集成四步算法结果
"""

import json
import sys
from pathlib import Path

def apply_resharding_results(results_file: str):
    """应用重分片结果到BlockEmulator"""
    
    # 读取Python算法输出
    with open(results_file, 'r', encoding='utf-8') as f:
        if results_file.endswith('.json'):
            results = json.load(f)
        else:
            import pickle
            with open(results_file, 'rb') as pf:
                results = pickle.load(pf)
    
    # 创建集成接口
    from blockemulator_integration_interface import BlockEmulatorIntegrationInterface
    interface = BlockEmulatorIntegrationInterface()
    
    # 应用结果
    status = interface.apply_four_step_results_to_blockemulator(results)
    
    print(f"应用状态: {status}")
    return status

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python blockemulator_bridge.py <results_file>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    apply_resharding_results(results_file)
