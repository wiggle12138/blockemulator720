#!/usr/bin/env python3
"""
简化版EvolveGCN接口，用于快速测试Go-Python集成
"""

import json
import argparse
import sys
import time
from pathlib import Path

def quick_test_pipeline(input_data):
    """快速测试流水线，不进行复杂计算"""
    print("[QUICK_TEST] Starting simplified pipeline...")
    
    # 简单的分片逻辑：按节点ID的hash分配
    partition_map = {}
    node_count = len(input_data.get("node_features", []))
    
    for i, node_feature in enumerate(input_data.get("node_features", [])):
        node_id = node_feature.get("node_id", f"node_{i}")
        # 简单的轮询分片
        shard_id = i % 2  # 分配到0或1分片
        partition_map[node_id] = shard_id
    
    # 计算跨分片边数
    edges = input_data.get("edges", [])
    cross_shard_edges = 0
    
    for edge in edges:
        if len(edge) >= 2:
            src_node, dst_node = edge[0], edge[1]
            if (src_node in partition_map and dst_node in partition_map and 
                partition_map[src_node] != partition_map[dst_node]):
                cross_shard_edges += 1
    
    return {
        "success": True,
        "partition_map": partition_map,
        "cross_shard_edges": cross_shard_edges,
        "metrics": {
            "total_nodes": node_count,
            "total_edges": len(edges),
            "cross_shard_edges": cross_shard_edges,
            "cross_shard_rate": cross_shard_edges / max(len(edges), 1),
            "load_balance": 1.0,
            "security_score": 0.85
        },
        "performance_score": 1.0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "suggestions": ["Quick test completed successfully"]
    }

def main():
    parser = argparse.ArgumentParser(description='Quick EvolveGCN Interface for Testing')
    parser.add_argument('--input', required=True, help='Input JSON file')
    parser.add_argument('--output', required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    try:
        print(f"[QUICK_TEST] Reading input: {args.input}")
        
        # 读取输入文件
        if not Path(args.input).exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
            
        with open(args.input, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        print(f"[QUICK_TEST] Processing {len(input_data.get('node_features', []))} nodes")
        
        # 执行快速测试流水线
        result = quick_test_pipeline(input_data)
        
        # 写入输出文件
        print(f"[QUICK_TEST] Writing output: {args.output}")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        print("[QUICK_TEST] Pipeline completed successfully")
        print(f"[QUICK_TEST] Processed {result['metrics']['total_nodes']} nodes")
        print(f"[QUICK_TEST] Cross-shard edges: {result['cross_shard_edges']}")
        
        return 0
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, indent=2)
        except:
            pass
            
        print(f"[QUICK_TEST] Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
