#!/usr/bin/env python3
"""
BlockEmulator <-> EvolveGCN接口
Go程序通过此脚本调用Python的EvolveGCN算法
输入: JSON格式的节点特征和交易图
输出: JSON格式的分片结果
"""

import sys
import json
import argparse
import traceback
from pathlib import Path
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

# 添加路径
sys.path.append('.')
sys.path.append('./partition')
sys.path.append('./partition/feature')
sys.path.append('./muti_scale')
sys.path.append('./evolve_GCN')
sys.path.append('./evolve_GCN/models')
sys.path.append('./feedback')

try:
    from integrated_four_step_pipeline import OriginalIntegratedFourStepPipeline
except ImportError as e:
    print(f"错误: 无法导入四步流水线: {e}")
    sys.exit(1)

class EvolveGCNGoInterface:
    """Go程序调用EvolveGCN的接口"""
    
    def __init__(self):
        self.pipeline = None
        
    def initialize_pipeline(self, config=None):
        """初始化EvolveGCN流水线"""
        try:
            self.pipeline = OriginalIntegratedFourStepPipeline(config)
            return True
        except Exception as e:
            print(f"初始化流水线失败: {e}")
            return False
    
    def run_four_step_pipeline(self, input_data):
        """
        运行四步EvolveGCN流水线
        
        输入格式:
        {
            "node_features": [
                {"node_id": "addr1", "features": [0.1, 0.2, ...], "metadata": {...}},
                ...
            ],
            "transaction_graph": {
                "edges": [["addr1", "addr2", weight], ...],
                "metadata": {...}
            },
            "config": {...}
        }
        
        输出格式:
        {
            "success": true,
            "partition_map": {"addr1": 0, "addr2": 1, ...},
            "cross_shard_edges": 123,
            "metrics": {...},
            "timestamp": "..."
        }
        """
        if not self.pipeline:
            return {"success": False, "error": "Pipeline not initialized"}
            
        try:
            # 解析输入数据
            node_features = input_data.get("node_features", [])
            transaction_graph = input_data.get("transaction_graph", {})
            config = input_data.get("config", {})
            
            # 创建输入数据结构
            pipeline_input = self._prepare_pipeline_input(node_features, transaction_graph)
            
            # 运行完整流水线
            print("[START] 启动四步EvolveGCN流水线...")
            result = self.pipeline.run_complete_pipeline_with_data(pipeline_input)
            
            # 转换输出格式
            output = self._format_pipeline_output(result)
            
            print(f"[SUCCESS] EvolveGCN流水线完成，跨分片边数: {output.get('cross_shard_edges', 0)}")
            return output
            
        except Exception as e:
            error_msg = f"EvolveGCN流水线执行失败: {str(e)}"
            print(f"[ERROR] {error_msg}")
            traceback.print_exc()
            return {"success": False, "error": error_msg}
    
    def _prepare_pipeline_input(self, node_features, transaction_graph):
        """准备流水线输入数据"""
        # 转换节点特征格式
        processed_features = []
        for node in node_features:
            node_data = {
                'node_id': node['node_id'],
                'features': np.array(node['features']),
                'shard_info': node.get('metadata', {})
            }
            processed_features.append(node_data)
        
        # 转换交易图格式
        edges = transaction_graph.get('edges', [])
        graph_data = {
            'edges': edges,
            'nodes': [node['node_id'] for node in node_features],
            'edge_weights': [edge[2] if len(edge) > 2 else 1.0 for edge in edges]
        }
        
        return {
            'node_features': processed_features,
            'transaction_graph': graph_data,
            'metadata': transaction_graph.get('metadata', {})
        }
    
    def _format_pipeline_output(self, result):
        """格式化流水线输出"""
        if not result or not result.get('success'):
            return {"success": False, "error": "Pipeline execution failed"}
        
        # 提取分片结果
        final_sharding = result.get('final_sharding', {})
        partition_map = {}
        cross_shard_edges = 0
        
        for shard_id, shard_info in final_sharding.items():
            if isinstance(shard_info, dict) and 'nodes' in shard_info:
                for node_id in shard_info['nodes']:
                    partition_map[str(node_id)] = int(shard_id)
        
        # 计算跨分片边数
        if 'metrics' in result:
            cross_shard_edges = result['metrics'].get('cross_shard_edges', 0)
        
        return {
            "success": True,
            "partition_map": partition_map,
            "cross_shard_edges": cross_shard_edges,
            "metrics": result.get('metrics', {}),
            "performance_score": result.get('performance_score', 0.0),
            "timestamp": result.get('timestamp', ''),
            "suggestions": result.get('suggestions', [])
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='EvolveGCN Go Interface')
    parser.add_argument('--input', '-i', required=True, help='输入JSON文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出JSON文件路径')
    parser.add_argument('--config', '-c', help='配置文件路径')
    
    args = parser.parse_args()
    
    try:
        # 读取输入数据
        with open(args.input, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # 读取配置（如果提供）
        config = None
        if args.config and Path(args.config).exists():
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
        
        # 创建接口实例
        interface = EvolveGCNGoInterface()
        
        # 初始化流水线
        if not interface.initialize_pipeline(config):
            result = {"success": False, "error": "Failed to initialize pipeline"}
        else:
            # 运行流水线
            result = interface.run_four_step_pipeline(input_data)
        
        # 保存结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 输出结果状态
        if result.get('success'):
            print(f"[SUCCESS] 成功完成EvolveGCN处理")
            print(f"[DATA] 分片映射: {len(result.get('partition_map', {}))} 个节点")
            print(f"[LOOP] 跨分片边数: {result.get('cross_shard_edges', 0)}")
            sys.exit(0)
        else:
            print(f"[ERROR] EvolveGCN处理失败: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        error_result = {"success": False, "error": str(e)}
        
        # 尝试保存错误结果
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, ensure_ascii=False, indent=2)
        except:
            pass
        
        print(f"[ERROR] 脚本执行错误: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
