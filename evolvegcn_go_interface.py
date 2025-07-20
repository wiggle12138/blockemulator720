#!/usr/bin/env python3
"""
BlockEmulator <-> EvolveGCN Interface
Go program calls Python EvolveGCN algorithm through this script
Input: JSON format node features and transaction graph
[CHAR][CHAR]: JSON[CHAR][CHAR][CHAR][SHARD][RESULT]
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

# [CHAR][CHAR][PATH]
sys.path.append('.')
sys.path.append('./partition')
sys.path.append('./partition/feature')
sys.path.append('./muti_scale')
sys.path.append('./evolve_GCN')
sys.path.append('./evolve_GCN/models')
sys.path.append('./feedback')

try:
    # 优先使用真实的四步分片系统集成
    from real_integrated_four_step_pipeline import RealIntegratedFourStepPipeline as OriginalIntegratedFourStepPipeline
    print("[SUCCESS] 导入真实四步分片系统集成")
except ImportError as e1:
    try:
        # 如果真实系统不可用，回退到原来的实现
        from integrated_four_step_pipeline import OriginalIntegratedFourStepPipeline
        print("[WARNING] 使用简化版四步分片系统 (真实系统不可用)")
    except ImportError as e2:
        print(f"[ERROR] 无法导入任何四步分片系统: 真实系统={e1}, 简化系统={e2}")
        sys.exit(1)

class EvolveGCNGoInterface:
    """Go[CHAR][CHAR][CALL]EvolveGCN[CHAR][CHAR][CHAR]"""
    
    def __init__(self):
        self.pipeline = None
        
    def initialize_pipeline(self, config=None):
        """[CHAR][CHAR][CHAR]EvolveGCN[CHAR][CHAR][CHAR]"""
        try:
            self.pipeline = OriginalIntegratedFourStepPipeline(config)
            return True
        except Exception as e:
            print(f"[CHAR][CHAR][CHAR][CHAR][CHAR][CHAR][FAILED]: {e}")
            return False
    
    def run_four_step_pipeline(self, input_data):
        """
        [CHAR][CHAR][CHAR][CHAR]EvolveGCN[CHAR][CHAR][CHAR]
        
        [CHAR][CHAR][CHAR][CHAR]:
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
        
        [CHAR][CHAR][CHAR][CHAR]:
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
            # [CHAR][CHAR][CHAR][CHAR][DATA]
            node_features = input_data.get("node_features", [])
            transaction_graph = input_data.get("transaction_graph", {})
            config = input_data.get("config", {})
            
            # [CHAR][CHAR][CHAR][CHAR][DATA][CHAR][CHAR]
            pipeline_input = self._prepare_pipeline_input(node_features, transaction_graph)
            
            # [CHAR][CHAR][CHAR][CHAR][CHAR][CHAR][CHAR]
            print("[START] [LAUNCH][CHAR][CHAR]EvolveGCN[CHAR][CHAR][CHAR]...")
            result = self.pipeline.run_complete_pipeline_with_data(pipeline_input)
            
            # [CHAR][CHAR][CHAR][CHAR][CHAR][CHAR]
            output = self._format_pipeline_output(result)
            
            print(f"[SUCCESS] EvolveGCN流水线完成，跨分片边数: {output.get('cross_shard_edges', 0)}")
            return output
            
        except Exception as e:
            error_msg = f"EvolveGCN流水线执行失败: {str(e)}"
            print(f"[ERROR] {error_msg}")
            traceback.print_exc()
            return {"success": False, "error": error_msg}
    
    def _prepare_pipeline_input(self, node_features, transaction_graph):
        """[CHAR][CHAR][CHAR][CHAR][CHAR][CHAR][CHAR][DATA]"""
        # [CHAR][CHAR][NODE][FEATURE][CHAR][CHAR]
        processed_features = []
        for node in node_features:
            node_data = {
                'node_id': node['node_id'],
                'features': np.array(node['features']),
                'shard_info': node.get('metadata', {})
            }
            processed_features.append(node_data)
        
        # [CHAR][CHAR][CHAR][CHAR][CHAR][CHAR][CHAR]
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
        """[CHAR][CHAR][CHAR][CHAR][CHAR][CHAR][CHAR][CHAR]"""
        if not result or not result.get('success'):
            return {"success": False, "error": "Pipeline execution failed"}
        
        # [CHAR][CHAR][SHARD][RESULT]
        final_sharding = result.get('final_sharding', {})
        partition_map = {}
        cross_shard_edges = 0
        
        for shard_id, shard_info in final_sharding.items():
            if isinstance(shard_info, dict) and 'nodes' in shard_info:
                for node_id in shard_info['nodes']:
                    partition_map[str(node_id)] = int(shard_id)
        
        # [CHAR][CHAR][CHAR][SHARD][CHAR][CHAR]
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
    """[CHAR][FUNC]"""
    parser = argparse.ArgumentParser(description='EvolveGCN Go Interface')
    parser.add_argument('--input', '-i', required=True, help='[CHAR][CHAR]JSON[FILE][PATH]')
    parser.add_argument('--output', '-o', required=True, help='[CHAR][CHAR]JSON[FILE][PATH]')
    parser.add_argument('--config', '-c', help='[CONFIG][FILE][PATH]')
    
    args = parser.parse_args()
    
    try:
        # [READ][CHAR][CHAR][DATA]
        with open(args.input, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # [READ][CONFIG][CHAR][CHAR][CHAR][CHAR][CHAR][CHAR]
        config = None
        if args.config and Path(args.config).exists():
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
        
        # [CHAR][CHAR][CHAR][CHAR][CHAR][CHAR]
        interface = EvolveGCNGoInterface()
        
        # [CHAR][CHAR][CHAR][CHAR][CHAR][CHAR]
        if not interface.initialize_pipeline(config):
            result = {"success": False, "error": "Failed to initialize pipeline"}
        else:
            # [CHAR][CHAR][CHAR][CHAR][CHAR]
            result = interface.run_four_step_pipeline(input_data)
        
        # [SAVE][RESULT]
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # [CHAR][CHAR][RESULT][CHAR][CHAR]
        if result.get('success'):
            print(f"[SUCCESS] 成功完成EvolveGCN处理")
            print(f"[DATA] 分片映射: {len(result.get('partition_map', {}))} 个节点")
            print(f"[METRICS] 跨分片边数: {result.get('cross_shard_edges', 0)}")
            sys.exit(0)
        else:
            print(f"[ERROR] EvolveGCN处理失败: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        error_result = {"success": False, "error": str(e)}
        
        # [CHAR][CHAR][SAVE][ERROR][RESULT]
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, ensure_ascii=False, indent=2)
        except:
            pass
        
        print(f"[ERROR] [CHAR][CHAR][EXECUTE][ERROR]: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
