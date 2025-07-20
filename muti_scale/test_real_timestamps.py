"""
测试第二步多尺度对比学习的真实时间步处理
验证与BlockEmulator的时间戳集成
"""

import torch
import numpy as np
import time
from pathlib import Path
import sys

# 添加路径
sys.path.append(str(Path(__file__).parent))

from step2_config import Step2Config
from realtime_mscia import RealtimeMSCIAProcessor


def test_real_timestamp_processing():
    """测试真实时间戳处理功能"""
    print("=== 测试真实时间戳处理 ===")
    
    # 获取配置
    config = Step2Config().get_blockemulator_integration_config()
    
    # 初始化处理器
    processor = RealtimeMSCIAProcessor(config)
    
    # 测试不同类型的时间戳
    test_cases = [
        # (逻辑时间步, 真实时间戳, 描述)
        (0, None, "纯逻辑时间戳"),
        (1, time.time(), "Unix时间戳"),
        (2, 1234.567, "相对时间戳（秒）"),
        (3, 12345678, "大数值时间戳"),
        (4, 0.5, "小数值时间戳")
    ]
    
    for logical_ts, real_ts, description in test_cases:
        print(f"\n--- 测试: {description} ---")
        print(f"输入: 逻辑={logical_ts}, 真实={real_ts}")
        
        # 测试时间戳处理
        processed_ts = processor._process_real_timestamp(logical_ts, real_ts)
        normalized_ts = processor._normalize_timestamp_for_model(processed_ts)
        
        print(f"处理后: {processed_ts}")
        print(f"标准化: {normalized_ts}")
        
        # 模拟第一步输出
        num_nodes = 20
        mock_step1_output = {
            'f_classic': torch.randn(num_nodes, 128),
            'f_graph': torch.randn(num_nodes, 96),
            'node_mapping': {i: f"node_{i}" for i in range(num_nodes)},
            'metadata': {
                'num_nodes': num_nodes,
                'test_case': description
            }
        }
        
        # 处理第二步
        try:
            result = processor.process_step1_output(
                mock_step1_output, 
                timestamp=logical_ts,
                blockemulator_timestamp=real_ts
            )
            
            print(f"✓ 处理成功:")
            print(f"  输出形状: {result['temporal_embeddings'].shape}")
            print(f"  损失: {result['loss'].item():.4f}")
            print(f"  元数据包含真实时间: {'real_timestamp' in result['metadata']}")
            
        except Exception as e:
            print(f"[ERROR] 处理失败: {e}")


def test_time_window_with_real_timestamps():
    """测试真实时间戳的时间窗口处理"""
    print("\n=== 测试真实时间戳时间窗口 ===")
    
    config = Step2Config().get_blockemulator_integration_config()
    processor = RealtimeMSCIAProcessor(config)
    
    # 模拟一系列带有真实时间戳的数据
    base_time = time.time()
    time_sequence = [
        (0, base_time),
        (1, base_time + 2.5),      # 2.5秒后
        (2, base_time + 5.1),      # 5.1秒后
        (3, base_time + 7.8),      # 7.8秒后
        (4, base_time + 10.2),     # 10.2秒后
        (5, base_time + 12.9)      # 12.9秒后
    ]
    
    results = []
    
    for logical_ts, real_ts in time_sequence:
        print(f"\n--- 时间步 {logical_ts} (真实时间: {real_ts:.2f}) ---")
        
        num_nodes = 30
        mock_step1_output = {
            'f_classic': torch.randn(num_nodes, 128),
            'f_graph': torch.randn(num_nodes, 96),
            'node_mapping': {i: f"node_{i}" for i in range(num_nodes)},
            'metadata': {
                'num_nodes': num_nodes,
                'sequence_position': logical_ts
            }
        }
        
        result = processor.process_step1_output(
            mock_step1_output,
            timestamp=logical_ts,
            blockemulator_timestamp=real_ts
        )
        
        results.append(result)
        
        # 检查时间窗口信息
        temporal_context = result['metadata'].get('temporal_context', {})
        print(f"时间窗口大小: {temporal_context.get('window_size', 0)}")
        print(f"时间戳序列: {temporal_context.get('timestamps', [])}")
    
    print(f"\n✓ 完成 {len(results)} 个时间步的处理")
    print(f"最终时间窗口大小: {results[-1]['metadata']['temporal_context']['window_size']}")


def test_blockemulator_integration_format():
    """测试BlockEmulator集成格式"""
    print("\n=== 测试BlockEmulator集成格式 ===")
    
    config = Step2Config().get_blockemulator_integration_config()
    processor = RealtimeMSCIAProcessor(config)
    
    # 模拟真实的BlockEmulator数据格式
    current_time = time.time()
    
    # 第一步的典型输出（来自blockemulator_adapter.py）
    blockemulator_step1_output = {
        'f_classic': torch.randn(100, 128),    # 经典特征
        'f_graph': torch.randn(100, 96),       # 图特征
        'f_reduced': torch.randn(100, 64),     # 精简特征（第二步不直接使用）
        'node_mapping': {i: f"shard_0_node_{i}" for i in range(100)},
        'metadata': {
            'processing_time': 0.123,
            'nodes_per_second': 813.0,
            'shard_info': {'total_shards': 4, 'current_shard': 0},
            'feature_extraction_method': 'unified_extractor',
            'blockemulator_timestamp': current_time,
            'simulation_round': 42
        }
    }
    
    # 处理第二步
    result = processor.process_step1_output(
        blockemulator_step1_output,
        timestamp=42,  # 模拟区块链回合数
        blockemulator_timestamp=current_time
    )
    
    print("✓ BlockEmulator格式处理成功:")
    print(f"  输入节点数: 100")
    print(f"  输出嵌入: {result['temporal_embeddings'].shape}")
    print(f"  保留映射: {len(result['node_mapping'])} 个节点")
    print(f"  时间信息完整: {result['metadata']['real_time_processed']}")
    print(f"  原始BE时间戳: {result['metadata']['real_timestamp']}")
    print(f"  处理后时间戳: {result['metadata']['processed_timestamp']}")
    
    # 显示输出格式兼容性
    format_info = processor.get_output_format_info()
    print(f"\n第二步输出格式兼容性:")
    print(f"  下一步兼容: {format_info['next_step_compatibility']}")
    print(f"  时间戳保存: {format_info['metadata_enrichment']}")


if __name__ == "__main__":
    print("开始测试第二步多尺度对比学习的真实时间步处理...")
    
    try:
        test_real_timestamp_processing()
        test_time_window_with_real_timestamps()
        test_blockemulator_integration_format()
        
        print("\n🎉 所有测试完成！第二步现在支持真实时间步处理。")
        
    except Exception as e:
        print(f"\n[ERROR] 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
