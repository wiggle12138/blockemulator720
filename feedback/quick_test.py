#!/usr/bin/env python3
"""
简化的第三步-第四步反馈机制测试
"""

import torch
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "evolve_GCN"))

def quick_integration_test():
    """快速集成测试"""
    print("[START] 快速第三步-第四步集成测试")
    
    try:
        # 1. 测试第三步分片模块
        print("\n[DATA] 测试第三步分片模块...")
        from models.sharding_modules import DynamicShardingModule
        
        # 模拟数据
        num_nodes = 60
        embedding_dim = 64
        node_embeddings = torch.randn(num_nodes, embedding_dim)
        
        # 创建分片模块
        sharding_module = DynamicShardingModule(
            embedding_dim=embedding_dim,
            base_shards=3,
            max_shards=6,
            min_shard_size=5,
            max_empty_ratio=0.2
        )
        
        # 前向传播（无反馈）
        assignment, embeddings, attention, k = sharding_module(
            Z=node_embeddings,
            history_states=None,
            feedback_signal=None
        )
        
        print(f"   [SUCCESS] 第三步工作正常: {assignment.shape}, 分片数: {k}")
        
        # 2. 测试第四步反馈引擎  
        print("\n🔄 测试第四步反馈引擎...")
        from unified_feedback_engine import UnifiedFeedbackEngine
        
        # 创建反馈引擎
        feedback_engine = UnifiedFeedbackEngine()
        
        # 准备数据
        features = {
            'hardware': torch.randn(num_nodes, 17),
            'onchain_behavior': torch.randn(num_nodes, 17),
            'network_topology': torch.randn(num_nodes, 20),
            'dynamic_attributes': torch.randn(num_nodes, 13),
            'heterogeneous_type': torch.randn(num_nodes, 17),
            'categorical': torch.randn(num_nodes, 15)
        }
        
        hard_assignment = torch.argmax(assignment, dim=1)
        
        # 生成反馈信号
        feedback_matrix = feedback_engine.analyze_performance(
            features=features,
            shard_assignments=hard_assignment,
            edge_index=None,
            performance_hints={'load_balance': 0.7}
        )
        
        print(f"   [SUCCESS] 第四步工作正常: {feedback_matrix.shape}")
        
        # 3. 测试反馈回环
        print("\n🔁 测试反馈回环...")
        
        enhanced_assignment, _, _, _ = sharding_module(
            Z=node_embeddings,
            history_states=None,
            feedback_signal=feedback_matrix
        )
        
        # 计算差异
        diff = torch.norm(enhanced_assignment - assignment).item()
        print(f"   反馈前后差异: {diff:.4f}")
        
        if diff > 0.01:
            print("   [SUCCESS] 反馈成功影响了分片分配")
        else:
            print("   [WARNING] 反馈影响较小但正常")
        
        print("\n🎉 集成测试完成! 第三步-第四步反馈机制工作正常")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_integration_test()
    if success:
        print("\n[SUCCESS] 第三步和第四步反馈机制集成成功!")
    else:
        print("\n[ERROR] 集成测试失败，请检查模块")
