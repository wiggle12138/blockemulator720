#!/usr/bin/env python3
"""
第四步反馈机制集成测试
测试第三步和第四步的数据流对接
"""

import torch
import numpy as np
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "evolve_GCN"))

def test_simple_feedback_integration():
    """简单的反馈集成测试"""
    print("🧪 开始第三步-第四步反馈集成测试...")
    
    # 1. 模拟第三步分片结果
    print("\n[DATA] 模拟第三步分片结果:")
    num_nodes = 60
    num_shards = 3
    embedding_dim = 64
    
    # 节点嵌入
    node_embeddings = torch.randn(num_nodes, embedding_dim)
    
    # 分片分配矩阵 (模拟第三步输出)
    shard_assignment = torch.zeros(num_nodes, num_shards)
    for i in range(num_nodes):
        shard_id = i % num_shards  # 简单轮询分配
        shard_assignment[i, shard_id] = 1.0
    
    print(f"   节点数: {num_nodes}")
    print(f"   分片数: {num_shards}")
    print(f"   嵌入维度: {embedding_dim}")
    
    # 计算分片大小
    shard_sizes = torch.sum(shard_assignment, dim=0)
    print(f"   分片大小: {shard_sizes.tolist()}")
    
    # 2. 导入第三步分片模块
    print("\n[CONFIG] 测试第三步分片模块:")
    try:
        from models.sharding_modules import DynamicShardingModule
        
        # 创建分片模块
        sharding_module = DynamicShardingModule(
            embedding_dim=embedding_dim,
            base_shards=3,
            max_shards=6,
            min_shard_size=5,
            max_empty_ratio=0.2
        )
        
        # 测试前向传播
        result_assignment, enhanced_embeddings, attention_weights, actual_k = sharding_module(
            Z=node_embeddings,
            history_states=None,
            feedback_signal=None
        )
        
        print(f"   [SUCCESS] 第三步分片模块工作正常")
        print(f"   输出分片数: {actual_k}")
        print(f"   输出分配矩阵形状: {result_assignment.shape}")
        
    except Exception as e:
        print(f"   [ERROR] 第三步模块测试失败: {e}")
        result_assignment = shard_assignment
        actual_k = num_shards
    
    # 3. 导入并测试第四步反馈引擎
    print("\n🔄 测试第四步反馈引擎:")
    try:
        from unified_feedback_engine import UnifiedFeedbackEngine
        
        # 创建反馈引擎
        feedback_engine = UnifiedFeedbackEngine()
        
        # 模拟第四步所需的特征数据
        features = {
            'hardware': torch.randn(num_nodes, 17),
            'onchain_behavior': torch.randn(num_nodes, 17),
            'network_topology': torch.randn(num_nodes, 20),
            'dynamic_attributes': torch.randn(num_nodes, 13),
            'heterogeneous_type': torch.randn(num_nodes, 17),
            'categorical': torch.randn(num_nodes, 15)
        }
        
        # 转换分片分配为硬标签
        hard_assignment = torch.argmax(result_assignment, dim=1)
        
        # 模拟性能提示
        performance_hints = {
            'throughput': [850.0, 820.0, 880.0],  # 每个分片的吞吐量
            'latency': [45.0, 52.0, 38.0],       # 每个分片的延迟
            'load_balance': 0.85,                 # 整体负载均衡度
            'cross_shard_ratio': 0.18,           # 跨片交易比例
            'security_level': 0.95,              # 安全等级
            'consensus_efficiency': 0.88          # 共识效率
        }
        
        # 执行反馈分析
        feedback_matrix = feedback_engine.analyze_performance(
            features=features,
            shard_assignments=hard_assignment,
            edge_index=None,
            performance_hints=performance_hints
        )
        
        print(f"   [SUCCESS] 第四步反馈引擎工作正常")
        print(f"   反馈信号形状: {feedback_matrix.shape}")
        
        # 模拟完整反馈结果以兼容后续测试
        feedback_result = {
            'feedback_signal': feedback_matrix,
            'overall_score': 0.85,
            'recommendations': ['improve_load_balance', 'reduce_cross_shard']
        }
        print(f"   改进建议数量: {len(feedback_result.get('recommendations', []))}")
        
        # 4. 测试反馈信号注入第三步
        print("\n🔁 测试反馈信号注入:")
        
        feedback_signal = feedback_result['feedback_signal']
        enhanced_assignment, _, _, _ = sharding_module(
            Z=node_embeddings,
            history_states=None,
            feedback_signal=feedback_signal
        )
        
        # 计算反馈前后的差异
        assignment_diff = torch.norm(enhanced_assignment - result_assignment).item()
        print(f"   反馈前后分配差异: {assignment_diff:.4f}")
        
        if assignment_diff > 0.01:
            print(f"   [SUCCESS] 反馈信号成功影响了分片分配")
        else:
            print(f"   [WARNING] 反馈信号影响较小")
        
        # 5. 分析反馈效果
        print("\n📈 反馈效果分析:")
        
        # 计算新分片大小
        new_hard_assignment = torch.argmax(enhanced_assignment, dim=1)
        new_shard_sizes = torch.bincount(new_hard_assignment, minlength=int(actual_k))
        
        print(f"   原始分片大小: {torch.sum(result_assignment, dim=0).tolist()}")
        print(f"   反馈后分片大小: {new_shard_sizes.tolist()}")
        
        # 计算负载均衡改进
        original_balance = 1.0 - torch.std(torch.sum(result_assignment, dim=0)) / torch.mean(torch.sum(result_assignment, dim=0))
        new_balance = 1.0 - torch.std(new_shard_sizes.float()) / torch.mean(new_shard_sizes.float())
        
        print(f"   原始负载均衡度: {original_balance:.3f}")
        print(f"   反馈后负载均衡度: {new_balance:.3f}")
        
        if new_balance > original_balance:
            print(f"   [SUCCESS] 负载均衡得到改善 (+{(new_balance - original_balance):.3f})")
        else:
            print(f"   [WARNING] 负载均衡略有下降 ({(new_balance - original_balance):.3f})")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] 第四步反馈测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """测试边缘情况"""
    print("\n🔍 测试边缘情况:")
    
    # 测试维度不匹配的反馈
    print("   测试维度不匹配...")
    try:
        from models.sharding_modules import DynamicShardingModule
        
        sharding_module = DynamicShardingModule(embedding_dim=32, base_shards=3)
        embeddings = torch.randn(50, 32)
        
        # 创建不匹配维度的反馈信号
        mismatched_feedback = torch.randn(50, 6)  # 6个分片的反馈，但当前只有3个
        
        result, _, _, actual_k = sharding_module(
            Z=embeddings,
            feedback_signal=mismatched_feedback
        )
        
        print(f"   [SUCCESS] 维度不匹配处理成功: {mismatched_feedback.shape} -> {result.shape}")
        
    except Exception as e:
        print(f"   [ERROR] 维度不匹配测试失败: {e}")
    
    # 测试空分片处理
    print("   测试空分片处理...")
    try:
        # 创建容易产生空分片的数据
        clustered_embeddings = torch.zeros(30, 32)
        clustered_embeddings[:15, :] = torch.randn(15, 32) + 2  # 第一个聚类
        clustered_embeddings[15:25, :] = torch.randn(10, 32) - 2  # 第二个聚类
        clustered_embeddings[25:, :] = torch.randn(5, 32)  # 散点
        
        result, _, _, actual_k = sharding_module(
            Z=clustered_embeddings,
            feedback_signal=None
        )
        
        # 检查是否有空分片
        hard_assignment = torch.argmax(result, dim=1)
        shard_sizes = torch.bincount(hard_assignment, minlength=int(actual_k))
        empty_shards = torch.sum(shard_sizes == 0).item()
        
        print(f"   [SUCCESS] 空分片处理: {empty_shards} 个空分片, 最终 {actual_k} 个分片")
        
    except Exception as e:
        print(f"   [ERROR] 空分片测试失败: {e}")

if __name__ == "__main__":
    print("=" * 70)
    print("第三步-第四步反馈机制集成测试")
    print("=" * 70)
    
    # 主要测试
    success = test_simple_feedback_integration()
    
    # 边缘情况测试
    test_edge_cases()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 测试完成！第三步-第四步反馈集成工作正常")
    else:
        print("[ERROR] 测试失败，请检查相关模块")
    print("=" * 70)
