"""
空分片处理机制验证脚本
专门测试第三步DynamicShardingModule的空分片检测与处理功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from evolve_GCN.models.sharding_modules import DynamicShardingModule
from evolve_GCN.models.temporal_conv import TemporalConvNet


def generate_test_data(num_nodes=100, embedding_dim=128, scenario="normal"):
    """
    生成测试数据，模拟不同场景
    
    Args:
        num_nodes: 节点数量
        embedding_dim: 嵌入维度
        scenario: 测试场景 - "normal", "clustered", "sparse", "extreme_sparse"
    """
    torch.manual_seed(42)
    
    if scenario == "normal":
        # 正常分布的嵌入
        embeddings = torch.randn(num_nodes, embedding_dim)
        
    elif scenario == "clustered":
        # 聚类分布的嵌入（容易产生空分片）
        embeddings = torch.zeros(num_nodes, embedding_dim)
        cluster_centers = torch.randn(3, embedding_dim) * 2
        
        nodes_per_cluster = num_nodes // 3
        for i, center in enumerate(cluster_centers):
            start_idx = i * nodes_per_cluster
            end_idx = start_idx + nodes_per_cluster if i < 2 else num_nodes
            embeddings[start_idx:end_idx] = center + torch.randn(end_idx - start_idx, embedding_dim) * 0.3
            
    elif scenario == "sparse":
        # 极度聚集的嵌入（大部分节点聚集在少数点附近）
        embeddings = torch.randn(num_nodes, embedding_dim) * 0.1
        # 90%的节点聚集在原点附近
        main_cluster_size = int(num_nodes * 0.9)
        embeddings[:main_cluster_size] = torch.randn(main_cluster_size, embedding_dim) * 0.05
        
    elif scenario == "extreme_sparse":
        # 极端稀疏场景（几乎所有节点都在同一点）
        embeddings = torch.randn(num_nodes, embedding_dim) * 0.01
        
    # 生成历史状态（负载均衡度, 跨片交易率, 安全阈值）
    history_states = [
        torch.tensor([0.8, 0.15, 0.9]),  # 较好的平衡
        torch.tensor([0.6, 0.25, 0.85]), # 中等平衡
        torch.tensor([0.4, 0.35, 0.8])   # 较差的平衡
    ]
    
    return embeddings, history_states


def test_empty_shard_handling():
    """测试空分片处理机制"""
    print("=" * 70)
    print("空分片处理机制验证测试")
    print("=" * 70)
    
    # 测试参数
    embedding_dim = 128
    base_shards = 5
    max_shards = 10
    
    # 初始化分片模块
    sharding_module = DynamicShardingModule(
        embedding_dim=embedding_dim,
        base_shards=base_shards,
        max_shards=max_shards,
        min_shard_size=8,      # 最小分片大小
        max_empty_ratio=0.3    # 最大空分片比例30%
    )
    
    # 测试场景
    scenarios = [
        ("正常分布", "normal", 100),
        ("聚类分布", "clustered", 100),
        ("稀疏分布", "sparse", 80),
        ("极端稀疏", "extreme_sparse", 60),
        ("小节点集", "clustered", 30)
    ]
    
    for scenario_name, scenario_type, num_nodes in scenarios:
        print(f"\n{'='*20} {scenario_name} ({num_nodes}个节点) {'='*20}")
        
        # 生成测试数据
        embeddings, history_states = generate_test_data(num_nodes, embedding_dim, scenario_type)
        
        # 前向传播
        with torch.no_grad():
            S_t, enhanced_embeddings, attention_weights, K_t = sharding_module(
                embeddings, history_states
            )
        
        # 分析分片结果
        hard_assignment = torch.argmax(S_t, dim=1)
        shard_sizes = torch.bincount(hard_assignment, minlength=int(K_t))
        
        # 统计信息
        empty_shards = torch.sum(shard_sizes == 0).item()
        min_shard_size = torch.min(shard_sizes[shard_sizes > 0]).item()
        max_shard_size = torch.max(shard_sizes).item()
        avg_shard_size = torch.mean(shard_sizes[shard_sizes > 0].float()).item()
        
        print(f"  最终分片数: {K_t}")
        print(f"  分片大小分布: {shard_sizes.tolist()}")
        print(f"  空分片数量: {empty_shards}")
        print(f"  最小分片大小: {min_shard_size}")
        print(f"  最大分片大小: {max_shard_size}")
        print(f"  平均分片大小: {avg_shard_size:.1f}")
        
        # 健康度检查
        empty_ratio = empty_shards / K_t
        size_variance = torch.var(shard_sizes[shard_sizes > 0].float()).item()
        
        print(f"  空分片比例: {empty_ratio:.1%}")
        print(f"  分片大小方差: {size_variance:.1f}")
        
        # 健康度评级
        if empty_ratio == 0 and min_shard_size >= sharding_module.min_shard_size:
            health_status = "[SUCCESS] 健康"
        elif empty_ratio <= 0.2 and min_shard_size >= sharding_module.min_shard_size // 2:
            health_status = "[WARNING]  可接受"
        else:
            health_status = "[ERROR] 需要改进"
            
        print(f"  健康状态: {health_status}")


def test_progressive_adjustment():
    """测试分片数渐进调整功能"""
    print(f"\n{'='*70}")
    print("分片数渐进调整测试")
    print("=" * 70)
    
    embedding_dim = 128
    num_nodes = 80
    base_shards = 4
    
    sharding_module = DynamicShardingModule(
        embedding_dim=embedding_dim,
        base_shards=base_shards,
        max_shards=12,
        min_shard_size=6,
        max_empty_ratio=0.25
    )
    
    embeddings, _ = generate_test_data(num_nodes, embedding_dim, "clustered")
    
    # 模拟多个时间步的历史状态变化
    history_sequence = [
        [torch.tensor([0.9, 0.1, 0.95])],  # t=1: 高平衡，预期增加分片
        [torch.tensor([0.7, 0.2, 0.9])],   # t=2: 中平衡
        [torch.tensor([0.4, 0.4, 0.8])],   # t=3: 低平衡，预期减少分片
        [torch.tensor([0.3, 0.5, 0.75])],  # t=4: 更低平衡
        [torch.tensor([0.6, 0.25, 0.85])], # t=5: 恢复中等平衡
    ]
    
    print("时间步\t预测分片数\t实际分片数\t分片数变化\t空分片数")
    print("-" * 60)
    
    prev_k = base_shards
    
    for t, history_states in enumerate(history_sequence, 1):
        with torch.no_grad():
            S_t, _, _, K_t = sharding_module(embeddings, history_states)
        
        # 分析结果
        hard_assignment = torch.argmax(S_t, dim=1)
        shard_sizes = torch.bincount(hard_assignment, minlength=K_t)
        empty_shards = torch.sum(shard_sizes == 0).item()
        
        change = K_t - prev_k
        change_str = f"+{change}" if change > 0 else str(change) if change < 0 else "0"
        
        print(f"t={t}\t\t{getattr(sharding_module, 'prev_shard_count', '?')}\t\t{K_t}\t\t{change_str}\t\t{empty_shards}")
        
        prev_k = K_t


def test_feedback_integration():
    """测试反馈信号融合功能"""
    print(f"\n{'='*70}")
    print("反馈信号融合测试")
    print("=" * 70)
    
    embedding_dim = 128
    num_nodes = 60
    base_shards = 4
    
    sharding_module = DynamicShardingModule(
        embedding_dim=embedding_dim,
        base_shards=base_shards,
        max_shards=8
    )
    
    embeddings, history_states = generate_test_data(num_nodes, embedding_dim, "normal")
    
    # 生成模拟的第四步反馈信号
    # 假设反馈信号建议某些节点应该重新分配
    feedback_signal = torch.randn(num_nodes, base_shards)
    feedback_signal = torch.softmax(feedback_signal, dim=1)
    
    print("测试有无反馈信号的分片结果差异:")
    
    # 无反馈信号
    with torch.no_grad():
        S_no_feedback, _, _, K_no_feedback = sharding_module(embeddings, history_states)
    
    # 生成与实际分片数匹配的反馈信号
    matched_feedback_signal = torch.randn(num_nodes, K_no_feedback)
    matched_feedback_signal = torch.softmax(matched_feedback_signal, dim=1)
    
    # 有反馈信号
    with torch.no_grad():
        S_with_feedback, _, _, K_with_feedback = sharding_module(
            embeddings, history_states, matched_feedback_signal
        )
    
    print(f"  无反馈 - 分片数: {K_no_feedback}, 形状: {S_no_feedback.shape}")
    print(f"  有反馈 - 分片数: {K_with_feedback}, 形状: {S_with_feedback.shape}")
    
    # 计算分配差异 - 处理可能的维度不匹配
    if S_no_feedback.shape == S_with_feedback.shape:
        assignment_diff = torch.mean(torch.abs(S_no_feedback - S_with_feedback)).item()
        print(f"  分配矩阵差异: {assignment_diff:.4f}")
        
        if assignment_diff > 0.01:
            print("  [SUCCESS] 反馈信号成功影响了分片分配")
        else:
            print("  [WARNING] 反馈信号影响较小")
    else:
        print("  [WARNING] 分片数发生变化，无法直接比较")
        # 比较分片分布
        no_feedback_dist = torch.sum(S_no_feedback, dim=0).tolist()
        with_feedback_dist = torch.sum(S_with_feedback, dim=0).tolist()
        print(f"  无反馈分片分布: {[f'{x:.1f}' for x in no_feedback_dist]}")
        print(f"  有反馈分片分布: {[f'{x:.1f}' for x in with_feedback_dist]}")
    
    print(f"  反馈融合系数: {sharding_module.feedback_alpha.item():.4f}")
    
    # 测试不匹配维度的反馈处理
    print("\n测试维度不匹配的反馈信号处理:")
    mismatched_feedback = torch.softmax(torch.randn(num_nodes, 6), dim=1)  # 固定6个分片
    
    try:
        with torch.no_grad():
            S_mismatched, _, _, K_mismatched = sharding_module(
                embeddings, history_states, mismatched_feedback
            )
        print(f"  [SUCCESS] 成功处理维度不匹配反馈: {mismatched_feedback.shape} -> {S_mismatched.shape}")
    except Exception as e:
        print(f"  [ERROR] 处理维度不匹配反馈失败: {e}")


if __name__ == "__main__":
    print("开始验证空分片处理机制...")
    
    # 运行各项测试
    test_empty_shard_handling()
    test_progressive_adjustment()
    test_feedback_integration()
    
    print(f"\n{'='*70}")
    print("验证完成！")
    print("=" * 70)
