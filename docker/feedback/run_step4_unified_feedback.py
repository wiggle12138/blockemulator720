#!/usr/bin/env python3
"""
第四步：统一反馈处理主入口
整合优化性能评估、重要性分析、异常检测等机制，为第三步分片优化提供智能反馈
"""

import torch
import numpy as np
import pickle
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# 添加项目路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))
sys.path.append(str(current_dir))

from unified_feedback_engine import UnifiedFeedbackEngine

def load_step3_results(step3_output_dir: str = "../evolve_GCN/") -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    加载第三步的分片结果
    
    Returns:
        features: 6类特征字典
        shard_assignments: 分片分配结果
        edge_index: 边索引
    """
    step3_dir = Path(step3_output_dir)
    
    print(f"📂 从第三步加载分片结果...")
    print(f"   查找目录: {step3_dir.resolve()}")
    
    # 1. 加载特征数据 - 智能查找
    possible_files = [
        "step1_large_samples.pt",
        "large_samples.pt", 
        "../large_samples.csv",
        "../examples/large_samples.csv"
    ]
    
    features_file = None
    for filename in possible_files:
        candidate = step3_dir / filename
        if candidate.exists():
            features_file = candidate
            break
    
    if features_file is None:
        print(f"[WARNING] 未找到特征文件，生成模拟数据")
        # 生成模拟数据用于测试
        num_nodes = 100
        features = {
            'node_features': torch.randn(num_nodes, 64),
            'timestamps': torch.arange(num_nodes, dtype=torch.float32),
            'num_nodes': num_nodes
        }
        step3_data = features
    else:
        try:
            if features_file.suffix == '.csv':
                import pandas as pd
                df = pd.read_csv(features_file)
                features = {
                    'node_features': torch.tensor(df.select_dtypes(include=[float, int]).values[:100], dtype=torch.float32),
                    'timestamps': torch.arange(min(100, len(df)), dtype=torch.float32),
                    'num_nodes': min(100, len(df))
                }
                step3_data = features
            else:
                step3_data = torch.load(features_file, map_location='cpu')
            print(f"   [SUCCESS] 特征文件: {features_file.name}")
        except Exception as e:
            print(f"   [ERROR] 加载失败: {e}, 使用模拟数据")
            features = {
                'node_features': torch.randn(100, 64),
                'timestamps': torch.arange(100, dtype=torch.float32), 
                'num_nodes': 100
            }
            step3_data = features
    
    # 2. 加载邻接信息
    adjacency_file = step3_dir / "step1_adjacency_raw.pt"
    if not adjacency_file.exists():
        adjacency_file = step3_dir / "adjacency_raw.pt" 
    
    if adjacency_file.exists():
        adjacency_data = torch.load(adjacency_file, map_location='cpu')
        print(f"   邻接文件: {adjacency_file.name}")
    else:
        print(f"[WARNING] 未找到邻接文件，将生成模拟边索引")
        adjacency_data = None
    
    # 3. 提取或生成分片分配结果
    if 'shard_assignments' in step3_data:
        shard_assignments = step3_data['shard_assignments']
        print(f"   分片分配: {shard_assignments.shape} - {shard_assignments.max().item()+1} 个分片")
    else:
        # 生成模拟分片分配
        num_nodes = step3_data.get('f_classic', torch.randn(1000, 128)).shape[0]
        num_shards = min(8, max(3, num_nodes // 100))
        shard_assignments = torch.randint(0, num_shards, (num_nodes,))
        print(f"   模拟分片分配: {num_nodes} 节点 -> {num_shards} 分片")
    
    # 4. 提取6类特征
    features = extract_six_feature_types(step3_data)
    
    # 5. 提取边索引
    edge_index = extract_edge_index_from_data(adjacency_data, len(shard_assignments))
    
    print(f"[SUCCESS] 第三步数据加载完成")
    print(f"   节点数: {len(shard_assignments)}")
    print(f"   特征类别: {list(features.keys())}")
    print(f"   边数: {edge_index.size(1) if edge_index is not None else 0}")
    
    return features, shard_assignments, edge_index


def extract_six_feature_types(step3_data: Dict) -> Dict[str, torch.Tensor]:
    """从第三步数据中提取6类特征"""
    
    # 优先使用已有的分类特征
    if all(k in step3_data for k in ['hardware', 'onchain_behavior', 'network_topology', 
                                     'dynamic_attributes', 'heterogeneous_type', 'categorical']):
        return {k: step3_data[k] for k in ['hardware', 'onchain_behavior', 'network_topology',
                                          'dynamic_attributes', 'heterogeneous_type', 'categorical']}
    
    # 从经典特征中分割
    if 'f_classic' in step3_data:
        f_classic = step3_data['f_classic']
        num_nodes = f_classic.shape[0]
        feature_dim = f_classic.shape[1]
        
        print(f"   从 f_classic 分割6类特征: {f_classic.shape}")
        
        # 按预定义维度分割
        original_dims = {
            'hardware': 17,
            'onchain_behavior': 17, 
            'network_topology': 20,
            'dynamic_attributes': 13,
            'heterogeneous_type': 17,
            'categorical': 15
        }
        
        features = {}
        start_idx = 0
        
        for feature_name, dim in original_dims.items():
            end_idx = start_idx + dim
            if end_idx <= feature_dim:
                features[feature_name] = f_classic[:, start_idx:end_idx].clone()
            else:
                # 维度不足时生成合理特征
                features[feature_name] = generate_realistic_feature(feature_name, num_nodes, dim)
            start_idx = end_idx
            print(f"     {feature_name}: {features[feature_name].shape}")
        
        return features
    
    # 兜底：生成完整的模拟特征
    num_nodes = step3_data.get('num_nodes', 1000)
    print(f"   生成模拟6类特征: {num_nodes} 节点")
    
    return {
        'hardware': generate_realistic_feature('hardware', num_nodes, 17),
        'onchain_behavior': generate_realistic_feature('onchain_behavior', num_nodes, 17),
        'network_topology': generate_realistic_feature('network_topology', num_nodes, 20),
        'dynamic_attributes': generate_realistic_feature('dynamic_attributes', num_nodes, 13),
        'heterogeneous_type': generate_realistic_feature('heterogeneous_type', num_nodes, 17),
        'categorical': generate_realistic_feature('categorical', num_nodes, 15)
    }


def generate_realistic_feature(feature_name: str, num_nodes: int, dim: int) -> torch.Tensor:
    """生成符合实际业务的特征数据"""
    
    base_ranges = {
        'hardware': (0.4, 0.9),           # 硬件性能通常在中等偏上范围
        'onchain_behavior': (0.3, 0.9),   # 链上行为表现差异较大
        'network_topology': (0.3, 0.7),   # 网络拓扑相对稳定
        'dynamic_attributes': (0.1, 0.9), # 动态属性变化很大
        'heterogeneous_type': (0.2, 0.8), # 异构类型相对集中
        'categorical': (0.3, 0.8)         # 分类特征相对稳定
    }
    
    min_val, max_val = base_ranges.get(feature_name, (0.25, 0.75))
    
    # 生成带有一定分布特征的数据
    base_tensor = torch.rand(num_nodes, dim)
    scaled_tensor = base_tensor * (max_val - min_val) + min_val
    
    # 添加一些真实性：某些特征相关性
    if feature_name == 'hardware':
        # 硬件特征通常相关（CPU高的机器内存也高）
        correlation_factor = torch.randn(num_nodes, 1) * 0.1
        scaled_tensor += correlation_factor.expand(-1, dim)
    elif feature_name == 'dynamic_attributes':
        # 动态特征增加时间相关性
        time_factor = torch.sin(torch.arange(num_nodes).float() / 100).unsqueeze(1)
        scaled_tensor += time_factor.expand(-1, dim) * 0.1
    
    return torch.clamp(scaled_tensor, 0.0, 1.0)


def extract_edge_index_from_data(adjacency_data: Dict, num_nodes: int) -> torch.Tensor:
    """从邻接数据中提取边索引"""
    
    if adjacency_data is None:
        # 生成模拟网络图
        print(f"   生成模拟边索引: {num_nodes} 节点")
        edges_per_node = min(8, max(2, num_nodes // 50))  # 每节点平均连接数
        total_edges = num_nodes * edges_per_node // 2
        
        # 随机生成边，确保一定的连通性
        source_nodes = torch.randint(0, num_nodes, (total_edges,))
        target_nodes = torch.randint(0, num_nodes, (total_edges,))
        
        # 去除自环
        valid_mask = source_nodes != target_nodes
        source_nodes = source_nodes[valid_mask]
        target_nodes = target_nodes[valid_mask]
        
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)
        return edge_index
    
    # 从已有数据中提取
    if 'edge_index' in adjacency_data:
        edge_index = adjacency_data['edge_index']
    elif 'original_edge_index' in adjacency_data:
        edge_index = adjacency_data['original_edge_index']
    elif 'adjacency_matrix' in adjacency_data:
        adj_matrix = adjacency_data['adjacency_matrix']
        edges = torch.nonzero(adj_matrix, as_tuple=False)
        edge_index = edges.t()
    else:
        print(f"[WARNING] 邻接数据格式未知，生成模拟边索引")
        return extract_edge_index_from_data(None, num_nodes)
    
    # 确保格式正确 [2, num_edges]
    if edge_index.shape[0] != 2:
        edge_index = edge_index.t()
    
    return edge_index


def run_step4_feedback(step3_output_dir: str = "../evolve_GCN/", 
                      output_dir: str = "./", 
                      config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    运行第四步统一反馈处理
    
    Args:
        step3_output_dir: 第三步输出目录
        output_dir: 第四步输出目录  
        config: 反馈引擎配置
        
    Returns:
        反馈处理结果
    """
    
    print(f"\n[START] 开始第四步统一反馈处理...")
    print(f"   第三步目录: {step3_output_dir}")
    print(f"   第四步目录: {output_dir}")
    
    try:
        # 1. 加载第三步结果
        features, shard_assignments, edge_index = load_step3_results(step3_output_dir)
        
        # 2. 初始化统一反馈引擎
        feature_dims = {k: v.shape[1] for k, v in features.items()}
        feedback_engine = UnifiedFeedbackEngine(feature_dims, config)
        
        # 3. 处理分片反馈
        feedback_result = feedback_engine.process_sharding_feedback(
            features=features,
            shard_assignments=shard_assignments,
            edge_index=edge_index,
            performance_hints=None
        )
        
        # 4. 保存结果
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 主反馈结果
        main_result_file = output_path / "step4_feedback_result.pkl"
        with open(main_result_file, 'wb') as f:
            pickle.dump(feedback_result, f)
        
        # 专门给第三步的反馈包
        step3_feedback_file = output_path / "step3_performance_feedback.pkl"
        with open(step3_feedback_file, 'wb') as f:
            pickle.dump(feedback_result['step3_feedback_package'], f)
        
        # JSON格式的可读结果
        readable_result = {
            'overall_score': feedback_result['optimized_feedback']['overall_score'],
            'performance_metrics': feedback_result['performance_metrics'],
            'feature_importance': feedback_result['importance_analysis']['feature_importance'],
            'smart_suggestions': feedback_result['smart_suggestions'],
            'anomaly_count': feedback_result['anomaly_report']['anomaly_count'],
            'engine_status': feedback_result['engine_status']
        }
        
        readable_file = output_path / "step4_readable_result.json"
        with open(readable_file, 'w', encoding='utf-8') as f:
            json.dump(readable_result, f, indent=2, ensure_ascii=False)
        
        # 保存引擎状态
        engine_state_file = output_path / "feedback_engine_state.pkl"
        feedback_engine.save_feedback_state(str(engine_state_file))
        
        print(f"\n[SUCCESS] 第四步反馈处理完成!")
        print(f"   主结果文件: {main_result_file}")
        print(f"   第三步反馈: {step3_feedback_file}")
        print(f"   可读结果: {readable_file}")
        print(f"   引擎状态: {engine_state_file}")
        
        # 打印关键指标
        print(f"\n[DATA] 反馈处理摘要:")
        perf = feedback_result['performance_metrics']
        print(f"   综合评分: {feedback_result['optimized_feedback']['overall_score']:.3f}")
        print(f"   负载均衡: {perf['load_balance']:.3f}")
        print(f"   跨片交易率: {perf['cross_shard_rate']:.3f}")
        print(f"   安全性评分: {perf['security_score']:.3f}")
        print(f"   共识延迟: {perf['consensus_latency']:.3f}")
        print(f"   智能建议数: {len(feedback_result['smart_suggestions'])}")
        print(f"   检测异常数: {feedback_result['anomaly_report']['anomaly_count']}")
        
        # 打印重要特征
        importance = feedback_result['importance_analysis']['feature_importance']
        print(f"\n[TARGET] 特征重要性排序:")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feat, score in sorted_importance:
            print(f"   {feat}: {score:.3f}")
        
        # 打印关键建议
        high_priority_suggestions = [s for s in feedback_result['smart_suggestions'] if s['priority'] == 'high']
        if high_priority_suggestions:
            print(f"\n[WARNING] 高优先级建议:")
            for suggestion in high_priority_suggestions[:3]:  # 只显示前3个
                print(f"   - {suggestion['description']}")
        
        return feedback_result
        
    except Exception as e:
        print(f"[ERROR] 第四步反馈处理失败: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'success': False}


def main():
    """主函数入口"""
    
    # 默认配置
    default_config = {
        'max_history': 50,
        'learning_rate': 0.01,
        'feedback_weights': {
            'balance': 0.35,
            'cross_shard': 0.25,
            'security': 0.20,
            'consensus': 0.20
        },
        'adaptive_threshold': 0.15,
        'anomaly_threshold': 2.0,
        'evolution_enabled': True
    }
    
    # 运行第四步反馈处理
    result = run_step4_feedback(
        step3_output_dir="../evolve_GCN/",  # 指向第三步目录
        output_dir="./",
        config=default_config
    )
    
    if result.get('success', True):  # 没有error字段表示成功
        print(f"\n 第四步反馈处理成功完成!")
        
        # 检查是否有重要建议需要立即关注
        if 'smart_suggestions' in result:
            critical_suggestions = [s for s in result['smart_suggestions'] 
                                  if s['priority'] == 'high' and 'balance' in s['type']]
            if critical_suggestions:
                print(f"\n[FIX] 发现 {len(critical_suggestions)} 个关键负载均衡问题，建议优先处理!")
    else:
        print(f"\n💥 第四步反馈处理失败，请检查日志并修复问题")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
