#!/usr/bin/env python3
"""
真正的真实四步分片系统测试脚本
修复了导入问题，确保调用真实的系统组件
"""

import sys
import time
import json
import logging
import torch
from pathlib import Path
from typing import Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_real_step1():
    """测试真实的第一步：特征提取"""
    logger.info("[STEP1] 测试真实特征提取...")
    
    try:
        # 使用绝对导入，避免相对导入问题
        sys.path.insert(0, str(Path("partition/feature").absolute()))
        
        from partition.feature.system_integration_pipeline import BlockEmulatorStep1Pipeline
        
        # 创建模拟的NodeFeaturesModule
        class MockNodeFeaturesModule:
            def GetAllCollectedData(self):
                mock_data = []
                for i in range(20):
                    node_data = {
                        'ShardID': i % 4,
                        'NodeID': i,
                        'Timestamp': int(time.time() * 1000),
                        'RequestID': f"req_{i}",
                        'NodeState': {
                            'Static': {
                                'ResourceCapacity': {
                                    'Hardware': {
                                        'CPU': {'CoreCount': 4, 'Architecture': 'amd64'},
                                        'Memory': {'TotalCapacity': 8, 'Type': 'DDR4', 'Bandwidth': 50.0},
                                        'Storage': {'Capacity': 100, 'Type': 'SSD', 'ReadWriteSpeed': 500.0},
                                        'Network': {'UpstreamBW': 100.0, 'DownstreamBW': 1000.0, 'Latency': 50.0}
                                    }
                                },
                                'NetworkTopology': {
                                    'GeoLocation': {'Timezone': 'UTC+8'},
                                    'Connections': {
                                        'IntraShardConn': 3, 'InterShardConn': 2,
                                        'WeightedDegree': 5.0, 'ActiveConn': 4
                                    },
                                    'ShardAllocation': {'Adaptability': 0.7}
                                },
                                'HeterogeneousType': {
                                    'NodeType': 'full_node',
                                    'FunctionTags': 'consensus,validation',
                                    'SupportedFuncs': {'Functions': 'tx_processing'},
                                    'Application': {
                                        'CurrentState': 'active',
                                        'LoadMetrics': {'TxFrequency': 100, 'StorageOps': 50}
                                    }
                                }
                            },
                            'Dynamic': {
                                'OnChainBehavior': {
                                    'TransactionCapability': {
                                        'AvgTPS': 50.0,
                                        'CrossShardTx': {'InterNodeVolume': '1MB', 'InterShardVolume': '5MB'},
                                        'ConfirmationDelay': 100.0,  # 改为数值
                                        'ResourcePerTx': {
                                            'CPUPerTx': 0.1, 'MemPerTx': 0.05,
                                            'DiskPerTx': 0.02, 'NetworkPerTx': 0.01
                                        }
                                    },
                                    'BlockGeneration': {
                                        'AvgInterval': 5.0, 'IntervalStdDev': 1.0  # 改为数值
                                    },
                                    'TransactionTypes': {
                                        'NormalTxRatio': 0.8, 'ContractTxRatio': 0.2
                                    },
                                    'Consensus': {
                                        'ParticipationRate': 0.9, 'TotalReward': 100.0, 'SuccessRate': 0.95
                                    },
                                    'SmartContractUsage': {'InvocationFrequency': 0},
                                    'EconomicContribution': {'FeeContributionRatio': 0.01}
                                },
                                'DynamicAttributes': {
                                    'Compute': {
                                        'CPUUsage': 30.0, 'MemUsage': 40.0, 'ResourceFlux': 0.1
                                    },
                                    'Storage': {
                                        'Available': 80.0, 'Utilization': 20.0
                                    },
                                    'Network': {
                                        'LatencyFlux': 0.05, 'AvgLatency': 50.0, 'BandwidthUsage': 0.3  # 改为数值
                                    },
                                    'Transactions': {
                                        'Frequency': 10, 'ProcessingDelay': 200.0  # 改为数值
                                    }
                                }
                            }
                        }
                    }
                    mock_data.append(node_data)
                return mock_data
        
        # 初始化真实的第一步管道
        pipeline = BlockEmulatorStep1Pipeline()
        mock_module = MockNodeFeaturesModule()
        
        # 执行真实的特征提取
        result = pipeline.extract_features_from_system(
            node_features_module=mock_module,
            experiment_name="real_test"
        )
        
        logger.info(f"[SUCCESS] 第一步完成 - 特征: {result['features'].shape}")
        
        return {
            'f_classic': result['features'],
            'f_graph': result.get('adjacency_matrix', torch.eye(result['features'].shape[0])),
            'node_mapping': result.get('node_mapping', {}),
            'metadata': result['metadata']
        }
        
    except Exception as e:
        logger.error(f"[ERROR] 第一步失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_real_step2(step1_data):
    """测试真实的第二步：多尺度对比学习"""
    logger.info("[STEP2] 测试真实多尺度对比学习...")
    
    try:
        sys.path.insert(0, str(Path("muti_scale").absolute()))
        
        from muti_scale.realtime_mscia import RealtimeMSCIAProcessor
        from muti_scale.step2_config import Step2Config
        
        # 使用真实配置
        config = Step2Config().get_blockemulator_integration_config()
        processor = RealtimeMSCIAProcessor(config)
        
        # 执行真实的第二步处理
        result = processor.process_step1_output(
            step1_data,
            timestamp=1,
            blockemulator_timestamp=time.time()
        )
        
        logger.info(f"[SUCCESS] 第二步完成 - 嵌入: {result['temporal_embeddings'].shape}")
        
        return {
            'temporal_embeddings': result['temporal_embeddings'],
            'node_mapping': result['node_mapping'],
            'metadata': result['metadata']
        }
        
    except Exception as e:
        logger.error(f"[ERROR] 第二步失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_real_step3(step2_data):
    """测试真实的第三步：EvolveGCN分片"""
    logger.info("[STEP3] 测试真实EvolveGCN分片...")
    
    try:
        sys.path.insert(0, str(Path("evolve_GCN/models").absolute()))
        
        from evolve_GCN.models.sharding_modules import DynamicShardingModule
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embeddings = step2_data['temporal_embeddings'].to(device)
        
        # 初始化真实的分片模块
        sharding_module = DynamicShardingModule(
            embedding_dim=embeddings.shape[1],
            base_shards=3,
            max_shards=6
        ).to(device)
        
        # 执行真实的分片决策
        shard_assignments, enhanced_embeddings, attention_weights, predicted_num_shards = sharding_module(
            embeddings, 
            history_states=[], 
            feedback_signal=None
        )
        
        hard_assignment = torch.argmax(shard_assignments, dim=1)
        unique_shards, shard_counts = torch.unique(hard_assignment, return_counts=True)
        
        logger.info(f"[SUCCESS] 第三步完成 - 预测分片: {predicted_num_shards}, 实际分片: {len(unique_shards)}")
        
        return {
            'shard_assignments': shard_assignments,
            'hard_assignment': hard_assignment,
            'predicted_num_shards': predicted_num_shards,
            'enhanced_embeddings': enhanced_embeddings,
            'shard_counts': shard_counts,
            'unique_shards': unique_shards
        }
        
    except Exception as e:
        logger.error(f"[ERROR] 第三步失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_real_step4(step3_data, step1_data, step2_data):
    """测试真实的第四步：统一反馈引擎"""
    logger.info("[STEP4] 测试真实统一反馈引擎...")
    
    try:
        sys.path.insert(0, str(Path("feedback").absolute()))
        
        from feedback.unified_feedback_engine import UnifiedFeedbackEngine
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化真实的统一反馈引擎
        feedback_engine = UnifiedFeedbackEngine(device=device)
        
        # 准备6类真实特征数据
        num_nodes = step1_data['f_classic'].shape[0]
        features = {
            'hardware': step1_data['f_classic'][:, :17].to(device),
            'onchain_behavior': step1_data['f_classic'][:, 17:34].to(device),
            'network_topology': step1_data['f_classic'][:, 34:54].to(device),
            'dynamic_attributes': step1_data['f_classic'][:, 54:67].to(device),
            'heterogeneous_type': step1_data['f_classic'][:, 67:84].to(device),
            'categorical': step1_data['f_classic'][:, 84:99].to(device) if step1_data['f_classic'].shape[1] > 84 else torch.randn(num_nodes, 15).to(device)
        }
        
        # 构建边索引
        edge_index = torch.tensor([[i, (i+1) % num_nodes] for i in range(num_nodes)]).t().to(device)
        
        # 准备性能提示
        performance_hints = {
            'predicted_shards': step3_data['predicted_num_shards'],
            'actual_shards': len(step3_data['unique_shards']),
            'shard_sizes': step3_data['shard_counts'].tolist(),
            'load_balance_hint': 1.0 - (step3_data['shard_counts'].std() / (step3_data['shard_counts'].mean() + 1e-8)).item()
        }
        
        # 执行真实的统一反馈分析
        feedback_result = feedback_engine.process_sharding_feedback(
            features=features,
            shard_assignments=step3_data['hard_assignment'].to(device),
            edge_index=edge_index,
            performance_hints=performance_hints
        )
        
        logger.info(f"[SUCCESS] 第四步完成 - 综合评分: {feedback_result['optimized_feedback']['overall_score']:.3f}")
        logger.info(f"   负载均衡: {feedback_result['performance_metrics']['load_balance']:.3f}")
        logger.info(f"   跨片交易率: {feedback_result['performance_metrics']['cross_shard_rate']:.3f}")
        logger.info(f"   安全评分: {feedback_result['performance_metrics']['security_score']:.3f}")
        
        return {
            'feedback_result': feedback_result,
            'feedback_signal': feedback_result['step3_feedback_package']['assignment_guidance'],
            'performance_metrics': feedback_result['performance_metrics']
        }
        
    except Exception as e:
        logger.error(f"[ERROR] 第四步失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_real_integration_loop(step3_data, step4_data):
    """测试真实的第三步-第四步集成循环"""
    logger.info("[LOOP] 测试真实的集成循环...")
    
    try:
        sys.path.insert(0, str(Path("evolve_GCN/models").absolute()))
        from evolve_GCN.models.sharding_modules import DynamicShardingModule
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embeddings = step3_data['enhanced_embeddings'].to(device)
        
        # 初始化分片模块
        sharding_module = DynamicShardingModule(
            embedding_dim=embeddings.shape[1],
            base_shards=3,
            max_shards=6
        ).to(device)
        
        # 获取反馈信号
        feedback_signal = step4_data['feedback_signal'].to(device) if step4_data else None
        
        best_performance = 0.0
        iterations = 3
        
        for iteration in range(iterations):
            logger.info(f"   迭代 {iteration + 1}/{iterations}")
            
            # 第三步：使用反馈的动态分片
            shard_assignments, enhanced_embeddings, attention_weights, predicted_num_shards = sharding_module(
                embeddings, 
                history_states=[], 
                feedback_signal=feedback_signal
            )
            
            hard_assignment = torch.argmax(shard_assignments, dim=1)
            unique_shards, shard_counts = torch.unique(hard_assignment, return_counts=True)
            
            # 计算性能指标
            balance_score = 1.0 - (shard_counts.std() / (shard_counts.mean() + 1e-8)).item()
            
            # 计算跨片率
            num_nodes = embeddings.shape[0]
            edge_index = torch.tensor([[i, (i+1) % num_nodes] for i in range(num_nodes)]).t().to(device)
            cross_shard_edges = (hard_assignment[edge_index[0]] != hard_assignment[edge_index[1]]).float()
            cross_rate = torch.mean(cross_shard_edges).item()
            
            performance_score = 0.6 * balance_score + 0.4 * (1.0 - cross_rate)
            
            logger.info(f"     分片数: {len(unique_shards)}, 负载均衡: {balance_score:.3f}, 跨片率: {cross_rate:.3f}")
            logger.info(f"     性能分数: {performance_score:.3f}")
            
            if performance_score > best_performance:
                best_performance = performance_score
                logger.info("     [SUCCESS] 性能改善!")
            
            # 更新嵌入
            embeddings = enhanced_embeddings + torch.randn_like(enhanced_embeddings) * 0.01
        
        logger.info(f"[SUCCESS] 集成循环完成 - 最佳性能: {best_performance:.3f}")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] 集成循环失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    logger.info("[START] 真实四步分片系统测试")
    logger.info("=" * 60)
    
    start_time = time.time()
    results = {}
    
    # 测试第一步
    step1_result = test_real_step1()
    results['step1'] = 'PASS' if step1_result else 'FAIL'
    
    if not step1_result:
        logger.error("[ERROR] 第一步失败，停止测试")
        return False
    
    # 测试第二步
    step2_result = test_real_step2(step1_result)
    results['step2'] = 'PASS' if step2_result else 'FAIL'
    
    if not step2_result:
        logger.error("[ERROR] 第二步失败，停止测试")
        return False
    
    # 测试第三步
    step3_result = test_real_step3(step2_result)
    results['step3'] = 'PASS' if step3_result else 'FAIL'
    
    if not step3_result:
        logger.error("[ERROR] 第三步失败，停止测试")
        return False
    
    # 测试第四步
    step4_result = test_real_step4(step3_result, step1_result, step2_result)
    results['step4'] = 'PASS' if step4_result else 'FAIL'
    
    # 测试集成循环
    integration_result = test_real_integration_loop(step3_result, step4_result)
    results['integration'] = 'PASS' if integration_result else 'FAIL'
    
    # 总结
    end_time = time.time()
    passed = sum(1 for r in results.values() if r == 'PASS')
    total = len(results)
    
    logger.info("=" * 60)
    logger.info("[SUMMARY] 测试总结:")
    logger.info(f"   用时: {end_time - start_time:.2f}秒")
    logger.info(f"   通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    for test, result in results.items():
        status = "[SUCCESS]" if result == "PASS" else "[FAILED]"
        logger.info(f"   {status} {test}: {result}")
    
    if passed == total:
        logger.info("[SUCCESS] 所有测试通过！真实四步分片系统运行正常")
        return True
    else:
        logger.info("[WARNING] 部分测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
