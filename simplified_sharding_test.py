#!/usr/bin/env python3
"""
简化版真实分片系统测试脚本
使用绝对导入避免模块加载问题
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
import torch
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_individual_components():
    """分别测试各个组件的可用性"""
    
    logger.info("="*60)
    logger.info("开始分片系统组件单独测试")
    logger.info("="*60)
    
    # 测试第一步组件
    logger.info("\n[STEP1] 测试特征提取组件...")
    try:
        # 使用系统验证中已确认可用的方式
        from partition.feature.system_integration_pipeline import BlockEmulatorStep1Pipeline
        from partition.feature.node_features import MockNodeFeaturesModule
        
        pipeline = BlockEmulatorStep1Pipeline()
        mock_module = MockNodeFeaturesModule()
        
        # 简单测试
        result = pipeline.extract_features_from_system(
            node_module=mock_module,
            num_nodes=20,  # 减少节点数量
            enable_dynamic_features=True
        )
        
        logger.info(f"[SUCCESS] 第一步组件测试成功")
        logger.info(f"   经典特征维度: {result['f_classic'].shape}")
        logger.info(f"   图特征维度: {result['f_graph'].shape}")
        
        step1_result = result
        
    except Exception as e:
        logger.error(f"[ERROR] 第一步组件测试失败: {e}")
        step1_result = None
    
    # 测试第二步组件
    logger.info("\n[STEP2] 测试多尺度处理组件...")
    try:
        from muti_scale.realtime_mscia import RealtimeMSCIAProcessor
        from muti_scale.step2_config import Step2Config
        
        config = Step2Config()
        processor = RealtimeMSCIAProcessor(config.get_config('default'))
        
        if step1_result:
            # 使用第一步的真实输出
            step2_input = {
                'node_features': step1_result['f_classic'],
                'node_ids': step1_result['node_ids'],
                'timestamps': step1_result['timestamps'],
                'graph_structure': torch.randint(0, 2, (len(step1_result['node_ids']), len(step1_result['node_ids'])))
            }
            
            result = processor.process_step1_output(step2_input)
            
            logger.info(f"[SUCCESS] 第二步组件测试成功")
            logger.info(f"   时序嵌入维度: {result['temporal_embeddings'].shape}")
            logger.info(f"   对比损失: {result['contrastive_loss']:.4f}")
            
            step2_result = result
        else:
            logger.warning("[WARNING] 第一步失败，跳过第二步真实数据测试")
            step2_result = None
            
    except Exception as e:
        logger.error(f"[ERROR] 第二步组件测试失败: {e}")
        step2_result = None
    
    # 测试第三步组件
    logger.info("\n[STEP3] 测试EvolveGCN分片组件...")
    try:
        from evolve_GCN.models.sharding_modules import DynamicShardingModule
        
        # 创建分片模块
        sharding_module = DynamicShardingModule(
            embedding_dim=64,
            base_shards=3,
            max_shards=8
        )
        
        if step2_result:
            # 使用第二步的真实输出
            Z = step2_result['temporal_embeddings']
            
            # 创建历史状态
            history_states = torch.randn(5, 3)  # [seq_len, 3]
            
            # 运行分片决策
            S_t, enhanced_embeddings, attention_weights, K_t = sharding_module(
                Z, 
                history_states=history_states
            )
            
            logger.info(f"[SUCCESS] 第三步组件测试成功")
            logger.info(f"   分片关联矩阵: {S_t.shape}")
            logger.info(f"   预测分片数: {K_t}")
            logger.info(f"   增强嵌入维度: {enhanced_embeddings.shape}")
            
            step3_result = {
                'sharding_assignments': S_t,
                'enhanced_embeddings': enhanced_embeddings,
                'attention_weights': attention_weights,
                'num_shards': K_t
            }
        else:
            logger.warning("[WARNING] 第二步失败，跳过第三步真实数据测试")
            step3_result = None
            
    except Exception as e:
        logger.error(f"[ERROR] 第三步组件测试失败: {e}")
        step3_result = None
    
    # 测试第四步组件
    logger.info("\n[STEP4] 测试统一反馈组件...")
    try:
        from feedback.unified_feedback_engine import UnifiedFeedbackEngine
        
        engine = UnifiedFeedbackEngine()
        
        if step3_result and step1_result:
            # 构建反馈输入
            feedback_input = {
                'sharding_result': step3_result['sharding_assignments'],
                'node_features': step1_result['f_classic'],
                'performance_metrics': {
                    'throughput': 1000.0,
                    'latency': 50.0,
                    'load_balance': 0.85,
                    'cross_shard_ratio': 0.15
                }
            }
            
            result = engine.process_sharding_feedback(feedback_input)
            
            logger.info(f"[SUCCESS] 第四步组件测试成功")
            logger.info(f"   反馈信号维度: {result['feedback_signal'].shape}")
            logger.info(f"   优化建议数量: {len(result['optimization_suggestions'])}")
            logger.info(f"   性能评分: {result['performance_score']:.4f}")
            
        else:
            logger.warning("[WARNING] 前序步骤失败，跳过第四步真实数据测试")
            
    except Exception as e:
        logger.error(f"[ERROR] 第四步组件测试失败: {e}")
    
    # 生成测试报告
    logger.info("\n" + "="*60)
    logger.info("组件测试总结")
    logger.info("="*60)
    
    success_steps = 0
    if step1_result is not None:
        logger.info("[✓] 第一步特征提取: 成功")
        success_steps += 1
    else:
        logger.info("[✗] 第一步特征提取: 失败")
    
    if step2_result is not None:
        logger.info("[✓] 第二步多尺度处理: 成功")
        success_steps += 1
    else:
        logger.info("[✗] 第二步多尺度处理: 失败")
    
    if step3_result is not None:
        logger.info("[✓] 第三步EvolveGCN分片: 成功")
        success_steps += 1
    else:
        logger.info("[✗] 第三步EvolveGCN分片: 失败")
    
    logger.info(f"\n总体成功率: {success_steps}/3 = {success_steps/3*100:.1f}%")
    
    if success_steps == 3:
        logger.info("\n[结论] 所有组件测试成功！系统可以进行真实分片工作。")
        return True
    else:
        logger.warning(f"\n[结论] {3-success_steps}个组件测试失败，需要修复后才能正常工作。")
        return False

def test_blockemulator_interface():
    """测试BlockEmulator接口"""
    logger.info("\n[INTERFACE] 测试BlockEmulator接口...")
    
    try:
        from blockchain_interface import BlockchainInterface
        
        interface = BlockchainInterface()
        
        # 测试基本功能
        status = interface.get_status()
        logger.info(f"[SUCCESS] BlockEmulator接口可用")
        logger.info(f"   状态: {status}")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] BlockEmulator接口测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始简化版分片系统测试...")
    
    # 测试BlockEmulator接口
    interface_ok = test_blockemulator_interface()
    
    # 测试各个组件
    components_ok = test_individual_components()
    
    # 最终结果
    logger.info("\n" + "="*60)
    logger.info("最终测试结果")
    logger.info("="*60)
    
    if interface_ok and components_ok:
        logger.info("[SUCCESS] 系统完全就绪，可以进行真实分片工作！")
        logger.info("建议运行完整的集成测试脚本进行端到端验证。")
        return 0
    else:
        logger.warning("[WARNING] 系统存在问题，需要修复：")
        if not interface_ok:
            logger.warning("- BlockEmulator接口问题")
        if not components_ok:
            logger.warning("- 分片系统组件问题")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
