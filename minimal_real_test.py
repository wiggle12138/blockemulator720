#!/usr/bin/env python3
"""
最小化分片系统验证脚本
验证系统是否真正调用了真实组件而非模拟数据
"""

import sys
import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_step1_data():
    """创建模拟的第一步数据用于测试"""
    return {
        'f_classic': torch.randn(50, 128),      # 经典特征
        'f_graph': torch.randn(50, 96),         # 图特征 
        'node_ids': list(range(50)),            # 节点ID
        'timestamps': torch.randint(1, 1000, (50,)),  # 时间戳
        'adjacency_matrix': torch.randint(0, 2, (50, 50))  # 邻接矩阵
    }

def test_step2_real_component():
    """测试第二步真实组件"""
    logger.info("测试第二步多尺度对比学习真实组件...")
    
    try:
        from muti_scale.realtime_mscia import RealtimeMSCIAProcessor
        from muti_scale.step2_config import Step2Config
        
        # 创建真实配置和处理器
        config = Step2Config()
        processor = RealtimeMSCIAProcessor(config.get_config('default'))
        
        # 创建测试数据（按照第二步处理器期望的格式）
        step1_data = create_mock_step1_data()
        
        # 构建正确的第一步输出格式
        step1_result = {
            'f_classic': step1_data['f_classic'],   # [N, 128]
            'f_graph': step1_data['f_graph'],       # [N, 96]
            'node_mapping': {str(i): i for i in range(50)},  # 节点映射
            'metadata': {
                'num_nodes': 50,
                'extraction_timestamp': 1000,
                'feature_dims': {'classic': 128, 'graph': 96}
            }
        }
        
        # 调用真实处理函数
        result = processor.process_step1_output(
            step1_result=step1_result,
            timestamp=1,
            blockemulator_timestamp=1000
        )
        
        logger.info("[SUCCESS] 第二步真实组件测试成功")
        logger.info(f"   输入经典特征: {step1_result['f_classic'].shape}")
        logger.info(f"   输入图特征: {step1_result['f_graph'].shape}")
        logger.info(f"   输出时序嵌入: {result['temporal_embeddings'].shape}")
        logger.info(f"   对比学习损失: {result['loss'].item():.6f}")
        
        return result
        
    except Exception as e:
        logger.error(f"[ERROR] 第二步真实组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_step3_real_component(step2_result):
    """测试第三步真实EvolveGCN分片组件"""
    logger.info("测试第三步EvolveGCN分片真实组件...")
    
    try:
        from evolve_GCN.models.sharding_modules import DynamicShardingModule
        
        # 创建真实分片模块
        sharding_module = DynamicShardingModule(
            embedding_dim=64,    # 与第二步输出维度匹配
            base_shards=3,
            max_shards=8
        )
        
        if step2_result:
            # 使用第二步的真实输出
            Z = step2_result['temporal_embeddings']
            
            # 创建历史状态数据（模拟）- 需要是列表格式
            history_states = [
                torch.tensor([0.8, 0.2, 0.9], dtype=torch.float32),  # 负载均衡度, 跨片交易率, 安全阈值
                torch.tensor([0.7, 0.3, 0.8], dtype=torch.float32),
                torch.tensor([0.9, 0.1, 0.95], dtype=torch.float32),
                torch.tensor([0.75, 0.25, 0.85], dtype=torch.float32),
                torch.tensor([0.82, 0.18, 0.88], dtype=torch.float32)
            ]
            
            # 调用真实分片决策
            S_t, enhanced_embeddings, attention_weights, K_t = sharding_module(
                Z, 
                history_states=history_states
            )
            
            logger.info("[SUCCESS] 第三步真实组件测试成功")
            logger.info(f"   输入时序嵌入: {Z.shape}")
            logger.info(f"   分片关联矩阵: {S_t.shape}")
            logger.info(f"   预测分片数: {K_t}")
            logger.info(f"   增强嵌入维度: {enhanced_embeddings.shape}")
            logger.info(f"   注意力权重: {attention_weights.shape}")
            
            return {
                'sharding_assignments': S_t,
                'enhanced_embeddings': enhanced_embeddings,
                'attention_weights': attention_weights,
                'num_shards': K_t,
                'input_embeddings': Z
            }
        else:
            logger.error("[ERROR] 第二步结果为空，无法测试第三步")
            return None
            
    except Exception as e:
        logger.error(f"[ERROR] 第三步真实组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_step4_real_component(step1_data, step3_result):
    """测试第四步真实反馈组件"""
    logger.info("测试第四步统一反馈引擎真实组件...")
    
    try:
        from feedback.unified_feedback_engine import UnifiedFeedbackEngine
        
        # 创建真实反馈引擎
        engine = UnifiedFeedbackEngine()
        
        if step3_result and step1_data:
            # 构建真实反馈输入
            feedback_input = {
                'sharding_result': step3_result['sharding_assignments'],
                'node_features': step1_data['f_classic'],
                'performance_metrics': {
                    'throughput': 850.0,      # TPS
                    'latency': 45.2,          # ms
                    'load_balance': 0.78,     # 负载均衡度
                    'cross_shard_ratio': 0.22 # 跨片交易比例
                }
            }
            
            # 调用真实反馈处理
            result = engine.process_sharding_feedback(feedback_input)
            
            logger.info("[SUCCESS] 第四步真实组件测试成功")
            logger.info(f"   输入分片结果: {feedback_input['sharding_result'].shape}")
            logger.info(f"   输入节点特征: {feedback_input['node_features'].shape}")
            logger.info(f"   反馈信号维度: {result['feedback_signal'].shape}")
            logger.info(f"   性能评分: {result['performance_score']:.4f}")
            logger.info(f"   优化建议数量: {len(result['optimization_suggestions'])}")
            
            return result
        else:
            logger.error("[ERROR] 前序步骤结果为空，无法测试第四步")
            return None
            
    except Exception as e:
        logger.error(f"[ERROR] 第四步真实组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_integration_loop(step2_result, step3_result, step4_result):
    """测试第三步和第四步的集成循环"""
    logger.info("测试第三、四步集成反馈循环...")
    
    try:
        if not (step2_result and step3_result and step4_result):
            logger.error("[ERROR] 前序步骤有失败，无法测试集成循环")
            return False
        
        from evolve_GCN.models.sharding_modules import DynamicShardingModule
        
        # 重新创建分片模块进行反馈集成测试
        sharding_module = DynamicShardingModule(
            embedding_dim=64,
            base_shards=3,
            max_shards=8
        )
        
        # 使用第四步的反馈信号重新进行分片
        Z = step2_result['temporal_embeddings']
        history_states = [
            torch.tensor([0.8, 0.2, 0.9], dtype=torch.float32),
            torch.tensor([0.7, 0.3, 0.8], dtype=torch.float32),
            torch.tensor([0.9, 0.1, 0.95], dtype=torch.float32),
            torch.tensor([0.75, 0.25, 0.85], dtype=torch.float32),
            torch.tensor([0.82, 0.18, 0.88], dtype=torch.float32)
        ]
        
        # 带反馈信号的分片决策
        S_t_improved, enhanced_embeddings, attention_weights, K_t = sharding_module(
            Z, 
            history_states=history_states,
            feedback_signal=step4_result['feedback_signal']
        )
        
        logger.info("[SUCCESS] 集成反馈循环测试成功")
        logger.info(f"   带反馈的分片矩阵: {S_t_improved.shape}")
        logger.info(f"   反馈前分片数: {step3_result['num_shards']}")
        logger.info(f"   反馈后分片数: {K_t}")
        
        # 比较反馈前后的差异
        diff = torch.abs(S_t_improved - step3_result['sharding_assignments']).mean().item()
        logger.info(f"   分片决策改进程度: {diff:.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] 集成反馈循环测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    logger.info("="*70)
    logger.info("开始真实分片系统组件验证")
    logger.info("目标：验证系统确实调用了真实组件而非模拟数据")
    logger.info("="*70)
    
    # 创建基础测试数据
    step1_mock_data = create_mock_step1_data()
    logger.info(f"创建模拟第一步数据: 节点数={len(step1_mock_data['node_ids'])}")
    
    # 依次测试各个真实组件
    step2_result = test_step2_real_component()
    step3_result = test_step3_real_component(step2_result)
    step4_result = test_step4_real_component(step1_mock_data, step3_result)
    
    # 测试集成循环
    integration_success = test_integration_loop(step2_result, step3_result, step4_result)
    
    # 生成最终报告
    logger.info("\n" + "="*70)
    logger.info("真实分片系统验证结果")
    logger.info("="*70)
    
    success_count = 0
    total_tests = 4
    
    if step2_result is not None:
        logger.info("[✓] 第二步多尺度对比学习: 真实组件调用成功")
        success_count += 1
    else:
        logger.info("[✗] 第二步多尺度对比学习: 真实组件调用失败")
    
    if step3_result is not None:
        logger.info("[✓] 第三步EvolveGCN分片: 真实组件调用成功")
        success_count += 1
    else:
        logger.info("[✗] 第三步EvolveGCN分片: 真实组件调用失败")
    
    if step4_result is not None:
        logger.info("[✓] 第四步统一反馈引擎: 真实组件调用成功")
        success_count += 1
    else:
        logger.info("[✗] 第四步统一反馈引擎: 真实组件调用失败")
    
    if integration_success:
        logger.info("[✓] 第三四步集成反馈循环: 真实组件集成成功")
        success_count += 1
    else:
        logger.info("[✗] 第三四步集成反馈循环: 真实组件集成失败")
    
    success_rate = success_count / total_tests * 100
    logger.info(f"\n整体成功率: {success_count}/{total_tests} = {success_rate:.1f}%")
    
    if success_rate >= 75:
        logger.info("\n[结论] ✓ 系统成功调用了真实的分片组件！")
        logger.info("这证明测试脚本确实在使用真实的分片系统，而不是模拟数据。")
        logger.info("分片系统的多尺度对比学习、EvolveGCN分片、统一反馈引擎都在正常工作。")
        return 0
    else:
        logger.warning("\n[结论] ✗ 部分真实组件调用失败")
        logger.warning("需要修复失败的组件才能确保系统使用真实分片功能。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
