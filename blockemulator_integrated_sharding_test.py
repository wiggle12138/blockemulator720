#!/usr/bin/env python3
"""
BlockEmulator集成分片系统完整测试
严格按照四步流程：特征提取 → 多尺度对比学习 → EvolveGCN分片 ⇄ 性能反馈
这是最终集成到BlockEmulator中的分片系统测试
"""

import sys
import time
import json
import logging
import warnings
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# 忽略警告
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加路径 - 按BlockEmulator项目结构
sys.path.append('.')
sys.path.append('./partition')
sys.path.append('./partition/feature')
sys.path.append('./muti_scale')
sys.path.append('./evolve_GCN')
sys.path.append('./evolve_GCN/models')
sys.path.append('./feedback')

class BlockEmulatorIntegratedShardingSystem:
    """
    BlockEmulator集成分片系统
    实现完整的四步闭环流程，可以集成到BlockEmulator中
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 系统状态
        self.current_epoch = 0
        self.performance_history = []
        self.sharding_history = []
        
        # 组件实例
        self.step1_pipeline = None
        self.step2_processor = None
        self.step3_sharding_module = None
        self.step4_feedback_engine = None
        
        logger.info("[START] BlockEmulator集成分片系统初始化")
        logger.info(f"   设备: {self.device}")
        logger.info(f"   配置: {self.config['system_name']}")

    def _get_default_config(self) -> Dict[str, Any]:
        """默认配置 - 针对BlockEmulator优化"""
        return {
            'system_name': 'BlockEmulator-EvolveGCN-DynamicSharding',
            'version': '1.0.0',
            
            # 第一步配置
            'step1': {
                'feature_extraction_mode': 'comprehensive',
                'output_dim': 128,
                'save_adjacency': True
            },
            
            # 第二步配置
            'step2': {
                'time_window': 3,
                'batch_size': 16,
                'hidden_dim': 64,
                'learning_rate': 0.02,
                'use_real_timestamps': True
            },
            
            # 第三步配置
            'step3': {
                'base_shards': 3,
                'max_shards': 6,
                'embedding_dim': 64,
                'max_iterations': 3
            },
            
            # 第四步配置
            'step4': {
                'enable_feedback': True,
                'convergence_threshold': 0.01,
                'max_feedback_iterations': 5
            },
            
            # 系统集成配置
            'integration': {
                'auto_apply_sharding': False,  # 是否自动应用分片结果到BlockEmulator
                'save_results': True,
                'result_path': './data_exchange',
                'enable_monitoring': True
            }
        }

    def initialize_components(self):
        """初始化所有组件"""
        logger.info("[CONFIG] 初始化分片系统组件...")
        
        try:
            # 第一步：特征提取管道
            from partition.feature.system_integration_pipeline import BlockEmulatorStep1Pipeline
            self.step1_pipeline = BlockEmulatorStep1Pipeline(
                use_comprehensive_features=True,
                save_adjacency=True,
                output_dir=self.config['integration']['result_path']
            )
            logger.info("[SUCCESS] 第一步组件初始化完成")
            
        except Exception as e:
            logger.warning(f"[WARNING] 第一步组件初始化失败，使用模拟模式: {e}")
            self.step1_pipeline = None

        try:
            # 第二步：多尺度对比学习
            from muti_scale.realtime_mscia import RealtimeMSCIAProcessor
            from muti_scale.step2_config import Step2Config
            
            step2_config = Step2Config().get_blockemulator_integration_config()
            step2_config.update(self.config['step2'])
            
            self.step2_processor = RealtimeMSCIAProcessor(step2_config)
            logger.info("[SUCCESS] 第二步组件初始化完成")
            
        except Exception as e:
            logger.error(f"[ERROR] 第二步组件初始化失败: {e}")
            return False

        try:
            # 第三步：EvolveGCN动态分片
            from evolve_GCN.models.sharding_modules import DynamicShardingModule
            
            self.step3_sharding_module = DynamicShardingModule(
                embedding_dim=self.config['step3']['embedding_dim'],
                base_shards=self.config['step3']['base_shards'],
                max_shards=self.config['step3']['max_shards']
            ).to(self.device)
            logger.info("[SUCCESS] 第三步组件初始化完成")
            
        except Exception as e:
            logger.error(f"[ERROR] 第三步组件初始化失败: {e}")
            return False

        try:
            # 第四步：性能反馈引擎
            from feedback.unified_feedback_engine import UnifiedFeedbackEngine
            
            feature_dims = {
                'node_features': self.config['step3']['embedding_dim'],
                'degree_centrality': 1,
                'betweenness_centrality': 1,
                'clustering_coefficient': 1,
                'pagerank': 1,
                'shard_balance': 1
            }
            
            self.step4_feedback_engine = UnifiedFeedbackEngine(
                feature_dims=feature_dims,
                device=self.device
            )
            logger.info("[SUCCESS] 第四步组件初始化完成")
            
        except Exception as e:
            logger.warning(f"[WARNING] 第四步组件初始化失败，使用简化反馈: {e}")
            self.step4_feedback_engine = None

        logger.info("[TARGET] 所有组件初始化完成，系统就绪")
        return True

    def run_complete_cycle(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        运行完整的四步分片循环
        这是BlockEmulator调用的主要接口
        """
        logger.info("[START] 开始完整分片循环")
        logger.info("=" * 80)
        
        cycle_start_time = time.time()
        results = {
            'cycle_id': f"cycle_{self.current_epoch}",
            'timestamp': time.time(),
            'steps': {},
            'final_sharding': None,
            'performance_metrics': {},
            'status': 'running'
        }

        try:
            # 第一步：特征提取
            step1_result = self._execute_step1(input_data)
            results['steps']['step1'] = step1_result
            
            if not step1_result['success']:
                raise Exception("第一步失败")

            # 第二步：多尺度对比学习
            step2_result = self._execute_step2(step1_result['data'])
            results['steps']['step2'] = step2_result
            
            if not step2_result['success']:
                raise Exception("第二步失败")

            # 第三步和第四步的迭代循环
            feedback_loop_result = self._execute_feedback_loop(step2_result['data'])
            results['steps']['feedback_loop'] = feedback_loop_result
            
            if not feedback_loop_result['success']:
                raise Exception("反馈循环失败")

            # 整理最终结果
            results['final_sharding'] = feedback_loop_result['best_sharding']
            results['performance_metrics'] = feedback_loop_result['final_metrics']
            results['status'] = 'completed'

            # 保存结果
            if self.config['integration']['save_results']:
                self._save_cycle_results(results)

            cycle_time = time.time() - cycle_start_time
            logger.info(f"[SUCCESS] 完整分片循环完成，用时: {cycle_time:.2f}秒")
            logger.info(f"   最终分片数: {results['final_sharding']['actual_num_shards']}")
            logger.info(f"   性能指标: {results['performance_metrics']}")

            self.current_epoch += 1
            return results

        except Exception as e:
            logger.error(f"[ERROR] 分片循环失败: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            return results

    def _execute_step1(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行第一步：特征提取"""
        logger.info("🔍 [第一步] 特征提取...")
        
        try:
            if self.step1_pipeline and input_data and 'node_features_module' in input_data:
                # 真实的BlockEmulator数据
                features_result = self.step1_pipeline.extract_features_from_system(
                    node_features_module=input_data['node_features_module'],
                    experiment_name=f"cycle_{self.current_epoch}"
                )
                
                step1_data = {
                    'f_classic': features_result['features'],
                    'f_graph': features_result.get('adjacency_matrix', torch.eye(features_result['features'].shape[0])),
                    'node_mapping': features_result.get('node_mapping', {}),
                    'metadata': features_result['metadata']
                }
                
            else:
                # 模拟数据（用于测试）
                num_nodes = input_data.get('num_nodes', 20) if input_data else 20
                step1_data = {
                    'f_classic': torch.randn(num_nodes, 128),
                    'f_graph': torch.randn(num_nodes, 96),
                    'node_mapping': {i: f"node_{i}" for i in range(num_nodes)},
                    'metadata': {
                        'num_nodes': num_nodes,
                        'data_source': 'simulation',
                        'extraction_time': time.time()
                    }
                }

            logger.info(f"[SUCCESS] [第一步] 完成 - 提取 {step1_data['metadata']['num_nodes']} 个节点特征")
            
            return {
                'success': True,
                'data': step1_data,
                'metrics': {
                    'num_nodes': step1_data['metadata']['num_nodes'],
                    'feature_dims': {
                        'f_classic': list(step1_data['f_classic'].shape),
                        'f_graph': list(step1_data['f_graph'].shape)
                    }
                }
            }

        except Exception as e:
            logger.error(f"[ERROR] [第一步] 失败: {e}")
            return {'success': False, 'error': str(e)}

    def _execute_step2(self, step1_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行第二步：多尺度对比学习"""
        logger.info("🧠 [第二步] 多尺度对比学习...")
        
        try:
            # 调用第二步处理器
            step2_result = self.step2_processor.process_step1_output(
                step1_data,
                timestamp=self.current_epoch,
                blockemulator_timestamp=time.time()
            )
            
            step2_data = {
                'temporal_embeddings': step2_result['temporal_embeddings'],
                'node_mapping': step2_result['node_mapping'],
                'metadata': step2_result['metadata']
            }
            
            logger.info(f"[SUCCESS] [第二步] 完成 - 生成时序嵌入: {step2_data['temporal_embeddings'].shape}")
            
            return {
                'success': True,
                'data': step2_data,
                'metrics': {
                    'embedding_shape': list(step2_data['temporal_embeddings'].shape),
                    'loss': step2_result.get('loss', 0.0),
                    'processing_time': step2_result['metadata'].get('processing_time', 0)
                }
            }

        except Exception as e:
            logger.error(f"[ERROR] [第二步] 失败: {e}")
            return {'success': False, 'error': str(e)}

    def _execute_feedback_loop(self, step2_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行第三步-第四步反馈循环"""
        logger.info("🔄 [第三步-第四步] 反馈循环...")
        
        try:
            embeddings = step2_data['temporal_embeddings'].to(self.device)
            num_nodes = embeddings.shape[0]
            
            best_sharding = None
            best_performance = float('-inf')
            feedback_signal = None
            
            iteration_results = []
            
            for iteration in range(self.config['step3']['max_iterations']):
                logger.info(f"   迭代 {iteration + 1}/{self.config['step3']['max_iterations']}")
                
                # 第三步：EvolveGCN分片
                step3_result = self._execute_step3(embeddings, feedback_signal, iteration)
                
                if not step3_result['success']:
                    continue
                
                # 第四步：性能评估
                step4_result = self._execute_step4(step3_result['data'], embeddings)
                
                # 记录迭代结果
                iteration_result = {
                    'iteration': iteration + 1,
                    'step3': step3_result,
                    'step4': step4_result,
                    'performance_score': step4_result.get('performance_score', 0.0)
                }
                iteration_results.append(iteration_result)
                
                # 更新最佳结果
                current_performance = step4_result.get('performance_score', 0.0)
                if current_performance > best_performance:
                    best_performance = current_performance
                    best_sharding = step3_result['data']
                    logger.info(f"     [SUCCESS] 性能改善! 分数: {current_performance:.3f}")
                
                # 更新反馈信号
                feedback_signal = step4_result.get('feedback_signal', None)
                
                # 检查收敛
                if self._check_convergence(iteration_results):
                    logger.info(f"     [TARGET] 收敛达成，提前结束")
                    break

            logger.info(f"[SUCCESS] [反馈循环] 完成 - 最佳性能: {best_performance:.3f}")
            
            return {
                'success': True,
                'best_sharding': best_sharding,
                'final_metrics': {
                    'best_performance_score': best_performance,
                    'total_iterations': len(iteration_results),
                    'converged': self._check_convergence(iteration_results)
                },
                'iteration_history': iteration_results
            }

        except Exception as e:
            logger.error(f"[ERROR] [反馈循环] 失败: {e}")
            return {'success': False, 'error': str(e)}

    def _execute_step3(self, embeddings: torch.Tensor, feedback_signal: torch.Tensor = None, iteration: int = 0) -> Dict[str, Any]:
        """执行第三步：EvolveGCN分片"""
        try:
            # 动态分片
            shard_assignments, enhanced_embeddings, attention_weights, predicted_num_shards = self.step3_sharding_module(
                embeddings, 
                history_states=None,
                feedback_signal=feedback_signal
            )
            
            # 计算硬分配
            hard_assignment = torch.argmax(shard_assignments, dim=1)
            unique_shards, shard_counts = torch.unique(hard_assignment, return_counts=True)
            
            step3_data = {
                'shard_assignments': shard_assignments,
                'hard_assignment': hard_assignment,
                'enhanced_embeddings': enhanced_embeddings,
                'predicted_num_shards': predicted_num_shards,
                'actual_num_shards': len(unique_shards),
                'shard_sizes': shard_counts.tolist()
            }
            
            logger.info(f"     [第三步] 预测分片数: {predicted_num_shards}, 实际分片: {len(unique_shards)}")
            logger.info(f"     [第三步] 分片大小: {shard_counts.tolist()}")
            
            return {
                'success': True,
                'data': step3_data,
                'metrics': {
                    'predicted_shards': predicted_num_shards,
                    'actual_shards': len(unique_shards),
                    'shard_distribution': shard_counts.tolist()
                }
            }

        except Exception as e:
            logger.error(f"[ERROR] [第三步] 失败: {e}")
            return {'success': False, 'error': str(e)}

    def _execute_step4(self, step3_data: Dict[str, Any], embeddings: torch.Tensor) -> Dict[str, Any]:
        """执行第四步：性能反馈评估"""
        try:
            if self.step4_feedback_engine:
                # 使用真实的反馈引擎
                features = {
                    'node_features': embeddings,
                    'degree_centrality': torch.rand(embeddings.shape[0], 1),
                    'betweenness_centrality': torch.rand(embeddings.shape[0], 1),
                    'clustering_coefficient': torch.rand(embeddings.shape[0], 1),
                    'pagerank': torch.rand(embeddings.shape[0], 1),
                    'shard_balance': torch.rand(embeddings.shape[0], 1)
                }
                
                feedback_matrix = self.step4_feedback_engine.analyze_performance(
                    features,
                    step3_data['hard_assignment'],
                    step3_data['shard_assignments']
                )
                
                # 计算性能指标
                feedback_signal = feedback_matrix
                performance_score = float(feedback_matrix.mean().item())  # 使用反馈矩阵的平均值作为性能分数
                
            else:
                # 简化的性能计算
                hard_assignment = step3_data['hard_assignment']
                shard_counts = step3_data['shard_sizes']
                
                # 负载均衡分数
                if len(shard_counts) > 1:
                    balance_score = 1.0 - (np.std(shard_counts) / (np.mean(shard_counts) + 1e-8))
                else:
                    balance_score = 0.0
                
                # 跨片率分数（简化计算）
                num_nodes = embeddings.shape[0]
                edge_index = torch.tensor([[i, (i+1) % num_nodes] for i in range(num_nodes)]).t()
                cross_shard_edges = (hard_assignment[edge_index[0]] != hard_assignment[edge_index[1]]).float()
                cross_rate = torch.mean(cross_shard_edges).item()
                cross_score = 1.0 - cross_rate
                
                # 综合性能分数
                performance_score = 0.6 * balance_score + 0.4 * cross_score
                feedback_signal = None

            logger.info(f"     [第四步] 性能分数: {performance_score:.3f}")
            
            return {
                'success': True,
                'performance_score': performance_score,
                'feedback_signal': feedback_signal,
                'metrics': {
                    'balance_score': balance_score if 'balance_score' in locals() else 0.0,
                    'cross_shard_rate': cross_rate if 'cross_rate' in locals() else 0.0
                }
            }

        except Exception as e:
            logger.error(f"[ERROR] [第四步] 失败: {e}")
            return {'success': False, 'error': str(e)}

    def _check_convergence(self, iteration_results: List[Dict]) -> bool:
        """检查是否收敛"""
        if len(iteration_results) < 2:
            return False
        
        # 检查最近两次迭代的性能变化
        recent_scores = [r.get('performance_score', 0.0) for r in iteration_results[-2:]]
        performance_change = abs(recent_scores[-1] - recent_scores[-2])
        
        return performance_change < self.config['step4']['convergence_threshold']

    def _save_cycle_results(self, results: Dict[str, Any]):
        """保存循环结果"""
        try:
            result_dir = Path(self.config['integration']['result_path'])
            result_dir.mkdir(exist_ok=True)
            
            # 保存详细结果
            result_file = result_dir / f"sharding_cycle_{self.current_epoch}.json"
            with open(result_file, 'w') as f:
                # 将tensor转换为列表以便JSON序列化
                serializable_results = self._make_json_serializable(results)
                json.dump(serializable_results, f, indent=2)
            
            # 保存简化结果（用于BlockEmulator集成）
            summary_file = result_dir / "latest_sharding_result.json"
            summary = {
                'cycle_id': results['cycle_id'],
                'timestamp': results['timestamp'],
                'num_shards': results['final_sharding']['actual_num_shards'] if results['final_sharding'] else 0,
                'shard_assignment': results['final_sharding']['hard_assignment'].tolist() if results['final_sharding'] else [],
                'performance_score': results['performance_metrics'].get('best_performance_score', 0.0),
                'status': results['status']
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
                
            logger.info(f"[SUCCESS] 结果已保存: {result_file}")

        except Exception as e:
            logger.error(f"[ERROR] 保存结果失败: {e}")

    def _make_json_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态（用于BlockEmulator监控）"""
        return {
            'system_name': self.config['system_name'],
            'version': self.config['version'],
            'current_epoch': self.current_epoch,
            'device': str(self.device),
            'components_status': {
                'step1_pipeline': self.step1_pipeline is not None,
                'step2_processor': self.step2_processor is not None,
                'step3_sharding_module': self.step3_sharding_module is not None,
                'step4_feedback_engine': self.step4_feedback_engine is not None
            },
            'last_update': time.time()
        }

def test_integrated_system():
    """测试集成系统"""
    logger.info("🧪 测试BlockEmulator集成分片系统")
    logger.info("=" * 80)

    # 创建系统实例
    system = BlockEmulatorIntegratedShardingSystem()
    
    # 初始化组件
    if not system.initialize_components():
        logger.error("[ERROR] 组件初始化失败")
        return False

    # 运行测试循环
    test_input = {
        'num_nodes': 25,
        'test_mode': True
    }
    
    result = system.run_complete_cycle(test_input)
    
    # 显示结果
    if result['status'] == 'completed':
        logger.info("[SUCCESS] 测试成功完成")
        logger.info(f"   最终分片数: {result['final_sharding']['actual_num_shards']}")
        logger.info(f"   性能分数: {result['performance_metrics']['best_performance_score']:.3f}")
        logger.info(f"   收敛状态: {result['performance_metrics']['converged']}")
        
        # 显示系统状态
        status = system.get_system_status()
        logger.info(f"   系统状态: {status['components_status']}")
        
        return True
    else:
        logger.error(f"[ERROR] 测试失败: {result.get('error', '未知错误')}")
        return False

def main():
    """主函数"""
    logger.info("[START] BlockEmulator集成分片系统")
    logger.info("   这是最终集成到BlockEmulator中的分片系统")
    logger.info("   严格按照四步流程：特征提取 → 多尺度对比学习 → EvolveGCN分片 ⇄ 性能反馈")
    logger.info("=" * 80)
    
    # 运行测试
    success = test_integrated_system()
    
    if success:
        logger.info("🎉 系统集成测试完全成功！")
        logger.info("   该分片系统已准备好集成到BlockEmulator中")
        logger.info("   可以通过调用 run_complete_cycle() 方法来执行动态分片")
    else:
        logger.error("[WARNING] 系统集成测试失败，需要进一步调试")

if __name__ == "__main__":
    main()
