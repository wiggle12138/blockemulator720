#!/usr/bin/env python3
"""
BlockEmulator 集成测试脚本
连接 BlockEmulator 主系统与 EvolveGCN/feedback 模块
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from blockchain_interface import blockchain_interface, BlockchainDataFormat
from test_iterative_sharding_feedback import IterativeShardingTest
from examples.full_pipeline_demo import FullIntegratedPipelineDemo

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BlockEmulatorIntegration:
    """BlockEmulator 集成控制器"""
    
    def __init__(self, config_path: str = "python_config.json"):
        self.config = self.load_config(config_path)
        self.is_running = False
        
        # 初始化模块
        self.evolve_gcn_test = None
        self.feedback_demo = None
        
        # 统计信息
        self.stats = {
            "iterations": 0,
            "successful_feedbacks": 0,
            "errors": 0,
            "start_time": None
        }
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Config loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "enable_evolve_gcn": True,
            "enable_feedback": True,
            "python_path": "python",
            "module_path": "./",
            "max_iterations": 10,
            "epochs_per_iteration": 8,
            "data_exchange_dir": "./data_exchange",
            "output_interval": 30,
            "continuous_mode": True,
            "log_level": "INFO"
        }
    
    def initialize_modules(self):
        """初始化 Python 模块"""
        try:
            if self.config.get("enable_evolve_gcn", True):
                logger.info("Initializing EvolveGCN module...")
                self.evolve_gcn_test = IterativeShardingTest()
                self.evolve_gcn_test.load_real_data()
                self.evolve_gcn_test.setup_modules()
                logger.info("EvolveGCN module initialized")
            
            if self.config.get("enable_feedback", True):
                logger.info("Initializing Feedback module...")
                self.feedback_demo = FullIntegratedPipelineDemo()
                logger.info("Feedback module initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize modules: {e}")
            raise
    
    def run_single_iteration(self) -> Dict[str, Any]:
        """运行单次迭代"""
        try:
            # 读取区块链数据
            blockchain_data = blockchain_interface.read_blockchain_data()
            
            if not blockchain_data:
                logger.debug("No blockchain data available")
                return {"status": "no_data"}
            
            logger.info(f"Processing {len(blockchain_data)} blockchain data files")
            
            # 处理数据
            results = self.process_blockchain_data(blockchain_data)
            
            # 写入反馈结果
            blockchain_interface.write_feedback_results(results)
            
            # 更新统计信息
            self.stats["iterations"] += 1
            self.stats["successful_feedbacks"] += 1
            
            logger.info(f"Iteration {self.stats['iterations']} completed successfully")
            
            return {"status": "success", "results": results}
            
        except Exception as e:
            logger.error(f"Error in iteration: {e}")
            self.stats["errors"] += 1
            return {"status": "error", "error": str(e)}
    
    def process_blockchain_data(self, blockchain_data: List[Dict]) -> Dict[str, Any]:
        """处理区块链数据"""
        logger.info("Processing blockchain data...")
        
        # 合并所有数据
        all_transactions = []
        all_performance = []
        all_shard_info = []
        
        for data in blockchain_data:
            all_transactions.extend(data.get('transactions', []))
            all_performance.append(data.get('performance', {}))
            all_shard_info.append(data.get('shard_info', {}))
        
        # 运行 EvolveGCN 测试
        if self.evolve_gcn_test and self.config.get("enable_evolve_gcn", True):
            logger.info("Running EvolveGCN dynamic sharding...")
            try:
                gcn_results = self.evolve_gcn_test.run_iterative_test(
                    max_iterations=self.config.get("max_iterations", 10),
                    epochs_per_iteration=self.config.get("epochs_per_iteration", 8)
                )
                
                # 运行反馈模块
                if self.feedback_demo and self.config.get("enable_feedback", True):
                    logger.info("Running feedback module...")
                    feedback_results = self.feedback_demo.run_demo()
                    
                    # 合并结果
                    combined_results = self.combine_results(gcn_results, feedback_results)
                else:
                    combined_results = self.format_gcn_results(gcn_results)
                
                return combined_results
                
            except Exception as e:
                logger.error(f"Error in EvolveGCN processing: {e}")
                return self.create_error_result(str(e))
        
        # 如果只运行反馈模块
        elif self.feedback_demo and self.config.get("enable_feedback", True):
            logger.info("Running feedback module only...")
            try:
                feedback_results = self.feedback_demo.run_demo()
                return self.format_feedback_results(feedback_results)
            except Exception as e:
                logger.error(f"Error in feedback processing: {e}")
                return self.create_error_result(str(e))
        
        return self.create_error_result("No modules enabled")
    
    def combine_results(self, gcn_results: Dict, feedback_results: Dict) -> Dict[str, Any]:
        """合并 EvolveGCN 和反馈结果"""
        return BlockchainDataFormat.create_feedback_data(
            shard_assignments=self.extract_shard_assignments(gcn_results),
            performance_score=gcn_results.get('performance_score', 0.0),
            load_balance_score=gcn_results.get('load_balance_score', 0.0),
            security_score=gcn_results.get('security_score', 0.0),
            cross_shard_ratio=gcn_results.get('cross_shard_ratio', 0.0),
            recommendations=self.generate_recommendations(gcn_results, feedback_results),
            optimized_sharding=self.optimize_sharding_config(gcn_results)
        )
    
    def format_gcn_results(self, gcn_results: Dict) -> Dict[str, Any]:
        """格式化 EvolveGCN 结果"""
        return BlockchainDataFormat.create_feedback_data(
            shard_assignments=self.extract_shard_assignments(gcn_results),
            performance_score=gcn_results.get('performance_score', 0.0),
            load_balance_score=gcn_results.get('load_balance_score', 0.0),
            security_score=gcn_results.get('security_score', 0.0),
            cross_shard_ratio=gcn_results.get('cross_shard_ratio', 0.0),
            recommendations=self.generate_recommendations(gcn_results),
            optimized_sharding=self.optimize_sharding_config(gcn_results)
        )
    
    def format_feedback_results(self, feedback_results: Dict) -> Dict[str, Any]:
        """格式化反馈结果"""
        return BlockchainDataFormat.create_feedback_data(
            shard_assignments={},
            performance_score=feedback_results.get('performance_score', 0.0),
            load_balance_score=feedback_results.get('load_balance_score', 0.0),
            security_score=feedback_results.get('security_score', 0.0),
            cross_shard_ratio=feedback_results.get('cross_shard_ratio', 0.0),
            recommendations=feedback_results.get('recommendations', []),
            optimized_sharding={}
        )
    
    def create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            "timestamp": int(time.time()),
            "status": "error",
            "error": error_msg,
            "shard_assignments": {},
            "performance_score": 0.0,
            "load_balance_score": 0.0,
            "security_score": 0.0,
            "cross_shard_ratio": 0.0,
            "recommendations": [f"Error occurred: {error_msg}"],
            "optimized_sharding": {}
        }
    
    def extract_shard_assignments(self, results: Dict) -> Dict[int, int]:
        """提取分片分配"""
        assignments = {}
        if 'shard_assignments' in results:
            if isinstance(results['shard_assignments'], list):
                for node_id, shard_id in enumerate(results['shard_assignments']):
                    assignments[node_id] = int(shard_id)
            elif isinstance(results['shard_assignments'], dict):
                assignments = {int(k): int(v) for k, v in results['shard_assignments'].items()}
        return assignments
    
    def generate_recommendations(self, gcn_results: Dict, feedback_results: Dict = None) -> List[str]:
        """生成推荐"""
        recommendations = []
        
        # 基于 EvolveGCN 结果的推荐
        if gcn_results.get('load_balance_score', 0) < 0.7:
            recommendations.append("建议重新平衡分片负载分布")
        
        if gcn_results.get('cross_shard_ratio', 0) > 0.3:
            recommendations.append("建议优化跨分片交易处理机制")
        
        if gcn_results.get('security_score', 0) < 0.8:
            recommendations.append("建议增强分片安全性措施")
        
        # 基于反馈结果的推荐
        if feedback_results:
            if feedback_results.get('performance_score', 0) < 0.6:
                recommendations.append("建议调整共识参数以提高性能")
        
        if not recommendations:
            recommendations.append("当前分片配置运行良好")
        
        return recommendations
    
    def optimize_sharding_config(self, results: Dict) -> Dict[int, Dict]:
        """优化分片配置"""
        configs = {}
        
        if 'optimized_sharding' in results:
            for shard_id, config in results['optimized_sharding'].items():
                configs[int(shard_id)] = {
                    "shard_id": int(shard_id),
                    "node_ids": config.get('node_ids', []),
                    "load_score": config.get('load_score', 0.0),
                    "capacity": config.get('capacity', 100)
                }
        
        return configs
    
    def run_continuous_mode(self):
        """运行连续模式"""
        logger.info("Starting continuous mode...")
        self.is_running = True
        self.stats["start_time"] = time.time()
        
        # 更新状态
        blockchain_interface.update_status("running", {
            "mode": "continuous",
            "config": self.config,
            "stats": self.stats
        })
        
        while self.is_running:
            try:
                # 运行一次迭代
                result = self.run_single_iteration()
                
                # 更新状态
                blockchain_interface.update_status("processing", {
                    "iteration": self.stats["iterations"],
                    "result": result,
                    "stats": self.stats
                })
                
                # 定期清理旧数据
                if self.stats["iterations"] % 10 == 0:
                    blockchain_interface.cleanup_old_data()
                
                # 等待下一次迭代
                time.sleep(self.config.get("output_interval", 30))
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping...")
                break
            except Exception as e:
                logger.error(f"Error in continuous mode: {e}")
                self.stats["errors"] += 1
                time.sleep(10)  # 错误后等待更长时间
        
        self.is_running = False
        blockchain_interface.update_status("stopped", {"stats": self.stats})
        logger.info("Continuous mode stopped")
    
    def run_single_mode(self):
        """运行单次模式"""
        logger.info("Running single mode...")
        
        self.stats["start_time"] = time.time()
        
        # 更新状态
        blockchain_interface.update_status("running", {
            "mode": "single",
            "config": self.config
        })
        
        # 运行一次迭代
        result = self.run_single_iteration()
        
        # 输出结果
        if result["status"] == "success":
            results = result["results"]
            logger.info("[SUCCESS] 集成测试完成，结果:")
            logger.info(f"  - 性能分数: {results.get('performance_score', 0):.3f}")
            logger.info(f"  - 负载均衡: {results.get('load_balance_score', 0):.3f}")
            logger.info(f"  - 安全分数: {results.get('security_score', 0):.3f}")
            logger.info(f"  - 跨分片比例: {results.get('cross_shard_ratio', 0):.3f}")
            logger.info(f"  - 推荐建议: {len(results.get('recommendations', []))} 条")
        else:
            logger.error(f"[ERROR] 集成测试失败: {result.get('error', 'Unknown error')}")
        
        blockchain_interface.update_status("completed", {
            "result": result,
            "stats": self.stats
        })
    
    def stop(self):
        """停止运行"""
        self.is_running = False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="BlockEmulator Python Integration")
    parser.add_argument("--config", default="python_config.json", help="配置文件路径")
    parser.add_argument("--mode", choices=["single", "continuous"], default="continuous", help="运行模式")
    parser.add_argument("--max_iterations", type=int, help="最大迭代次数")
    parser.add_argument("--epochs_per_iteration", type=int, help="每轮epoch数")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.log_level:
        logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    print("[START] 启动 BlockEmulator Python 集成模块")
    print(f"运行模式: {args.mode}")
    print(f"配置文件: {args.config}")
    
    try:
        # 初始化集成控制器
        integration = BlockEmulatorIntegration(args.config)
        
        # 覆盖配置文件中的参数
        if args.max_iterations:
            integration.config["max_iterations"] = args.max_iterations
        if args.epochs_per_iteration:
            integration.config["epochs_per_iteration"] = args.epochs_per_iteration
        
        # 初始化模块
        integration.initialize_modules()
        
        # 运行
        if args.mode == "continuous":
            integration.run_continuous_mode()
        else:
            integration.run_single_mode()
            
    except KeyboardInterrupt:
        print("\n[WARNING] 接收到中断信号，正在关闭...")
        if 'integration' in locals():
            integration.stop()
    except Exception as e:
        print(f"[ERROR] 运行错误: {e}")
        logger.error(f"Main error: {e}", exc_info=True)
        blockchain_interface.update_status("error", {"error": str(e)})
    finally:
        blockchain_interface.update_status("stopped")
        print("[SUCCESS] 程序已停止")

if __name__ == "__main__":
    main()
