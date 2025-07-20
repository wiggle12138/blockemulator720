#!/usr/bin/env python3
"""
简化的Python集成脚本 - 用于测试和运行反馈优化分片模块
支持EvolveGCN和Feedback集成系统
"""

import json
import os
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
import signal
import sys

# 导入配置加载器
try:
    from config_loader import ConfigLoader, load_system_config, get_config
except ImportError:
    print("[WARNING] Warning: config_loader not found, using basic configuration")
    ConfigLoader = None

# 尝试导入numpy，如果不存在则使用内置random
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    import random
    HAS_NUMPY = False
    print("[WARNING] Warning: numpy not found, using built-in random instead")

# 设置日志
def setup_logging(log_level: str = "INFO"):
    """设置日志配置"""
    level = getattr(logging, log_level.upper())
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('integration.log')
        ]
    )
    return logging.getLogger(__name__)

class MockBlockchainInterface:
    """模拟区块链接口，用于测试"""
    
    def __init__(self):
        self.data_exchange_dir = Path("./data_exchange")
        self.data_exchange_dir.mkdir(exist_ok=True)
        
    def read_blockchain_data(self):
        """读取区块链数据"""
        # 模拟返回一些测试数据
        return [
            {"block_height": 1, "transactions": 100, "shard_id": 0},
            {"block_height": 2, "transactions": 150, "shard_id": 1},
            {"block_height": 3, "transactions": 200, "shard_id": 2}
        ]
    
    def write_feedback_results(self, results):
        """写入反馈结果"""
        output_file = self.data_exchange_dir / "feedback_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def update_status(self, status, data):
        """更新状态"""
        status_file = self.data_exchange_dir / "status.json"
        with open(status_file, 'w') as f:
            json.dump({"status": status, "data": data}, f, indent=2)
    
    def cleanup_old_data(self):
        """清理旧数据"""
        # 模拟清理操作
        pass

class BlockchainDataFormat:
    """区块链数据格式化类"""
    
    @staticmethod
    def create_feedback_data(performance_scores, optimization_results):
        """创建反馈数据"""
        return {
            "performance_scores": performance_scores,
            "optimization_results": optimization_results,
            "timestamp": time.time()
        }

class SimplifiedBlockEmulatorIntegration:
    """简化的 BlockEmulator 集成控制器"""
    
    def __init__(self, config_path: str = None):
        # 初始化配置
        if ConfigLoader and config_path is None:
            # 使用高级配置加载器
            self.config_loader = ConfigLoader()
            self.config = load_system_config()
            log_level = self.config_loader.get_log_level()
        else:
            # 使用基础配置
            self.config = self.load_basic_config(config_path)
            log_level = self.config.get("log_level", "INFO")
        
        # 设置日志
        self.logger = setup_logging(log_level)
        
        self.is_running = False
        
        # 统计信息
        self.stats = {
            "iterations": 0,
            "successful_feedbacks": 0,
            "errors": 0,
            "start_time": None
        }
        
        # 模拟数据生成器
        self.data_generator = SimulatedDataGenerator()
        
        # 初始化区块链接口
        try:
            import blockchain_interface
            self.blockchain_interface = blockchain_interface
        except ImportError:
            self.logger.warning("blockchain_interface not found, using mock interface")
            self.blockchain_interface = MockBlockchainInterface()
    
    def load_basic_config(self, config_path: str = None) -> Dict[str, Any]:
        """加载基础配置文件"""
        if config_path is None:
            config_files = [
                "integration_config.json",
                "python_config.json",
                "evolve_gcn_feedback_config.json"
            ]
            for cf in config_files:
                if os.path.exists(cf):
                    config_path = cf
                    break
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load config: {e}")
        
        return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "modules": {
                "enable_evolve_gcn": True,
                "enable_feedback": True,
                "enable_integration": True
            },
            "environment": {
                "python_path": "python",
                "module_path": "./",
                "data_exchange_dir": "./data_exchange"
            },
            "integration": {
                "max_iterations": 10,
                "epochs_per_iteration": 8,
                "output_interval": 30,
                "continuous_mode": True
            },
            "logging": {
                "level": "INFO"
            }
        }
    
    def get_config_value(self, key: str, default=None):
        """获取配置值，支持嵌套键"""
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def run_single_iteration(self) -> Dict[str, Any]:
        """运行单次迭代"""
        try:
            # 读取区块链数据
            blockchain_data = self.blockchain_interface.read_blockchain_data()
            
            if not blockchain_data:
                self.logger.debug("No blockchain data available")
                return {"status": "no_data"}
            
            self.logger.info(f"Processing {len(blockchain_data)} blockchain data files")
            
            # 处理数据
            results = self.process_blockchain_data(blockchain_data)
            
            # 写入反馈结果
            self.blockchain_interface.write_feedback_results(results)
            
            # 更新统计信息
            self.stats["iterations"] += 1
            self.stats["successful_feedbacks"] += 1
            
            self.logger.info(f"Iteration {self.stats['iterations']} completed successfully")
            
            return {"status": "success", "results": results}
            
        except Exception as e:
            self.logger.error(f"Error in iteration: {e}")
            self.stats["errors"] += 1
            return {"status": "error", "error": str(e)}
    
    def process_blockchain_data(self, blockchain_data: List[Dict]) -> Dict[str, Any]:
        """处理区块链数据 - 简化版本"""
        self.logger.info("Processing blockchain data with simplified algorithms...")
        
        # 模拟 EvolveGCN 和 Feedback 处理
        return self.simulate_advanced_processing(blockchain_data)
    
    def simulate_advanced_processing(self, blockchain_data: List[Dict]) -> Dict[str, Any]:
        """模拟高级处理算法"""
        self.logger.info("Running simulated EvolveGCN and Feedback processing...")
        
        # 模拟性能评估
        performance_scores = self.data_generator.generate_performance_scores()
        
        # 模拟优化建议
        optimization_results = self.data_generator.generate_optimization_suggestions(
            len(blockchain_data)
        )
        
        # 创建反馈数据
        return BlockchainDataFormat.create_feedback_data(
            performance_scores, 
            optimization_results
        )
    
    def run_continuous_mode(self):
        """运行连续模式"""
        self.logger.info("Starting continuous mode...")
        self.is_running = True
        self.stats["start_time"] = time.time()
        
        # 更新状态
        self.blockchain_interface.update_status("running", {
            "mode": "continuous",
            "start_time": self.stats["start_time"]
        })
        
        try:
            max_iterations = self.get_config_value("integration.max_iterations", 10)
            output_interval = self.get_config_value("integration.output_interval", 30)
            
            for iteration in range(max_iterations):
                if not self.is_running:
                    break
                
                # 更新处理状态
                self.blockchain_interface.update_status("processing", {
                    "iteration": iteration + 1,
                    "total_iterations": max_iterations
                })
                
                # 运行单次迭代
                result = self.run_single_iteration()
                
                # 定期清理旧数据
                if iteration % 5 == 0:
                    self.blockchain_interface.cleanup_old_data()
                
                # 等待下一次迭代
                time.sleep(output_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, stopping...")
            self.is_running = False
            
        except Exception as e:
            self.logger.error(f"Error in continuous mode: {e}")
            self.is_running = False
        
        finally:
            # 更新最终状态
            self.blockchain_interface.update_status("stopped", {"stats": self.stats})
            self.logger.info("Continuous mode stopped")
    
    def run_single_mode(self):
        """运行单次模式"""
        self.logger.info("Running single mode...")
        self.stats["start_time"] = time.time()
        
        # 更新状态
        self.blockchain_interface.update_status("running", {
            "mode": "single",
            "start_time": self.stats["start_time"]
        })
        
        # 运行单次迭代
        result = self.run_single_iteration()
        
        # 显示结果
        if result["status"] == "success":
            results = result["results"]
            self.logger.info("[SUCCESS] 简化集成测试完成，结果:")
            self.logger.info(f"  - 性能分数: {results.get('performance_score', 0):.3f}")
            self.logger.info(f"  - 负载均衡: {results.get('load_balance_score', 0):.3f}")
            self.logger.info(f"  - 安全分数: {results.get('security_score', 0):.3f}")
            self.logger.info(f"  - 跨分片比例: {results.get('cross_shard_ratio', 0):.3f}")
            self.logger.info(f"  - 推荐建议: {len(results.get('recommendations', []))} 条")
            
            # 显示建议
            for i, rec in enumerate(results.get('recommendations', [])[:3], 1):
                self.logger.info(f"    {i}. {rec}")
        else:
            self.logger.error(f"[ERROR] 简化集成测试失败: {result.get('error', 'Unknown error')}")
        
        return result
    
    def stop(self):
        """停止运行"""
        self.is_running = False

class SimulatedDataGenerator:
    """模拟数据生成器"""
    
    def __init__(self):
        pass
    
    def generate_performance_scores(self) -> Dict[str, float]:
        """生成性能分数"""
        if HAS_NUMPY:
            return {
                "performance_score": float(np.random.uniform(0.6, 0.95)),
                "load_balance_score": float(np.random.uniform(0.7, 0.9)),
                "security_score": float(np.random.uniform(0.8, 0.95)),
                "cross_shard_ratio": float(np.random.uniform(0.1, 0.3))
            }
        else:
            return {
                "performance_score": random.uniform(0.6, 0.95),
                "load_balance_score": random.uniform(0.7, 0.9),
                "security_score": random.uniform(0.8, 0.95),
                "cross_shard_ratio": random.uniform(0.1, 0.3)
            }
    
    def generate_optimization_suggestions(self, data_size: int) -> Dict[str, Any]:
        """生成优化建议"""
        suggestions = [
            "增加分片数量以提高并行处理能力",
            "优化跨分片交易路由算法",
            "调整验证节点分布以提高安全性",
            "实施动态负载均衡策略",
            "优化交易池管理机制"
        ]
        
        if HAS_NUMPY:
            selected_count = int(np.random.uniform(2, 5))
            selected_suggestions = np.random.choice(suggestions, selected_count, replace=False).tolist()
        else:
            selected_count = random.randint(2, 5)
            selected_suggestions = random.sample(suggestions, selected_count)
        
        return {
            "recommendations": selected_suggestions,
            "optimization_score": random.uniform(0.6, 0.9),
            "estimated_improvement": random.uniform(0.1, 0.3)
        }

def signal_handler(signum, frame):
    """信号处理器"""
    global integration_controller
    print(f"\n收到信号 {signum}，正在停止...")
    if integration_controller:
        integration_controller.stop()
    sys.exit(0)

# 全局变量
integration_controller = None

def main():
    """主函数"""
    global integration_controller
    
    parser = argparse.ArgumentParser(description='简化的BlockEmulator集成脚本')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--mode', choices=['single', 'continuous'], default='single', 
                       help='运行模式：single（单次）或continuous（连续）')
    parser.add_argument('--iterations', type=int, default=10, help='最大迭代次数')
    parser.add_argument('--interval', type=int, default=30, help='输出间隔（秒）')
    args = parser.parse_args()
    
    # 设置信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 创建集成控制器
        integration_controller = SimplifiedBlockEmulatorIntegration(args.config)
        
        # 运行相应模式
        if args.mode == 'continuous':
            integration_controller.run_continuous_mode()
        else:
            integration_controller.run_single_mode()
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
