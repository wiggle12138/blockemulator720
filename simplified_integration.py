#!/usr/bin/env python3
"""
简化的Python集成脚本 - 用于测试和运行反馈优化分片模块
"""

import json
import os
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
from config_loader import ConfigLoader, load_system_config, get_config

# 尝试导入numpy，如果不存在则使用内置random
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    import random
    HAS_NUMPY = False
    print("[WARNING] Warning: numpy not found, using built-in random instead")

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

# 设置日志
def setup_logging(log_level: str = "INFO"):
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

class SimplifiedBlockEmulatorIntegration:
    """简化的 BlockEmulator 集成控制器"""
    
    def __init__(self, config_path: str = None):
        # 初始化配置加载器
        self.config_loader = ConfigLoader()
        
        # 加载配置
        if config_path:
            self.config = self.config_loader.load_config(config_path)
        else:
            # 自动查找配置文件
            self.config = load_system_config()
        
        # 设置日志
        self.logger = setup_logging(self.config_loader.get_log_level())
        
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
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件（保持向后兼容）"""
        try:
            return self.config_loader.load_config(config_path)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
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
        
        # 基础统计
        total_nodes = len(blockchain_data)
        total_transactions = sum(len(data.get('transactions', [])) for data in blockchain_data)
        
        # 模拟分片分配（基于负载均衡）
        shard_assignments = self.simulate_load_balanced_sharding(blockchain_data)
        
        # 模拟性能评估
        performance_score = self.simulate_performance_evaluation(blockchain_data)
        load_balance_score = self.simulate_load_balance_evaluation(shard_assignments)
        security_score = self.simulate_security_evaluation(blockchain_data)
        cross_shard_ratio = self.simulate_cross_shard_ratio(blockchain_data, shard_assignments)
        
        # 生成推荐
        recommendations = self.generate_recommendations(
            performance_score, load_balance_score, security_score, cross_shard_ratio
        )
        
        # 优化分片配置
        optimized_sharding = self.optimize_sharding_config(shard_assignments, blockchain_data)
        
        return BlockchainDataFormat.create_feedback_data(
            shard_assignments=shard_assignments,
            performance_score=performance_score,
            load_balance_score=load_balance_score,
            security_score=security_score,
            cross_shard_ratio=cross_shard_ratio,
            recommendations=recommendations,
            optimized_sharding=optimized_sharding
        )
    
    def simulate_load_balanced_sharding(self, blockchain_data: List[Dict]) -> Dict[int, int]:
        """模拟负载均衡分片"""
        assignments = {}
        
        # 计算每个节点的负载
        node_loads = {}
        for data in blockchain_data:
            node_id = data.get('node_id', 0)
            transactions = len(data.get('transactions', []))
            performance = data.get('performance', {})
            load = transactions * performance.get('latency', 1.0) / max(performance.get('tps', 1.0), 1.0)
            node_loads[node_id] = load
        
        # 简单的负载均衡分片算法
        sorted_nodes = sorted(node_loads.items(), key=lambda x: x[1])
        shard_count = min(4, max(2, len(sorted_nodes) // 3))  # 2-4个分片
        
        for i, (node_id, load) in enumerate(sorted_nodes):
            assignments[node_id] = i % shard_count
        
        return assignments
    
    def simulate_performance_evaluation(self, blockchain_data: List[Dict]) -> float:
        """模拟性能评估"""
        if not blockchain_data:
            return 0.0
        
        total_tps = sum(data.get('performance', {}).get('tps', 0) for data in blockchain_data)
        total_latency = sum(data.get('performance', {}).get('latency', 0) for data in blockchain_data)
        
        avg_tps = total_tps / len(blockchain_data)
        avg_latency = total_latency / len(blockchain_data)
        
        # 简化的性能评分：TPS越高越好，延迟越低越好
        performance_score = min(1.0, (avg_tps / 100.0) * (50.0 / max(avg_latency, 1.0)))
        
        return max(0.0, min(1.0, performance_score))
    
    def simulate_load_balance_evaluation(self, shard_assignments: Dict[int, int]) -> float:
        """模拟负载均衡评估"""
        if not shard_assignments:
            return 0.0
        
        # 计算每个分片的节点数
        shard_counts = {}
        for node_id, shard_id in shard_assignments.items():
            shard_counts[shard_id] = shard_counts.get(shard_id, 0) + 1
        
        if not shard_counts:
            return 0.0
        
        # 计算负载均衡分数：分片间节点数越平均越好
        avg_nodes = sum(shard_counts.values()) / len(shard_counts)
        variance = sum((count - avg_nodes) ** 2 for count in shard_counts.values()) / len(shard_counts)
        
        # 方差越小，负载越均衡
        load_balance_score = max(0.0, 1.0 - (variance / avg_nodes))
        
        return max(0.0, min(1.0, load_balance_score))
    
    def simulate_security_evaluation(self, blockchain_data: List[Dict]) -> float:
        """模拟安全性评估"""
        if not blockchain_data:
            return 0.0
        
        # 基于节点数量和分片信息评估安全性
        total_nodes = len(blockchain_data)
        active_nodes = sum(len(data.get('shard_info', {}).get('active_nodes', [])) for data in blockchain_data)
        
        # 节点越多，安全性越高
        security_score = min(1.0, active_nodes / max(total_nodes * 2, 1))
        
        return max(0.0, min(1.0, security_score))
    
    def simulate_cross_shard_ratio(self, blockchain_data: List[Dict], shard_assignments: Dict[int, int]) -> float:
        """模拟跨分片交易比例"""
        if not blockchain_data or not shard_assignments:
            return 0.0
        
        # 简化的跨分片比例计算
        total_transactions = sum(len(data.get('transactions', [])) for data in blockchain_data)
        cross_shard_tx = random.randint(0, total_transactions // 4)  # 模拟跨分片交易
        
        return cross_shard_tx / max(total_transactions, 1)
    
    def generate_recommendations(self, performance_score: float, load_balance_score: float, 
                               security_score: float, cross_shard_ratio: float) -> List[str]:
        """生成推荐"""
        recommendations = []
        
        if performance_score < 0.6:
            recommendations.append("建议优化共识算法参数以提高性能")
        
        if load_balance_score < 0.7:
            recommendations.append("建议重新平衡分片负载分布")
        
        if cross_shard_ratio > 0.3:
            recommendations.append("建议优化跨分片交易处理机制")
        
        if security_score < 0.8:
            recommendations.append("建议增强分片安全性措施")
        
        if performance_score >= 0.8 and load_balance_score >= 0.8:
            recommendations.append("当前分片配置运行良好")
        
        return recommendations
    
    def optimize_sharding_config(self, shard_assignments: Dict[int, int], blockchain_data: List[Dict]) -> Dict[int, Dict]:
        """优化分片配置"""
        configs = {}
        
        # 按分片组织节点
        shard_nodes = {}
        for node_id, shard_id in shard_assignments.items():
            if shard_id not in shard_nodes:
                shard_nodes[shard_id] = []
            shard_nodes[shard_id].append(node_id)
        
        # 为每个分片生成配置
        for shard_id, node_ids in shard_nodes.items():
            # 计算分片负载
            shard_load = 0.0
            for node_id in node_ids:
                node_data = next((data for data in blockchain_data if data.get('node_id') == node_id), {})
                transactions = len(node_data.get('transactions', []))
                shard_load += transactions
            
            configs[shard_id] = {
                "shard_id": shard_id,
                "node_ids": node_ids,
                "load_score": min(1.0, shard_load / max(len(node_ids) * 50, 1)),
                "capacity": len(node_ids) * 100
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
            logger.info("[SUCCESS] 简化集成测试完成，结果:")
            logger.info(f"  - 性能分数: {results.get('performance_score', 0):.3f}")
            logger.info(f"  - 负载均衡: {results.get('load_balance_score', 0):.3f}")
            logger.info(f"  - 安全分数: {results.get('security_score', 0):.3f}")
            logger.info(f"  - 跨分片比例: {results.get('cross_shard_ratio', 0):.3f}")
            logger.info(f"  - 推荐建议: {len(results.get('recommendations', []))} 条")
            
            for i, rec in enumerate(results.get('recommendations', []), 1):
                logger.info(f"    {i}. {rec}")
                
        else:
            logger.error(f"[ERROR] 简化集成测试失败: {result.get('error', 'Unknown error')}")
        
        blockchain_interface.update_status("completed", {
            "result": result,
            "stats": self.stats
        })
    
    def stop(self):
        """停止运行"""
        self.is_running = False

class SimulatedDataGenerator:
    """模拟数据生成器"""
    
    def __init__(self):
        self.node_counter = 0
    
    def generate_sample_data(self, num_nodes: int = 4) -> List[Dict]:
        """生成示例数据"""
        data = []
        
        for i in range(num_nodes):
            node_data = {
                "timestamp": int(time.time()),
                "node_id": i,
                "shard_id": i % 2,  # 简单的分片分配
                "transactions": [
                    {
                        "tx_id": f"tx_{i}_{j}_{int(time.time())}",
                        "from": f"addr_{i}_{j}_from",
                        "to": f"addr_{i}_{j}_to",
                        "value": random.uniform(1.0, 1000.0),
                        "gas_used": random.randint(21000, 100000),
                        "status": "confirmed"
                    }
                    for j in range(random.randint(1, 10))
                ],
                "performance": {
                    "tps": random.uniform(50.0, 200.0),
                    "latency": random.uniform(10.0, 100.0),
                    "cross_shard_tx": random.randint(0, 5),
                    "total_tx": random.randint(10, 50),
                    "block_time": random.uniform(1.0, 10.0),
                    "queue_length": random.randint(0, 10)
                },
                "shard_info": {
                    "shard_id": i % 2,
                    "node_count": num_nodes,
                    "active_nodes": [i],
                    "load_balance": random.uniform(0.5, 1.0),
                    "security_score": random.uniform(0.7, 1.0)
                }
            }
            data.append(node_data)
        
        return data

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Simplified BlockEmulator Python Integration")
    parser.add_argument("--config", default="python_config.json", help="配置文件路径")
    parser.add_argument("--mode", choices=["single", "continuous"], default="single", help="运行模式")
    parser.add_argument("--max_iterations", type=int, help="最大迭代次数")
    parser.add_argument("--epochs_per_iteration", type=int, help="每轮epoch数")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    parser.add_argument("--generate_sample", action="store_true", help="生成示例数据")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.log_level:
        logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    print("[START] 启动简化的 BlockEmulator Python 集成模块")
    print(f"运行模式: {args.mode}")
    print(f"配置文件: {args.config}")
    
    try:
        # 初始化集成控制器
        integration = SimplifiedBlockEmulatorIntegration(args.config)
        
        # 覆盖配置文件中的参数
        if args.max_iterations:
            integration.config["max_iterations"] = args.max_iterations
        if args.epochs_per_iteration:
            integration.config["epochs_per_iteration"] = args.epochs_per_iteration
        
        # 如果需要生成示例数据
        if args.generate_sample:
            logger.info("Generating sample data...")
            sample_data = integration.data_generator.generate_sample_data()
            for data in sample_data:
                filename = f"blockchain_data_{data['shard_id']}_{data['node_id']}_{int(time.time())}.json"
                filepath = Path(integration.config["data_exchange_dir"]) / filename
                filepath.parent.mkdir(exist_ok=True)
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"Sample data saved to {filepath}")
        
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
