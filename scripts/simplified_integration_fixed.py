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
import numpy as np

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

class BlockchainInterface:
    """简化的区块链接口"""
    
    def __init__(self, data_dir: str = "./data_exchange"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 输入输出文件路径
        self.input_pattern = "blockchain_data_*.json"
        self.output_file = self.data_dir / "feedback_results.json"
        self.status_file = self.data_dir / "python_status.json"
        
        # 初始化状态
        self.update_status("initialized")
    
    def update_status(self, status: str, details: Dict = None):
        """更新状态"""
        status_data = {
            "timestamp": int(time.time()),
            "status": status,
            "details": details or {}
        }
        
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            print(f"Error updating status: {e}")
    
    def read_blockchain_data(self) -> List[Dict]:
        """读取区块链数据"""
        data_files = list(self.data_dir.glob(self.input_pattern))
        blockchain_data = []
        
        for file_path in data_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    blockchain_data.append(data)
                # 读取后删除文件
                file_path.unlink()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        return blockchain_data
    
    def write_feedback_results(self, feedback_data: Dict):
        """写入反馈结果"""
        try:
            feedback_data["generated_at"] = int(time.time())
            with open(self.output_file, 'w') as f:
                json.dump(feedback_data, f, indent=2)
        except Exception as e:
            print(f"Error writing feedback results: {e}")

class SimplifiedIntegration:
    def __init__(self, config_file: str = "python_config.json"):
        self.config = self.load_config(config_file)
        self.logger = setup_logging(self.config.get("log_level", "INFO"))
        self.blockchain_interface = BlockchainInterface(self.config.get("data_exchange_dir", "./data_exchange"))
        
        self.logger.info("[START] SimplifiedIntegration initialized")
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_file} not found, using defaults")
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
    
    def generate_sample_data(self) -> List[Dict[str, Any]]:
        """生成示例区块链数据"""
        self.logger.info("[DATA] Generating sample blockchain data...")
        
        sample_data = []
        for shard_id in range(2):  # 2个分片
            for node_id in range(4):  # 每个分片4个节点
                data = {
                    "timestamp": int(time.time()),
                    "node_id": node_id,
                    "shard_id": shard_id,
                    "transactions": [
                        {
                            "tx_id": f"tx_{shard_id}_{node_id}_{i}",
                            "from": f"addr_{np.random.randint(1000)}",
                            "to": f"addr_{np.random.randint(1000)}",
                            "value": np.random.uniform(0.1, 100.0),
                            "gas_used": np.random.randint(21000, 100000),
                            "status": "success" if np.random.random() > 0.1 else "failed"
                        }
                        for i in range(np.random.randint(10, 50))
                    ],
                    "performance": {
                        "tps": np.random.uniform(100, 1000),
                        "latency": np.random.uniform(0.1, 2.0),
                        "cross_shard_tx": np.random.randint(0, 20),
                        "total_tx": np.random.randint(50, 200),
                        "block_time": np.random.uniform(1.0, 15.0),
                        "queue_length": np.random.randint(0, 100)
                    },
                    "shard_info": {
                        "shard_id": shard_id,
                        "node_count": 4,
                        "active_nodes": list(range(4)),
                        "load_balance": np.random.uniform(0.5, 1.0),
                        "security_score": np.random.uniform(0.7, 1.0)
                    }
                }
                sample_data.append(data)
        
        # 保存示例数据
        data_dir = Path(self.config.get("data_exchange_dir", "./data_exchange"))
        data_dir.mkdir(exist_ok=True)
        
        for i, data in enumerate(sample_data):
            filename = f"blockchain_data_{data['shard_id']}_{data['node_id']}.json"
            filepath = data_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        self.logger.info(f"[SUCCESS] Generated {len(sample_data)} sample data files")
        return sample_data
    
    def run_evolve_gcn_simulation(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """模拟运行EvolveGCN分片优化"""
        self.logger.info("🧠 Running EvolveGCN simulation...")
        
        # 计算性能指标
        total_tps = sum(d['performance']['tps'] for d in data)
        avg_latency = np.mean([d['performance']['latency'] for d in data])
        cross_shard_ratio = sum(d['performance']['cross_shard_tx'] for d in data) / sum(d['performance']['total_tx'] for d in data)
        
        # 模拟分片优化结果
        num_nodes = len(data)
        shard_assignments = {}
        
        # 简单的负载均衡分配
        for i, node_data in enumerate(data):
            # 基于性能分配分片
            if node_data['performance']['tps'] > 500:
                shard_assignments[node_data['node_id']] = 0  # 高性能节点到分片0
            else:
                shard_assignments[node_data['node_id']] = 1  # 低性能节点到分片1
        
        # 计算优化分数
        performance_score = min(total_tps / 2000, 1.0)  # 归一化TPS
        load_balance_score = 1.0 - abs(0.5 - len([v for v in shard_assignments.values() if v == 0]) / len(shard_assignments))
        security_score = np.mean([d['shard_info']['security_score'] for d in data])
        
        results = {
            "algorithm": "EvolveGCN",
            "performance_score": performance_score,
            "load_balance_score": load_balance_score,
            "security_score": security_score,
            "cross_shard_ratio": cross_shard_ratio,
            "shard_assignments": shard_assignments,
            "total_tps": total_tps,
            "avg_latency": avg_latency,
            "optimization_time": np.random.uniform(0.5, 2.0)
        }
        
        self.logger.info(f"[SUCCESS] EvolveGCN simulation completed - Performance: {performance_score:.3f}")
        return results
    
    def run_feedback_optimization(self, evolve_results: Dict[str, Any], data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """运行反馈优化"""
        self.logger.info("🔄 Running feedback optimization...")
        
        # 基于EvolveGCN结果进行反馈优化
        current_performance = evolve_results['performance_score']
        
        # 模拟反馈调整
        recommendations = []
        if current_performance < 0.7:
            recommendations.append("建议增加高性能节点到热点分片")
        if evolve_results['load_balance_score'] < 0.8:
            recommendations.append("建议重新平衡分片负载分布")
        if evolve_results['cross_shard_ratio'] > 0.2:
            recommendations.append("建议优化跨分片交易路由")
        
        # 生成优化的分片配置
        optimized_sharding = {}
        for shard_id in [0, 1]:
            nodes_in_shard = [k for k, v in evolve_results['shard_assignments'].items() if v == shard_id]
            optimized_sharding[shard_id] = {
                "shard_id": shard_id,
                "node_ids": nodes_in_shard,
                "load_score": np.random.uniform(0.7, 1.0),
                "capacity": len(nodes_in_shard) * 100
            }
        
        feedback_results = {
            "algorithm": "FeedbackOptimization",
            "base_performance": current_performance,
            "optimized_performance": min(current_performance + 0.1, 1.0),
            "recommendations": recommendations,
            "optimized_sharding": optimized_sharding,
            "improvement_ratio": 0.1,
            "feedback_time": np.random.uniform(0.2, 1.0)
        }
        
        self.logger.info(f"[SUCCESS] Feedback optimization completed - Improvement: {feedback_results['improvement_ratio']:.3f}")
        return feedback_results
    
    def run_integrated_optimization(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """运行集成优化"""
        self.logger.info("🌟 Running integrated optimization...")
        
        # 运行EvolveGCN
        evolve_results = self.run_evolve_gcn_simulation(data)
        
        # 运行反馈优化
        feedback_results = self.run_feedback_optimization(evolve_results, data)
        
        # 整合结果
        integrated_results = {
            "timestamp": int(time.time()),
            "mode": "integrated",
            "shard_assignments": evolve_results['shard_assignments'],
            "performance_score": feedback_results['optimized_performance'],
            "load_balance_score": evolve_results['load_balance_score'],
            "security_score": evolve_results['security_score'],
            "cross_shard_ratio": evolve_results['cross_shard_ratio'],
            "recommendations": feedback_results['recommendations'],
            "optimized_sharding": feedback_results['optimized_sharding'],
            "total_optimization_time": evolve_results['optimization_time'] + feedback_results['feedback_time'],
            "components": {
                "evolve_gcn": evolve_results,
                "feedback": feedback_results
            }
        }
        
        self.logger.info("[SUCCESS] Integrated optimization completed")
        return integrated_results
    
    def run_single_test(self, generate_sample: bool = True):
        """运行单次测试"""
        self.logger.info("🔄 Running single test...")
        
        # 生成或加载数据
        if generate_sample:
            data = self.generate_sample_data()
        else:
            data = self.blockchain_interface.read_blockchain_data()
        
        if not data:
            self.logger.error("[ERROR] No data available for processing")
            return
        
        # 运行优化
        if self.config.get("enable_evolve_gcn", True) and self.config.get("enable_feedback", True):
            results = self.run_integrated_optimization(data)
        elif self.config.get("enable_evolve_gcn", True):
            results = self.run_evolve_gcn_simulation(data)
        elif self.config.get("enable_feedback", True):
            # 需要先运行EvolveGCN作为基础
            evolve_results = self.run_evolve_gcn_simulation(data)
            results = self.run_feedback_optimization(evolve_results, data)
        else:
            self.logger.error("[ERROR] No optimization modules enabled")
            return
        
        # 保存结果
        self.blockchain_interface.write_feedback_results(results)
        
        # 显示结果摘要
        self.display_results_summary(results)
        
        return results
    
    def display_results_summary(self, results: Dict[str, Any]):
        """显示结果摘要"""
        print("\n" + "="*60)
        print("[DATA] OPTIMIZATION RESULTS SUMMARY")
        print("="*60)
        print(f"🕒 Timestamp: {results.get('timestamp', 'N/A')}")
        print(f"[TARGET] Mode: {results.get('mode', 'single')}")
        print(f"📈 Performance Score: {results.get('performance_score', 0):.3f}")
        print(f"⚖️ Load Balance Score: {results.get('load_balance_score', 0):.3f}")
        print(f"🔒 Security Score: {results.get('security_score', 0):.3f}")
        print(f"🔄 Cross-Shard Ratio: {results.get('cross_shard_ratio', 0):.3f}")
        
        if 'recommendations' in results:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        if 'optimized_sharding' in results:
            print("\n[TARGET] Optimized Sharding:")
            for shard_id, config in results['optimized_sharding'].items():
                print(f"  Shard {shard_id}: {len(config['node_ids'])} nodes, Load: {config['load_score']:.3f}")
        
        print("\n" + "="*60)
    
    def run_continuous_mode(self):
        """运行连续模式"""
        self.logger.info("🔄 Running continuous mode...")
        
        iteration = 0
        while True:
            try:
                print(f"\n🔄 Iteration {iteration + 1}")
                
                # 生成新数据
                data = self.generate_sample_data()
                
                # 运行优化
                results = self.run_integrated_optimization(data)
                
                # 保存结果
                self.blockchain_interface.write_feedback_results(results)
                
                # 显示简要结果
                print(f"[SUCCESS] Iteration {iteration + 1} completed - Performance: {results['performance_score']:.3f}")
                
                iteration += 1
                
                # 等待下一轮
                time.sleep(self.config.get("output_interval", 30))
                
            except KeyboardInterrupt:
                self.logger.info("[WARNING] Received interrupt signal, stopping continuous mode...")
                break
            except Exception as e:
                self.logger.error(f"[ERROR] Error in continuous mode: {e}")
                time.sleep(10)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Simplified Python Integration for BlockEmulator")
    parser.add_argument("--mode", choices=["single", "continuous"], default="single", 
                       help="运行模式: single (单次) 或 continuous (连续)")
    parser.add_argument("--generate_sample", action="store_true", 
                       help="生成示例数据")
    parser.add_argument("--config", default="python_config.json", 
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    print("[START] Starting BlockEmulator Simplified Python Integration")
    print(f"📋 Mode: {args.mode}")
    print(f"⚙️ Config: {args.config}")
    
    # 初始化集成器
    integration = SimplifiedIntegration(args.config)
    
    try:
        if args.mode == "single":
            integration.run_single_test(generate_sample=args.generate_sample)
        else:
            integration.run_continuous_mode()
            
    except KeyboardInterrupt:
        print("\n[WARNING] Received interrupt signal, shutting down...")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        logging.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
