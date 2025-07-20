#!/usr/bin/env python3
"""
区块链系统与 Python 模块的数据接口
用于 BlockEmulator 与 EvolveGCN/feedback 模块的数据交换
"""
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BlockchainInterface:
    """区块链系统与 Python 模块的接口"""
    
    def __init__(self, data_dir: str = "./data_exchange"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 输入输出文件路径
        self.input_pattern = "blockchain_data_*.json"
        self.output_file = self.data_dir / "feedback_results.json"
        self.status_file = self.data_dir / "python_status.json"
        self.config_file = self.data_dir / "runtime_config.json"
        
        # 初始化状态
        self.update_status("initialized")
        logger.info(f"BlockchainInterface initialized with data_dir: {self.data_dir}")
    
    def update_status(self, status: str, details: Optional[Dict] = None):
        """更新 Python 模块状态"""
        status_data = {
            "timestamp": int(time.time()),
            "status": status,
            "details": details or {},
            "pid": os.getpid()
        }
        
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
            logger.debug(f"Status updated: {status}")
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
    
    def read_blockchain_data(self) -> List[Dict[str, Any]]:
        """读取区块链数据"""
        data_files = list(self.data_dir.glob(self.input_pattern))
        blockchain_data = []
        
        for file_path in data_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    blockchain_data.append(data)
                    logger.info(f"Loaded data from {file_path}")
                
                # 读取后删除文件以避免重复处理
                file_path.unlink()
                
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
        
        return blockchain_data
    
    def write_feedback_results(self, feedback_data: Dict[str, Any]):
        """写入反馈结果"""
        try:
            # 添加时间戳
            feedback_data["generated_at"] = int(time.time())
            
            with open(self.output_file, 'w') as f:
                json.dump(feedback_data, f, indent=2)
            
            logger.info(f"Feedback results written to {self.output_file}")
            
        except Exception as e:
            logger.error(f"Error writing feedback results: {e}")
    
    def read_runtime_config(self) -> Dict[str, Any]:
        """读取运行时配置"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error reading runtime config: {e}")
        
        return {}
    
    def write_runtime_config(self, config: Dict[str, Any]):
        """写入运行时配置"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info("Runtime config updated")
        except Exception as e:
            logger.error(f"Error writing runtime config: {e}")
    
    def cleanup_old_data(self, max_age_hours: int = 24):
        """清理旧数据文件"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        for file_path in self.data_dir.glob("*.json"):
            try:
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    logger.info(f"Cleaned up old file: {file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up {file_path}: {e}")
    
    def get_latest_blockchain_data(self) -> Optional[Dict[str, Any]]:
        """获取最新的区块链数据"""
        data_files = list(self.data_dir.glob(self.input_pattern))
        if not data_files:
            return None
        
        # 按修改时间排序，获取最新的
        latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            # 读取后删除文件
            latest_file.unlink()
            return data
            
        except Exception as e:
            logger.error(f"Error reading latest data from {latest_file}: {e}")
            return None
    
    def is_go_system_running(self) -> bool:
        """检查 Go 系统是否在运行"""
        # 通过检查是否有新的数据文件来判断
        data_files = list(self.data_dir.glob(self.input_pattern))
        if not data_files:
            return False
        
        # 检查最新文件是否在合理时间内创建
        latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
        age = time.time() - latest_file.stat().st_mtime
        
        return age < 120  # 2分钟内有新数据则认为系统在运行

# 全局接口实例
blockchain_interface = BlockchainInterface()

# 数据格式定义
class BlockchainDataFormat:
    """区块链数据格式定义"""
    
    @staticmethod
    def create_blockchain_data(node_id: int, shard_id: int, 
                              transactions: List[Dict], 
                              performance: Dict,
                              shard_info: Dict) -> Dict[str, Any]:
        """创建标准的区块链数据格式"""
        return {
            "timestamp": int(time.time()),
            "node_id": node_id,
            "shard_id": shard_id,
            "transactions": transactions,
            "performance": performance,
            "shard_info": shard_info
        }
    
    @staticmethod
    def create_feedback_data(shard_assignments: Dict[int, int],
                            performance_score: float,
                            load_balance_score: float,
                            security_score: float,
                            cross_shard_ratio: float,
                            recommendations: List[str],
                            optimized_sharding: Dict[int, Dict]) -> Dict[str, Any]:
        """创建标准的反馈数据格式"""
        return {
            "timestamp": int(time.time()),
            "shard_assignments": shard_assignments,
            "performance_score": performance_score,
            "load_balance_score": load_balance_score,
            "security_score": security_score,
            "cross_shard_ratio": cross_shard_ratio,
            "recommendations": recommendations,
            "optimized_sharding": optimized_sharding
        }
