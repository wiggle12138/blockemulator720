import json
import os
from pathlib import Path
import logging

class ConfigLoader:
    """配置文件加载器，支持多种配置文件格式"""
    
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.config = {}
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_file=None):
        """加载配置文件"""
        if config_file is None:
            # 按优先级查找配置文件
            config_files = [
                "integration_config.json",
                "evolve_gcn_feedback_config.json", 
                "python_config.json"
            ]
            
            for config_file in config_files:
                if os.path.exists(config_file):
                    self.config_path = config_file
                    break
            else:
                raise FileNotFoundError("未找到配置文件")
        else:
            self.config_path = config_file
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
                self.logger.info(f"成功加载配置文件: {self.config_path}")
                return self.config
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise
    
    def get(self, key, default=None):
        """获取配置值，支持嵌套键名"""
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_evolve_gcn_config(self):
        """获取EvolveGCN配置"""
        return self.get('evolve_gcn', {})
    
    def get_feedback_config(self):
        """获取反馈配置"""
        return self.get('feedback', {})
    
    def get_integration_config(self):
        """获取集成配置"""
        return self.get('integration', {})
    
    def get_blockchain_config(self):
        """获取区块链配置"""
        return self.get('blockchain', {})
    
    def is_module_enabled(self, module_name):
        """检查模块是否启用"""
        return self.get(f'modules.enable_{module_name}', False)
    
    def get_data_exchange_dir(self):
        """获取数据交换目录"""
        return self.get('environment.data_exchange_dir', './data_exchange')
    
    def get_log_level(self):
        """获取日志级别"""
        return self.get('logging.level', 'INFO')
    
    def validate_config(self):
        """验证配置文件的完整性"""
        required_keys = [
            'modules.enable_evolve_gcn',
            'modules.enable_feedback',
            'environment.python_path',
            'environment.module_path'
        ]
        
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"配置文件缺少必要的键: {missing_keys}")
        
        return True
    
    def save_config(self, config_path=None):
        """保存配置文件"""
        if config_path is None:
            config_path = self.config_path
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            self.logger.info(f"配置文件已保存: {config_path}")
        except Exception as e:
            self.logger.error(f"保存配置文件失败: {e}")
            raise
    
    def update_config(self, updates):
        """更新配置"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
        return self.config

# 全局配置实例
config_loader = ConfigLoader()

def load_system_config(config_file=None):
    """加载系统配置"""
    return config_loader.load_config(config_file)

def get_config(key, default=None):
    """获取配置值"""
    return config_loader.get(key, default)

# 示例使用
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 加载配置
        config = load_system_config()
        
        # 验证配置
        config_loader.validate_config()
        
        # 获取各模块配置
        evolve_gcn_config = config_loader.get_evolve_gcn_config()
        feedback_config = config_loader.get_feedback_config()
        integration_config = config_loader.get_integration_config()
        
        print("=== 系统配置加载完成 ===")
        print(f"EvolveGCN启用: {config_loader.is_module_enabled('evolve_gcn')}")
        print(f"反馈模块启用: {config_loader.is_module_enabled('feedback')}")
        print(f"数据交换目录: {config_loader.get_data_exchange_dir()}")
        print(f"日志级别: {config_loader.get_log_level()}")
        
    except Exception as e:
        print(f"配置加载失败: {e}")
