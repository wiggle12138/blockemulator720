# partition.feature 包初始化文件
"""
BlockEmulator 特征提取模块
"""

# 设置包结构，解决相对导入问题
import sys
import os
from pathlib import Path

# 确保当前路径在sys.path中
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 预加载核心模块，避免循环导入
try:
    # 重要：先加载基础配置模块
    from . import config
    from . import nodeInitialize
    
    # 再加载依赖这些基础模块的高级模块
    from . import data_processor
    from . import graph_builder
    from . import feature_extractor
    from . import blockemulator_adapter
    from . import system_integration_pipeline
    
    print("[Partition.Feature] 包加载成功")
    
except ImportError as e:
    print(f"[Partition.Feature] 包加载警告: {e}")
    # 如果相对导入失败，设置标志位让各个模块使用绝对导入
    _USE_ABSOLUTE_IMPORT = True
else:
    _USE_ABSOLUTE_IMPORT = False
