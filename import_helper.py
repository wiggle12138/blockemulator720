#!/usr/bin/env python3
"""
导入助手模块 - 解决partition/feature模块的相对导入问题
统一处理不同导入环境下的模块加载
"""

import sys
import importlib.util
from pathlib import Path
from typing import Any, Optional, Dict
import warnings

class ImportHelper:
    """
    导入助手类 - 智能解决相对导入问题
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """初始化导入助手"""
        self.base_path = base_path or Path(__file__).parent
        self.loaded_modules: Dict[str, Any] = {}
        self._setup_paths()
    
    def _setup_paths(self):
        """设置导入路径"""
        # 添加必要的路径到sys.path
        paths_to_add = [
            str(self.base_path.absolute()),
            str((self.base_path / "partition").absolute()),  
            str((self.base_path / "partition" / "feature").absolute())
        ]
        
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
    
    def safe_import(self, module_name: str, file_path: Optional[Path] = None, 
                   package_name: Optional[str] = None) -> Optional[Any]:
        """
        安全导入模块
        
        Args:
            module_name: 模块名称
            file_path: 模块文件路径（可选）
            package_name: 包名称（用于相对导入）
            
        Returns:
            导入的模块，失败则返回None
        """
        # 如果已经加载过，直接返回
        cache_key = f"{package_name}.{module_name}" if package_name else module_name
        if cache_key in self.loaded_modules:
            return self.loaded_modules[cache_key]
        
        module = None
        
        # 方法1: 尝试包导入
        if package_name:
            try:
                full_name = f"{package_name}.{module_name}"
                module = importlib.import_module(full_name)
                print(f"[SUCCESS] 包导入成功: {full_name}")
            except ImportError as e:
                print(f"  包导入失败: {e}")
        
        # 方法2: 尝试直接导入
        if not module:
            try:
                module = importlib.import_module(module_name)
                print(f"[SUCCESS] 直接导入成功: {module_name}")
            except ImportError as e:
                print(f"  直接导入失败: {e}")
        
        # 方法3: 尝试文件导入
        if not module and file_path and file_path.exists():
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # 添加到sys.modules中以供其他模块引用
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    print(f"[SUCCESS] 文件导入成功: {file_path}")
            except Exception as e:
                print(f"  文件导入失败: {e}")
        
        # 缓存结果
        if module:
            self.loaded_modules[cache_key] = module
        
        return module
    
    def load_feature_modules(self) -> Dict[str, Any]:
        """
        加载feature模块的所有依赖
        
        Returns:
            加载的模块字典
        """
        feature_path = self.base_path / "partition" / "feature"
        modules = {}
        
        print("[DEBUG] 开始加载feature模块依赖...")
        
        # 按依赖顺序加载
        load_order = [
            ("config", "config.py"),
            ("nodeInitialize", "nodeInitialize.py"),
            ("data_processor", "data_processor.py"), 
            ("graph_builder", "graph_builder.py"),
            ("sliding_window_extractor", "sliding_window_extractor.py"),
            ("feature_extractor", "feature_extractor.py"),
            ("blockemulator_adapter", "blockemulator_adapter.py"),
            ("system_integration_pipeline", "system_integration_pipeline.py")
        ]
        
        for module_name, file_name in load_order:
            file_path = feature_path / file_name
            if file_path.exists():
                print(f"  加载 {module_name}...")
                module = self.safe_import(
                    module_name=module_name,
                    file_path=file_path,
                    package_name="partition.feature"
                )
                if module:
                    modules[module_name] = module
                    # 同时添加到全局命名空间
                    sys.modules[f"partition.feature.{module_name}"] = module
                else:
                    print(f"  [WARNING] {module_name} 加载失败")
            else:
                print(f"  [WARNING] {file_name} 文件不存在")
        
        print(f"[SUCCESS] 成功加载 {len(modules)}/{len(load_order)} 个模块")
        return modules

def get_step1_pipeline_class():
    """
    获取Step1流水线类的统一接口
    
    Returns:
        BlockEmulatorStep1Pipeline类或None
    """
    print("[INIT] 尝试获取Step1流水线类...")
    
    helper = ImportHelper()
    
    # 加载所有依赖模块
    modules = helper.load_feature_modules()
    
    # 尝试获取主类
    if "system_integration_pipeline" in modules:
        pipeline_module = modules["system_integration_pipeline"]
        pipeline_class = getattr(pipeline_module, 'BlockEmulatorStep1Pipeline', None)
        if pipeline_class:
            print("[SUCCESS] 成功获取真实Step1流水线类")
            return pipeline_class
    
    print("❌ 无法获取Step1流水线类")
    return None

def test_import_helper():
    """测试导入助手"""
    print("=== 导入助手测试 ===")
    
    pipeline_class = get_step1_pipeline_class()
    if pipeline_class:
        print(f"[SUCCESS] 测试成功: {pipeline_class}")
        # 尝试创建实例
        try:
            instance = pipeline_class()
            print(f"[SUCCESS] 实例创建成功: {type(instance)}")
        except Exception as e:
            print(f"[WARNING] 实例创建失败: {e}")
    else:
        print("❌ 测试失败")

if __name__ == "__main__":
    test_import_helper()
