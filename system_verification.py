"""
系统验证脚本
验证4步分片系统的所有组件是否正确配置和可用
"""

import sys
import os
import json
import importlib
from pathlib import Path
import traceback

def check_system_components():
    """检查所有系统组件是否可用"""
    print("开始检查系统组件...")
    
    components_status = {
        'step1_pipeline': False,
        'step2_mscia': False,
        'step3_evolvegcn': False,
        'step4_feedback': False,
        'blockemulator_interface': False
    }
    
    # 检查第一步 - 特征提取管道
    try:
        from partition.feature.system_integration_pipeline import BlockEmulatorStep1Pipeline
        pipeline = BlockEmulatorStep1Pipeline()
        print("第一步特征提取管道: 可用")
        components_status['step1_pipeline'] = True
    except Exception as e:
        print(f"第一步特征提取管道: 不可用 - {e}")
    
    # 检查第二步 - 多尺度对比学习
    try:
        from muti_scale.realtime_mscia import RealtimeMSCIAProcessor
        from muti_scale.step2_config import Step2Config
        config = Step2Config()
        processor = RealtimeMSCIAProcessor(config.get_config('default'))
        print("第二步多尺度处理器: 可用")
        components_status['step2_mscia'] = True
    except Exception as e:
        print(f"第二步多尺度处理器: 不可用 - {e}")
    
    # 检查第三步 - EvolveGCN分片模块
    try:
        from evolve_GCN.models.sharding_modules import DynamicShardingModule
        # 创建一个测试实例
        module = DynamicShardingModule(
            embedding_dim=64,
            base_shards=3,
            max_shards=10
        )
        print("第三步EvolveGCN分片模块: 可用")
        components_status['step3_evolvegcn'] = True
    except Exception as e:
        print(f"第三步EvolveGCN分片模块: 不可用 - {e}")
    
    # 检查第四步 - 统一反馈引擎
    try:
        from feedback.unified_feedback_engine import UnifiedFeedbackEngine
        engine = UnifiedFeedbackEngine()
        print("第四步统一反馈引擎: 可用")
        components_status['step4_feedback'] = True
    except Exception as e:
        print(f"第四步统一反馈引擎: 不可用 - {e}")
    
    # 检查BlockEmulator接口
    try:
        from blockchain_interface import BlockchainInterface
        interface = BlockchainInterface()
        print("BlockEmulator接口: 可用")
        components_status['blockemulator_interface'] = True
    except Exception as e:
        print(f"BlockEmulator接口: 不可用 - {e}")
    
    return components_status

def check_configuration_files():
    """检查配置文件是否存在和有效"""
    print("\n检查配置文件...")
    
    config_files = [
        'real_system_test_config.json',
        'paramsConfig.json',
        'python_config.json'
    ]
    
    config_status = {}
    
    for config_file in config_files:
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                print(f"配置文件 {config_file}: 有效")
                config_status[config_file] = True
            else:
                print(f"配置文件 {config_file}: 不存在")
                config_status[config_file] = False
        except Exception as e:
            print(f"配置文件 {config_file}: 无效 - {e}")
            config_status[config_file] = False
    
    return config_status

def check_dependencies():
    """检查Python依赖包"""
    print("\n检查Python依赖...")
    
    required_packages = [
        'torch',
        'numpy',
        'pandas',
        'networkx',
        'sklearn',
        'matplotlib'
    ]
    
    dependency_status = {}
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"依赖包 {package}: 已安装")
            dependency_status[package] = True
        except ImportError:
            print(f"依赖包 {package}: 未安装")
            dependency_status[package] = False
    
    return dependency_status

def generate_system_report(components, configs, dependencies):
    """生成系统状态报告"""
    print("\n" + "="*50)
    print("系统组件状态报告")
    print("="*50)
    
    # 组件状态
    print("\n系统组件:")
    for component, status in components.items():
        status_text = "可用" if status else "不可用"
        print(f"  {component}: {status_text}")
    
    # 配置文件状态
    print("\n配置文件:")
    for config, status in configs.items():
        status_text = "有效" if status else "无效/不存在"
        print(f"  {config}: {status_text}")
    
    # 依赖状态
    print("\n依赖包:")
    for dep, status in dependencies.items():
        status_text = "已安装" if status else "未安装"
        print(f"  {dep}: {status_text}")
    
    # 总体状态
    all_components_ok = all(components.values())
    all_configs_ok = all(configs.values())
    critical_deps_ok = all(dependencies[dep] for dep in ['torch', 'numpy', 'pandas'])
    
    print("\n总体状态:")
    print(f"  系统组件: {'全部正常' if all_components_ok else '存在问题'}")
    print(f"  配置文件: {'全部正常' if all_configs_ok else '存在问题'}")
    print(f"  关键依赖: {'全部正常' if critical_deps_ok else '存在问题'}")
    
    system_ready = all_components_ok and critical_deps_ok
    print(f"\n系统就绪状态: {'就绪' if system_ready else '未就绪'}")
    
    if not system_ready:
        print("\n建议修复以下问题:")
        if not all_components_ok:
            print("- 检查并修复不可用的系统组件")
        if not critical_deps_ok:
            print("- 安装缺失的关键Python依赖包")
    
    return system_ready

def main():
    """主函数"""
    print("系统验证开始...")
    
    try:
        # 检查各个组件
        components_status = check_system_components()
        config_status = check_configuration_files()
        dependency_status = check_dependencies()
        
        # 生成报告
        system_ready = generate_system_report(
            components_status,
            config_status,
            dependency_status
        )
        
        # 保存验证结果
        import datetime
        verification_result = {
            'timestamp': str(datetime.datetime.now()),
            'components': components_status,
            'configurations': config_status,
            'dependencies': dependency_status,
            'system_ready': system_ready
        }
        
        with open('system_verification_report.json', 'w', encoding='utf-8') as f:
            json.dump(verification_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n验证报告已保存到: system_verification_report.json")
        
        return 0 if system_ready else 1
        
    except Exception as e:
        print(f"系统验证过程中出现错误: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
