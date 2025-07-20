#!/usr/bin/env python3
"""
完整系统流程演示
演示从特征提取到BlockEmulator分片应用的完整流程
"""

import sys
import os
import time
import json
import pickle
import torch
from pathlib import Path

def main():
    """演示完整的四步骤闭环流程"""
    print("🎮 BlockEmulator四步骤闭环集成系统演示")
    print("=" * 60)
    
    print("📋 系统流程:")
    print("  [STEP1] 第一步：从BlockEmulator获取特征数据")
    print("  [STEP2] 第二步：多尺度对比学习生成时序嵌入")
    print("  [STEP3] 第三步：EvolveGCN动态分片优化")
    print("  [STEP4] 第四步：性能反馈评估")
    print("  🔄 第三步⇄第四步多轮迭代优化")
    print("  🔗 分片结果应用到BlockEmulator系统")
    print()
    
    try:
        # 步骤1: 运行四步骤闭环流水线
        print("[START] 启动四步骤闭环流水线...")
        from integrated_four_step_pipeline import OriginalIntegratedFourStepPipeline
        
        pipeline = OriginalIntegratedFourStepPipeline()
        pipeline.run_complete_pipeline()
        
        print("\n[SUCCESS] 四步骤闭环流水线执行完成")
        
        # 步骤2: 手动演示BlockEmulator集成接口
        print("\n" + "="*60)
        print("🔗 手动演示BlockEmulator集成接口")
        print("="*60)
        
        demo_blockemulator_integration()
        
        print("\n[SUCCESS] 完整系统演示完成!")
        
    except Exception as e:
        print(f"[ERROR] 演示执行失败: {e}")
        import traceback
        traceback.print_exc()


def demo_blockemulator_integration():
    """演示BlockEmulator集成接口"""
    
    try:
        from blockemulator_integration_interface import BlockEmulatorIntegrationInterface
        
        print("[CONFIG] 创建BlockEmulator集成接口...")
        interface = BlockEmulatorIntegrationInterface()
        
        # 创建模拟的四步算法结果
        print("\n[DATA] 准备模拟的四步算法结果...")
        demo_results = create_demo_results()
        
        # 应用结果到BlockEmulator
        print("\n[TARGET] 应用结果到BlockEmulator...")
        status = interface.apply_four_step_results_to_blockemulator(demo_results)
        
        # 显示结果
        print("\n📋 应用状态:")
        print(json.dumps(status, indent=2, ensure_ascii=False))
        
        # 创建兼容桥梁
        print("\n🌉 创建兼容桥梁...")
        bridge_path = interface.create_compatibility_bridge()
        
        print(f"\n[SUCCESS] BlockEmulator集成演示完成")
        print(f"   兼容桥梁: {bridge_path}")
        
    except ImportError as e:
        print(f"[WARNING] 无法导入BlockEmulator集成接口: {e}")
    except Exception as e:
        print(f"[ERROR] BlockEmulator集成演示失败: {e}")


def create_demo_results():
    """创建演示用的四步算法结果"""
    
    # 模拟100个节点分配到4个分片
    num_nodes = 100
    num_shards = 4
    
    # 创建相对均衡的分片分配
    shard_assignments = []
    for i in range(num_nodes):
        shard_id = i % num_shards
        shard_assignments.append(shard_id)
    
    # 添加一些随机性来模拟算法优化结果
    import random
    for _ in range(10):  # 随机调整10个节点的分片
        node_id = random.randint(0, num_nodes - 1)
        new_shard = random.randint(0, num_shards - 1)
        shard_assignments[node_id] = new_shard
    
    # 构建完整的结果结构
    demo_results = {
        # 分片分配结果
        'shard_assignments': shard_assignments,
        
        # 性能指标
        'performance_metrics': {
            'load_balance': 0.85,        # 负载均衡评分
            'cross_shard_rate': 0.15,    # 跨分片交易率
            'security_score': 0.92,      # 安全性评分
            'consensus_latency': 125.5    # 共识延迟(ms)
        },
        
        # 优化分片信息
        'optimized_sharding': {},
        
        # 优化反馈
        'optimized_feedback': {
            'overall_score': 0.88
        },
        
        # 智能建议
        'smart_suggestions': [
            '当前分片配置表现良好',
            '建议继续监控跨分片交易率',
            '可适当优化节点间通信延迟'
        ],
        
        # 异常报告
        'anomaly_report': {
            'anomaly_count': 2
        }
    }
    
    # 构建优化分片信息
    for shard_id in range(num_shards):
        node_ids = [i for i, s in enumerate(shard_assignments) if s == shard_id]
        demo_results['optimized_sharding'][str(shard_id)] = {
            'node_ids': node_ids,
            'load_score': 0.8 + random.uniform(-0.1, 0.1),
            'capacity': len(node_ids) * 10
        }
    
    print(f"[SUCCESS] 创建演示结果: {num_nodes}节点 → {num_shards}分片")
    shard_distribution = {}
    for shard_id in shard_assignments:
        shard_distribution[shard_id] = shard_distribution.get(shard_id, 0) + 1
    print(f"   分片分布: {shard_distribution}")
    
    return demo_results


def check_system_requirements():
    """检查系统需求"""
    print("🔍 检查系统需求...")
    
    requirements = {
        'torch': 'PyTorch深度学习框架',
        'numpy': 'NumPy数值计算库',
        'pandas': 'Pandas数据处理库',
        'pathlib': 'Path路径处理库'
    }
    
    missing = []
    for package, description in requirements.items():
        try:
            __import__(package)
            print(f"  [SUCCESS] {package}: {description}")
        except ImportError:
            print(f"  [ERROR] {package}: {description} - 缺失")
            missing.append(package)
    
    if missing:
        print(f"\n[WARNING] 缺失依赖: {', '.join(missing)}")
        print("请安装缺失的依赖包")
        return False
    
    print("[SUCCESS] 所有需求满足")
    return True


def show_system_architecture():
    """显示系统架构"""
    print("\n🏗️ 系统架构:")
    print("""
    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
    │  BlockEmulator  │───▶│   特征提取(步骤1)  │───▶│ 对比学习(步骤2) │
    │  (Go系统)        │    │  CSV/实时数据     │    │   时序嵌入      │
    └─────────────────┘    └──────────────────┘    └─────────────────┘
            ▲                                              │
            │                                              ▼
            │                ┌─────────────────┐    ┌─────────────────┐
            │                │  性能反馈(步骤4) │◀───│ EvolveGCN(步骤3)│
            │                │   评估优化      │    │   动态分片      │
            │                └─────────────────┘    └─────────────────┘
            │                        ▲                       │
            │                        │                       │
            │                   ┌─────────┐                  │
            │                   │ 闭环迭代 │                  │
            │                   └─────────┘                  │
            │                                               │
            │    ┌─────────────────────────────────────────────┘
            │    │
            │    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │            BlockEmulator集成接口                           │
    │   • 分片结果格式转换                                        │
    │   • PartitionModifiedMap消息                              │
    │   • AccountTransferMsg账户迁移                            │
    │   • 重分片触发和监控                                        │
    └─────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    print("🌟 开始系统演示...")
    
    # 显示架构
    show_system_architecture()
    
    # 检查需求
    if not check_system_requirements():
        sys.exit(1)
    
    # 运行主演示
    main()
    
    print("\n🎉 系统演示完成!")
    print("💡 提示: 在真实环境中，请确保BlockEmulator系统正在运行以实现完整的分片应用流程。")
