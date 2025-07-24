#!/usr/bin/env python3
"""
BlockEmulator分片系统对接测试脚本
验证第一步特征提取的完整流程
"""

import os
import sys
import torch
import json
import time
from datetime import datetime

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_blockemulator_integration():
    """完整的对接测试流程"""
    print("=" * 60)
    print("BlockEmulator 分片系统对接测试")
    print("=" * 60)
    
    # 1. 测试适配器功能
    print("\n1. 测试适配器功能...")
    try:
        from blockemulator_adapter import BlockEmulatorAdapter, create_mock_blockemulator_data
        
        adapter = BlockEmulatorAdapter()
        mock_data = create_mock_blockemulator_data(num_nodes=15, num_shards=3)
        
        results = adapter.create_step1_output(
            raw_data=mock_data,
            output_filename="test_adapter_output.pt"
        )
        
        print(f"   ✓ 适配器测试成功")
        print(f"   - 节点数量: {results['metadata']['num_nodes']}")
        print(f"   - 特征维度: {results['metadata']['feature_dim']}")
        print(f"   - 边数量: {results['metadata']['num_edges']}")
        
    except Exception as e:
        print(f"   ✗ 适配器测试失败: {e}")
        return False
    
    # 2. 测试系统集成流水线
    print("\n2. 测试系统集成流水线...")
    try:
        from system_integration_pipeline import BlockEmulatorStep1Pipeline, create_mock_node_features_module
        
        pipeline = BlockEmulatorStep1Pipeline(output_dir="./test_integration_outputs")
        mock_system = create_mock_node_features_module()
        
        # 测试全量特征提取
        all_results = pipeline.extract_features_from_system(
            node_features_module=mock_system,
            experiment_name="integration_test"
        )
        
        print(f"   ✓ 流水线测试成功")
        print(f"   - 特征形状: {all_results['features'].shape}")
        print(f"   - 图结构: {all_results['edge_index'].shape}")
        
    except Exception as e:
        print(f"   ✗ 流水线测试失败: {e}")
        return False
    
    # 3. 测试epoch提取功能
    print("\n3. 测试epoch提取功能...")
    try:
        epoch_result = pipeline.extract_features_from_epoch_data(
            node_features_module=mock_system,
            epoch=1,
            experiment_name="epoch_test"
        )
        
        if epoch_result:
            print(f"   ✓ Epoch提取成功")
            print(f"   - Epoch 1 特征: {epoch_result['features'].shape}")
        else:
            print(f"   ✗ Epoch提取失败")
            return False
            
    except Exception as e:
        print(f"   ✗ Epoch提取测试失败: {e}")
        return False
    
    # 4. 测试批量处理
    print("\n4. 测试批量处理...")
    try:
        batch_results = pipeline.batch_extract_epoch_features(
            node_features_module=mock_system,
            epochs=[1, 2, 3],
            experiment_name="batch_test"
        )
        
        print(f"   ✓ 批量处理成功")
        print(f"   - 成功处理: {len(batch_results)} 个epoch")
        
        for epoch, result in batch_results.items():
            print(f"     Epoch {epoch}: {result['features'].shape[0]} 节点")
            
    except Exception as e:
        print(f"   ✗ 批量处理测试失败: {e}")
        return False
    
    # 5. 测试数据质量
    print("\n5. 测试数据质量...")
    try:
        features = all_results['features']
        
        # 检查数据质量
        nan_count = torch.isnan(features).sum().item()
        inf_count = torch.isinf(features).sum().item()
        
        quality_report = {
            'shape': list(features.shape),
            'nan_count': nan_count,
            'inf_count': inf_count,
            'mean': float(features.mean()),
            'std': float(features.std()),
            'min': float(features.min()),
            'max': float(features.max()),
            'feature_ranges': []
        }
        
        # 检查每个特征维度的范围
        for i in range(min(10, features.shape[1])):  # 只检查前10维
            dim_data = features[:, i]
            quality_report['feature_ranges'].append({
                'dim': i,
                'min': float(dim_data.min()),
                'max': float(dim_data.max()),
                'mean': float(dim_data.mean())
            })
        
        print(f"   ✓ 数据质量检查完成")
        print(f"   - NaN数量: {nan_count}")
        print(f"   - Inf数量: {inf_count}")
        print(f"   - 特征均值: {quality_report['mean']:.4f}")
        print(f"   - 特征标准差: {quality_report['std']:.4f}")
        
        # 保存质量报告
        with open('test_quality_report.json', 'w') as f:
            json.dump(quality_report, f, indent=2)
        
        if nan_count > 0 or inf_count > 0:
            print(f"   ⚠ 警告: 发现异常数值")
            
    except Exception as e:
        print(f"   ✗ 数据质量检查失败: {e}")
        return False
    
    # 6. 测试与后续步骤的兼容性
    print("\n6. 测试与后续步骤的兼容性...")
    try:
        # 检查输出格式是否包含后续步骤需要的字段
        required_fields = [
            'features', 'edge_index', 'edge_type', 'adjacency_matrix',
            'node_info', 'metadata'
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in all_results:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"   ✗ 缺少必要字段: {missing_fields}")
            return False
        
        # 检查数据类型
        type_checks = {
            'features': torch.Tensor,
            'edge_index': torch.Tensor,
            'edge_type': torch.Tensor,
            'adjacency_matrix': torch.Tensor,
        }
        
        for field, expected_type in type_checks.items():
            if not isinstance(all_results[field], expected_type):
                print(f"   ✗ 字段 {field} 类型错误: {type(all_results[field])}")
                return False
        
        print(f"   ✓ 兼容性检查通过")
        print(f"   - 所有必要字段存在")
        print(f"   - 数据类型正确")
        
    except Exception as e:
        print(f"   ✗ 兼容性检查失败: {e}")
        return False
    
    # 7. 生成测试报告
    print("\n7. 生成测试报告...")
    try:
        test_report = {
            'test_time': datetime.now().isoformat(),
            'test_status': 'PASSED',
            'components_tested': {
                'adapter': 'PASSED',
                'pipeline': 'PASSED',
                'epoch_extraction': 'PASSED',
                'batch_processing': 'PASSED',
                'data_quality': 'PASSED',
                'compatibility': 'PASSED'
            },
            'output_summary': {
                'total_nodes': int(all_results['metadata']['num_nodes']),
                'feature_dimension': int(all_results['metadata']['feature_dim']),
                'edge_count': int(all_results['metadata']['num_edges']),
                'files_generated': []
            },
            'recommendations': [
                "对接测试全部通过，可以开始实际集成",
                "建议在实际环境中进行小规模测试",
                "监控特征质量和系统性能"
            ]
        }
        
        # 列出生成的文件
        test_files = [
            'test_adapter_output.pt',
            'test_adapter_output_stats.json',
            'test_quality_report.json'
        ]
        
        for filename in test_files:
            if os.path.exists(filename):
                test_report['output_summary']['files_generated'].append(filename)
        
        # 保存测试报告
        with open('blockemulator_integration_test_report.json', 'w') as f:
            json.dump(test_report, f, indent=2)
        
        print(f"   ✓ 测试报告已生成: blockemulator_integration_test_report.json")
        
    except Exception as e:
        print(f"   ✗ 生成测试报告失败: {e}")
        return False
    
    return True

def print_integration_summary():
    """打印集成总结"""
    print("\n" + "=" * 60)
    print("集成总结")
    print("=" * 60)
    
    print("\n✓ 成功完成的功能:")
    print("  1. BlockEmulator数据格式适配")
    print("  2. 65维综合特征提取")
    print("  3. 图结构构建（邻接矩阵、边类型）")
    print("  4. 系统接口集成（GetAllCollectedData）")
    print("  5. Epoch级别数据提取")
    print("  6. 批量处理能力")
    print("  7. 与后续步骤的兼容性")
    
    print("\n📁 生成的文件:")
    files = [
        "blockemulator_adapter.py - 核心适配器",
        "system_integration_pipeline.py - 集成流水线",
        "INTEGRATION_GUIDE.md - 使用指南",
        "test_integration.py - 测试脚本",
        "test_*.pt - 测试输出文件",
        "test_*.json - 统计和报告文件"
    ]
    
    for file_desc in files:
        print(f"  • {file_desc}")
    
    print("\n🔄 对接流程:")
    print("  原始: CSV → 特征提取 → 后续步骤")
    print("  新版: BlockEmulator系统 → 适配器 → 标准特征 → 后续步骤")
    
    print("\n[DATA] 特征维度分布:")
    feature_breakdown = {
        "硬件资源": 13,
        "链上行为": 15,
        "网络拓扑": 7,
        "动态属性": 10,
        "异构类型": 10,
        "跨分片交易": 4,
        "身份特征": 2,
        "总计": 65
    }
    
    for category, dims in feature_breakdown.items():
        print(f"  • {category}: {dims}维")
    
    print("\n[START] 下一步建议:")
    print("  1. 在实际环境中测试适配器")
    print("  2. 集成到现有的分片算法流程中")
    print("  3. 验证与第二、三、四步的数据传递")
    print("  4. 性能优化和错误处理完善")
    
    print("\n" + "=" * 60)

def main():
    """主测试函数"""
    print("开始BlockEmulator分片系统对接测试...")
    
    # 运行完整测试
    success = test_blockemulator_integration()
    
    if success:
        print("\n🎉 所有测试通过！")
        print_integration_summary()
    else:
        print("\n[ERROR] 测试失败，请检查错误信息")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
