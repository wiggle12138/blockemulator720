"""
测试BlockEmulator适配器的功能
验证实时特征提取能力和输出格式
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from blockemulator_adapter import BlockEmulatorAdapter, create_mock_emulator_data
import torch
import json
import time

def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试BlockEmulator适配器基本功能 ===\n")
    
    # 1. 初始化
    print("1. 初始化适配器...")
    adapter = BlockEmulatorAdapter()
    print("✓ 初始化成功\n")
    
    # 2. 创建测试数据
    print("2. 创建测试数据...")
    test_data = create_mock_emulator_data(num_nodes=5, num_shards=2)
    print(f"✓ 创建了 {len(test_data)} 个节点的测试数据\n")
    
    # 3. 特征提取
    print("3. 执行特征提取...")
    features = adapter.extract_features_realtime(test_data)
    print("✓ 特征提取完成\n")
    
    # 4. 验证输出格式
    print("4. 验证输出格式...")
    expected_keys = ['f_classic', 'f_graph', 'f_reduced', 'node_mapping', 'metadata']
    for key in expected_keys:
        assert key in features, f"缺少输出键: {key}"
    print("✓ 输出格式验证通过\n")
    
    # 5. 验证特征维度
    print("5. 验证特征维度...")
    f_classic = features['f_classic']
    f_graph = features['f_graph']
    f_reduced = features['f_reduced']
    
    assert f_classic.shape == (5, 128), f"f_classic维度错误: {f_classic.shape}"
    assert f_graph.shape == (5, 96), f"f_graph维度错误: {f_graph.shape}"
    assert f_reduced.shape == (5, 64), f"f_reduced维度错误: {f_reduced.shape}"
    print("✓ 特征维度验证通过\n")
    
    # 6. 验证元数据
    print("6. 验证元数据...")
    metadata = features['metadata']
    assert metadata['total_nodes'] == 5, f"节点数量错误: {metadata['total_nodes']}"
    assert metadata['num_shards'] == 2, f"分片数量错误: {metadata['num_shards']}"
    print("✓ 元数据验证通过\n")
    
    return features

def test_performance():
    """测试性能"""
    print("=== 测试处理性能 ===\n")
    
    adapter = BlockEmulatorAdapter()
    
    # 测试不同规模的数据
    test_sizes = [10, 50, 100]
    
    for size in test_sizes:
        print(f"测试 {size} 个节点...")
        
        # 创建测试数据
        test_data = create_mock_emulator_data(num_nodes=size, num_shards=4)
        
        # 计时特征提取
        start_time = time.time()
        features = adapter.extract_features_realtime(test_data)
        processing_time = time.time() - start_time
        
        speed = size / processing_time
        print(f"  处理时间: {processing_time:.3f}秒")
        print(f"  处理速度: {speed:.1f} 节点/秒")
        print(f"  输出形状: F_classic={features['f_classic'].shape}, F_graph={features['f_graph'].shape}")
        print()

def test_file_output():
    """测试文件输出功能"""
    print("=== 测试文件输出功能 ===\n")
    
    adapter = BlockEmulatorAdapter()
    
    # 创建测试数据
    test_data = create_mock_emulator_data(num_nodes=8, num_shards=2)
    
    # 特征提取
    features = adapter.extract_features_realtime(test_data)
    
    # 保存文件
    output_dir = "test_output"
    saved_files = adapter.save_features_for_next_steps(features, output_dir)
    
    print(f"保存的文件:")
    for key, path in saved_files.items():
        file_exists = os.path.exists(path)
        file_size = os.path.getsize(path) if file_exists else 0
        print(f"  {key}: {path} ({'存在' if file_exists else '不存在'}, {file_size} bytes)")
    
    # 验证文件内容
    print(f"\n验证文件内容...")
    
    # 检查torch文件
    if 'f_classic_pt' in saved_files:
        loaded_f_classic = torch.load(saved_files['f_classic_pt'])
        assert torch.equal(loaded_f_classic, features['f_classic']), "f_classic文件内容不匹配"
        print("✓ f_classic.pt 文件验证通过")
    
    # 检查JSON文件
    if 'metadata' in saved_files:
        with open(saved_files['metadata'], 'r', encoding='utf-8') as f:
            loaded_metadata = json.load(f)
        assert loaded_metadata['total_nodes'] == features['metadata']['total_nodes'], "metadata文件内容不匹配"
        print("✓ metadata.json 文件验证通过")
    
    print()

def test_compatibility():
    """测试与后续步骤的兼容性"""
    print("=== 测试兼容性 ===\n")
    
    adapter = BlockEmulatorAdapter()
    
    # 获取输出格式信息
    format_info = adapter.get_step1_output_info()
    
    print("第一步输出格式信息:")
    print(json.dumps(format_info, indent=2, ensure_ascii=False))
    print()
    
    # 验证关键字段
    assert format_info['step'] == 1, "步骤编号错误"
    assert 'output_dimensions' in format_info, "缺少输出维度信息"
    assert 'next_step_compatibility' in format_info, "缺少兼容性信息"
    
    print("✓ 兼容性验证通过\n")

def test_edge_cases():
    """测试边界情况"""
    print("=== 测试边界情况 ===\n")
    
    adapter = BlockEmulatorAdapter()
    
    # 测试空数据
    print("1. 测试空数据...")
    empty_features = adapter.extract_features_realtime([])
    assert empty_features['f_classic'].shape[0] == 0, "空数据处理错误"
    print("✓ 空数据处理正确\n")
    
    # 测试缺失字段的数据
    print("2. 测试缺失字段数据...")
    incomplete_data = [{
        "ShardID": 0,
        "NodeID": 1,
        "NodeState": {
            "Static": {},  # 空的静态特征
            "Dynamic": {}  # 空的动态特征
        }
    }]
    
    features = adapter.extract_features_realtime(incomplete_data)
    assert features['f_classic'].shape[0] == 1, "缺失字段数据处理错误"
    print("✓ 缺失字段数据处理正确\n")
    
    # 测试异常数据类型
    print("3. 测试异常数据类型...")
    malformed_data = [{
        "ShardID": "invalid",  # 错误的数据类型
        "NodeID": None,
        "NodeState": "not_a_dict"
    }]
    
    try:
        features = adapter.extract_features_realtime(malformed_data)
        print("✓ 异常数据类型处理正确\n")
    except Exception as e:
        print(f"异常数据类型处理失败: {e}\n")

def main():
    """主测试函数"""
    print("开始测试BlockEmulator适配器...\n")
    
    try:
        # 运行各项测试
        test_basic_functionality()
        test_performance()
        test_file_output()
        test_compatibility()
        test_edge_cases()
        
        print("=== 所有测试通过！ ===\n")
        
        # 运行完整演示
        print("运行完整演示...")
        from blockemulator_adapter import demo_realtime_feature_extraction
        demo_realtime_feature_extraction()
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
