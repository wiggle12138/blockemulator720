"""
统一的第一步到第二步处理流水线
连接BlockEmulator特征提取和多尺度对比学习
"""

import torch
import time
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os

# 添加路径以导入相关模块
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "partition" / "feature"))
sys.path.append(str(Path(__file__).parent / "muti_scale"))

from realtime_mscia import RealtimeMSCIAProcessor
from All_Final import train_mscia, generate_final_embeddings


class Step1ToStep2Pipeline:
    """第一步到第二步的统一处理流水线"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化流水线
        
        Args:
            config: 配置参数，如果为None则使用默认配置
        """
        self.config = config or self._get_default_config()
        self.step2_processor = None
        self.temporal_data_cache = []  # 缓存时序数据
        
        print(f"第一步到第二步流水线初始化完成")
        print(f"配置: {self.config}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 第二步多尺度对比学习配置
            'time_window': 5,
            'batch_size': 32,
            'input_dim': 128,      # 来自第一步f_classic的维度
            'hidden_dim': 64,
            'time_dim': 16,
            'k_ratio': 0.9,
            'alpha': 0.3,
            'beta': 0.4,
            'gamma': 0.3,
            'lr': 0.02,
            'weight_decay': 9e-6,
            'tau': 0.09,
            'max_timestamp': 10000,
            'epochs': 50,          # 减少训练轮数以适应实时处理
            
            # 数据处理配置
            'use_realtime_mode': True,
            'cache_temporal_data': True,
            'save_intermediate_results': True
        }
    
    def initialize_step2_processor(self):
        """初始化第二步处理器"""
        if self.step2_processor is None:
            self.step2_processor = RealtimeMSCIAProcessor(self.config)
            print("✓ 第二步处理器初始化完成")
    
    def process_step1_to_step2(self, step1_result: Dict[str, torch.Tensor], 
                              timestamp: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        处理第一步输出，执行第二步多尺度对比学习
        
        Args:
            step1_result: 第一步输出
                - 'f_classic': [N, 128] 经典特征
                - 'f_graph': [N, 96] 图特征
                - 'f_reduced': [N, 64] 精简特征
                - 'node_mapping': Dict 节点映射
                - 'metadata': Dict 元数据
            timestamp: 当前时间戳
        
        Returns:
            第二步输出:
                - 'temporal_embeddings': [N, 64] 时序嵌入
                - 'loss': 对比学习损失
                - 'node_mapping': 节点映射
                - 'metadata': 增强的元数据
        """
        print(f"\n=== 第一步到第二步处理流水线 ===")
        
        # 确保第二步处理器已初始化
        self.initialize_step2_processor()
        
        # 如果没有提供时间戳，使用当前时间
        if timestamp is None:
            timestamp = int(time.time())
        
        # 验证第一步输出格式
        required_keys = ['f_classic', 'f_graph', 'node_mapping', 'metadata']
        missing_keys = [key for key in required_keys if key not in step1_result]
        if missing_keys:
            raise ValueError(f"第一步输出缺少必要键: {missing_keys}")
        
        # 检查特征维度
        f_classic = step1_result['f_classic']
        f_graph = step1_result['f_graph']
        
        print(f"第一步输出验证:")
        print(f"  F_classic: {f_classic.shape}")
        print(f"  F_graph: {f_graph.shape}")
        print(f"  节点数量: {len(step1_result['node_mapping'])}")
        print(f"  时间戳: {timestamp}")
        
        # 缓存时序数据
        if self.config.get('cache_temporal_data', True):
            self.temporal_data_cache.append({
                'timestamp': timestamp,
                'features': f_classic.clone(),
                'graph_features': f_graph.clone(),
                'metadata': step1_result['metadata'].copy()
            })
            
            # 保持缓存大小
            max_cache_size = self.config.get('time_window', 5) * 2
            if len(self.temporal_data_cache) > max_cache_size:
                self.temporal_data_cache.pop(0)
        
        # 处理第二步
        step2_result = self.step2_processor.process_step1_output(step1_result, timestamp)
        
        # 增强元数据
        step2_result['metadata'].update({
            'pipeline_timestamp': timestamp,
            'step1_to_step2_processing': True,
            'cached_temporal_steps': len(self.temporal_data_cache),
            'processing_mode': 'realtime' if self.config.get('use_realtime_mode') else 'batch'
        })
        
        print(f"✓ 第一步到第二步处理完成")
        print(f"  时序嵌入: {step2_result['temporal_embeddings'].shape}")
        print(f"  损失值: {step2_result['loss'].item():.4f}")
        
        return step2_result
    
    def batch_process_multiple_steps(self, step1_results: List[Dict[str, torch.Tensor]], 
                                   timestamps: Optional[List[int]] = None) -> List[Dict[str, torch.Tensor]]:
        """
        批量处理多个第一步输出
        
        Args:
            step1_results: 多个第一步输出的列表
            timestamps: 对应的时间戳列表
        
        Returns:
            多个第二步输出的列表
        """
        print(f"\n=== 批量处理 {len(step1_results)} 个时间步 ===")
        
        if timestamps is None:
            # 生成连续时间戳
            base_time = int(time.time())
            timestamps = [base_time + i * 10 for i in range(len(step1_results))]
        
        step2_results = []
        
        for i, (step1_result, timestamp) in enumerate(zip(step1_results, timestamps)):
            print(f"\n--- 处理时间步 {i+1}/{len(step1_results)} (timestamp: {timestamp}) ---")
            
            try:
                step2_result = self.process_step1_to_step2(step1_result, timestamp)
                step2_results.append(step2_result)
                
            except Exception as e:
                print(f"错误: 处理时间步 {i+1} 失败: {e}")
                # 创建空结果以保持列表长度一致
                empty_result = self._create_empty_step2_result(step1_result)
                step2_results.append(empty_result)
        
        print(f"\n✓ 批量处理完成: {len(step2_results)} 个结果")
        return step2_results
    
    def _create_empty_step2_result(self, step1_result: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """创建空的第二步结果"""
        num_nodes = step1_result['f_classic'].shape[0]
        return {
            'temporal_embeddings': torch.zeros(num_nodes, self.config['hidden_dim']),
            'loss': torch.tensor(0.0),
            'node_mapping': step1_result['node_mapping'],
            'metadata': {
                **step1_result['metadata'],
                'processing_failed': True,
                'error_fallback': True
            }
        }
    
    def get_temporal_summary(self) -> Dict[str, Any]:
        """获取时序处理摘要"""
        if not self.temporal_data_cache:
            return {'status': 'no_data'}
        
        timestamps = [data['timestamp'] for data in self.temporal_data_cache]
        
        return {
            'status': 'active',
            'cached_steps': len(self.temporal_data_cache),
            'timestamp_range': {
                'min': min(timestamps),
                'max': max(timestamps),
                'span_seconds': max(timestamps) - min(timestamps)
            },
            'average_nodes_per_step': np.mean([
                data['features'].shape[0] for data in self.temporal_data_cache
            ]),
            'feature_dimensions': {
                'classic': self.temporal_data_cache[0]['features'].shape[1],
                'graph': self.temporal_data_cache[0]['graph_features'].shape[1]
            }
        }
    
    def save_results(self, step2_results: List[Dict[str, torch.Tensor]], 
                    output_dir: str = "step1_to_step2_output"):
        """保存处理结果"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存时序嵌入
        all_embeddings = []
        all_timestamps = []
        
        for result in step2_results:
            if 'temporal_embeddings' in result:
                all_embeddings.append(result['temporal_embeddings'])
                all_timestamps.append(result['metadata'].get('pipeline_timestamp', 0))
        
        if all_embeddings:
            # 保存为张量
            embeddings_tensor = torch.stack(all_embeddings)  # [T, N, D]
            torch.save(embeddings_tensor, output_path / "temporal_embeddings_sequence.pt")
            
            # 保存时间戳信息
            timestamp_info = {
                'timestamps': all_timestamps,
                'sequence_length': len(all_timestamps),
                'embedding_shape': embeddings_tensor.shape
            }
            
            import json
            with open(output_path / "timestamp_info.json", 'w') as f:
                json.dump(timestamp_info, f, indent=2)
            
            print(f"✓ 结果已保存到 {output_path}")
            print(f"  时序嵌入序列: {embeddings_tensor.shape}")
            print(f"  时间跨度: {len(all_timestamps)} 个时间步")
        
        return output_path


def demo_pipeline():
    """演示第一步到第二步流水线"""
    print("=== 第一步到第二步流水线演示 ===")
    
    # 初始化流水线
    pipeline = Step1ToStep2Pipeline()
    
    # 模拟多个时间步的第一步输出
    time_steps = [100, 110, 120, 130, 140]
    step1_outputs = []
    
    for i, timestamp in enumerate(time_steps):
        print(f"\n--- 生成模拟第一步输出 {i+1} (timestamp: {timestamp}) ---")
        
        # 模拟节点数量变化（真实场景中可能有节点加入/离开）
        num_nodes = np.random.randint(80, 120)
        
        mock_step1_output = {
            'f_classic': torch.randn(num_nodes, 128) * 0.5,  # 标准化的特征
            'f_graph': torch.randn(num_nodes, 96) * 0.3,
            'f_reduced': torch.randn(num_nodes, 64) * 0.4,
            'node_mapping': {j: f"node_{timestamp}_{j}" for j in range(num_nodes)},
            'metadata': {
                'num_nodes': num_nodes,
                'extraction_timestamp': timestamp,
                'source': 'blockemulator_demo',
                'step1_processing_time': np.random.uniform(0.1, 0.5)
            }
        }
        
        step1_outputs.append(mock_step1_output)
    
    # 批量处理
    step2_results = pipeline.batch_process_multiple_steps(step1_outputs, time_steps)
    
    # 显示结果摘要
    print(f"\n=== 处理结果摘要 ===")
    temporal_summary = pipeline.get_temporal_summary()
    print(f"时序摘要: {temporal_summary}")
    
    for i, result in enumerate(step2_results):
        print(f"时间步 {i+1}:")
        print(f"  时序嵌入: {result['temporal_embeddings'].shape}")
        print(f"  损失: {result['loss'].item():.4f}")
        print(f"  节点数: {len(result['node_mapping'])}")
    
    # 保存结果
    save_path = pipeline.save_results(step2_results)
    print(f"\n✓ 演示完成，结果保存在: {save_path}")


if __name__ == "__main__":
    demo_pipeline()
