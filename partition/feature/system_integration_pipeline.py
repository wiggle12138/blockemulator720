"""
BlockEmulator系统对接主流水线
集成系统的GetAllCollectedData()接口，实现第一步特征提取的完整流程
"""

import torch
import numpy as np
import json
import sys
import os
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import time

# 导入适配器
try:
    from .blockemulator_adapter import BlockEmulatorAdapter
except ImportError:
    try:
        from blockemulator_adapter import BlockEmulatorAdapter
    except ImportError:
        import sys
        import importlib.util
        from pathlib import Path
        
        # 使用绝对路径导入适配器
        adapter_path = Path(__file__).parent / "blockemulator_adapter.py"
        if adapter_path.exists():
            spec = importlib.util.spec_from_file_location("blockemulator_adapter", adapter_path)
            if spec and spec.loader:
                adapter_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(adapter_module)
                BlockEmulatorAdapter = getattr(adapter_module, 'BlockEmulatorAdapter', None)
            else:
                raise ImportError("无法加载BlockEmulatorAdapter")
        else:
            raise ImportError(f"适配器文件不存在: {adapter_path}")

class BlockEmulatorStep1Pipeline:
    """BlockEmulator系统第一步特征提取流水线"""
    
    def __init__(self, 
                 use_comprehensive_features: bool = True,
                 save_adjacency: bool = True,
                 output_dir: str = "./step1_outputs"):
        """
        初始化流水线
        
        Args:
            use_comprehensive_features: 是否使用全面特征提取
            save_adjacency: 是否保存邻接矩阵信息
            output_dir: 输出目录
        """
        self.use_comprehensive_features = use_comprehensive_features
        self.save_adjacency = save_adjacency
        self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化适配器
        self.adapter = BlockEmulatorAdapter()
        
        # 特征维度配置
        self.feature_dims = {
            'comprehensive': 65,  # 综合特征维度
            'hardware': 13,       # 硬件特征
            'onchain': 15,        # 链上行为特征  
            'topology': 7,        # 网络拓扑特征
            'dynamic': 10,        # 动态属性特征
            'heterogeneous': 10,  # 异构类型特征
            'crossshard': 4,      # 跨分片特征
            'identity': 2         # 身份特征
        }
        
        print(f"[Step1 Pipeline] 初始化完成，输出目录: {output_dir}")
    
    def extract_features_from_system(self, 
                                   node_features_module,
                                   experiment_name: str = "default") -> Dict[str, torch.Tensor]:
        """
        从BlockEmulator系统提取特征
        
        Args:
            node_features_module: NodeFeaturesModule实例
            experiment_name: 实验名称
            
        Returns:
            包含特征数据的字典
        """
        print(f"[Step1 Pipeline] 开始从系统提取特征，实验: {experiment_name}")
        
        # 1. 获取系统收集的原始数据
        try:
            # 检查输入类型
            if isinstance(node_features_module, dict):
                # 新的数据接口格式：直接是转换后的数据
                raw_node_data = node_features_module.get('node_features', [])
                print(f"[Step1 Pipeline] 接收字典格式数据: {len(raw_node_data)} 个节点")
            else:
                # 传统的NodeFeaturesModule接口
                raw_node_data = node_features_module.GetAllCollectedData()
                print(f"[Step1 Pipeline] 成功获取 {len(raw_node_data)} 个节点的原始数据")
        except Exception as e:
            print(f"[Step1 Pipeline] 获取系统数据失败: {e}")
            raise
        
        # 2. 转换为适配器可处理的格式
        system_data = self._convert_system_data_format(raw_node_data)
        
        # 3. 使用适配器处理数据
        output_filename = os.path.join(self.output_dir, f"step1_{experiment_name}_features.pt")
        results = self.adapter.extract_features_realtime(system_data)
        
        # 3.5 确保边索引存在（为兼容性）
        if 'edge_index' not in results:
            results['edge_index'] = self._generate_default_edge_index(len(system_data))
            results['edge_type'] = torch.zeros(results['edge_index'].shape[1], dtype=torch.long)
        
        # 4. 保存详细信息
        if self.save_adjacency:
            self._save_detailed_adjacency_info(results, experiment_name)
        
        # 5. 生成兼容性输出
        compat_results = self._create_compatibility_output(results, experiment_name)
        
        return compat_results
    
    def _convert_system_data_format(self, raw_node_data) -> List[Dict[str, Any]]:
        """
        将系统原始数据转换为适配器期望的格式
        
        Args:
            raw_node_data: 系统返回的ReplyNodeStateMsg列表
            
        Returns:
            转换后的数据列表
        """
        converted_data = []
        
        for node_msg in raw_node_data:
            # 提取数据字段
            converted_item = {
                'ShardID': getattr(node_msg, 'ShardID', 0),
                'NodeID': getattr(node_msg, 'NodeID', 0),
                'Timestamp': getattr(node_msg, 'Timestamp', int(time.time() * 1000)),
                'RequestID': getattr(node_msg, 'RequestID', ''),
                'NodeState': {
                    'Static': self._extract_static_features(node_msg),
                    'Dynamic': self._extract_dynamic_features(node_msg)
                }
            }
            converted_data.append(converted_item)
        
        return converted_data
    
    def _extract_static_features(self, node_msg) -> Dict[str, Any]:
        """提取静态特征"""
        try:
            static_data = node_msg.NodeState.Static
            
            return {
                'ResourceCapacity': {
                    'Hardware': {
                        'CPU': {
                            'CoreCount': getattr(static_data.ResourceCapacity.Hardware.CPU, 'CoreCount', 1),
                            'Architecture': getattr(static_data.ResourceCapacity.Hardware.CPU, 'Architecture', 'amd64')
                        },
                        'Memory': {
                            'TotalCapacity': getattr(static_data.ResourceCapacity.Hardware.Memory, 'TotalCapacity', 2),
                            'Type': getattr(static_data.ResourceCapacity.Hardware.Memory, 'Type', 'DDR4'),
                            'Bandwidth': getattr(static_data.ResourceCapacity.Hardware.Memory, 'Bandwidth', 50.0)
                        },
                        'Storage': {
                            'Capacity': getattr(static_data.ResourceCapacity.Hardware.Storage, 'Capacity', 1),
                            'Type': getattr(static_data.ResourceCapacity.Hardware.Storage, 'Type', 'SSD'),
                            'ReadWriteSpeed': getattr(static_data.ResourceCapacity.Hardware.Storage, 'ReadWriteSpeed', 500.0)
                        },
                        'Network': {
                            'UpstreamBW': getattr(static_data.ResourceCapacity.Hardware.Network, 'UpstreamBW', 100.0),
                            'DownstreamBW': getattr(static_data.ResourceCapacity.Hardware.Network, 'DownstreamBW', 1000.0),
                            'Latency': getattr(static_data.ResourceCapacity.Hardware.Network, 'Latency', '50ms')
                        }
                    }
                },
                'NetworkTopology': {
                    'GeoLocation': {
                        'Timezone': getattr(static_data.NetworkTopology.GeoLocation, 'Timezone', 'UTC+8')
                    },
                    'Connections': {
                        'IntraShardConn': getattr(static_data.NetworkTopology.Connections, 'IntraShardConn', 3),
                        'InterShardConn': getattr(static_data.NetworkTopology.Connections, 'InterShardConn', 2),
                        'WeightedDegree': getattr(static_data.NetworkTopology.Connections, 'WeightedDegree', 5.0),
                        'ActiveConn': getattr(static_data.NetworkTopology.Connections, 'ActiveConn', 4)
                    },
                    'ShardAllocation': {
                        'Adaptability': getattr(static_data.NetworkTopology.ShardAllocation, 'Adaptability', 0.7)
                    }
                },
                'HeterogeneousType': {
                    'NodeType': getattr(static_data.HeterogeneousType, 'NodeType', 'full_node'),
                    'FunctionTags': getattr(static_data.HeterogeneousType, 'FunctionTags', 'consensus,validation'),
                    'SupportedFuncs': {
                        'Functions': getattr(static_data.HeterogeneousType.SupportedFuncs, 'Functions', 'tx_processing')
                    },
                    'Application': {
                        'CurrentState': getattr(static_data.HeterogeneousType.Application, 'CurrentState', 'active'),
                        'LoadMetrics': {
                            'TxFrequency': getattr(static_data.HeterogeneousType.Application.LoadMetrics, 'TxFrequency', 100),
                            'StorageOps': getattr(static_data.HeterogeneousType.Application.LoadMetrics, 'StorageOps', 50)
                        }
                    }
                }
            }
        except Exception as e:
            print(f"[Step1 Pipeline] 提取静态特征失败: {e}")
            return self._get_default_static_features()
    
    def _extract_dynamic_features(self, node_msg) -> Dict[str, Any]:
        """提取动态特征"""
        try:
            dynamic_data = node_msg.NodeState.Dynamic
            
            return {
                'OnChainBehavior': {
                    'TransactionCapability': {
                        'AvgTPS': getattr(dynamic_data.OnChainBehavior.TransactionCapability, 'AvgTPS', 0.0),
                        'CrossShardTx': {
                            'InterNodeVolume': getattr(dynamic_data.OnChainBehavior.TransactionCapability.CrossShardTx, 'InterNodeVolume', ''),
                            'InterShardVolume': getattr(dynamic_data.OnChainBehavior.TransactionCapability.CrossShardTx, 'InterShardVolume', '')
                        },
                        'ConfirmationDelay': getattr(dynamic_data.OnChainBehavior.TransactionCapability, 'ConfirmationDelay', '100ms'),
                        'ResourcePerTx': {
                            'CPUPerTx': getattr(dynamic_data.OnChainBehavior.TransactionCapability.ResourcePerTx, 'CPUPerTx', 0.1),
                            'MemPerTx': getattr(dynamic_data.OnChainBehavior.TransactionCapability.ResourcePerTx, 'MemPerTx', 0.05),
                            'DiskPerTx': getattr(dynamic_data.OnChainBehavior.TransactionCapability.ResourcePerTx, 'DiskPerTx', 0.02),
                            'NetworkPerTx': getattr(dynamic_data.OnChainBehavior.TransactionCapability.ResourcePerTx, 'NetworkPerTx', 0.01)
                        }
                    },
                    'BlockGeneration': {
                        'AvgInterval': getattr(dynamic_data.OnChainBehavior.BlockGeneration, 'AvgInterval', '5.0s'),
                        'IntervalStdDev': getattr(dynamic_data.OnChainBehavior.BlockGeneration, 'IntervalStdDev', '1.0s')
                    },
                    'TransactionTypes': {
                        'NormalTxRatio': getattr(dynamic_data.OnChainBehavior.TransactionTypes, 'NormalTxRatio', 0.8),
                        'ContractTxRatio': getattr(dynamic_data.OnChainBehavior.TransactionTypes, 'ContractTxRatio', 0.2)
                    },
                    'Consensus': {
                        'ParticipationRate': getattr(dynamic_data.OnChainBehavior.Consensus, 'ParticipationRate', 0.9),
                        'TotalReward': getattr(dynamic_data.OnChainBehavior.Consensus, 'TotalReward', 100.0),
                        'SuccessRate': getattr(dynamic_data.OnChainBehavior.Consensus, 'SuccessRate', 0.95)
                    },
                    'SmartContractUsage': {
                        'InvocationFrequency': getattr(dynamic_data.OnChainBehavior.SmartContractUsage, 'InvocationFrequency', 0)
                    },
                    'EconomicContribution': {
                        'FeeContributionRatio': getattr(dynamic_data.OnChainBehavior.EconomicContribution, 'FeeContributionRatio', 0.01)
                    }
                },
                'DynamicAttributes': {
                    'Compute': {
                        'CPUUsage': getattr(dynamic_data.DynamicAttributes.Compute, 'CPUUsage', 30.0),
                        'MemUsage': getattr(dynamic_data.DynamicAttributes.Compute, 'MemUsage', 40.0),
                        'ResourceFlux': getattr(dynamic_data.DynamicAttributes.Compute, 'ResourceFlux', 0.1)
                    },
                    'Storage': {
                        'Available': getattr(dynamic_data.DynamicAttributes.Storage, 'Available', 80.0),
                        'Utilization': getattr(dynamic_data.DynamicAttributes.Storage, 'Utilization', 20.0)
                    },
                    'Network': {
                        'LatencyFlux': getattr(dynamic_data.DynamicAttributes.Network, 'LatencyFlux', 0.05),
                        'AvgLatency': getattr(dynamic_data.DynamicAttributes.Network, 'AvgLatency', '50ms'),
                        'BandwidthUsage': getattr(dynamic_data.DynamicAttributes.Network, 'BandwidthUsage', 0.3)
                    },
                    'Transactions': {
                        'Frequency': getattr(dynamic_data.DynamicAttributes.Transactions, 'Frequency', 10),
                        'ProcessingDelay': getattr(dynamic_data.DynamicAttributes.Transactions, 'ProcessingDelay', '200ms')
                    }
                }
            }
        except Exception as e:
            print(f"[Step1 Pipeline] 提取动态特征失败: {e}")
            return self._get_default_dynamic_features()
    
    def _get_default_static_features(self) -> Dict[str, Any]:
        """获取默认静态特征"""
        return {
            'ResourceCapacity': {
                'Hardware': {
                    'CPU': {'CoreCount': 2, 'Architecture': 'amd64'},
                    'Memory': {'TotalCapacity': 4, 'Type': 'DDR4', 'Bandwidth': 50.0},
                    'Storage': {'Capacity': 2, 'Type': 'SSD', 'ReadWriteSpeed': 1000.0},
                    'Network': {'UpstreamBW': 100.0, 'DownstreamBW': 1000.0, 'Latency': '50ms'}
                }
            },
            'NetworkTopology': {
                'GeoLocation': {'Timezone': 'UTC+8'},
                'Connections': {
                    'IntraShardConn': 3, 'InterShardConn': 2, 
                    'WeightedDegree': 5.0, 'ActiveConn': 4
                },
                'ShardAllocation': {'Adaptability': 0.7}
            },
            'HeterogeneousType': {
                'NodeType': 'full_node',
                'FunctionTags': 'consensus,validation',
                'SupportedFuncs': {'Functions': 'tx_processing'},
                'Application': {
                    'CurrentState': 'active',
                    'LoadMetrics': {'TxFrequency': 100, 'StorageOps': 50}
                }
            }
        }
    
    def _get_default_dynamic_features(self) -> Dict[str, Any]:
        """获取默认动态特征"""
        return {
            'OnChainBehavior': {
                'TransactionCapability': {
                    'AvgTPS': 50.0,
                    'CrossShardTx': {'InterNodeVolume': '', 'InterShardVolume': ''},
                    'ConfirmationDelay': '100ms',
                    'ResourcePerTx': {
                        'CPUPerTx': 0.1, 'MemPerTx': 0.05, 
                        'DiskPerTx': 0.02, 'NetworkPerTx': 0.01
                    }
                },
                'BlockGeneration': {
                    'AvgInterval': '5.0s', 'IntervalStdDev': '1.0s'
                },
                'TransactionTypes': {
                    'NormalTxRatio': 0.8, 'ContractTxRatio': 0.2
                },
                'Consensus': {
                    'ParticipationRate': 0.9, 'TotalReward': 100.0, 'SuccessRate': 0.95
                },
                'SmartContractUsage': {'InvocationFrequency': 0},
                'EconomicContribution': {'FeeContributionRatio': 0.01}
            },
            'DynamicAttributes': {
                'Compute': {'CPUUsage': 30.0, 'MemUsage': 40.0, 'ResourceFlux': 0.1},
                'Storage': {'Available': 80.0, 'Utilization': 20.0},
                'Network': {
                    'LatencyFlux': 0.05, 'AvgLatency': '50ms', 'BandwidthUsage': 0.3
                },
                'Transactions': {'Frequency': 10, 'ProcessingDelay': '200ms'}
            }
        }
    
    def _save_detailed_adjacency_info(self, results: Dict[str, torch.Tensor], experiment_name: str):
        """保存详细的邻接矩阵信息"""
        adjacency_info = {
            'generation_time': datetime.now().isoformat(),
            'experiment_name': experiment_name,
            'graph_metadata': {
                'num_nodes': results['metadata']['total_nodes'],
                'num_edges': results['metadata'].get('num_edges', 0),
                'feature_dim': results['metadata'].get('feature_dim', 128)
            },
            'edge_statistics': {
                'total_edges': int(results['edge_index'].shape[1]),
                'edge_types': {
                    'intra_shard': int((results['edge_type'] == 0).sum()),
                    'inter_shard_adjacent': int((results['edge_type'] == 1).sum()),
                    'inter_shard_distant': int((results['edge_type'] == 2).sum())
                }
            },
            'node_distribution': {
                'shard_counts': {}
            }
        }
        
        # 统计分片分布 - 使用安全的方式
        if 'node_info' in results and 'shard_ids' in results['node_info']:
            shard_ids = results['node_info']['shard_ids'].numpy()
            for shard_id in np.unique(shard_ids):
                adjacency_info['node_distribution']['shard_counts'][str(shard_id)] = int(np.sum(shard_ids == shard_id))
        else:
            # 没有分片信息时，假设所有节点在分片0
            num_nodes = results['metadata'].get('total_nodes', results['metadata'].get('num_nodes', 0))
            adjacency_info['node_distribution']['shard_counts']['0'] = num_nodes
        
        # 保存邻接信息
        adjacency_filename = os.path.join(self.output_dir, f"step1_{experiment_name}_adjacency_info.json")
        with open(adjacency_filename, 'w', encoding='utf-8') as f:
            json.dump(adjacency_info, f, indent=2, ensure_ascii=False)
        
        print(f"[Step1 Pipeline] 邻接信息已保存到: {adjacency_filename}")
    
    def _create_compatibility_output(self, results: Dict[str, torch.Tensor], experiment_name: str) -> Dict[str, torch.Tensor]:
        """创建与后续步骤兼容的输出格式"""
        # 确保输出格式与后续步骤兼容
        # 使用f_classic作为主要特征，因为它是最完整的128维特征
        main_features = results.get('f_classic', results.get('f_comprehensive', results.get('features')))
        
        compat_results = {
            # 第二步需要的格式
            'features': main_features,  # [N, 128] 主要特征
            'edge_index': results['edge_index'],     # [2, E] 边索引
            'edge_type': results['edge_type'],       # [E] 边类型
            
            # 第三步需要的格式
            'node_features': main_features,  # 节点特征
            'adjacency_matrix': self._edge_index_to_adjacency(
                results['edge_index'], results['metadata']['total_nodes']
            ),  # 邻接矩阵
            
            # 兼容性 - 添加所有可用特征
            'f_classic': results.get('f_classic', main_features),
            'f_graph': results.get('f_graph', main_features[:, :96] if main_features.shape[1] >= 96 else main_features),
            'f_reduced': results.get('f_reduced', main_features[:, :64] if main_features.shape[1] >= 64 else main_features),
            'f_comprehensive': main_features,  # 添加期望的comprehensive特征
            
            # 元数据
            'metadata': results['metadata'],
            'node_info': results.get('node_mapping', {}),
            
            # 兼容性字段
            'num_nodes': results['metadata']['total_nodes'],
            'num_features': results['metadata'].get('feature_dim', 128),
            'timestamp': results['metadata'].get('timestamp', results['metadata']['processing_timestamp']),
            'data_source': 'blockemulator_system'
        }
        
        # 保存兼容性输出
        compat_filename = os.path.join(self.output_dir, f"step1_{experiment_name}_compatible.pt")
        torch.save(compat_results, compat_filename)
        print(f"[Step1 Pipeline] 兼容性输出已保存到: {compat_filename}")
        
        return compat_results
    
    def _generate_default_edge_index(self, num_nodes: int) -> torch.Tensor:
        """生成默认的边索引（环形连接）"""
        if num_nodes <= 1:
            return torch.empty((2, 0), dtype=torch.long)
        
        # 创建环形连接：每个节点连接到下一个节点
        edges = []
        for i in range(num_nodes):
            next_node = (i + 1) % num_nodes
            edges.append([i, next_node])
            edges.append([next_node, i])  # 双向连接
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        return edge_index
    
    def _edge_index_to_adjacency(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """将边索引转换为邻接矩阵"""
        adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        if edge_index.shape[1] > 0:
            adjacency[edge_index[0], edge_index[1]] = 1.0
        return adjacency
    
    def extract_features_from_epoch_data(self, 
                                       node_features_module,
                                       epoch: int,
                                       experiment_name: str = "epoch") -> Optional[Dict[str, torch.Tensor]]:
        """
        从特定epoch提取特征
        
        Args:
            node_features_module: NodeFeaturesModule实例
            epoch: epoch编号
            experiment_name: 实验名称前缀
            
        Returns:
            特征数据字典，如果epoch无数据则返回None
        """
        print(f"[Step1 Pipeline] 提取 epoch {epoch} 的特征数据...")
        
        try:
            # 获取指定epoch的数据
            epoch_data, exists = node_features_module.GetEpochData(epoch)
            
            if not exists or not epoch_data:
                print(f"[Step1 Pipeline] Epoch {epoch} 无可用数据")
                return None
            
            print(f"[Step1 Pipeline] Epoch {epoch} 包含 {len(epoch_data)} 个节点的数据")
            
            # 转换数据格式
            system_data = self._convert_system_data_format(epoch_data)
            
            # 处理数据
            epoch_experiment_name = f"{experiment_name}_epoch_{epoch}"
            output_filename = os.path.join(self.output_dir, f"step1_{epoch_experiment_name}_features.pt")
            results = self.adapter.create_step1_output(
                raw_data=system_data,
                output_filename=output_filename
            )
            
            # 保存详细信息
            if self.save_adjacency:
                self._save_detailed_adjacency_info(results, epoch_experiment_name)
            
            # 生成兼容性输出
            compat_results = self._create_compatibility_output(results, epoch_experiment_name)
            
            return compat_results
            
        except Exception as e:
            print(f"[Step1 Pipeline] 提取 epoch {epoch} 数据失败: {e}")
            return None
    
    def batch_extract_epoch_features(self, 
                                    node_features_module,
                                    epochs: List[int],
                                    experiment_name: str = "batch") -> Dict[int, Dict[str, torch.Tensor]]:
        """
        批量提取多个epoch的特征
        
        Args:
            node_features_module: NodeFeaturesModule实例
            epochs: epoch列表
            experiment_name: 实验名称前缀
            
        Returns:
            epoch到特征数据的映射
        """
        print(f"[Step1 Pipeline] 批量提取 {len(epochs)} 个epoch的特征...")
        
        epoch_results = {}
        successful_epochs = []
        failed_epochs = []
        
        for epoch in epochs:
            result = self.extract_features_from_epoch_data(
                node_features_module, epoch, experiment_name
            )
            
            if result is not None:
                epoch_results[epoch] = result
                successful_epochs.append(epoch)
            else:
                failed_epochs.append(epoch)
        
        print(f"[Step1 Pipeline] 批量提取完成:")
        print(f"  - 成功: {len(successful_epochs)} 个epoch: {successful_epochs}")
        print(f"  - 失败: {len(failed_epochs)} 个epoch: {failed_epochs}")
        
        # 保存批量处理摘要
        batch_summary = {
            'generation_time': datetime.now().isoformat(),
            'experiment_name': experiment_name,
            'total_epochs': len(epochs),
            'successful_epochs': successful_epochs,
            'failed_epochs': failed_epochs,
            'success_rate': len(successful_epochs) / len(epochs) if epochs else 0,
            'epoch_statistics': {}
        }
        
        # 添加每个成功epoch的统计信息
        for epoch, result in epoch_results.items():
            batch_summary['epoch_statistics'][str(epoch)] = {
                'num_nodes': result['metadata']['num_nodes'],
                'num_edges': result['metadata']['num_edges'],
                'feature_dim': result['metadata']['feature_dim']
            }
        
        # 保存批量摘要
        summary_filename = os.path.join(self.output_dir, f"step1_{experiment_name}_batch_summary.json")
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)
        
        print(f"[Step1 Pipeline] 批量处理摘要已保存到: {summary_filename}")
        
        return epoch_results
    
    def get_step1_output_for_step2(self, experiment_name: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        获取第一步的输出用于第二步处理
        
        Args:
            experiment_name: 实验名称
            
        Returns:
            特征数据字典
        """
        compat_filename = os.path.join(self.output_dir, f"step1_{experiment_name}_compatible.pt")
        
        if os.path.exists(compat_filename):
            try:
                results = torch.load(compat_filename)
                print(f"[Step1 Pipeline] 已加载第一步输出: {compat_filename}")
                return results
            except Exception as e:
                print(f"[Step1 Pipeline] 加载第一步输出失败: {e}")
                return None
        else:
            print(f"[Step1 Pipeline] 第一步输出文件不存在: {compat_filename}")
            return None


def create_mock_node_features_module():
    """创建模拟的NodeFeaturesModule用于测试"""
    from blockemulator_adapter import create_mock_blockemulator_data
    
    class MockNodeFeaturesModule:
        def __init__(self):
            # 创建模拟数据
            self.all_data = create_mock_blockemulator_data(num_nodes=30, num_shards=5)
            self.epoch_data = {
                1: self.all_data[:10],
                2: self.all_data[10:20], 
                3: self.all_data[20:30]
            }
        
        def GetAllCollectedData(self):
            """模拟获取所有数据"""
            # 将字典转换为具有属性的对象
            class MockMsg:
                def __init__(self, data):
                    self.ShardID = data['ShardID']
                    self.NodeID = data['NodeID']
                    self.Timestamp = data['Timestamp']
                    self.RequestID = data['RequestID']
                    self.NodeState = self._create_node_state(data['NodeState'])
                
                def _create_node_state(self, state_data):
                    class NodeState:
                        def __init__(self, static_data, dynamic_data):
                            self.Static = self._create_static(static_data)
                            self.Dynamic = self._create_dynamic(dynamic_data)
                        
                        def _create_static(self, data):
                            return self._create_nested_object(data)
                        
                        def _create_dynamic(self, data):
                            return self._create_nested_object(data)
                        
                        def _create_nested_object(self, data):
                            class NestedObject:
                                pass
                            
                            obj = NestedObject()
                            for key, value in data.items():
                                if isinstance(value, dict):
                                    setattr(obj, key, self._create_nested_object(value))
                                else:
                                    setattr(obj, key, value)
                            return obj
                    
                    return NodeState(state_data['Static'], state_data['Dynamic'])
            
            return [MockMsg(data) for data in self.all_data]
        
        def GetEpochData(self, epoch):
            """模拟获取epoch数据"""
            if epoch in self.epoch_data:
                return [self._create_mock_msg(data) for data in self.epoch_data[epoch]], True
            return [], False
        
        def _create_mock_msg(self, data):
            """创建模拟消息对象（简化版本）"""
            class MockMsg:
                def __init__(self, data):
                    for key, value in data.items():
                        setattr(self, key, value)
            return MockMsg(data)
    
    return MockNodeFeaturesModule()


def main():
    """测试BlockEmulator第一步流水线"""
    print("=== BlockEmulator 第一步流水线测试 ===")
    
    # 创建流水线
    pipeline = BlockEmulatorStep1Pipeline(
        use_comprehensive_features=True,
        save_adjacency=True,
        output_dir="./test_step1_outputs"
    )
    
    # 创建模拟的系统模块
    mock_system = create_mock_node_features_module()
    
    # 测试1: 提取所有特征
    print("\n1. 测试提取所有特征...")
    all_results = pipeline.extract_features_from_system(
        node_features_module=mock_system,
        experiment_name="test_all"
    )
    print(f"   结果: {all_results['features'].shape} 特征张量")
    
    # 测试2: 提取单个epoch特征
    print("\n2. 测试提取单个epoch特征...")
    epoch_result = pipeline.extract_features_from_epoch_data(
        node_features_module=mock_system,
        epoch=1,
        experiment_name="test_single"
    )
    if epoch_result:
        print(f"   结果: {epoch_result['features'].shape} 特征张量")
    
    # 测试3: 批量提取epoch特征
    print("\n3. 测试批量提取epoch特征...")
    batch_results = pipeline.batch_extract_epoch_features(
        node_features_module=mock_system,
        epochs=[1, 2, 3, 4],  # epoch 4 应该失败
        experiment_name="test_batch"
    )
    print(f"   结果: 成功提取 {len(batch_results)} 个epoch的特征")
    
    # 测试4: 获取第一步输出用于第二步
    print("\n4. 测试获取第一步输出...")
    step2_input = pipeline.get_step1_output_for_step2("test_all")
    if step2_input:
        print(f"   结果: 成功加载 {step2_input['features'].shape} 特征数据")
    
    print("\n=== 测试完成 ===")
    print("生成的文件在 ./test_step1_outputs/ 目录中")


if __name__ == "__main__":
    main()
