"""
第四步：统一反馈引擎 - 整合性能评估、重要性分析、特征进化
消除冗余，提升智能性，专门为第三步分片优化服务
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import normalized_mutual_info_score
from typing import Dict, List, Tuple, Any, Optional
import json
import pickle
from pathlib import Path
from collections import defaultdict, deque

class UnifiedFeedbackEngine:
    """统一反馈引擎 - 第四步性能反馈核心"""
    
    def __init__(self, feature_dims: Dict[str, int] = None, config: Dict[str, Any] = None, device: str = None):
        self.feature_dims = feature_dims or self._get_default_feature_dims()
        self.config = config or self._get_default_config()
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化核心组件
        self.performance_evaluator = SmartPerformanceEvaluator(self.feature_dims)
        self.importance_analyzer = AdaptiveImportanceAnalyzer(self.feature_dims)
        self.feedback_optimizer = FeedbackOptimizer(self.config)
        
        # 将所有神经网络模型迁移到指定设备
        self.performance_evaluator = self.performance_evaluator.to(self.device)
        if hasattr(self.importance_analyzer, 'to'):
            self.importance_analyzer = self.importance_analyzer.to(self.device)
        if hasattr(self.feedback_optimizer, 'to'):
            self.feedback_optimizer = self.feedback_optimizer.to(self.device)
        
        # 历史状态管理
        self.performance_history = deque(maxlen=self.config['max_history'])
        self.feedback_trend = deque(maxlen=20)
        self.anomaly_detector = AnomalyDetector()
        
        print(f"   统一反馈引擎已初始化 (设备: {self.device})")
        print(f"   支持特征类别: {list(self.feature_dims.keys())}")
        print(f"   配置参数: 历史窗口={self.config['max_history']}, 学习率={self.config['learning_rate']}")

    def _get_default_feature_dims(self) -> Dict[str, int]:
        """获取默认的6类特征维度"""
        return {
            'hardware': 17,           # 硬件特征
            'onchain_behavior': 17,   # 链上行为特征  
            'network_topology': 20,   # 网络拓扑特征
            'dynamic_attributes': 13, # 动态属性特征
            'heterogeneous_type': 17, # 异构类型特征
            'categorical': 15         # 分类特征
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'max_history': 50,
            'learning_rate': 0.01,
            'feedback_weights': {
                'balance': 0.35,      # 负载均衡权重
                'cross_shard': 0.25,  # 跨片率权重  
                'security': 0.20,     # 安全性权重
                'consensus': 0.20     # 共识延迟权重
            },
            'adaptive_threshold': 0.15,
            'anomaly_threshold': 2.0,
            'evolution_enabled': True
        }

    def process_sharding_feedback(self, 
                                  features: Dict[str, torch.Tensor],
                                  shard_assignments: torch.Tensor,
                                  edge_index: torch.Tensor = None,
                                  performance_hints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理分片反馈的主入口
        
        Args:
            features: 6类特征字典
            shard_assignments: 分片分配结果 [num_nodes]
            edge_index: 边索引 [2, num_edges]
            performance_hints: 性能提示（来自第三步）
            
        Returns:
            统一反馈结果，包含给第三步的优化建议
        """
        print(f"\n 开始处理第四步分片反馈...")
        
        try:
            # 确保所有输入张量在正确设备上
            features = {k: v.to(self.device) for k, v in features.items()}
            shard_assignments = shard_assignments.to(self.device)
            if edge_index is not None:
                edge_index = edge_index.to(self.device)
            
            # 从输入数据中提取节点数和分片数
            num_nodes = shard_assignments.shape[0]
            n_shards = int(shard_assignments.max().item()) + 1
            
            # 保存当前的节点数和分片数，供后续使用
            self._last_num_nodes = num_nodes
            self._last_n_shards = n_shards
            
            print(f" 当前分片配置: {num_nodes} 个节点, {n_shards} 个分片")
            
            # 1. 性能评估
            performance_metrics = self.performance_evaluator.evaluate_comprehensive(
                features, shard_assignments, edge_index, performance_hints
            )
            
            # 2. 重要性分析
            importance_analysis = self.importance_analyzer.analyze_feature_importance(
                features, shard_assignments, performance_metrics
            )
            
            # 3. 反馈优化
            optimized_feedback = self.feedback_optimizer.optimize_feedback(
                performance_metrics, importance_analysis, self.performance_history
            )
            
            # 4. 异常检测
            anomaly_report = self.anomaly_detector.detect_anomalies(
                performance_metrics, self.feedback_trend
            )
            
            # 5. 智能建议生成
            smart_suggestions = self._generate_smart_suggestions(
                performance_metrics, importance_analysis, anomaly_report
            )
            
            # 6. 更新历史记录
            self._update_history(performance_metrics, optimized_feedback)
            
            # 7. 构建给第三步的反馈包 (传递正确的节点数和分片数)
            step3_feedback = self._build_step3_feedback_package(
                optimized_feedback, smart_suggestions, importance_analysis,
                num_nodes=num_nodes, n_shards=n_shards
            )
            
            result = {
                'performance_metrics': performance_metrics,
                'importance_analysis': importance_analysis,
                'optimized_feedback': optimized_feedback,
                'anomaly_report': anomaly_report,
                'smart_suggestions': smart_suggestions,
                'step3_feedback_package': step3_feedback,
                'engine_status': self._get_engine_status()
            }
            
            print(f"   第四步反馈处理完成")
            print(f"   性能评分: {optimized_feedback['overall_score']:.3f}")
            print(f"   异常检测: {len(anomaly_report['detected_anomalies'])} 个异常")
            print(f"   智能建议: {len(smart_suggestions)} 条建议")
            
            return result
            
        except Exception as e:
            print(f"[ERROR] 第四步反馈处理失败: {e}")
            import traceback
            traceback.print_exc()
            return self._get_error_fallback()

    def _generate_smart_suggestions(self, 
                                   performance_metrics: Dict[str, Any],
                                   importance_analysis: Dict[str, Any],
                                   anomaly_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成智能优化建议"""
        suggestions = []
        
        # 基于性能指标的建议
        if performance_metrics['load_balance'] < 0.3:
            suggestions.append({
                'type': 'load_balance',
                'priority': 'high',
                'description': '负载均衡严重不足，建议增加硬件特征权重',
                'action': 'increase_hardware_weight',
                'target_improvement': 0.5,
                'affected_features': ['hardware', 'dynamic_attributes']
            })
        
        if performance_metrics['cross_shard_rate'] > 0.8:
            suggestions.append({
                'type': 'cross_shard',
                'priority': 'medium',
                'description': '跨片交易率过高，建议优化分类特征',
                'action': 'optimize_categorical_features',
                'target_improvement': 0.3,
                'affected_features': ['categorical', 'network_topology']
            })
        
        # 基于重要性分析的建议
        low_importance_features = [
            feat for feat, score in importance_analysis['feature_importance'].items() 
            if score < 0.1
        ]
        if low_importance_features:
            suggestions.append({
                'type': 'feature_importance',
                'priority': 'low',
                'description': f'特征重要性偏低: {low_importance_features}',
                'action': 'enhance_feature_representation',
                'target_improvement': 0.2,
                'affected_features': low_importance_features
            })
        
        # 基于异常检测的建议
        for anomaly in anomaly_report['detected_anomalies']:
            suggestions.append({
                'type': 'anomaly_fix',
                'priority': 'high',
                'description': f'检测到异常: {anomaly["description"]}',
                'action': 'investigate_and_fix',
                'target_improvement': 0.4,
                'affected_features': anomaly.get('affected_features', [])
            })
        
        return suggestions

    def _build_step3_feedback_package(self, 
                                     optimized_feedback: Dict[str, Any],
                                     smart_suggestions: List[Dict[str, Any]],
                                     importance_analysis: Dict[str, Any],
                                     num_nodes: int = None,
                                     n_shards: int = None) -> Dict[str, Any]:
        """构建专门给第三步使用的反馈包"""
        
        # 性能向量（用于分片优化）
        performance_vector = torch.tensor([
            optimized_feedback['metrics']['load_balance'],
            optimized_feedback['metrics']['cross_shard_rate'],
            optimized_feedback['metrics']['security_score'],
            optimized_feedback['metrics']['consensus_latency']
        ], dtype=torch.float32, device=self.device)
        
        # 特征权重调整建议
        feature_weight_adjustments = {}
        for feat, importance in importance_analysis['feature_importance'].items():
            if importance < 0.2:
                feature_weight_adjustments[feat] = max(0.1, importance * 1.5)  # 提升低重要性特征
            elif importance > 0.8:
                feature_weight_adjustments[feat] = min(1.0, importance * 0.9)  # 稍微降低过高权重
            else:
                feature_weight_adjustments[feat] = importance
        
        # 分片策略建议
        sharding_strategy_hints = {
            'min_shard_size': max(10, int(100 * optimized_feedback['metrics']['load_balance'])),
            'max_shard_count': min(20, int(15 / max(0.1, optimized_feedback['metrics']['cross_shard_rate']))),
            'balance_weight': optimized_feedback['metrics']['load_balance'],
            'topology_weight': 1.0 - optimized_feedback['metrics']['cross_shard_rate']
        }
        
        # 动态确定节点数和分片数（如果未提供）
        if num_nodes is None:
            num_nodes = getattr(self, '_last_num_nodes', 200)  # 默认200个节点
        if n_shards is None:
            n_shards = getattr(self, '_last_n_shards', 8)      # 默认8个分片
            
        # 生成分配指导矩阵 (num_nodes, 4)
        # 每个节点都有一个4维的性能指导向量
        assignment_guidance = performance_vector.unsqueeze(0).repeat(num_nodes, 1).to(self.device)
        
        # 根据节点索引和性能指标进行微调
        for i in range(num_nodes):
            # 添加一些基于节点位置的变化，让不同节点有略微不同的指导
            node_factor = (i % n_shards) / n_shards  # 0到1之间的因子
            assignment_guidance[i] = assignment_guidance[i] * (0.8 + 0.4 * node_factor)
        
        print(f" 生成反馈指导矩阵: {assignment_guidance.shape} (节点数={num_nodes}, 分片数={n_shards})")
        
        return {
            'performance_vector': performance_vector.tolist(),
            'overall_score': optimized_feedback['overall_score'],
            'feature_weight_adjustments': feature_weight_adjustments,
            'sharding_strategy_hints': sharding_strategy_hints,
            'priority_suggestions': [s for s in smart_suggestions if s['priority'] == 'high'],
            'feedback_confidence': optimized_feedback.get('confidence', 0.8),
            'timestamp': torch.tensor([len(self.performance_history)], dtype=torch.float32, device=self.device),
            'trend_indicator': self._calculate_trend_indicator(),
            'assignment_guidance': assignment_guidance  # 动态生成的分配指导矩阵
        }

    def _calculate_trend_indicator(self) -> float:
        """计算性能趋势指示器"""
        if len(self.feedback_trend) < 3:
            return 0.0
        
        recent_scores = list(self.feedback_trend)[-5:]
        if len(recent_scores) < 2:
            return 0.0
        
        # 计算趋势斜率
        x = np.arange(len(recent_scores))
        y = np.array(recent_scores)
        slope = np.polyfit(x, y, 1)[0]
        
        return float(np.clip(slope, -1.0, 1.0))

    def _update_history(self, performance_metrics: Dict[str, Any], optimized_feedback: Dict[str, Any]):
        """更新历史记录"""
        record = {
            'timestamp': len(self.performance_history),
            'performance': performance_metrics,
            'feedback': optimized_feedback,
            'overall_score': optimized_feedback['overall_score']
        }
        
        self.performance_history.append(record)
        self.feedback_trend.append(optimized_feedback['overall_score'])

    def _get_engine_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        return {
            'history_length': len(self.performance_history),
            'trend_length': len(self.feedback_trend),
            'last_score': self.feedback_trend[-1] if self.feedback_trend else 0.0,
            'avg_score': np.mean(list(self.feedback_trend)) if self.feedback_trend else 0.0,
            'config_active': True
        }

    def _get_error_fallback(self) -> Dict[str, Any]:
        """错误时的回退结果"""
        return {
            'performance_metrics': {'load_balance': 0.5, 'cross_shard_rate': 0.5, 
                                  'security_score': 0.5, 'consensus_latency': 0.5},
            'step3_feedback_package': {
                'performance_vector': [0.5, 0.5, 0.5, 0.5],
                'overall_score': 0.5,
                'feature_weight_adjustments': {feat: 1.0 for feat in self.feature_dims.keys()},
                'sharding_strategy_hints': {'min_shard_size': 20, 'max_shard_count': 10},
                'feedback_confidence': 0.1
            },
            'error': True
        }

    def save_feedback_state(self, filepath: str):
        """保存反馈引擎状态"""
        state = {
            'performance_history': list(self.performance_history),
            'feedback_trend': list(self.feedback_trend),
            'config': self.config,
            'feature_dims': self.feature_dims
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f" 反馈引擎状态已保存: {filepath}")

    def load_feedback_state(self, filepath: str):
        """加载反馈引擎状态"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.performance_history = deque(state['performance_history'], maxlen=self.config['max_history'])
            self.feedback_trend = deque(state['feedback_trend'], maxlen=20)
            print(f" 反馈引擎状态已加载: {filepath}")
        except Exception as e:
            print(f" 加载状态失败: {e}")

    def analyze_performance(self, 
                           features: Dict[str, torch.Tensor],
                           shard_assignments: torch.Tensor,
                           edge_index: torch.Tensor = None,
                           performance_hints: Dict[str, Any] = None) -> torch.Tensor:
        """
        分析性能并生成反馈信号矩阵
        
        Args:
            features: 6类特征字典
            shard_assignments: 分片分配结果 [num_nodes] 
            edge_index: 边索引 [2, num_edges]
            performance_hints: 性能提示
            
        Returns:
            反馈矩阵 [num_nodes, 4] (改为返回与assignment_guidance一致的维度)
        """
        print(f" 开始性能分析...")
        
        try:
            # 获取完整的反馈处理结果
            feedback_result = self.process_sharding_feedback(
                features, shard_assignments, edge_index, performance_hints
            )
            
            # 提取第三步所需的反馈矩阵
            if ('step3_feedback_package' in feedback_result and 
                'assignment_guidance' in feedback_result['step3_feedback_package']):
                
                feedback_matrix = feedback_result['step3_feedback_package']['assignment_guidance']
                print(f"      生成反馈矩阵: {feedback_matrix.shape}")
                
                # 确保返回正确类型的tensor
                if isinstance(feedback_matrix, list):
                    feedback_matrix = torch.tensor(feedback_matrix)
                
                return feedback_matrix
            else:
                print(f"     未找到assignment_guidance，使用降级处理")
                # 降级处理：基于性能指标构建简单反馈
                num_nodes = len(shard_assignments)
                
                # 基于负载均衡度构建反馈
                performance_metrics = self.performance_evaluator.evaluate_comprehensive(
                    features, shard_assignments, edge_index, performance_hints
                )
                
                # 构建4维性能向量给每个节点
                performance_vector = torch.tensor([
                    performance_metrics.get('load_balance', 0.5),
                    performance_metrics.get('cross_shard_rate', 0.5),
                    performance_metrics.get('security_score', 0.8),
                    performance_metrics.get('consensus_latency', 0.6)
                ], dtype=torch.float32, device=self.device)
                
                # 为每个节点复制这个性能向量
                feedback_matrix = performance_vector.unsqueeze(0).repeat(num_nodes, 1)
                
                print(f"     使用降级反馈矩阵: {feedback_matrix.shape}")
                return feedback_matrix
                
        except Exception as e:
            print(f"   [ERROR] 性能分析失败: {e}")
            import traceback
            traceback.print_exc()
            # 最终降级：返回均匀性能向量
            num_nodes = len(shard_assignments)
            return torch.ones(num_nodes, 4, dtype=torch.float32, device=self.device) * 0.5


class SmartPerformanceEvaluator(nn.Module):
    """智能性能评估器 - 支持6类特征的综合评估"""
    
    def __init__(self, feature_dims: Dict[str, int]):
        super().__init__()
        self.feature_dims = feature_dims
        
        # 智能权重（可学习）
        self.smart_weights = nn.ParameterDict({
            'hw_balance': nn.Parameter(torch.tensor(0.4)),
            'topo_balance': nn.Parameter(torch.tensor(0.3)),
            'dynamic_balance': nn.Parameter(torch.tensor(0.3)),
            'categorical_cross': nn.Parameter(torch.tensor(0.5)),
            'topo_cross': nn.Parameter(torch.tensor(0.3)),
            'hetero_cross': nn.Parameter(torch.tensor(0.2)),
            'onchain_security': nn.Parameter(torch.tensor(0.7)),
            'hetero_security': nn.Parameter(torch.tensor(0.3)),
        })
        
        # 性能预测网络
        total_dim = sum(feature_dims.values())
        self.performance_net = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4个核心指标
            nn.Sigmoid()
        )

    def evaluate_comprehensive(self, 
                             features: Dict[str, torch.Tensor],
                             shard_assignments: torch.Tensor,
                             edge_index: torch.Tensor = None,
                             hints: Dict[str, Any] = None) -> Dict[str, Any]:
        """综合性能评估"""
        
        # 基础指标计算
        load_balance = self._compute_load_balance(shard_assignments, features)
        cross_shard_rate = self._compute_cross_shard_rate(shard_assignments, edge_index)
        security_score = self._compute_security_score(features, shard_assignments)
        consensus_latency = self._compute_consensus_latency(features, shard_assignments)
        
        # 使用神经网络增强评估
        combined_features = torch.cat([features[k] for k in sorted(features.keys())], dim=1)
        # 确保神经网络输入在正确设备上
        combined_features = combined_features.to(next(self.performance_net.parameters()).device)
        nn_scores = self.performance_net(combined_features.mean(dim=0, keepdim=True)).squeeze()
        
        # 融合传统计算和神经网络结果
        alpha = 0.7  # 传统方法权重
        final_scores = {
            'load_balance': alpha * load_balance + (1-alpha) * nn_scores[0].item(),
            'cross_shard_rate': alpha * cross_shard_rate + (1-alpha) * nn_scores[1].item(),
            'security_score': alpha * security_score + (1-alpha) * nn_scores[2].item(),
            'consensus_latency': alpha * consensus_latency + (1-alpha) * nn_scores[3].item()
        }
        
        return final_scores

    def _compute_load_balance(self, shard_assignments: torch.Tensor, features: Dict[str, torch.Tensor]) -> float:
        """计算负载均衡指标"""
        try:
            shard_sizes = torch.bincount(shard_assignments, minlength=shard_assignments.max().item() + 1)
            if len(shard_sizes) <= 1:
                return 0.8  # 单分片情况
            
            # 考虑硬件特征的负载均衡
            hw_features = features.get('hardware', torch.randn(len(shard_assignments), 17, device=shard_assignments.device))
            hw_capacity = hw_features.mean(dim=1)  # 硬件容量
            
            weighted_sizes = []
            for shard_id in range(len(shard_sizes)):
                mask = shard_assignments == shard_id
                if mask.sum() > 0:
                    avg_capacity = hw_capacity[mask].mean()
                    weighted_size = shard_sizes[shard_id].float() / (avg_capacity + 1e-6)
                    weighted_sizes.append(weighted_size)
            
            if not weighted_sizes:
                return 0.5
            
            weighted_sizes = torch.tensor(weighted_sizes)
            # 修复单分片情况下的std警告
            if len(weighted_sizes) <= 1:
                return 0.8  # 单分片时给出较高的均衡分数
            
            balance_score = 1.0 - (weighted_sizes.std() / (weighted_sizes.mean() + 1e-6))
            return max(0.0, min(1.0, balance_score.item()))
            
        except Exception as e:
            print(f" 负载均衡计算异常: {e}")
            return 0.5

    def _compute_cross_shard_rate(self, shard_assignments: torch.Tensor, edge_index: torch.Tensor) -> float:
        """计算跨片交易率"""
        if edge_index is None or edge_index.size(1) == 0:
            return 0.1  # 默认低跨片率
        
        try:
            src_shards = shard_assignments[edge_index[0]]
            dst_shards = shard_assignments[edge_index[1]]
            cross_shard_edges = (src_shards != dst_shards).sum().item()
            total_edges = edge_index.size(1)
            
            cross_rate = cross_shard_edges / max(total_edges, 1)
            return min(1.0, max(0.0, cross_rate))
            
        except Exception as e:
            print(f" 跨片率计算异常: {e}")
            return 0.3

    def _compute_security_score(self, features: Dict[str, torch.Tensor], shard_assignments: torch.Tensor) -> float:
        """计算安全性评分"""
        try:
            onchain_features = features.get('onchain_behavior', torch.randn(len(shard_assignments), 17, device=shard_assignments.device))
            hetero_features = features.get('heterogeneous_type', torch.randn(len(shard_assignments), 17, device=shard_assignments.device))
            
            # 每个分片的安全性评估
            num_shards = shard_assignments.max().item() + 1
            shard_security_scores = []
            
            for shard_id in range(num_shards):
                mask = shard_assignments == shard_id
                if mask.sum() == 0:
                    continue
                
                # 基于链上行为的信誉度
                reputation = onchain_features[mask].mean(dim=0).mean().item()
                
                # 基于异构类型的多样性
                # 修复单节点分片的std警告
                if mask.sum() <= 1:
                    diversity = 0.5  # 单节点分片的默认多样性
                else:
                    diversity = hetero_features[mask].std(dim=0).mean().item()
                
                # 分片大小的安全性（太小或太大都不安全）
                shard_size = mask.sum().item()
                size_security = 1.0 - abs(shard_size - 50) / 100  # 最优50个节点
                size_security = max(0.0, min(1.0, size_security))
                
                shard_security = 0.5 * reputation + 0.3 * diversity + 0.2 * size_security
                shard_security_scores.append(shard_security)
            
            if not shard_security_scores:
                return 0.5
            
            overall_security = np.mean(shard_security_scores)
            return max(0.0, min(1.0, overall_security))
            
        except Exception as e:
            print(f" 安全性计算异常: {e}")
            return 0.5

    def _compute_consensus_latency(self, features: Dict[str, torch.Tensor], shard_assignments: torch.Tensor) -> float:
        """计算共识延迟评分（低延迟=高分）"""
        try:
            dynamic_features = features.get('dynamic_attributes', torch.randn(len(shard_assignments), 13, device=shard_assignments.device))
            onchain_features = features.get('onchain_behavior', torch.randn(len(shard_assignments), 17, device=shard_assignments.device))
            
            # 基于动态属性计算平均负载
            avg_load = dynamic_features.mean(dim=1)
            
            # 基于链上行为计算处理能力
            processing_capacity = onchain_features.mean(dim=1)
            
            # 每个分片的共识延迟估算
            num_shards = shard_assignments.max().item() + 1
            shard_latencies = []
            
            for shard_id in range(num_shards):
                mask = shard_assignments == shard_id
                if mask.sum() == 0:
                    continue
                
                shard_load = avg_load[mask].mean().item()
                shard_capacity = processing_capacity[mask].mean().item()
                shard_size = mask.sum().item()
                
                # 延迟建模：负载高、容量低、分片过大都会增加延迟
                estimated_latency = shard_load / (shard_capacity + 1e-6) + shard_size / 100
                latency_score = 1.0 / (1.0 + estimated_latency)  # 越低延迟分数越高
                shard_latencies.append(latency_score)
            
            if not shard_latencies:
                return 0.5
            
            overall_latency_score = np.mean(shard_latencies)
            return max(0.0, min(1.0, overall_latency_score))
            
        except Exception as e:
            print(f" 共识延迟计算异常: {e}")
            return 0.5


class AdaptiveImportanceAnalyzer:
    """自适应重要性分析器"""
    
    def __init__(self, feature_dims: Dict[str, int]):
        self.feature_dims = feature_dims
        self.importance_history = defaultdict(list)

    def analyze_feature_importance(self, 
                                  features: Dict[str, torch.Tensor],
                                  shard_assignments: torch.Tensor,
                                  performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """分析特征重要性"""
        
        importance_scores = {}
        
        # 基于性能关联度的重要性分析
        for feat_name, feat_tensor in features.items():
            try:
                # 使用互信息计算特征与性能的关联度
                feat_flat = feat_tensor.detach().cpu().flatten().numpy()
                perf_combined = np.array([
                    performance_metrics['load_balance'],
                    performance_metrics['cross_shard_rate'],
                    performance_metrics['security_score'],
                    performance_metrics['consensus_latency']
                ]).repeat(len(feat_flat) // 4 + 1)[:len(feat_flat)]
                
                if len(feat_flat) == len(perf_combined):
                    mi_score = mutual_info_regression(feat_flat.reshape(-1, 1), perf_combined)[0]
                    importance_scores[feat_name] = max(0.0, min(1.0, mi_score))
                else:
                    importance_scores[feat_name] = 0.5  # 默认中等重要性
                    
            except Exception as e:
                print(f" 特征 {feat_name} 重要性分析异常: {e}")
                importance_scores[feat_name] = 0.5
        
        # 更新历史记录
        for feat_name, score in importance_scores.items():
            self.importance_history[feat_name].append(score)
            if len(self.importance_history[feat_name]) > 20:
                self.importance_history[feat_name].pop(0)
        
        # 计算趋势
        importance_trends = {}
        for feat_name in importance_scores.keys():
            history = self.importance_history[feat_name]
            if len(history) >= 3:
                recent_avg = np.mean(history[-3:])
                earlier_avg = np.mean(history[:-3]) if len(history) > 3 else recent_avg
                trend = recent_avg - earlier_avg
                importance_trends[feat_name] = trend
            else:
                importance_trends[feat_name] = 0.0
        
        return {
            'feature_importance': importance_scores,
            'importance_trends': importance_trends,
            'stability_scores': {k: 1.0 - np.std(v) if len(v) > 1 else 1.0 
                               for k, v in self.importance_history.items()}
        }


class FeedbackOptimizer:
    """反馈优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adaptation_rate = config.get('learning_rate', 0.01)

    def optimize_feedback(self, 
                         performance_metrics: Dict[str, Any],
                         importance_analysis: Dict[str, Any],
                         history: deque) -> Dict[str, Any]:
        """优化反馈信号"""
        
        # 基础性能加权
        weights = self.config['feedback_weights']
        base_score = (
            weights['balance'] * performance_metrics['load_balance'] +
            weights['cross_shard'] * (1.0 - performance_metrics['cross_shard_rate']) +  # 跨片率越低越好
            weights['security'] * performance_metrics['security_score'] +
            weights['consensus'] * performance_metrics['consensus_latency']
        )
        
        # 重要性调整
        importance_bonus = np.mean(list(importance_analysis['feature_importance'].values()))
        importance_factor = 1.0 + 0.2 * (importance_bonus - 0.5)  # ±20%调整
        
        # 历史趋势调整
        trend_factor = 1.0
        if len(history) >= 3:
            recent_scores = [h['feedback']['overall_score'] for h in list(history)[-3:]]
            if len(recent_scores) == 3:
                if recent_scores[-1] > recent_scores[-2] > recent_scores[-3]:
                    trend_factor = 1.1  # 连续改善，给予奖励
                elif recent_scores[-1] < recent_scores[-2] < recent_scores[-3]:
                    trend_factor = 0.9  # 连续恶化，给予惩罚
        
        # 最终优化分数
        optimized_score = base_score * importance_factor * trend_factor
        optimized_score = max(0.0, min(1.0, optimized_score))
        
        # 置信度计算
        confidence = self._calculate_confidence(performance_metrics, importance_analysis, history)
        
        return {
            'overall_score': optimized_score,
            'base_score': base_score,
            'importance_factor': importance_factor,
            'trend_factor': trend_factor,
            'confidence': confidence,
            'metrics': performance_metrics
        }

    def _calculate_confidence(self, performance_metrics, importance_analysis, history) -> float:
        """计算反馈置信度"""
        confidence_factors = []
        
        # 基于性能指标的一致性
        perf_values = list(performance_metrics.values())
        perf_std = np.std(perf_values)
        consistency_factor = 1.0 - perf_std  # 标准差越小，一致性越高
        confidence_factors.append(max(0.0, consistency_factor))
        
        # 基于重要性稳定性
        stability_scores = list(importance_analysis['stability_scores'].values())
        avg_stability = np.mean(stability_scores)
        confidence_factors.append(avg_stability)
        
        # 基于历史记录长度
        history_factor = min(1.0, len(history) / 10)  # 历史越长，置信度越高
        confidence_factors.append(history_factor)
        
        overall_confidence = np.mean(confidence_factors)
        return max(0.1, min(1.0, overall_confidence))


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self):
        self.normal_range = {
            'load_balance': (0.3, 1.0),
            'cross_shard_rate': (0.0, 0.6),
            'security_score': (0.4, 1.0),
            'consensus_latency': (0.3, 1.0)
        }

    def detect_anomalies(self, performance_metrics: Dict[str, Any], trend_history: deque) -> Dict[str, Any]:
        """检测性能异常"""
        anomalies = []
        
        # 阈值检测
        for metric, value in performance_metrics.items():
            if metric in self.normal_range:
                min_val, max_val = self.normal_range[metric]
                if value < min_val:
                    anomalies.append({
                        'type': 'threshold_low',
                        'metric': metric,
                        'value': value,
                        'threshold': min_val,
                        'description': f'{metric} 值 {value:.3f} 低于正常范围 {min_val}',
                        'severity': 'high' if value < min_val * 0.5 else 'medium'
                    })
                elif value > max_val:
                    anomalies.append({
                        'type': 'threshold_high',
                        'metric': metric,
                        'value': value,
                        'threshold': max_val,
                        'description': f'{metric} 值 {value:.3f} 高于正常范围 {max_val}',
                        'severity': 'high' if value > max_val * 1.5 else 'medium'
                    })
        
        # 趋势异常检测
        if len(trend_history) >= 5:
            recent_trend = list(trend_history)[-5:]
            trend_slope = np.polyfit(range(len(recent_trend)), recent_trend, 1)[0]
            
            if trend_slope < -0.1:  # 快速下降
                anomalies.append({
                    'type': 'trend_decline',
                    'description': '性能快速下降趋势',
                    'slope': trend_slope,
                    'severity': 'high' if trend_slope < -0.2 else 'medium'
                })
            elif abs(trend_slope) < 0.01 and np.std(recent_trend) < 0.05:  # 长期停滞
                anomalies.append({
                    'type': 'trend_stagnant',
                    'description': '性能长期停滞',
                    'std': np.std(recent_trend),
                    'severity': 'low'
                })
        
        return {
            'detected_anomalies': anomalies,
            'anomaly_count': len(anomalies),
            'max_severity': max([a.get('severity', 'low') for a in anomalies], default='none'),
            'summary': f'检测到 {len(anomalies)} 个异常' if anomalies else '无异常检测'
        }
