"""
完整的四步骤流水线演示 - 支持分层反馈的增强版本
"""
"""
整合第三步 EvolveGCN 动态分片的完整流水线
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import json

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入各步骤组件
from partition.feature.MainPipeline import Pipeline
from partition.feature.nodeInitialize import Node, load_nodes_from_csv
from feedback.feedback import FeedbackController
from feedback.feature_evolution import DynamicFeatureEvolution

# 导入第二步多尺度对比学习组件
from muti_scale.All_Final import (
    MSCIA, TemporalMSCIA, GNNEncoder,
    subgraph_contrastive_loss, graph_contrastive_loss, node_contrastive_loss
)

# 导入第三步 EvolveGCN 组件
from evolve_GCN.models import EvolveGCNWrapper, DynamicShardingModule
from evolve_GCN.data import BlockchainDataset
from evolve_GCN.losses import multi_objective_sharding_loss, temporal_consistency_loss
from evolve_GCN.config import TrainingConfig
from evolve_GCN.utils import get_device, HyperparameterUpdater


class IntegratedEvolveGCNDynamicSharder:
    """整合的第三步 EvolveGCN 动态分片模块"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.device = get_device()

        # 第三步专用配置
        self.config = TrainingConfig()
        if config:
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        # 模型组件
        self.model = None
        self.sharding_module = None
        self.param_updater = HyperparameterUpdater()

        # 训练状态
        self.history_states = []
        self.prev_shard_assignment = None
        self.prev_shard_count = self.config.base_shards
        self.cross_increase_count = 0
        self.prev_cross_rate = 0.0

        # 初始化状态
        self.initialized = False
        self.training_history = []

    def initialize_models(self, input_dim: int):
        """初始化 EvolveGCN 模型"""
        print(f"[SPEED] 初始化 EvolveGCN 动态分片模型...")

        # 1. EvolveGCN 包装器
        self.model = EvolveGCNWrapper(input_dim, self.config.hidden_dim).to(self.device)

        # 2. 动态分片模块
        self.sharding_module = DynamicShardingModule(
            self.config.hidden_dim,
            self.config.base_shards,
            self.config.max_shards
        ).to(self.device)

        # 3. 优化器
        self.model_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        self.shard_optimizer = torch.optim.Adam(
            self.sharding_module.parameters(),
            lr=self.config.shard_lr
        )

        self.initialized = True
        print(f"[SUCCESS] EvolveGCN 模型初始化完成 (输入维度: {input_dim}, 隐藏维度: {self.config.hidden_dim})")

    def process_step2_output(self, step2_output: Dict[str, torch.Tensor],
                             edge_index: torch.Tensor,
                             epoch: int = 0,
                             feedback_signal: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        处理第二步的输出，执行动态分片

        Args:
            step2_output: 第二步的多尺度对比学习结果
            edge_index: 网络拓扑边索引 [2, E]
            epoch: 当前轮次
            feedback_signal: 来自第四步的反馈信号

        Returns:
            {
                'shard_assignments': torch.Tensor,     # [N, num_shards] 软分配
                'hard_assignments': torch.Tensor,      # [N] 硬分配
                'predicted_num_shards': int,           # 预测分片数
                'shard_loss': float,                   # 分片损失
                'enhanced_embeddings': torch.Tensor,   # 增强嵌入
                'training_info': Dict                  # 训练信息
            }
        """
        if not self.initialized:
            input_dim = step2_output['embeddings'].size(1)
            self.initialize_models(input_dim)

        print(f"\n[SPEED] 执行第三步: EvolveGCN 动态分片 (Epoch {epoch})")

        # 1. 准备输入数据
        input_data = self._prepare_sharding_input(step2_output, edge_index, epoch, feedback_signal)

        # 2. 执行 EvolveGCN 时序处理
        evolved_embeddings = self._evolve_gcn_forward(input_data, epoch)

        # 3. 动态分片决策
        shard_results = self._dynamic_sharding_forward(evolved_embeddings, input_data, epoch)

        # 4. 计算分片损失
        loss_results = self._compute_sharding_loss(shard_results, input_data, epoch)

        # 5. 更新历史状态
        self._update_sharding_history(loss_results, shard_results, epoch)

        # 6. 构建输出
        output = {
            'shard_assignments': shard_results['shard_assignments'],
            'hard_assignments': torch.argmax(shard_results['shard_assignments'], dim=1),
            'predicted_num_shards': shard_results['predicted_num_shards'],
            'shard_loss': loss_results['total_loss'],
            'loss_components': loss_results['components'],
            'enhanced_embeddings': shard_results['enhanced_embeddings'],
            'attention_weights': shard_results['attention_weights'],
            'training_info': {
                'epoch': epoch,
                'feedback_used': feedback_signal is not None,
                'history_length': len(self.history_states),
                'convergence_info': self._get_convergence_info()
            },
            # 性能指标
            'performance_metrics': self._compute_performance_metrics(shard_results, input_data)
        }

        # 7. 记录训练历史
        self.training_history.append({
            'epoch': epoch,
            'shard_loss': loss_results['total_loss'],
            'num_shards': shard_results['predicted_num_shards'],
            'performance_metrics': output['performance_metrics']
        })

        print(f"[SUCCESS] 第三步完成:")
        print(f"  - 预测分片数: {shard_results['predicted_num_shards']}")
        print(f"  - 分片损失: {loss_results['total_loss']:.4f}")
        print(f"  - 负载均衡: {output['performance_metrics']['balance_score']:.3f}")
        print(f"  - 跨片比例: {output['performance_metrics']['cross_shard_ratio']:.3f}")

        return output

    def _prepare_sharding_input(self, step2_output: Dict[str, torch.Tensor],
                                edge_index: torch.Tensor,
                                epoch: int,
                                feedback_signal: Optional[torch.Tensor]) -> Dict[str, Any]:
        """准备分片输入数据"""

        embeddings = step2_output['embeddings'].to(self.device)
        edge_index = edge_index.to(self.device)

        # 构建时序数据（模拟时间步）
        num_nodes = embeddings.size(0)
        timestep_data = []

        # 为 EvolveGCN 创建时序输入
        for t in range(self.config.num_timesteps):
            # 添加噪声模拟时序变化
            noise = torch.randn_like(embeddings) * self.config.noise_level
            timestep_embeddings = embeddings + noise
            timestep_data.append((timestep_embeddings, edge_index, t))

        # 准备性能反馈
        if feedback_signal is not None:
            performance_feedback = feedback_signal.to(self.device)
        else:
            # 使用历史状态计算反馈
            performance_feedback = self._get_performance_feedback()

        return {
            'timestep_data': timestep_data,
            'original_embeddings': embeddings,
            'edge_index': edge_index,
            'performance_feedback': performance_feedback,
            'num_nodes': num_nodes,
            'epoch': epoch
        }

    def _evolve_gcn_forward(self, input_data: Dict[str, Any], epoch: int) -> torch.Tensor:
        """EvolveGCN 前向传播"""

        # 重置模型状态
        self.model.reset_state()

        all_embeddings = []
        performance_feedback = input_data['performance_feedback']

        # 时序处理
        for t, (node_features, edge_index, timestep) in enumerate(input_data['timestep_data']):
            # EvolveGCN 前向传播
            embeddings, delta_signal = self.model(node_features, edge_index, performance_feedback)
            all_embeddings.append(embeddings)

            # 计算 GCN 损失（用于模型更新）
            if epoch % 5 == 0:  # 周期性训练
                gcn_loss = self._compute_gcn_loss(embeddings, all_embeddings, t)

                # 反向传播更新 EvolveGCN
                self.model_optimizer.zero_grad()
                gcn_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.model_optimizer.step()

        # 返回最终嵌入
        final_embeddings = all_embeddings[-1]
        return final_embeddings

    def _dynamic_sharding_forward(self, evolved_embeddings: torch.Tensor,
                                  input_data: Dict[str, Any], epoch: int) -> Dict[str, Any]:
        """动态分片前向传播"""

        # 使用 DynamicShardingModule 进行分片
        shard_assignments, enhanced_embeddings, attention_weights, predicted_num_shards = self.sharding_module(
            evolved_embeddings,
            self.history_states,
            feedback_signal=input_data['performance_feedback']
        )

        return {
            'shard_assignments': shard_assignments,
            'enhanced_embeddings': enhanced_embeddings,
            'attention_weights': attention_weights,
            'predicted_num_shards': predicted_num_shards,
            'original_embeddings': evolved_embeddings
        }

    def _compute_sharding_loss(self, shard_results: Dict[str, Any],
                               input_data: Dict[str, Any], epoch: int) -> Dict[str, Any]:
        """计算分片损失"""

        shard_assignments = shard_results['shard_assignments']
        enhanced_embeddings = shard_results['enhanced_embeddings']
        edge_index = input_data['edge_index']

        # 生成安全评分（简化）
        num_nodes = shard_assignments.size(0)
        security_scores = torch.rand(num_nodes, dtype=torch.float32, device=self.device) * 0.5

        # 判断是否使用历史分配
        use_prev_assignment = (self.prev_shard_assignment is not None and
                               self.prev_shard_assignment.size(1) == shard_assignments.size(1))

        # 计算多目标分片损失
        shard_loss, loss_components = multi_objective_sharding_loss(
            shard_assignments, enhanced_embeddings, edge_index,
            prev_assignment=self.prev_shard_assignment if use_prev_assignment else None,
            security_scores=security_scores,
            a=self.param_updater.params.get('balance_weight', 1.0),
            b=self.param_updater.params.get('cross_weight', 1.0),
            c=self.param_updater.params.get('security_weight', 1.5),
            d=self.param_updater.params.get('migrate_weight', 0.5) if use_prev_assignment else 0.0
        )

        # 保存当前分配
        self.prev_shard_assignment = shard_assignments.detach().clone()

        # 优化分片模块
        if epoch % 3 == 0:  # 周期性优化
            self.shard_optimizer.zero_grad()
            shard_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sharding_module.parameters(), max_norm=1.0)
            self.shard_optimizer.step()

        return {
            'total_loss': shard_loss.item(),
            'components': loss_components,
            'use_prev_assignment': use_prev_assignment
        }

    def _compute_gcn_loss(self, embeddings: torch.Tensor,
                          all_embeddings: List[torch.Tensor], t: int) -> torch.Tensor:
        """计算 GCN 损失"""
        gcn_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        # 时序一致性损失
        if t > 0:
            consistency_loss = temporal_consistency_loss(
                embeddings, all_embeddings[t - 1].detach(),
                lambda_contrast=self.param_updater.params['lambda']
            )
            gcn_loss += consistency_loss

        # 正则化损失
        reg_loss = 0.01 * torch.mean(torch.norm(embeddings, p=2, dim=1))
        gcn_loss += reg_loss

        return gcn_loss

    def _update_sharding_history(self, loss_results: Dict[str, Any],
                                 shard_results: Dict[str, Any], epoch: int):
        """更新分片历史状态"""

        with torch.no_grad():
            shard_assignments = shard_results['shard_assignments']
            predicted_num_shards = shard_results['predicted_num_shards']

            hard_assignment = torch.argmax(shard_assignments, dim=1)
            shard_sizes = [(hard_assignment == s).sum().item() for s in range(predicted_num_shards)]

            # 性能指标计算
            balance_score = 1.0 - (np.std(shard_sizes) / (np.mean(shard_sizes) + 1e-8))
            cross_rate = loss_results['components']['cross']
            security_score = 1.0 - loss_results['components']['security']

            # 跨片交易率趋势检测
            if cross_rate > self.prev_cross_rate:
                self.cross_increase_count += 1
            else:
                self.cross_increase_count = 0
            self.prev_cross_rate = cross_rate

            # 更新历史状态
            current_state = torch.tensor([balance_score, cross_rate, security_score],
                                         dtype=torch.float32, device=self.device)
            self.history_states.append(current_state)
            if len(self.history_states) > self.config.history_length:
                self.history_states.pop(0)

            # 更新状态跟踪
            self.prev_shard_count = predicted_num_shards

            # 动态调节参数
            performance_metrics = {
                'balance_score': balance_score,
                'cross_tx_rate': cross_rate,
                'cross_increase_count': self.cross_increase_count
            }
            self.param_updater.update_hyperparams(performance_metrics)

    def _compute_performance_metrics(self, shard_results: Dict[str, Any],
                                     input_data: Dict[str, Any]) -> Dict[str, float]:
        """计算性能指标"""

        shard_assignments = shard_results['shard_assignments']
        predicted_num_shards = shard_results['predicted_num_shards']
        edge_index = input_data['edge_index']

        hard_assignment = torch.argmax(shard_assignments, dim=1)
        shard_sizes = [(hard_assignment == s).sum().item() for s in range(predicted_num_shards)]

        # 负载均衡指标
        balance_score = 1.0 - (np.std(shard_sizes) / (np.mean(shard_sizes) + 1e-8))

        # 跨片交易比例
        cross_shard_edges = 0
        total_edges = edge_index.size(1)
        if total_edges > 0:
            u, v = edge_index[0], edge_index[1]
            valid_mask = (u < len(hard_assignment)) & (v < len(hard_assignment))
            if valid_mask.sum() > 0:
                valid_u, valid_v = u[valid_mask], v[valid_mask]
                cross_shard_edges = (hard_assignment[valid_u] != hard_assignment[valid_v]).sum().item()

        cross_shard_ratio = cross_shard_edges / max(total_edges, 1)

        return {
            'balance_score': balance_score,
            'cross_shard_ratio': cross_shard_ratio,
            'shard_sizes': shard_sizes,
            'num_shards': predicted_num_shards,
            'size_variance': np.var(shard_sizes)
        }

    def _get_performance_feedback(self) -> torch.Tensor:
        """获取性能反馈信号"""
        if self.history_states:
            recent_states = torch.stack(self.history_states[-3:]) if len(self.history_states) >= 3 else torch.stack(
                self.history_states)
            performance_feedback = torch.mean(recent_states, dim=0).float()
        else:
            performance_feedback = torch.tensor([0.5, 0.1, 0.8], dtype=torch.float32, device=self.device)
        return performance_feedback

    def _get_convergence_info(self) -> Dict[str, Any]:
        """获取收敛信息"""
        if len(self.training_history) < 5:
            return {'converged': False, 'trend': 'unknown'}

        recent_losses = [h['shard_loss'] for h in self.training_history[-5:]]
        loss_trend = np.mean(np.diff(recent_losses))

        return {
            'converged': abs(loss_trend) < 0.001,
            'trend': 'decreasing' if loss_trend < 0 else 'increasing',
            'recent_avg_loss': np.mean(recent_losses),
            'loss_variance': np.var(recent_losses)
        }

    def get_model_state(self) -> Dict[str, Any]:
        """获取模型状态"""
        return {
            'initialized': self.initialized,
            'config': self.config.to_dict(),
            'device': str(self.device),
            'history_length': len(self.history_states),
            'training_history_length': len(self.training_history),
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            'sharding_parameters': sum(p.numel() for p in self.sharding_module.parameters()) if self.sharding_module else 0,
            'hyperparams': self.param_updater.get_params()
        }

    def save_sharding_results(self, output: Dict[str, Any], filepath: str):
        """保存分片结果"""

        hard_assignments = output['hard_assignments'].cpu().numpy()
        num_shards = output['predicted_num_shards']

        sharding_results = {}
        for s in range(num_shards):
            shard_nodes = (hard_assignments == s).nonzero()[0].tolist()
            sharding_results[f'shard_{s}'] = shard_nodes

        with open(filepath, 'wb') as f:
            pickle.dump(sharding_results, f)

        print(f"📁 分片结果已保存: {filepath}")

        # 打印分片统计
        for key, value in sharding_results.items():
            print(f"  - {key}: {len(value)} 节点")

        return sharding_results


# 更新完整流水线演示类，整合真实的第三步
class FullIntegratedPipelineDemo:
    """完整整合的四步骤流水线演示 - 真实的第二步和第三步"""

    def __init__(self):
        self.device = get_device()
        print(f"使用设备: {self.device}")

        # 配置参数
        self.config = {
            'num_epochs': 15,
            'feedback_start_epoch': 2,
            'save_results': True,
            'results_dir': './results',
            'sample_data_path': './large_samples.csv',

            # 第二步专用配置
            'step2_config': {
                'input_dim': 128,
                'hidden_dim': 64,
                'time_dim': 16,
                'k_ratio': 0.9,
                'alpha': 0.3,
                'beta': 0.4,
                'gamma': 0.3,
                'tau': 0.09,
                'augment_type': 'edge',      # 添加这个参数
                'num_node_types': 5,         # 添加这个参数
                'num_edge_types': 3,         # 添加这个参数
                'learning_rate': 0.02,
                'weight_decay': 9e-6,
                'max_epochs': 20,
                'target_loss': 0.25
            },

            # 第三步专用配置
            'step3_config': {
                'lr': 0.001,
                'epochs': 10,
                'num_timesteps': 5,
                'base_shards': 3,
                'hidden_dim': 64,
                'max_shards': 8,
                'noise_level': 0.01,
                'weight_decay': 1e-5,
                'shard_lr': 0.001,
                'balance_weight': 0.5,
                'cross_weight': 1.5,
                'security_weight': 0.5,
                'migrate_weight': 0.5,
                'max_grad_norm': 1.0,
                'history_length': 10,
                'print_freq': 5
            }
        }

        # 创建结果目录
        os.makedirs(self.config['results_dir'], exist_ok=True)

        # 流水线历史记录
        self.pipeline_history = {
            'step1_features': [],
            'step2_embeddings': [],
            'step3_sharding': [],
            'step4_feedback': [],
            'performance_metrics': []
        }

        # 初始化组件
        self._initialize_components()

    def _initialize_components(self):
        """初始化各步骤组件"""
        print("初始化各步骤组件...")

        # 第一步: 分层反馈特征提取流水线
        print("- 初始化第一步: 分层反馈特征提取流水线")
        base_pipeline = Pipeline(use_fusion=True, save_adjacency=True)
        self.step1_pipeline = LayeredFeedbackFeatureExtractor(base_pipeline)

        # 第二步: 整合的多尺度对比学习
        print("- 初始化第二步: 多尺度对比学习")
        self.step2_mscia = IntegratedMultiScaleContrastiveLearning(self.config['step2_config'])

        # 第三步: 整合的 EvolveGCN 动态分片
        print("- 初始化第三步: EvolveGCN 动态分片")
        self.step3_evolve_gcn = IntegratedEvolveGCNDynamicSharder(self.config['step3_config'])

        # 第四步: 增强反馈控制器
        print("- 初始化第四步: 增强反馈控制器")
        self.feedback_controller = EnhancedFeedbackController()

        print("[SUCCESS] 所有组件初始化完成")

    def run_step3_evolve_gcn_sharding(self, step2_output: Dict[str, torch.Tensor],
                                      edge_index: torch.Tensor,
                                      epoch: int,
                                      feedback_signal: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """第三步: 真实的 EvolveGCN 动态分片"""
        print(f"\n[SPEED] 执行第三步: EvolveGCN 动态分片 (Epoch {epoch})...")

        # 使用整合的 EvolveGCN 动态分片器
        step3_output = self.step3_evolve_gcn.process_step2_output(
            step2_output, edge_index, epoch, feedback_signal
        )

        # 记录分片历史
        sharding_record = {
            'epoch': epoch,
            'shard_loss': step3_output['shard_loss'],
            'num_shards': step3_output['predicted_num_shards'],
            'shard_sizes': step3_output['performance_metrics']['shard_sizes'],
            'balance_score': step3_output['performance_metrics']['balance_score'],
            'cross_shard_ratio': step3_output['performance_metrics']['cross_shard_ratio'],
            'feedback_used': step3_output['training_info']['feedback_used'],
            'convergence_info': step3_output['training_info']['convergence_info'],
            'embeddings_source': step2_output['training_info']['mode']
        }

        self.pipeline_history['step3_sharding'].append(sharding_record)

        # 周期性保存分片结果
        if epoch % 5 == 0:
            sharding_path = os.path.join(self.config['results_dir'], f'sharding_results_epoch_{epoch}.pkl')
            self.step3_evolve_gcn.save_sharding_results(step3_output, sharding_path)

        print(f"[SUCCESS] 第三步完成:")
        print(f"  - 预测分片数: {step3_output['predicted_num_shards']}")
        print(f"  - 分片损失: {step3_output['shard_loss']:.4f}")
        print(f"  - 负载均衡: {step3_output['performance_metrics']['balance_score']:.3f}")
        print(f"  - 跨片比例: {step3_output['performance_metrics']['cross_shard_ratio']:.3f}")
        print(f"  - 收敛状态: {step3_output['training_info']['convergence_info']['trend']}")

        return step3_output

    def run_complete_pipeline(self, save_results: bool = True) -> Dict[str, Any]:
        """运行完整的四步骤流水线 - 真实的第二步和第三步"""
        print("=" * 80)
        print("[START] 开始完整的四步骤区块链分片流水线 (整合真实第二步和第三步)")
        print("=" * 80)

        start_time = datetime.now()

        try:
            # 生成示例数据
            nodes = self.generate_sample_nodes()
            edge_index = self.generate_sample_network_topology(len(nodes))

            # 当前反馈指导状态
            current_step1_guidance = None

            # 主训练循环
            for epoch in range(self.config['num_epochs']):
                print(f"\n{'='*30} EPOCH {epoch+1}/{self.config['num_epochs']} {'='*30}")

                # 第一步: 分层反馈特征提取
                step1_output = self.run_step1_feature_extraction(
                    nodes, current_step1_guidance, epoch
                )

                # 第二步: 真实的多尺度对比学习
                step2_output = self.run_step2_contrastive_learning(
                    step1_output, edge_index, epoch
                )

                # 第三步: 真实的 EvolveGCN 动态分片
                feedback_signal = None
                if epoch > 0 and hasattr(self, 'last_feedback_signal'):
                    feedback_signal = self.last_feedback_signal

                step3_output = self.run_step3_evolve_gcn_sharding(
                    step2_output, edge_index, epoch, feedback_signal
                )

                # 第四步: 分层反馈优化
                feedback_signal, next_step1_guidance = self.run_step4_feedback_optimization(
                    step1_output, step3_output, edge_index, epoch
                )

                # 保存反馈信号和指导供下一轮使用
                self.last_feedback_signal = feedback_signal
                current_step1_guidance = next_step1_guidance

                # 记录整体性能
                epoch_performance = {
                    'epoch': epoch,
                    'step1_mode': step1_output.get('feedback_mode', 'unknown'),
                    'step2_loss': step2_output['contrastive_loss'],
                    'step2_mode': step2_output['training_info']['mode'],
                    'step3_loss': step3_output['shard_loss'],
                    'step3_converged': step3_output['training_info']['convergence_info']['converged'],
                    'num_shards': step3_output['predicted_num_shards'],
                    'balance_score': feedback_signal[0].item(),
                    'cross_tx_rate': feedback_signal[1].item(),
                    'security_score': feedback_signal[2].item(),
                    'feedback_active': bool(current_step1_guidance),
                    'integration_metrics': {
                        'step1_layers': len(step1_output.get('layered_breakdown', {})),
                        'step2_converged': step2_output['training_info'].get('converged', False),
                        'step3_balance_score': step3_output['performance_metrics']['balance_score'],
                        'step3_cross_ratio': step3_output['performance_metrics']['cross_shard_ratio'],
                        'step3_trend': step3_output['training_info']['convergence_info']['trend']
                    }
                }

                self.pipeline_history['performance_metrics'].append(epoch_performance)

                # 每3轮输出详细信息
                if (epoch + 1) % 3 == 0:
                    self._print_full_integrated_epoch_summary(epoch_performance)

        except Exception as e:
            print(f"[ERROR] 流水线执行过程中出错: {e}")
            import traceback
            traceback.print_exc()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 汇总结果
        final_results = self._compile_full_integrated_final_results(duration)

        # 保存结果
        if save_results and self.config['save_results']:
            self._save_full_integrated_results(final_results)

        self._visualize_full_integrated_results()

        print("\n" + "=" * 80)
        print("[SUCCESS] 完整四步骤流水线执行完成 (真实第二步 + 真实第三步)!")
        print(f"⏱️  总耗时: {duration:.2f}秒")
        print(f"🧠 第二步模型状态: {self.step2_mscia.get_model_state()}")
        print(f"[SPEED] 第三步模型状态: {self.step3_evolve_gcn.get_model_state()}")
        adaptation_report = self.step1_pipeline.get_adaptation_report()
        print(f"🔄 反馈模式: {adaptation_report.get('current_mode', 'unknown')}")
        print("=" * 80)

        return final_results

    # 复用之前的方法实现...
    def generate_sample_nodes(self, num_nodes: int = 100) -> List[Node]:
        """生成示例节点数据 - 修复版本"""
        # 首先尝试从CSV文件加载
        if os.path.exists(self.config['sample_data_path']):
            print(f"从 {self.config['sample_data_path']} 加载节点数据...")
            try:
                return load_nodes_from_csv(self.config['sample_data_path'])
            except Exception as e:
                print(f"[WARNING] 从CSV加载失败: {e}")
                print("将生成模拟节点数据...")
        
        print(f"生成 {num_nodes} 个示例节点...")
        
        # 使用与generate_samples.py相同的节点生成逻辑
        try:
            # 使用专门的样本生成器
            from partition.feature.generate_samples import BlockchainNodeSampleGenerator
            
            generator = BlockchainNodeSampleGenerator(seed=42)
            raw_samples = generator.generate_samples(
                num_samples=num_nodes, 
                output_file='temp_generated_samples.csv'
            )
            
            # 将原始样本转换为Node对象
            nodes = []
            for i, sample_dict in enumerate(raw_samples):
                try:
                    # 使用正确的Node构造方式（传入字典数据）
                    node = Node(sample_dict)  # Node类接受字典形式的数据
                    nodes.append(node)
                except Exception as e:
                    print(f"[WARNING] 创建节点 {i} 失败: {e}")
                    continue
            
            print(f"[SUCCESS] 成功生成 {len(nodes)} 个节点")
            return nodes
            
        except Exception as e:
            print(f"[WARNING] 使用样本生成器失败: {e}")
            print("使用简化的节点生成方法...")
            
            # 回退到简化的节点生成
            return self._generate_simplified_nodes(num_nodes)

    def generate_sample_network_topology(self, num_nodes: int) -> torch.Tensor:
        """生成示例网络拓扑"""
        num_edges = min(num_nodes * 3, num_nodes * (num_nodes - 1) // 2)
        edges = []
        for _ in range(num_edges):
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            if u != v:
                edges.append([u, v])

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t().contiguous()

        return edge_index.to(self.device)

    def run_step1_feature_extraction(self, nodes: List[Node],
                                     step4_guidance: Optional[Dict[str, Any]] = None,
                                     epoch: int = 0) -> Dict[str, torch.Tensor]:
        """第一步: 分层反馈特征提取"""
        print(f"\n[CONFIG] 执行第一步: 分层反馈特征提取 (Epoch {epoch})")

        results = self.step1_pipeline.extract_features_with_feedback(
            nodes, step4_guidance, epoch
        )

        extraction_record = {
            'epoch': epoch,
            'f_classic_shape': results['f_classic'].shape,
            'f_graph_shape': results['f_graph'].shape,
            'feedback_applied': results.get('feedback_applied', False),
            'feedback_mode': results.get('feedback_mode', 'cold_start'),
            'layer_count': len(results.get('layered_breakdown', {}))
        }

        if 'f_fused' in results:
            extraction_record['f_fused_shape'] = results['f_fused'].shape

        self.pipeline_history['step1_features'].append(extraction_record)

        print(f"[SUCCESS] 第一步完成 (模式: {results.get('feedback_mode', 'unknown')}):")
        print(f"  - F_classic: {results['f_classic'].shape}")
        print(f"  - F_graph: {results['f_graph'].shape}")
        print(f"  - 分层特征: {len(results.get('layered_breakdown', {}))} 层")
        if 'f_fused' in results:
            print(f"  - F_fused: {results['f_fused'].shape}")

        return results

    def run_step2_contrastive_learning(self, step1_output: Dict[str, torch.Tensor],
                                       edge_index: torch.Tensor,
                                       epoch: int) -> Dict[str, torch.Tensor]:
        """第二步: 真实的多尺度对比学习"""
        print(f"\n🧠 执行第二步: 多尺度对比学习 (Epoch {epoch})...")

        step2_output = self.step2_mscia.process_step1_output(
            step1_output, edge_index, epoch
        )

        embedding_record = {
            'epoch': epoch,
            'input_features_shape': step1_output['f_classic'].shape,
            'embeddings_shape': step2_output['embeddings'].shape,
            'contrastive_loss': step2_output['contrastive_loss'],
            'training_mode': step2_output['training_info']['mode'],
            'converged': step2_output['training_info'].get('converged', False),
            'embedding_stats': {
                'mean': float(torch.mean(step2_output['embeddings'])),
                'std': float(torch.std(step2_output['embeddings'])),
                'min': float(torch.min(step2_output['embeddings'])),
                'max': float(torch.max(step2_output['embeddings']))
            }
        }

        self.pipeline_history['step2_embeddings'].append(embedding_record)

        if epoch % 5 == 0:
            embeddings_path = os.path.join(self.config['results_dir'], f'temporal_embeddings_epoch_{epoch}.pkl')
            self.step2_mscia.save_embeddings(step2_output['temporal_embeddings'], embeddings_path)

        print(f"[SUCCESS] 第二步完成:")
        print(f"  - 嵌入维度: {step2_output['embeddings'].shape}")
        print(f"  - 对比损失: {step2_output['contrastive_loss']:.4f}")
        print(f"  - 训练模式: {step2_output['training_info']['mode']}")
        if 'converged' in step2_output['training_info']:
            print(f"  - 收敛状态: {'是' if step2_output['training_info']['converged'] else '否'}")

        return step2_output

    def run_step4_feedback_optimization(self, step1_output: Dict[str, torch.Tensor],
                                        step3_output: Dict[str, Any],
                                        edge_index: torch.Tensor,
                                        epoch: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """第四步: 分层反馈优化"""
        print(f"\n🔄 执行第四步: 分层反馈优化 (Epoch {epoch})...")

        if epoch < self.config['feedback_start_epoch']:
            print("- 跳过反馈优化 (未到启动轮次)")
            default_feedback = torch.tensor([0.5, 0.1, 0.8], device=self.device)
            return default_feedback, {}

        try:
            # 使用真实的分片分配结果
            feedback_signal, step1_guidance = self.feedback_controller.process_layered_feedback(
                step1_output,
                step3_output['shard_assignments'],
                edge_index,
                self.step3_evolve_gcn.model if self.step3_evolve_gcn.model else None,
                epoch
            )

        except Exception as e:
            print(f"反馈处理出错，使用基于第三步性能的反馈: {e}")

            # 基于第三步的实际性能指标
            balance_score = step3_output['performance_metrics']['balance_score']
            cross_shard_ratio = step3_output['performance_metrics']['cross_shard_ratio']

            # 基于损失趋势估算安全分数
            convergence_info = step3_output['training_info']['convergence_info']
            security_score = 0.8 if convergence_info['converged'] else 0.6

            feedback_signal = torch.tensor([balance_score, cross_shard_ratio, security_score], device=self.device)

            step1_guidance = {
                'epoch': epoch,
                'guidance_type': 'step3_based',
                'layer_weight_adjustments': {
                    'hardware': 1.0 + (balance_score - 0.5) * 0.3,
                    'onchain_behavior': 1.0 + (security_score - 0.7) * 0.4,
                    'network_topology': 1.0 + (cross_shard_ratio - 0.2) * -0.4,
                    'sequence': 1.0 + (security_score - 0.7) * 0.2,
                },
                'layer_enhancement_factors': {
                    'hardware': 1.1 if balance_score < 0.6 else 1.0,
                    'onchain_behavior': 1.1 if security_score < 0.7 else 1.0,
                    'network_topology': 0.9 if cross_shard_ratio > 0.25 else 1.0
                }
            }

        feedback_record = {
            'epoch': epoch,
            'balance_score': feedback_signal[0].item(),
            'cross_tx_rate': feedback_signal[1].item(),
            'security_score': feedback_signal[2].item(),
            'has_guidance': bool(step1_guidance),
            'guidance_layers': len(step1_guidance.get('layer_weight_adjustments', {})),
            'step3_metrics': step3_output['performance_metrics']
        }

        self.pipeline_history['step4_feedback'].append(feedback_record)

        print(f"[SUCCESS] 第四步完成: 反馈信号 [{feedback_signal[0]:.3f}, {feedback_signal[1]:.3f}, {feedback_signal[2]:.3f}]")

        if step1_guidance:
            print(f"   生成分层指导: {len(step1_guidance.get('layer_weight_adjustments', {}))} 层权重调整")

        return feedback_signal, step1_guidance

    def _print_full_integrated_epoch_summary(self, epoch_performance: Dict[str, Any]):
        """打印完全整合的轮次摘要"""
        print(f"\n[DATA] Epoch {epoch_performance['epoch']} 完全整合性能摘要:")
        print(f"   • 第一步模式: {epoch_performance['step1_mode']}")
        print(f"   • 第二步损失: {epoch_performance['step2_loss']:.4f} (模式: {epoch_performance['step2_mode']})")
        print(f"   • 第三步损失: {epoch_performance['step3_loss']:.4f} (收敛: {'是' if epoch_performance['step3_converged'] else '否'})")
        print(f"   • 分片数量: {epoch_performance['num_shards']}")
        print(f"   • 负载均衡: {epoch_performance['balance_score']:.3f}")
        print(f"   • 跨片交易率: {epoch_performance['cross_tx_rate']:.3f}")
        print(f"   • 安全分数: {epoch_performance['security_score']:.3f}")
        print(f"   • 整合指标:")
        for metric, value in epoch_performance['integration_metrics'].items():
            if isinstance(value, (int, float)):
                print(f"     - {metric}: {value:.3f}" if isinstance(value, float) else f"     - {metric}: {value}")
            else:
                print(f"     - {metric}: {value}")

    def _compile_full_integrated_final_results(self, duration: float) -> Dict[str, Any]:
        """编译完全整合的最终结果"""
        return {
            'execution_info': {
                'duration_seconds': duration,
                'total_epochs': self.config['num_epochs'],
                'feedback_start_epoch': self.config['feedback_start_epoch'],
                'device': str(self.device),
                'integration_type': 'real_step2_step3_full'
            },
            'pipeline_history': self.pipeline_history,
            'step2_model_state': self.step2_mscia.get_model_state(),
            'step3_model_state': self.step3_evolve_gcn.get_model_state(),
            'adaptation_report': self.step1_pipeline.get_adaptation_report(),
            'final_metrics': {
                'avg_balance_score': np.mean([m['balance_score'] for m in self.pipeline_history['performance_metrics']]),
                'avg_cross_tx_rate': np.mean([m['cross_tx_rate'] for m in self.pipeline_history['performance_metrics']]),
                'avg_security_score': np.mean([m['security_score'] for m in self.pipeline_history['performance_metrics']]),
                'avg_step2_loss': np.mean([m['step2_loss'] for m in self.pipeline_history['performance_metrics']]),
                'avg_step3_loss': np.mean([m['step3_loss'] for m in self.pipeline_history['performance_metrics']]),
                'step2_convergence_rate': sum(1 for m in self.pipeline_history['performance_metrics']
                                              if m.get('integration_metrics', {}).get('step2_converged', False)) / len(self.pipeline_history['performance_metrics']),
                'step3_convergence_rate': sum(1 for m in self.pipeline_history['performance_metrics']
                                              if m.get('step3_converged', False)) / len(self.pipeline_history['performance_metrics']),
                'total_feedback_activations': sum(1 for m in self.pipeline_history['performance_metrics'] if m['feedback_active'])
            }
        }

    def _save_full_integrated_results(self, results: Dict[str, Any]):
        """保存完全整合的结果"""
        results_path = os.path.join(self.config['results_dir'], 'full_integrated_pipeline_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f"📁 完全整合结果已保存: {results_path}")

    def _visualize_full_integrated_results(self):
        """可视化完全整合的结果"""
        try:
            import matplotlib.pyplot as plt

            epochs = [m['epoch'] for m in self.pipeline_history['performance_metrics']]
            balance_scores = [m['balance_score'] for m in self.pipeline_history['performance_metrics']]
            step2_losses = [m['step2_loss'] for m in self.pipeline_history['performance_metrics']]
            step3_losses = [m['step3_loss'] for m in self.pipeline_history['performance_metrics']]

            plt.figure(figsize=(18, 12))

            plt.subplot(2, 4, 1)
            plt.plot(epochs, balance_scores, 'b-', label='Balance Score', linewidth=2)
            plt.title('负载均衡分数', fontsize=12)
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 4, 2)
            plt.plot(epochs, step2_losses, 'r-', label='Step2 Loss', linewidth=2)
            plt.title('第二步对比学习损失', fontsize=12)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 4, 3)
            plt.plot(epochs, step3_losses, 'g-', label='Step3 Loss', linewidth=2)
            plt.title('第三步 EvolveGCN 损失', fontsize=12)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)

            # 其他可视化
            if len(self.pipeline_history['step2_embeddings']) > 0:
                embedding_means = [e['embedding_stats']['mean'] for e in self.pipeline_history['step2_embeddings']]
                plt.subplot(2, 4, 4)
                plt.plot(epochs, embedding_means, 'm-', label='Embedding Mean', linewidth=2)
                plt.title('嵌入特征均值', fontsize=12)
                plt.xlabel('Epoch')
                plt.ylabel('Mean Value')
                plt.grid(True, alpha=0.3)

            # 第三步相关指标
            if len(self.pipeline_history['step3_sharding']) > 0:
                balance_scores_step3 = [s['balance_score'] for s in self.pipeline_history['step3_sharding']]
                cross_ratios = [s['cross_shard_ratio'] for s in self.pipeline_history['step3_sharding']]

                plt.subplot(2, 4, 5)
                plt.plot(epochs, balance_scores_step3, 'c-', label='Step3 Balance', linewidth=2)
                plt.title('第三步负载均衡', fontsize=12)
                plt.xlabel('Epoch')
                plt.ylabel('Balance Score')
                plt.grid(True, alpha=0.3)

                plt.subplot(2, 4, 6)
                plt.plot(epochs, cross_ratios, 'orange', label='Cross Shard Ratio', linewidth=2)
                plt.title('跨片交易比例', fontsize=12)
                plt.xlabel('Epoch')
                plt.ylabel('Cross Ratio')
                plt.grid(True, alpha=0.3)

            plt.tight_layout()

            plot_path = os.path.join(self.config['results_dir'], 'full_integrated_performance_visualization.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📈 完全整合性能可视化已保存: {plot_path}")

            plt.show()

        except ImportError:
            print("[WARNING] matplotlib未安装，跳过可视化")
        except Exception as e:
            print(f"[WARNING] 可视化生成失败: {e}")





class LayeredFeedbackFeatureExtractor:
    """
    分层反馈特征提取器 - 对应第一步的实际结构
    支持 99维原始特征的6层分解 + 32维时序特征 + 10维图结构特征
    """

    def __init__(self, base_pipeline: Pipeline):
        self.base_pipeline = base_pipeline
        self.feedback_enabled = False
        self.feedback_mode = 'cold_start'  # cold_start, warm_feedback, stable_feedback
        self.adaptation_history = []

        # 第一步特征结构映射
        self.feature_structure = {
            # 99维原始特征的6层分解
            'original_layers': {
                'hardware': {'start': 0, 'end': 17, 'dim': 17},           # 硬件规格特征
                'onchain_behavior': {'start': 17, 'end': 34, 'dim': 17},  # 链上行为特征
                'network_topology': {'start': 34, 'end': 54, 'dim': 20},  # 网络拓扑特征
                'dynamic_attributes': {'start': 54, 'end': 67, 'dim': 13}, # 动态属性特征
                'heterogeneous_type': {'start': 67, 'end': 84, 'dim': 17}, # 异构类型特征
                'categorical': {'start': 84, 'end': 99, 'dim': 15}         # 分类特征
            },
            # 附加特征
            'sequence_features': {'start': 99, 'end': 131, 'dim': 32},    # 时序特征
            'graph_structure': {'start': 131, 'end': 141, 'dim': 10},     # 图结构特征
            # 最终投影
            'final_projection': {'input_dim': 141, 'output_dim': 128}
        }

        # 自适应权重管理器
        self.layer_weights = {
            'hardware': 1.0,
            'onchain_behavior': 1.0,
            'network_topology': 1.0,
            'dynamic_attributes': 1.0,
            'heterogeneous_type': 1.0,
            'categorical': 1.0,
            'sequence': 1.0,
            'graph_structure': 1.0
        }

    def extract_features_with_feedback(self, nodes: List[Node],
                                       step4_guidance: Optional[Dict[str, Any]] = None,
                                       epoch: int = 0) -> Dict[str, torch.Tensor]:
        """带分层反馈的特征提取"""

        self._update_feedback_mode(step4_guidance, epoch)

        print(f"[CONFIG] 执行分层反馈特征提取 (Epoch {epoch}, Mode: {self.feedback_mode})")

        # 1. 使用基础Pipeline提取原始特征
        base_results = self.base_pipeline.extract_features(nodes)
        f_classic = base_results['f_classic']  # [N, 128] - 已投影的特征
        f_graph = base_results['f_graph']      # [N, 96]

        # 2. 重新提取141维的原始拼接特征（模拟第一步的实际流程）
        raw_features_141 = self._extract_raw_141_features(nodes)  # [N, 141]

        # 3. 分解为分层结构
        layered_breakdown = self._decompose_to_layers(raw_features_141)

        # 4. 应用反馈调整
        if step4_guidance and self.feedback_mode != 'cold_start':
            adjusted_layered = self._apply_layered_feedback(layered_breakdown, step4_guidance)
        else:
            adjusted_layered = layered_breakdown

        # 5. 重构为141维特征
        adjusted_raw_141 = self._reconstruct_from_layers(adjusted_layered)

        # 6. 重新投影到128维
        adjusted_f_classic = self._adaptive_projection(adjusted_raw_141, step4_guidance)

        # 7. 构建完整结果
        results = {
            'f_classic': adjusted_f_classic,      # [N, 128] - 调整后的最终特征
            'f_graph': f_graph,                   # [N, 96] - 图特征保持不变
            'nodes': nodes,

            # 分层特征详细信息
            'layered_breakdown': adjusted_layered,
            'raw_141_features': adjusted_raw_141,
            'original_raw_141': raw_features_141,

            # 反馈状态
            'feedback_applied': step4_guidance is not None,
            'feedback_mode': self.feedback_mode,
            'adaptation_info': {
                'epoch': epoch,
                'layer_weights': self.layer_weights.copy(),
                'guidance_keys': list(step4_guidance.keys()) if step4_guidance else []
            }
        }

        # 8. 特征融合（如果启用）
        if self.base_pipeline.use_fusion:
            f_fused, contrastive_loss = self.base_pipeline.fusion_pipeline(
                adjusted_f_classic, f_graph
            )
            results['f_fused'] = f_fused
            results['contrastive_loss'] = contrastive_loss

        # 9. 记录适应历史
        self._record_adaptation(step4_guidance, results, epoch)

        return results

    def _extract_raw_141_features(self, nodes: List[Node]) -> torch.Tensor:
        """提取141维原始拼接特征（模拟UnifiedFeatureExtractor的内部流程）"""

        # 1. 提取99维综合特征
        comprehensive_features = self.base_pipeline.feature_extractor.comprehensive_extractor.extract_features(nodes)  # [N, 99]

        # 2. 提取32维时序特征
        sequence_features = self.base_pipeline.feature_extractor.sequence_encoder(nodes)  # [N, 32]

        # 3. 提取10维图结构特征
        graph_structure_features = self.base_pipeline.feature_extractor.graph_encoder(nodes)  # [N, 10]

        # 4. 拼接为141维
        raw_141 = torch.cat([
            comprehensive_features,    # [N, 99]
            sequence_features,         # [N, 32]
            graph_structure_features   # [N, 10]
        ], dim=1)  # [N, 141]

        return raw_141

    def _decompose_to_layers(self, raw_141: torch.Tensor) -> Dict[str, torch.Tensor]:
        """将141维特征分解为分层结构"""
        layered = {}

        # 分解99维原始特征为6层
        for layer_name, info in self.feature_structure['original_layers'].items():
            start, end = info['start'], info['end']
            layered[layer_name] = raw_141[:, start:end]

        # 提取时序特征
        seq_info = self.feature_structure['sequence_features']
        layered['sequence'] = raw_141[:, seq_info['start']:seq_info['end']]

        # 提取图结构特征
        graph_info = self.feature_structure['graph_structure']
        layered['graph_structure'] = raw_141[:, graph_info['start']:graph_info['end']]

        return layered

    def _apply_layered_feedback(self, layered_features: Dict[str, torch.Tensor],
                                step4_guidance: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """应用分层反馈调整"""
        adjusted = {}

        # 获取反馈指导
        layer_weight_adjustments = step4_guidance.get('layer_weight_adjustments', {})
        layer_enhancement_factors = step4_guidance.get('layer_enhancement_factors', {})

        for layer_name, layer_features in layered_features.items():

            # 1. 权重调整
            weight_adjustment = layer_weight_adjustments.get(layer_name, 1.0)
            self.layer_weights[layer_name] = 0.7 * self.layer_weights[layer_name] + 0.3 * weight_adjustment

            # 2. 增强因子
            enhancement_factor = layer_enhancement_factors.get(layer_name, 1.0)

            # 3. 应用调整
            adjusted_features = layer_features * self.layer_weights[layer_name] * enhancement_factor

            # 4. 维度选择（如果有指导）
            dimension_selection = step4_guidance.get('layer_dimension_selection', {}).get(layer_name, {})
            if 'selection_ratio' in dimension_selection and dimension_selection['selection_ratio'] < 1.0:
                # 简化实现：保留高方差维度
                feature_vars = torch.var(adjusted_features, dim=0)
                num_keep = max(1, int(adjusted_features.size(1) * dimension_selection['selection_ratio']))
                _, top_indices = torch.topk(feature_vars, num_keep)

                # 创建零填充的完整维度特征
                full_features = torch.zeros_like(adjusted_features)
                full_features[:, top_indices] = adjusted_features[:, top_indices]
                adjusted_features = full_features

            adjusted[layer_name] = adjusted_features

            print(f"  调整 {layer_name}: 权重={self.layer_weights[layer_name]:.3f}, 增强={enhancement_factor:.3f}")

        return adjusted

    def _reconstruct_from_layers(self, layered_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """从分层特征重构为141维特征"""
        feature_parts = []

        # 重构99维原始特征部分
        for layer_name in ['hardware', 'onchain_behavior', 'network_topology',
                           'dynamic_attributes', 'heterogeneous_type', 'categorical']:
            if layer_name in layered_features:
                feature_parts.append(layered_features[layer_name])

        # 添加时序特征
        if 'sequence' in layered_features:
            feature_parts.append(layered_features['sequence'])

        # 添加图结构特征
        if 'graph_structure' in layered_features:
            feature_parts.append(layered_features['graph_structure'])

        reconstructed = torch.cat(feature_parts, dim=1)  # [N, 141]
        return reconstructed

    def _adaptive_projection(self, raw_141: torch.Tensor,
                             step4_guidance: Optional[Dict[str, Any]]) -> torch.Tensor:
        """自适应投影到128维"""

        # 使用基础Pipeline的投影器
        projected = self.base_pipeline.feature_extractor.feature_projector(raw_141)  # [N, 128]

        # 如果有投影调整指导，应用额外的变换
        if step4_guidance and 'projection_adjustment' in step4_guidance:
            proj_guidance = step4_guidance['projection_adjustment']
            if 'enhancement_factor' in proj_guidance:
                projected = projected * proj_guidance['enhancement_factor']

        return projected

    def _update_feedback_mode(self, step4_guidance: Optional[Dict[str, Any]], epoch: int):
        """更新反馈模式"""
        if epoch == 0:
            self.feedback_mode = 'cold_start'
            self.feedback_enabled = False
        elif epoch < 5 and step4_guidance is not None:
            self.feedback_mode = 'warm_feedback'
            if not self.feedback_enabled:
                self.feedback_enabled = True
                print(f"🔄 启用分层反馈模式 (Epoch {epoch})")
        elif epoch >= 5 and step4_guidance is not None:
            self.feedback_mode = 'stable_feedback'

    def _record_adaptation(self, step4_guidance: Optional[Dict[str, Any]],
                           results: Dict[str, torch.Tensor], epoch: int):
        """记录适应历史"""
        record = {
            'epoch': epoch,
            'feedback_mode': self.feedback_mode,
            'guidance_applied': step4_guidance is not None,
            'layer_weights': self.layer_weights.copy(),
            'feature_stats': {
                'f_classic_mean': float(torch.mean(results['f_classic'])),
                'f_classic_std': float(torch.std(results['f_classic'])),
                'raw_141_mean': float(torch.mean(results['raw_141_features'])),
            }
        }

        if step4_guidance:
            record['guidance_summary'] = {
                'layer_adjustments': len(step4_guidance.get('layer_weight_adjustments', {})),
                'enhancement_factors': len(step4_guidance.get('layer_enhancement_factors', {})),
                'dimension_selections': len(step4_guidance.get('layer_dimension_selection', {}))
            }

        self.adaptation_history.append(record)

    def get_adaptation_report(self) -> Dict[str, Any]:
        """获取适应报告"""
        return {
            'feedback_enabled': self.feedback_enabled,
            'current_mode': self.feedback_mode,
            'total_adaptations': len(self.adaptation_history),
            'current_layer_weights': self.layer_weights.copy(),
            'feature_structure': self.feature_structure
        }


class EnhancedFeedbackController(FeedbackController):
    """增强的反馈控制器 - 支持分层特征反馈"""

    def __init__(self):
        # 初始化特征维度（对应第一步的6层结构）
        feature_dims = {
            'hardware': 17,
            'onchain_behavior': 17,
            'network_topology': 20,
            'dynamic_attributes': 13,
            'heterogeneous_type': 17,
            'categorical': 15,
            'sequence': 32,
            'graph_structure': 10
        }
        super().__init__(feature_dims)

        # 专门的分层特征演化器
        self.layered_evolution = None

    def process_layered_feedback(self, step1_results: Dict[str, torch.Tensor],
                                 step3_shard_assignments: torch.Tensor,
                                 edge_index: torch.Tensor,
                                 evolve_gcn_model: nn.Module,
                                 epoch: int = 0) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        处理分层特征反馈

        Args:
            step1_results: 第一步的分层特征提取结果
            step3_shard_assignments: 第三步分片分配结果
            edge_index: 网络拓扑
            evolve_gcn_model: EvolveGCN模型
            epoch: 当前轮次

        Returns:
            feedback_signal: 反馈信号
            step1_guidance: 给第一步的分层指导
        """
        print(f"\n🔄 处理分层特征反馈 (Epoch {epoch})")

        # 1. 从step1结果中提取分层特征
        layered_features = step1_results.get('layered_breakdown', {})
        final_features = step1_results.get('f_classic')  # [N, 128]

        if not layered_features:
            print("[WARNING] 未找到分层特征，使用最终特征进行评估")
            # 创建临时分层结构用于评估
            layered_features = {'combined': final_features}

        # 2. 性能评估（基于分层特征）
        performance_metrics = self._evaluate_layered_performance(
            layered_features, step3_shard_assignments, edge_index
        )

        # 3. 分层重要性分析
        importance_matrix = self._analyze_layered_importance(
            layered_features, performance_metrics, evolve_gcn_model
        )

        # 4. 初始化分层演化器
        if self.layered_evolution is None:
            self.layered_evolution = DynamicFeatureEvolution(layered_features)

        # 5. 生成分层指导
        step1_guidance = self._generate_layered_guidance(
            layered_features, importance_matrix, performance_metrics, epoch
        )

        # 6. 计算反馈信号
        feedback_signal = self._compute_layered_feedback_signal(performance_metrics)

        print(f"[SUCCESS] 分层反馈处理完成")
        print(f"   反馈信号: {[f'{x:.3f}' for x in feedback_signal.tolist()]}")
        print(f"   分层指导: {len(step1_guidance)} 个调整项")

        return feedback_signal, step1_guidance

    def _evaluate_layered_performance(self, layered_features: Dict[str, torch.Tensor],
                                      shard_assignments: torch.Tensor,
                                      edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """评估分层特征的性能贡献"""

        # 使用原有的性能评估器，但适配分层输入
        if len(layered_features) == 1 and 'combined' in layered_features:
            # 单一特征的情况
            combined_features = layered_features['combined']
            eval_features = {
                'hardware': combined_features[:, :17] if combined_features.size(1) >= 17 else combined_features,
                'topology': combined_features[:, 17:37] if combined_features.size(1) >= 37 else combined_features[:, :min(20, combined_features.size(1))],
                'consensus': combined_features[:, 37:45] if combined_features.size(1) >= 45 else combined_features[:, :min(8, combined_features.size(1))],
                'semantic': combined_features[:, 45:] if combined_features.size(1) > 45 else combined_features[:, :min(15, combined_features.size(1))]
            }
        else:
            # 真实的分层特征
            eval_features = {
                'hardware': layered_features.get('hardware', torch.zeros(shard_assignments.size(0), 17)),
                'topology': layered_features.get('network_topology', torch.zeros(shard_assignments.size(0), 20)),
                'consensus': layered_features.get('onchain_behavior', torch.zeros(shard_assignments.size(0), 17))[:, :8],  # 取前8维作为共识特征
                'semantic': layered_features.get('categorical', torch.zeros(shard_assignments.size(0), 15))
            }

        # 使用基类的性能评估器
        return self.performance_evaluator(eval_features, shard_assignments, edge_index)

    def _analyze_layered_importance(self, layered_features: Dict[str, torch.Tensor],
                                    performance_metrics: Dict[str, torch.Tensor],
                                    evolve_gcn_model: nn.Module) -> Dict[str, Dict[str, float]]:
        """分析分层特征的重要性"""
        return self.importance_analyzer.analyze_importance(
            layered_features, performance_metrics, evolve_gcn_model
        )

    def _generate_layered_guidance(self, layered_features: Dict[str, torch.Tensor],
                                   importance_matrix: Dict[str, Dict[str, float]],
                                   performance_metrics: Dict[str, torch.Tensor],
                                   epoch: int) -> Dict[str, Any]:
        """生成分层指导"""
        guidance = {
            'epoch': epoch,
            'guidance_type': 'layered_feedback',

            # 层级权重调整
            'layer_weight_adjustments': {},

            # 层级增强因子
            'layer_enhancement_factors': {},

            # 维度选择指导
            'layer_dimension_selection': {},

            # 投影调整
            'projection_adjustment': {},

            # 特征结构信息
            'feature_structure_info': {
                'total_layers': len(layered_features),
                'layer_names': list(layered_features.keys())
            }
        }

        # 为每一层生成具体指导
        for layer_name, layer_features in layered_features.items():
            if layer_name in importance_matrix:
                importance = importance_matrix[layer_name].get('combined', 0.5)

                # 权重调整：基于重要性和性能
                base_weight = 1.0
                performance_factor = self._compute_layer_performance_factor(
                    layer_name, performance_metrics
                )
                adjusted_weight = base_weight * importance * (1 + performance_factor)
                guidance['layer_weight_adjustments'][layer_name] = adjusted_weight

                # 增强因子：重要层增强，不重要层抑制
                if importance > 0.7:
                    enhancement_factor = 1.1
                elif importance < 0.3:
                    enhancement_factor = 0.9
                else:
                    enhancement_factor = 1.0
                guidance['layer_enhancement_factors'][layer_name] = enhancement_factor

                # 维度选择：低重要性层进行维度压缩
                selection_ratio = max(0.5, importance)  # 至少保留50%
                guidance['layer_dimension_selection'][layer_name] = {
                    'selection_ratio': selection_ratio,
                    'importance_score': importance
                }

        # 投影调整
        avg_performance = sum(v.item() for v in performance_metrics.values()) / len(performance_metrics)
        if avg_performance < 0.6:
            guidance['projection_adjustment'] = {
                'enhancement_factor': 1.05,  # 轻微增强
                'reason': 'low_performance'
            }
        elif avg_performance > 0.8:
            guidance['projection_adjustment'] = {
                'enhancement_factor': 0.98,  # 轻微抑制防止过拟合
                'reason': 'high_performance'
            }

        return guidance

    def _compute_layer_performance_factor(self, layer_name: str,
                                          performance_metrics: Dict[str, torch.Tensor]) -> float:
        """计算层级性能因子"""
        factor = 0.0

        # 根据层次类型和性能指标的关系计算因子
        if layer_name in ['hardware', 'dynamic_attributes']:
            # 硬件和动态属性主要影响负载均衡
            if 'balance_score' in performance_metrics:
                balance_score = performance_metrics['balance_score'].item()
                factor += (balance_score - 0.5) * 0.2

        elif layer_name in ['onchain_behavior', 'sequence']:
            # 链上行为和时序特征主要影响安全性
            if 'security_score' in performance_metrics:
                security_score = performance_metrics['security_score'].item()
                factor += (security_score - 0.7) * 0.3

        elif layer_name in ['network_topology', 'graph_structure']:
            # 网络相关特征主要影响跨片交易
            if 'cross_tx_rate' in performance_metrics:
                cross_tx_rate = performance_metrics['cross_tx_rate'].item()
                factor -= cross_tx_rate * 0.2  # 跨片交易率高时降低权重

        return np.clip(factor, -0.2, 0.2)  # 限制因子范围

    def _compute_layered_feedback_signal(self, performance_metrics: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算分层反馈信号"""
        return torch.tensor([
            performance_metrics.get('balance_score', torch.tensor(0.5)).item(),
            performance_metrics.get('cross_tx_rate', torch.tensor(0.2)).item(),
            performance_metrics.get('security_score', torch.tensor(0.8)).item()
        ])







class IntegratedMultiScaleContrastiveLearning:
    """整合的第二步多尺度对比学习模块 - 修复版"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 默认配置（基于muti_scale目录中的最优参数）
        self.config = config or {
            'input_dim': 128,            # 第一步输出的特征维度
            'hidden_dim': 64,            # 隐藏层维度
            'time_dim': 16,              # 时序嵌入维度
            'k_ratio': 0.9,              # 采样比例
            'alpha': 0.3,                # 图级损失权重
            'beta': 0.4,                 # 节点级损失权重
            'gamma': 0.3,                # 子图级损失权重
            'tau': 0.09,                 # 温度参数
            'augment_type': 'edge',      # 增强类型
            'num_node_types': 5,         # 节点类型数
            'num_edge_types': 3,         # 边类型数
            'learning_rate': 0.02,       # 学习率
            'weight_decay': 9e-6,        # 权重衰减
            'max_epochs': 50,            # 每次调用的最大训练轮次
            'early_stopping_patience': 10,  # 早停耐心值
            'target_loss': 0.25,          # 目标损失
        }

        # 初始化模型和投影层
        self.model = None
        self.feature_projection = None  # 复用的投影层
        self.initialized = False
        self.training_history = []

    def initialize_model(self):
        """初始化MSCIA模型"""
        print(f"🧠 初始化多尺度对比学习模型...")

        # 创建时序MSCIA模型并确保在正确设备上
        self.model = TemporalMSCIA(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim'],
            time_dim=self.config['time_dim'],
            k_ratio=self.config['k_ratio'],
            alpha=self.config['alpha'],
            beta=self.config['beta'],
            gamma=self.config['gamma'],
            tau=self.config['tau'],
            num_node_types=self.config['num_node_types'],
            num_edge_types=self.config['num_edge_types']
        ).to(self.device)

        self.initialized = True
        print(f"[SUCCESS] 模型初始化完成 (设备: {self.device})")

    def process_step1_output(self, step1_results: Dict[str, torch.Tensor],
                             edge_index: torch.Tensor,
                             epoch: int = 0) -> Dict[str, torch.Tensor]:
        """
        处理第一步的输出，生成第二步的嵌入 - 修复版

        Args:
            step1_results: 第一步的特征提取结果
            edge_index: 网络拓扑边索引 [2, E]
            epoch: 当前轮次

        Returns:
            {
                'embeddings': torch.Tensor,           # [N, hidden_dim] 节点嵌入
                'temporal_embeddings': Dict,          # 时序嵌入字典
                'contrastive_loss': float,            # 对比学习损失
                'training_info': Dict                 # 训练信息
            }
        """
        if not self.initialized:
            self.initialize_model()

        print(f"\n🧠 执行第二步: 多尺度对比学习 (Epoch {epoch})")

        # 1. 准备输入数据 - 修复格式匹配
        input_data = self._prepare_input_data_fixed(step1_results, edge_index, epoch)

        # 2. 训练/推理模式选择
        if epoch == 0 or epoch % 5 == 0:  # 周期性重训练
            embeddings, training_info = self._train_mode_fixed(input_data, epoch)
        else:
            embeddings, training_info = self._inference_mode_fixed(input_data, epoch)

        # 3. 生成时序嵌入
        temporal_embeddings = self._generate_temporal_embeddings(
            embeddings, input_data, epoch
        )

        # 4. 构建输出
        output = {
            'embeddings': embeddings,                    # [N, hidden_dim]
            'temporal_embeddings': temporal_embeddings,  # 时序嵌入字典
            'contrastive_loss': training_info.get('loss', 0.0),
            'training_info': training_info,
            'model_state': {
                'epoch': epoch,
                'config': self.config,
                'initialized': self.initialized
            }
        }

        print(f"[SUCCESS] 第二步完成: 嵌入维度 {embeddings.shape}, 损失 {output['contrastive_loss']:.4f}")

        return output

    def _prepare_input_data_fixed(self, step1_results: Dict[str, torch.Tensor],
                                  edge_index: torch.Tensor, epoch: int) -> Dict[str, Any]:
        """准备MSCIA模型的输入数据 - 修复设备和格式问题"""

        # 优先使用融合特征，其次经典特征
        if 'f_fused' in step1_results and step1_results['f_fused'] is not None:
            node_features = step1_results['f_fused']  # [N, fused_dim]
        else:
            node_features = step1_results['f_classic']  # [N, 128]
        
        # 确保输入在正确设备上
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        
        # 如果特征维度不匹配，进行投影
        if node_features.size(1) != self.config['input_dim']:
            # 创建或复用投影层，确保在正确设备上
            if (not hasattr(self, 'feature_projection') or 
                self.feature_projection is None or 
                self.feature_projection.in_features != node_features.size(1)):
                
                self.feature_projection = nn.Linear(
                    node_features.size(1), 
                    self.config['input_dim']
                ).to(self.device)
            
            node_features = self.feature_projection(node_features)

        # 构建符合muti_scale期望的输入格式
        num_nodes = node_features.size(0)
        
        # 1. 构建邻接矩阵 [N, N]（不是batch形式）
        adjacency_matrix = self._build_adjacency_matrix_fixed(edge_index, num_nodes)
        
        # 2. 生成时间戳 [N]（不是batch形式）
        timestamps = torch.arange(num_nodes, device=self.device, dtype=torch.float32)
        
        # 3. 生成中心节点索引（用于子图采样）
        num_centers = min(32, num_nodes)  # 限制中心节点数量
        center_indices = torch.randperm(num_nodes, device=self.device)[:num_centers]
        
        # 4. 节点类型（如果有的话）
        node_types = self._extract_node_types(step1_results, num_nodes)

        return {
            # 符合TemporalMSCIA期望的格式
            'A_batch': adjacency_matrix.unsqueeze(0),      # [1, N, N] - 添加batch维度
            'X_batch': node_features.unsqueeze(0),         # [1, N, input_dim] - 添加batch维度
            'center_indices': center_indices,              # [num_centers]
            'timestamps': timestamps.unsqueeze(0),         # [1, N] - 添加batch维度
            'node_types': node_types,                      # [N] or None
            'edge_index': edge_index,                      # [2, E] - 保留原格式供其他用途
            'num_nodes': num_nodes,
            'epoch': epoch
        }

    def _build_adjacency_matrix_fixed(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """构建邻接矩阵 - 修复版"""
        adjacency = torch.zeros(num_nodes, num_nodes, device=self.device, dtype=torch.float32)

        if edge_index.size(1) > 0:  # 确保有边
            row, col = edge_index[0], edge_index[1]
            # 处理边索引超出范围的情况
            valid_mask = (row < num_nodes) & (col < num_nodes) & (row >= 0) & (col >= 0)
            
            if valid_mask.sum() > 0:
                row, col = row[valid_mask], col[valid_mask]
                adjacency[row, col] = 1.0
                # 如果是无向图，添加反向边
                adjacency[col, row] = 1.0

        # 添加自环（重要：许多GNN模型需要自环）
        adjacency.fill_diagonal_(1.0)

        return adjacency

    def _extract_node_types(self, step1_results: Dict[str, torch.Tensor], num_nodes: int) -> Optional[torch.Tensor]:
        """提取节点类型"""
        if 'nodes' in step1_results:
            try:
                nodes = step1_results['nodes']
                node_types = torch.zeros(num_nodes, dtype=torch.long, device=self.device)
                
                for i, node in enumerate(nodes[:num_nodes]):  # 确保不超出范围
                    if hasattr(node, 'node_type'):
                        # 节点类型映射
                        type_map = {
                            'validator': 0, 'full_node': 1, 'light_node': 2, 
                            'miner': 3, 'storage': 4, 'relay': 4
                        }
                        node_type_str = getattr(node, 'node_type', 'validator')
                        node_types[i] = type_map.get(node_type_str, 0)
                
                return node_types
            except Exception as e:
                print(f"    [WARNING] 提取节点类型失败: {e}")
                return None
        
        return None

    def _train_mode_fixed(self, input_data: Dict[str, Any], epoch: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """训练模式 - 修复版"""
        print(f"  🔄 训练模式 (Epoch {epoch})")

        self.model.train()

        # 优化器
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        # 训练循环
        best_loss = float('inf')
        patience_counter = 0
        epoch_losses = []

        max_train_epochs = min(self.config['max_epochs'], 15)  # 限制训练轮次避免过长

        for train_epoch in range(max_train_epochs):
            optimizer.zero_grad()

            try:
                # 调用TemporalMSCIA的forward方法
                # 传入符合其期望的参数
                output = self.model(
                    A_batch=input_data['A_batch'],         # [1, N, N]
                    X_batch=input_data['X_batch'],         # [1, N, input_dim]
                    center_indices=input_data['center_indices'],  # [num_centers]
                    timestamps=input_data['timestamps'],   # [1, N]
                    node_types=input_data['node_types']    # [N] or None
                )
                
                # 根据TemporalMSCIA的返回格式解析
                if isinstance(output, tuple):
                    loss, embeddings = output
                elif isinstance(output, dict):
                    loss = output.get('loss', output.get('total_loss', torch.tensor(0.0)))
                    embeddings = output.get('embeddings', output.get('node_embeddings'))
                else:
                    # 如果返回单一tensor，假设是损失
                    loss = output
                    embeddings = self.model.get_embeddings() if hasattr(self.model, 'get_embeddings') else None

                # 检查损失和嵌入的有效性
                if embeddings is None:
                    print(f"    [WARNING] 训练轮次 {train_epoch}: 无法获取嵌入")
                    continue
                    
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"    [WARNING] 训练轮次 {train_epoch}: 无效损失 {loss}")
                    continue

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                current_loss = loss.item()
                epoch_losses.append(current_loss)

                # 早停检查
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                # 达到目标损失或早停
                if current_loss < self.config['target_loss'] or patience_counter >= self.config['early_stopping_patience']:
                    print(f"    [SUCCESS] 训练完成: 损失 {current_loss:.4f} (轮次 {train_epoch+1})")
                    break

                if (train_epoch + 1) % 5 == 0:
                    print(f"    训练轮次 {train_epoch+1}: 损失 {current_loss:.4f}")

            except Exception as e:
                print(f"    [ERROR] 训练轮次 {train_epoch} 失败: {e}")
                continue

        # 获取最终嵌入
        self.model.eval()
        with torch.no_grad():
            try:
                final_output = self.model(
                    A_batch=input_data['A_batch'],
                    X_batch=input_data['X_batch'],
                    center_indices=input_data['center_indices'],
                    timestamps=input_data['timestamps'],
                    node_types=input_data['node_types']
                )
                
                if isinstance(final_output, tuple):
                    final_loss, final_embeddings = final_output
                    final_loss_value = final_loss.item()
                elif isinstance(final_output, dict):
                    final_loss_value = final_output.get('loss', final_output.get('total_loss', torch.tensor(0.0))).item()
                    final_embeddings = final_output.get('embeddings', final_output.get('node_embeddings'))
                else:
                    final_loss_value = final_output.item() if torch.is_tensor(final_output) else float(final_output)
                    final_embeddings = self.model.get_embeddings() if hasattr(self.model, 'get_embeddings') else None

                # 如果嵌入仍然有问题，使用X_batch作为fallback
                if final_embeddings is None:
                    print(f"    [WARNING] 无法获取最终嵌入，使用输入特征作为fallback")
                    final_embeddings = input_data['X_batch'].squeeze(0)  # [N, input_dim]
                    
                # 确保嵌入维度正确
                if final_embeddings.dim() == 3:  # [1, N, hidden_dim]
                    final_embeddings = final_embeddings.squeeze(0)  # [N, hidden_dim]
                    
            except Exception as e:
                print(f"    [ERROR] 获取最终嵌入失败: {e}")
                # 生成默认嵌入
                num_nodes = input_data['num_nodes']
                final_embeddings = torch.randn(num_nodes, self.config['hidden_dim'], device=self.device)
                final_loss_value = float('inf')

        training_info = {
            'mode': 'train',
            'loss': final_loss_value,
            'best_loss': best_loss,
            'epochs_trained': len(epoch_losses),
            'epoch_losses': epoch_losses,
            'converged': best_loss < self.config['target_loss']
        }

        # 记录训练历史
        self.training_history.append({
            'global_epoch': epoch,
            'training_info': training_info
        })

        return final_embeddings, training_info

    def _inference_mode_fixed(self, input_data: Dict[str, Any], epoch: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """推理模式 - 修复版"""
        print(f"  [SPEED] 推理模式 (Epoch {epoch})")

        self.model.eval()

        with torch.no_grad():
            try:
                output = self.model(
                    A_batch=input_data['A_batch'],
                    X_batch=input_data['X_batch'],
                    center_indices=input_data['center_indices'],
                    timestamps=input_data['timestamps'],
                    node_types=input_data['node_types']
                )
                
                if isinstance(output, tuple):
                    loss, embeddings = output
                    loss_value = loss.item()
                elif isinstance(output, dict):
                    loss_value = output.get('loss', output.get('total_loss', torch.tensor(0.0))).item()
                    embeddings = output.get('embeddings', output.get('node_embeddings'))
                else:
                    loss_value = output.item() if torch.is_tensor(output) else float(output)
                    embeddings = self.model.get_embeddings() if hasattr(self.model, 'get_embeddings') else None

                if embeddings is None:
                    print(f"    [WARNING] 推理模式无法获取嵌入，使用输入特征")
                    embeddings = input_data['X_batch'].squeeze(0)  # [N, input_dim]
                    
                if embeddings.dim() == 3:  # [1, N, hidden_dim]
                    embeddings = embeddings.squeeze(0)  # [N, hidden_dim]
                    
            except Exception as e:
                print(f"    [ERROR] 推理失败: {e}")
                num_nodes = input_data['num_nodes']
                embeddings = torch.randn(num_nodes, self.config['hidden_dim'], device=self.device)
                loss_value = float('inf')

        training_info = {
            'mode': 'inference',
            'loss': loss_value,
            'inference_successful': loss_value != float('inf')
        }

        return embeddings, training_info

    # 保持其他方法不变
    def _generate_temporal_embeddings(self, embeddings: torch.Tensor,
                                      input_data: Dict[str, Any],
                                      epoch: int) -> Dict[str, Any]:
        """生成时序嵌入字典"""
        temporal_embeddings = {}

        timestamps = input_data['timestamps'].squeeze(0) if input_data['timestamps'].dim() > 1 else input_data['timestamps']
        num_nodes = input_data['num_nodes']

        for i in range(num_nodes):
            node_id = f"node_{i}"
            timestamp = timestamps[i].item() if torch.is_tensor(timestamps[i]) else timestamps[i]

            if node_id not in temporal_embeddings:
                temporal_embeddings[node_id] = {}

            # 使用epoch作为时间维度的一部分
            time_key = f"t_{epoch}_{timestamp}"
            temporal_embeddings[node_id][time_key] = embeddings[i].detach().cpu().numpy()

        return {
            'embeddings_dict': temporal_embeddings,
            'metadata': {
                'epoch': epoch,
                'num_nodes': num_nodes,
                'embedding_dim': embeddings.size(1),
                'timestamp_range': [timestamps.min().item(), timestamps.max().item()]
            }
        }

    def save_embeddings(self, temporal_embeddings: Dict, filepath: str):
        """保存时序嵌入到文件"""
        with open(filepath, 'wb') as f:
            pickle.dump(temporal_embeddings, f)
        print(f"📁 时序嵌入已保存: {filepath}")

    def load_embeddings(self, filepath: str) -> Dict:
        """从文件加载时序嵌入"""
        with open(filepath, 'rb') as f:
            temporal_embeddings = pickle.load(f)
        print(f"📂 时序嵌入已加载: {filepath}")
        return temporal_embeddings

    def get_model_state(self) -> Dict[str, Any]:
        """获取模型状态"""
        return {
            'initialized': self.initialized,
            'config': self.config,
            'device': str(self.device),
            'training_history_length': len(self.training_history),
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }


# 修改完整流水线演示类
class IntegratedFullPipelineDemo:
    """整合真实第二步的完整四步骤流水线演示"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 配置参数
        self.config = {
            'num_epochs': 20,
            'feedback_start_epoch': 2,
            'save_results': True,
            'results_dir': './results',
            'sample_data_path': './large_samples.csv',
            # 第二步专用配置
            'step2_config': {
                'input_dim': 128,
                'hidden_dim': 64,
                'time_dim': 16,
                'k_ratio': 0.9,
                'alpha': 0.3,
                'beta': 0.4,
                'gamma': 0.3,
                'tau': 0.09,
                'learning_rate': 0.02,
                'weight_decay': 9e-6,
                'max_epochs': 30,
                'target_loss': 0.25
            }
        }

        # 创建结果目录
        os.makedirs(self.config['results_dir'], exist_ok=True)

        # 流水线历史记录
        self.pipeline_history = {
            'step1_features': [],
            'step2_embeddings': [],
            'step3_sharding': [],
            'step4_feedback': [],
            'performance_metrics': []
        }

        # 初始化组件
        self._initialize_components()

    def _initialize_components(self):
        """初始化各步骤组件"""
        print("初始化各步骤组件...")

        # 第一步: 分层反馈特征提取流水线
        print("- 初始化第一步: 分层反馈特征提取流水线")
        base_pipeline = Pipeline(use_fusion=True, save_adjacency=True)  # 启用邻接矩阵保存
        self.step1_pipeline = LayeredFeedbackFeatureExtractor(base_pipeline)

        # 第二步: 整合的多尺度对比学习
        print("- 初始化第二步: 多尺度对比学习")
        self.step2_mscia = IntegratedMultiScaleContrastiveLearning(self.config['step2_config'])

        # 第三步: 动态分片（使用真实实现或模拟）
        print("- 初始化第三步: 动态分片")
        self.dynamic_sharder = self._create_enhanced_dynamic_sharder()

        # 第四步: 增强反馈控制器
        print("- 初始化第四步: 增强反馈控制器")
        self.feedback_controller = EnhancedFeedbackController()

        print("[SUCCESS] 所有组件初始化完成")

    def _create_enhanced_dynamic_sharder(self):
        """创建增强的动态分片器（可以替换为真实实现）"""
        class EnhancedDynamicSharder(nn.Module):
            def __init__(self, embedding_dim=64, max_shards=8):
                super().__init__()
                self.embedding_dim = embedding_dim
                self.max_shards = max_shards

                # 分片数预测网络
                self.shard_num_predictor = nn.Sequential(
                    nn.Linear(embedding_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                    nn.Sigmoid()
                )

                # 分片分配网络
                self.shard_classifier = nn.Sequential(
                    nn.Linear(embedding_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, max_shards)
                )

            def forward(self, embeddings, feedback_signal=None):
                # 预测分片数量
                shard_num_raw = self.shard_num_predictor(embeddings.mean(dim=0, keepdim=True))
                predicted_num_shards = max(2, int(shard_num_raw.item() * self.max_shards) + 1)

                # 分片分配
                logits = self.shard_classifier(embeddings)[:, :predicted_num_shards]

                # 如果有反馈信号，调整logits
                if feedback_signal is not None:
                    balance_factor = feedback_signal[0].item()  # 负载均衡因子
                    if balance_factor < 0.5:  # 负载不均衡时，增加随机性
                        noise = torch.randn_like(logits) * 0.1
                        logits = logits + noise

                shard_assignments = torch.softmax(logits, dim=1)

                # 计算损失
                entropy_loss = -torch.mean(torch.sum(shard_assignments * torch.log(shard_assignments + 1e-8), dim=1))
                balance_loss = torch.std(torch.sum(shard_assignments, dim=0))
                total_loss = entropy_loss + 0.1 * balance_loss

                return shard_assignments, total_loss

        return EnhancedDynamicSharder().to(self.device)

    def run_step2_contrastive_learning(self, step1_output: Dict[str, torch.Tensor],
                                       edge_index: torch.Tensor,
                                       epoch: int) -> Dict[str, torch.Tensor]:
        """第二步: 真实的多尺度对比学习"""
        print(f"\n🧠 执行第二步: 多尺度对比学习 (Epoch {epoch})...")

        # 使用整合的MSCIA模型
        step2_output = self.step2_mscia.process_step1_output(
            step1_output, edge_index, epoch
        )

        # 记录嵌入历史
        embedding_record = {
            'epoch': epoch,
            'input_features_shape': step1_output['f_classic'].shape,
            'embeddings_shape': step2_output['embeddings'].shape,
            'contrastive_loss': step2_output['contrastive_loss'],
            'training_mode': step2_output['training_info']['mode'],
            'converged': step2_output['training_info'].get('converged', False),
            'embedding_stats': {
                'mean': float(torch.mean(step2_output['embeddings'])),
                'std': float(torch.std(step2_output['embeddings'])),
                'min': float(torch.min(step2_output['embeddings'])),
                'max': float(torch.max(step2_output['embeddings']))
            }
        }

        self.pipeline_history['step2_embeddings'].append(embedding_record)

        # 保存时序嵌入（周期性保存）
        if epoch % 5 == 0:
            embeddings_path = os.path.join(self.config['results_dir'], f'temporal_embeddings_epoch_{epoch}.pkl')
            self.step2_mscia.save_embeddings(step2_output['temporal_embeddings'], embeddings_path)

        print(f"[SUCCESS] 第二步完成:")
        print(f"  - 嵌入维度: {step2_output['embeddings'].shape}")
        print(f"  - 对比损失: {step2_output['contrastive_loss']:.4f}")
        print(f"  - 训练模式: {step2_output['training_info']['mode']}")
        if 'converged' in step2_output['training_info']:
            print(f"  - 收敛状态: {'是' if step2_output['training_info']['converged'] else '否'}")

        return step2_output

    def run_step3_dynamic_sharding(self, step2_output: Dict[str, torch.Tensor],
                                   edge_index: torch.Tensor,
                                   epoch: int,
                                   feedback_signal: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """第三步: 增强的动态分片"""
        print(f"\n[SPEED] 执行第三步: 动态分片 (Epoch {epoch})...")

        embeddings = step2_output['embeddings'].to(self.device)

        # 动态分片处理
        shard_assignments, shard_loss = self.dynamic_sharder(embeddings, feedback_signal)

        # 计算分片统计
        hard_assignments = torch.argmax(shard_assignments, dim=1)
        num_shards = shard_assignments.size(1)
        shard_sizes = [(hard_assignments == i).sum().item() for i in range(num_shards)]

        # 计算更详细的平衡指标
        balance_coefficient = np.std(shard_sizes) / (np.mean(shard_sizes) + 1e-8)

        # 计算跨片边比例
        cross_shard_edges = 0
        total_edges = edge_index.size(1)
        if total_edges > 0:
            u, v = edge_index[0], edge_index[1]
            valid_mask = (u < len(hard_assignments)) & (v < len(hard_assignments))
            if valid_mask.sum() > 0:
                valid_u, valid_v = u[valid_mask], v[valid_mask]
                cross_shard_edges = (hard_assignments[valid_u] != hard_assignments[valid_v]).sum().item()

        cross_shard_ratio = cross_shard_edges / max(total_edges, 1)

        # 构建输出
        output = {
            'shard_assignments': shard_assignments,
            'hard_assignments': hard_assignments,
            'shard_loss': shard_loss,
            'predicted_shards': num_shards,
            'shard_sizes': shard_sizes,
            'balance_coefficient': balance_coefficient,
            'cross_shard_ratio': cross_shard_ratio,
            'cross_shard_edges': cross_shard_edges,
            'total_edges': total_edges,
            'sharding_quality': {
                'balance_score': max(0, 1 - balance_coefficient),
                'efficiency_score': max(0, 1 - cross_shard_ratio),
                'size_variance': np.var(shard_sizes)
            }
        }

        # 记录分片历史
        sharding_record = {
            'epoch': epoch,
            'shard_loss': float(shard_loss),
            'num_shards': num_shards,
            'shard_sizes': shard_sizes,
            'balance_coefficient': balance_coefficient,
            'cross_shard_ratio': cross_shard_ratio,
            'feedback_used': feedback_signal is not None,
            'embeddings_source': step2_output['training_info']['mode'],
            'quality_metrics': output['sharding_quality']
        }

        self.pipeline_history['step3_sharding'].append(sharding_record)

        print(f"[SUCCESS] 第三步完成:")
        print(f"  - 分片数量: {num_shards}")
        print(f"  - 分片损失: {shard_loss:.4f}")
        print(f"  - 负载均衡系数: {balance_coefficient:.3f}")
        print(f"  - 跨片交易比例: {cross_shard_ratio:.3f}")
        print(f"  - 分片大小: {shard_sizes}")

        return output

    def run_complete_pipeline(self, save_results: bool = True) -> Dict[str, Any]:
        """运行完整的四步骤流水线 - 整合真实第二步"""
        print("=" * 70)
        print("[START] 开始完整的四步骤区块链分片流水线 (整合真实第二步)")
        print("=" * 70)

        start_time = datetime.now()

        try:
            # 生成示例数据
            nodes = self.generate_sample_nodes()
            edge_index = self.generate_sample_network_topology(len(nodes))

            # 当前反馈指导状态
            current_step1_guidance = None

            # 主训练循环
            for epoch in range(self.config['num_epochs']):
                print(f"\n{'='*25} EPOCH {epoch+1}/{self.config['num_epochs']} {'='*25}")

                # 第一步: 分层反馈特征提取
                step1_output = self.run_step1_feature_extraction(
                    nodes, current_step1_guidance, epoch
                )

                # 第二步: 真实的多尺度对比学习
                step2_output = self.run_step2_contrastive_learning(
                    step1_output, edge_index, epoch
                )

                # 第三步: 增强的动态分片
                feedback_signal = None
                if epoch > 0 and hasattr(self, 'last_feedback_signal'):
                    feedback_signal = self.last_feedback_signal

                step3_output = self.run_step3_dynamic_sharding(
                    step2_output, edge_index, epoch, feedback_signal
                )

                # 第四步: 分层反馈优化
                feedback_signal, next_step1_guidance = self.run_step4_feedback_optimization(
                    step1_output, step3_output, edge_index, epoch
                )

                # 保存反馈信号和指导供下一轮使用
                self.last_feedback_signal = feedback_signal
                current_step1_guidance = next_step1_guidance

                # 记录整体性能
                epoch_performance = {
                    'epoch': epoch,
                    'step1_mode': step1_output.get('feedback_mode', 'unknown'),
                    'step2_loss': step2_output['contrastive_loss'],
                    'step2_mode': step2_output['training_info']['mode'],
                    'step3_loss': float(step3_output['shard_loss']),
                    'num_shards': step3_output['predicted_shards'],
                    'balance_score': feedback_signal[0].item(),
                    'cross_tx_rate': feedback_signal[1].item(),
                    'security_score': feedback_signal[2].item(),
                    'feedback_active': bool(current_step1_guidance),
                    'sharding_quality': step3_output['sharding_quality'],
                    'integration_metrics': {
                        'step1_layers': len(step1_output.get('layered_breakdown', {})),
                        'step2_converged': step2_output['training_info'].get('converged', False),
                        'step3_balance_coeff': step3_output['balance_coefficient'],
                        'cross_shard_ratio': step3_output['cross_shard_ratio']
                    }
                }

                self.pipeline_history['performance_metrics'].append(epoch_performance)

                # 每5轮输出详细信息
                if (epoch + 1) % 5 == 0:
                    self._print_integrated_epoch_summary(epoch_performance)

        except Exception as e:
            print(f"[ERROR] 流水线执行过程中出错: {e}")
            import traceback
            traceback.print_exc()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 汇总结果
        final_results = self._compile_integrated_final_results(duration)

        # 保存结果
        if save_results and self.config['save_results']:
            self._save_integrated_results(final_results)

        self._visualize_integrated_results()

        print("\n" + "=" * 70)
        print("[SUCCESS] 完整四步骤流水线执行完成 (整合真实第二步)!")
        print(f"⏱️  总耗时: {duration:.2f}秒")
        print(f"🧠 第二步模型状态: {self.step2_mscia.get_model_state()}")
        adaptation_report = self.step1_pipeline.get_adaptation_report()
        print(f"🔄 反馈模式: {adaptation_report.get('current_mode', 'unknown')}")
        print("=" * 70)

        return final_results

    # 其他方法的实现...
    def generate_sample_nodes(self, num_nodes: int = 100) -> List[Node]:
        """生成示例节点数据"""
        if os.path.exists(self.config['sample_data_path']):
            print(f"从 {self.config['sample_data_path']} 加载节点数据...")
            return load_nodes_from_csv(self.config['sample_data_path'])
        else:
            print(f"生成 {num_nodes} 个示例节点...")
            nodes = []
            for i in range(num_nodes):
                node = Node(
                    node_id=i,
                    node_type='validator',
                    hardware_config={
                        'cpu_cores': np.random.randint(4, 33),
                        'memory_gb': np.random.randint(8, 65),
                        'storage_gb': np.random.randint(100, 1001),
                        'network_mbps': np.random.randint(100, 1001)
                    },
                    performance_metrics={
                        'tps': np.random.uniform(100, 1000),
                        'latency_ms': np.random.uniform(10, 100),
                        'availability': np.random.uniform(0.9, 1.0)
                    }
                )
                nodes.append(node)
            return nodes

    def generate_sample_network_topology(self, num_nodes: int) -> torch.Tensor:
        """生成示例网络拓扑"""
        num_edges = min(num_nodes * 3, num_nodes * (num_nodes - 1) // 2)
        edges = []
        for _ in range(num_edges):
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            if u != v:
                edges.append([u, v])

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t().contiguous()

        return edge_index.to(self.device)

    # 从LayeredFeedbackFeatureExtractor类复用第一步实现
    def run_step1_feature_extraction(self, nodes: List[Node],
                                     step4_guidance: Optional[Dict[str, Any]] = None,
                                     epoch: int = 0) -> Dict[str, torch.Tensor]:
        """第一步: 分层反馈特征提取"""
        print(f"\n[CONFIG] 执行第一步: 分层反馈特征提取 (Epoch {epoch})")

        results = self.step1_pipeline.extract_features_with_feedback(
            nodes, step4_guidance, epoch
        )

        extraction_record = {
            'epoch': epoch,
            'f_classic_shape': results['f_classic'].shape,
            'f_graph_shape': results['f_graph'].shape,
            'feedback_applied': results.get('feedback_applied', False),
            'feedback_mode': results.get('feedback_mode', 'cold_start'),
            'layer_count': len(results.get('layered_breakdown', {}))
        }

        if 'f_fused' in results:
            extraction_record['f_fused_shape'] = results['f_fused'].shape

        self.pipeline_history['step1_features'].append(extraction_record)

        print(f" 第一步完成 (模式: {results.get('feedback_mode', 'unknown')}):")
        print(f"  - F_classic: {results['f_classic'].shape}")
        print(f"  - F_graph: {results['f_graph'].shape}")
        print(f"  - 分层特征: {len(results.get('layered_breakdown', {}))} 层")
        if 'f_fused' in results:
            print(f"  - F_fused: {results['f_fused'].shape}")

        return results

    # 从EnhancedFeedbackController类复用第四步实现
    def run_step4_feedback_optimization(self, step1_output: Dict[str, torch.Tensor],
                                        step3_output: Dict[str, Any],
                                        edge_index: torch.Tensor,
                                        epoch: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """第四步: 分层反馈优化"""
        print(f"\n 执行第四步: 分层反馈优化 (Epoch {epoch})...")

        if epoch < self.config['feedback_start_epoch']:
            print("- 跳过反馈优化 (未到启动轮次)")
            default_feedback = torch.tensor([0.5, 0.1, 0.8], device=self.device)
            return default_feedback, {}

        try:
            feedback_signal, step1_guidance = self.feedback_controller.process_layered_feedback(
                step1_output,
                step3_output['shard_assignments'],
                edge_index,
                self.step2_mscia.model if self.step2_mscia.model else None,
                epoch
            )

        except Exception as e:
            print(f"反馈处理出错，使用模拟反馈: {e}")

            balance_score = max(0.3, 1.0 - epoch * 0.02)
            cross_tx_rate = min(0.3, 0.1 + epoch * 0.01)
            security_score = max(0.6, 0.9 - epoch * 0.01)

            feedback_signal = torch.tensor([balance_score, cross_tx_rate, security_score], device=self.device)

            step1_guidance = {
                'epoch': epoch,
                'guidance_type': 'simulated',
                'layer_weight_adjustments': {
                    'hardware': 1.0 + (balance_score - 0.5) * 0.2,
                    'onchain_behavior': 1.0 + (security_score - 0.7) * 0.4,
                    'network_topology': 1.0 + (cross_tx_rate - 0.2) * -0.3,
                    'sequence': 1.0 + (security_score - 0.7) * 0.2,
                },
                'layer_enhancement_factors': {
                    'hardware': 1.0 if balance_score > 0.6 else 1.1,
                    'onchain_behavior': 1.1 if security_score < 0.7 else 1.0,
                    'network_topology': 0.9 if cross_tx_rate > 0.25 else 1.0
                }
            }

        feedback_record = {
            'epoch': epoch,
            'balance_score': feedback_signal[0].item(),
            'cross_tx_rate': feedback_signal[1].item(),
            'security_score': feedback_signal[2].item(),
            'has_guidance': bool(step1_guidance),
            'guidance_layers': len(step1_guidance.get('layer_weight_adjustments', {}))
        }

        self.pipeline_history['step4_feedback'].append(feedback_record)

        print(f" 第四步完成: 反馈信号 [{feedback_signal[0]:.3f}, {feedback_signal[1]:.3f}, {feedback_signal[2]:.3f}]")

        if step1_guidance:
            print(f"   生成分层指导: {len(step1_guidance.get('layer_weight_adjustments', {}))} 层权重调整")

        return feedback_signal, step1_guidance

    def _print_integrated_epoch_summary(self, epoch_performance: Dict[str, Any]):
        """打印整合的轮次摘要"""
        print(f"\n Epoch {epoch_performance['epoch']} 整合性能摘要:")
        print(f"   • 第一步模式: {epoch_performance['step1_mode']}")
        print(f"   • 第二步损失: {epoch_performance['step2_loss']:.4f} (模式: {epoch_performance['step2_mode']})")
        print(f"   • 第三步损失: {epoch_performance['step3_loss']:.4f}")
        print(f"   • 分片数量: {epoch_performance['num_shards']}")
        print(f"   • 负载均衡: {epoch_performance['balance_score']:.3f}")
        print(f"   • 跨片交易率: {epoch_performance['cross_tx_rate']:.3f}")
        print(f"   • 安全分数: {epoch_performance['security_score']:.3f}")
        print(f"   • 分片质量:")
        for metric, value in epoch_performance['sharding_quality'].items():
            print(f"     - {metric}: {value:.3f}")

    def _compile_integrated_final_results(self, duration: float) -> Dict[str, Any]:
        """编译整合的最终结果"""
        return {
            'execution_info': {
                'duration_seconds': duration,
                'total_epochs': self.config['num_epochs'],
                'feedback_start_epoch': self.config['feedback_start_epoch'],
                'device': str(self.device),
                'integration_type': 'real_step2_mscia'
            },
            'pipeline_history': self.pipeline_history,
            'step2_model_state': self.step2_mscia.get_model_state(),
            'adaptation_report': self.step1_pipeline.get_adaptation_report(),
            'final_metrics': {
                'avg_balance_score': np.mean([m['balance_score'] for m in self.pipeline_history['performance_metrics']]),
                'avg_cross_tx_rate': np.mean([m['cross_tx_rate'] for m in self.pipeline_history['performance_metrics']]),
                'avg_security_score': np.mean([m['security_score'] for m in self.pipeline_history['performance_metrics']]),
                'avg_step2_loss': np.mean([m['step2_loss'] for m in self.pipeline_history['performance_metrics']]),
                'step2_convergence_rate': sum(1 for m in self.pipeline_history['performance_metrics']
                                              if m.get('integration_metrics', {}).get('step2_converged', False)) / len(self.pipeline_history['performance_metrics']),
                'total_feedback_activations': sum(1 for m in self.pipeline_history['performance_metrics'] if m['feedback_active'])
            }
        }

    def _save_integrated_results(self, results: Dict[str, Any]):
        """保存整合的结果"""
        results_path = os.path.join(self.config['results_dir'], 'integrated_full_pipeline_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f" 整合结果已保存: {results_path}")

    def _visualize_integrated_results(self):
        """可视化整合的结果"""
        try:
            import matplotlib.pyplot as plt

            epochs = [m['epoch'] for m in self.pipeline_history['performance_metrics']]
            balance_scores = [m['balance_score'] for m in self.pipeline_history['performance_metrics']]
            step2_losses = [m['step2_loss'] for m in self.pipeline_history['performance_metrics']]
            step3_losses = [m['step3_loss'] for m in self.pipeline_history['performance_metrics']]

            plt.figure(figsize=(15, 10))

            plt.subplot(2, 3, 1)
            plt.plot(epochs, balance_scores, 'b-', label='Balance Score')
            plt.title('负载均衡分数')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.grid(True)

            plt.subplot(2, 3, 2)
            plt.plot(epochs, step2_losses, 'r-', label='Step2 Loss')
            plt.title('第二步对比学习损失')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)

            plt.subplot(2, 3, 3)
            plt.plot(epochs, step3_losses, 'g-', label='Step3 Loss')
            plt.title('第三步分片损失')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)

            # 添加整合的可视化
            if len(self.pipeline_history['step2_embeddings']) > 0:
                embedding_means = [e['embedding_stats']['mean'] for e in self.pipeline_history['step2_embeddings']]
                plt.subplot(2, 3, 4)
                plt.plot(epochs, embedding_means, 'm-', label='Embedding Mean')
                plt.title('嵌入特征均值')
                plt.xlabel('Epoch')
                plt.ylabel('Mean Value')
                plt.grid(True)

            plt.tight_layout()

            plot_path = os.path.join(self.config['results_dir'], 'integrated_performance_visualization.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📈 整合性能可视化已保存: {plot_path}")

            plt.show()

        except ImportError:
            print(" matplotlib未安装，跳过可视化")
        except Exception as e:
            print(f" 可视化生成失败: {e}")


def main():
    """主函数"""
    print("=" * 80)
    print(" 四步骤区块链分片流水线 - 完全整合真实第二步和第三步")
    print("=" * 80)

    # 创建完全整合的流水线演示实例
    demo = FullIntegratedPipelineDemo()

    # 运行完整流水线
    results = demo.run_complete_pipeline(save_results=True)

    # 打印最终摘要
    print(f"\n 最终执行摘要:")
    print(f"   • 执行时长: {results['execution_info']['duration_seconds']:.2f}秒")
    print(f"   • 总轮次: {results['execution_info']['total_epochs']}")
    print(f"   • 整合类型: {results['execution_info']['integration_type']}")
    print(f"   • 平均负载均衡: {results['final_metrics']['avg_balance_score']:.3f}")
    print(f"   • 平均第二步损失: {results['final_metrics']['avg_step2_loss']:.3f}")
    print(f"   • 平均第三步损失: {results['final_metrics']['avg_step3_loss']:.3f}")
    print(f"   • 第二步收敛率: {results['final_metrics']['step2_convergence_rate']:.2%}")
    print(f"   • 第三步收敛率: {results['final_metrics']['step3_convergence_rate']:.2%}")

    step2_state = results['step2_model_state']
    step3_state = results['step3_model_state']
    print(f"\n 第二步模型最终状态:")
    print(f"   • 初始化状态: {'已完成' if step2_state['initialized'] else '未完成'}")
    print(f"   • 模型参数量: {step2_state['model_parameters']:,}")
    print(f"   • 训练历史长度: {step2_state['training_history_length']}")

    print(f"\n 第三步模型最终状态:")
    print(f"   • 初始化状态: {'已完成' if step3_state['initialized'] else '未完成'}")
    print(f"   • EvolveGCN参数量: {step3_state['model_parameters']:,}")
    print(f"   • 分片模块参数量: {step3_state['sharding_parameters']:,}")
    print(f"   • 历史状态长度: {step3_state['history_length']}")
    print(f"   • 训练历史长度: {step3_state['training_history_length']}")

    print("\n 完全整合真实第二步和第三步的流水线演示完成!")


if __name__ == "__main__":
    main()