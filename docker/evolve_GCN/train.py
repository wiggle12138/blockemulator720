"""
主训练脚本 - EvolveGCN动态分片训练
"""
import os
import pickle
import torch
import torch.optim as optim
import numpy as np
import random

# 导入自定义模块
from utils import get_device, print_device_info, HyperparameterUpdater
from models import EvolveGCNWrapper, DynamicShardingModule
from data import BlockchainDataset
from losses import multi_objective_sharding_loss, temporal_consistency_loss
from config import default_config

def set_random_seed(seed=42):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ShardingTrainer:
    """动态分片训练器"""

    def __init__(self, config=None):
        self.config = config or default_config
        self.device = get_device()

        # 初始化组件
        self.dataset = None
        self.model = None
        self.sharding_module = None
        self.model_optimizer = None
        self.shard_optimizer = None
        self.param_updater = HyperparameterUpdater()

        # 训练状态
        self.history_states = []
        self.prev_shard_assignment = None
        self.prev_shard_count = self.config.base_shards
        self.cross_increase_count = 0
        self.prev_cross_rate = 0.0

    def setup(self):
        """初始化训练环境"""
        print("开始EvolveGCN动态分片训练...")
        print_device_info()
        self.config.print_config()

        # 1. 数据准备
        print("\n准备数据集...")
        self.dataset = BlockchainDataset(
            self.config.embedding_path,
            self.config.edge_index_path,
            num_timesteps=self.config.num_timesteps,
            noise_level=self.config.noise_level
        )

        input_dim = self.dataset.embedding_dim

        # 2. 模型初始化
        print("\n初始化模型...")
        self.model = EvolveGCNWrapper(input_dim, self.config.hidden_dim).to(self.device)
        self.sharding_module = DynamicShardingModule(
            embedding_dim=self.config.hidden_dim,
            base_shards=self.config.base_shards,
            max_shards=self.config.max_shards,
            min_shard_size=self.config.min_shard_size,
            max_empty_ratio=self.config.max_empty_ratio
        ).to(self.device)

        # 3. 优化器
        self.model_optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        self.shard_optimizer = optim.Adam(
            self.sharding_module.parameters(),
            lr=self.config.shard_lr
        )

        print(f"初始化完成: 输入维度={input_dim}, 隐藏维度={self.config.hidden_dim}")

    def train_single_epoch(self, epoch):
        """单个epoch的训练"""
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{self.config.epochs}")
        print(f"{'=' * 60}")

        # 重置模型状态
        self.model.reset_state()

        epoch_losses = {'gcn': 0.0, 'shard': 0.0, 'total': 0.0}
        all_embeddings = []
        all_delta_signals = []

        # 准备性能反馈信号
        performance_feedback = self._get_performance_feedback()

        # 时序处理
        for t in range(self.config.num_timesteps):
            node_features, edge_index, timestep = self.dataset[t]
            node_features = node_features.to(self.device)
            edge_index = edge_index.to(self.device)

            # EvolveGCN前向传播
            embeddings, delta_signal = self.model(node_features, edge_index, performance_feedback)
            all_embeddings.append(embeddings)
            all_delta_signals.append(delta_signal)

            # 计算GCN损失
            gcn_loss = self._compute_gcn_loss(embeddings, all_embeddings, t)
            epoch_losses['gcn'] += gcn_loss.item()

            # 进度输出
            if t % 3 == 0:
                print(f"  时间步 {t}: 嵌入形状 {embeddings.shape}, 损失 {gcn_loss.item():.4f}")

        # 分片决策
        shard_loss, loss_components, predicted_shards = self._compute_sharding_loss(
            all_embeddings[-1], epoch_losses
        )

        # 总损失计算和反向传播
        total_loss = self._compute_total_loss(all_embeddings) + shard_loss
        epoch_losses['total'] = total_loss.item()

        self._backward_and_optimize(total_loss)

        # 更新历史状态
        self._update_history_states(loss_components, predicted_shards)

        # 输出训练进度
        if epoch % self.config.print_freq == 0:
            self._print_epoch_summary(epoch, epoch_losses, loss_components, predicted_shards)

        return epoch_losses

    def _load_step4_feedback(self):
        """加载第四步性能反馈信号"""
        try:
            from pathlib import Path
            import pickle
            
            # 第四步反馈文件的可能位置
            feedback_paths = [
                Path("../feedback/step4_feedback_result.pkl"),
                Path("feedback/step4_feedback_result.pkl"), 
                Path("step4_feedback_result.pkl"),
                Path("../feedback/step3_performance_feedback.pkl"),
                Path("step3_performance_feedback.pkl")
            ]
            
            for feedback_path in feedback_paths:
                if feedback_path.exists():
                    with open(feedback_path, "rb") as f:
                        feedback_data = pickle.load(f)
                    
                    print(f"   [SUCCESS] 加载第四步反馈: {feedback_path}")
                    
                    # � 处理不同格式的反馈数据
                    if isinstance(feedback_data, dict):
                        # 新格式：直接包含反馈矩阵
                        if 'step3_feedback' in feedback_data and 'assignment_guidance' in feedback_data['step3_feedback']:
                            feedback_matrix = feedback_data['step3_feedback']['assignment_guidance']
                            if isinstance(feedback_matrix, torch.Tensor):
                                return feedback_matrix.to(self.device)
                        
                        # 旧格式：需要构建反馈矩阵
                        elif 'temporal_performance' in feedback_data:
                            # 基于性能向量构建简单反馈
                            num_nodes = self.dataset.num_nodes if hasattr(self.dataset, 'num_nodes') else 100
                            performance_score = feedback_data['temporal_performance'].get('combined_score', 0.5)
                            
                            # 根据性能分数调整分片倾向性
                            if hasattr(self, 'prev_shard_assignment') and self.prev_shard_assignment is not None:
                                prev_shards = self.prev_shard_assignment.size(1)
                                feedback_matrix = torch.ones(num_nodes, prev_shards, device=self.device) / prev_shards
                                
                                # 如果性能较差，鼓励重新分配
                                if performance_score < 0.7:
                                    # 添加一些随机扰动鼓励探索
                                    noise = torch.randn_like(feedback_matrix) * 0.1
                                    feedback_matrix = torch.softmax(feedback_matrix + noise, dim=1)
                                
                                return feedback_matrix
                    
                    break
            
            # 如果没有找到反馈文件，返回None
            return None
            
        except Exception as e:
            print(f"   [WARNING]  加载第四步反馈失败: {e}")
            return None
        except Exception as e:
            print(f"[WARNING] 第四步反馈加载失败: {e}")
            return None
    
    def _get_performance_feedback(self):
        """获取性能反馈信号 - 优先使用第四步反馈"""
        
        # [FIX] 优先尝试第四步反馈
        step4_feedback = self._load_step4_feedback()
        if step4_feedback is not None:
            print(f"[DATA] 使用第四步反馈，维度: {step4_feedback.shape}")
            return step4_feedback
        
        # 原有的历史状态反馈作为备选
        if self.history_states:
            recent_states = torch.stack(self.history_states[-3:]) if len(self.history_states) >= 3 else torch.stack(
                self.history_states)
            performance_feedback = torch.mean(recent_states, dim=0).float().to(self.device)
            print(f"[DATA] 使用历史状态反馈，维度: {performance_feedback.shape}")
        else:
            # [FIX] 使用11维默认反馈匹配第四步格式
            performance_feedback = torch.tensor([
                0.5, 0.1, 0.8, 0.6,  # 核心4维：负载均衡, 跨片率, 安全性, 特征质量
                0.7, 0.6, 0.8, 0.5, 0.6, 0.7,  # 6维特征质量
                0.65  # 综合分数
            ], dtype=torch.float32, device=self.device)
            print(f"[DATA] 使用默认反馈，维度: {performance_feedback.shape}")
        
        return performance_feedback
    def _compute_gcn_loss(self, embeddings, all_embeddings, t):
        """计算GCN损失"""
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

    def _compute_sharding_loss(self, final_embeddings, epoch_losses):
        """计算分片损失 - 集成第四步反馈"""
        # 🔄 加载第四步反馈信号
        feedback_signal = self._load_step4_feedback()
        
        # 动态分片 - 使用优化后的模块
        shard_assignment, enhanced_embeddings, attention_weights, predicted_shards = self.sharding_module(
            Z=final_embeddings, 
            history_states=self.history_states, 
            feedback_signal=feedback_signal
        )

        print(f" 预测分片数: {predicted_shards} (上一轮: {self.prev_shard_count})")
        if feedback_signal is not None:
            print(f" 使用第四步反馈: {feedback_signal.shape}")

        # 生成安全评分（模拟）
        security_scores = torch.rand(final_embeddings.size(0), dtype=torch.float32, device=self.device) * 0.5

        # 判断是否使用历史分配
        use_prev_assignment = (self.prev_shard_assignment is not None and
                               self.prev_shard_assignment.size(1) == shard_assignment.size(1))

        # 计算分片损失
        shard_loss, loss_components = multi_objective_sharding_loss(
            shard_assignment, enhanced_embeddings, self.dataset.edge_index.to(self.device),
            prev_assignment=self.prev_shard_assignment if use_prev_assignment else None,
            security_scores=security_scores,
            a=self.param_updater.params.get('balance_weight', 1.0),
            b=self.param_updater.params.get('cross_weight', 1.0),
            c=self.param_updater.params.get('security_weight', 1.5),
            d=self.param_updater.params.get('migrate_weight', 0.5) if use_prev_assignment else 0.0
        )

        epoch_losses['shard'] = shard_loss.item()

        # 保存当前分配
        self.prev_shard_assignment = shard_assignment.detach().clone()
        self.prev_shard_count = predicted_shards

        return shard_loss, loss_components, predicted_shards

    def _compute_total_loss(self, all_embeddings):
        """计算总的GCN损失"""
        gcn_total_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        for i, emb in enumerate(all_embeddings):
            if i > 0:
                consistency_loss = temporal_consistency_loss(
                    emb, all_embeddings[i - 1].detach(),
                    lambda_contrast=self.param_updater.params['lambda']
                )
                gcn_total_loss += consistency_loss

            reg_loss = 0.01 * torch.mean(torch.norm(emb, p=2, dim=1))
            gcn_total_loss += reg_loss

        return gcn_total_loss

    def _backward_and_optimize(self, total_loss):
        """反向传播和优化"""
        self.model_optimizer.zero_grad()
        self.shard_optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.sharding_module.parameters(), max_norm=self.config.max_grad_norm)

        self.model_optimizer.step()
        self.shard_optimizer.step()

    def _update_history_states(self, loss_components, predicted_shards):
        """更新历史状态"""
        with torch.no_grad():
            hard_assignment = torch.argmax(self.prev_shard_assignment, dim=1)
            shard_sizes = [(hard_assignment == s).sum().item() for s in range(predicted_shards)]

            # 性能指标计算
            balance_score = 1.0 - (np.std(shard_sizes) / (np.mean(shard_sizes) + 1e-8))
            cross_rate = loss_components['cross']
            security_score = 1.0 - loss_components['security']

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
            self.prev_shard_count = predicted_shards

            # 动态调节参数
            performance_metrics = {
                'balance_score': balance_score,
                'cross_tx_rate': cross_rate,
                'cross_increase_count': self.cross_increase_count
            }
            self.param_updater.update_hyperparams(performance_metrics)

    def _print_epoch_summary(self, epoch, epoch_losses, loss_components, predicted_shards):
        """打印epoch总结"""
        hard_assignment = torch.argmax(self.prev_shard_assignment, dim=1)
        shard_sizes = [(hard_assignment == s).sum().item() for s in range(predicted_shards)]

        balance_score = 1.0 - (np.std(shard_sizes) / (np.mean(shard_sizes) + 1e-8))
        cross_rate = loss_components['cross']
        security_score = 1.0 - loss_components['security']

        print(f"\nEpoch {epoch + 1} 总结:")
        print(f"  总损失: {epoch_losses['total']:.4f}")
        print(f"  GCN损失: {epoch_losses['gcn']:.4f}")
        print(f"  分片损失: {epoch_losses['shard']:.4f}")
        print(f"  损失组件: {loss_components}")
        print(f"  分片大小: {shard_sizes}")
        print(f"  性能指标: 均衡={balance_score:.3f}, 跨片={cross_rate:.3f}, 安全={security_score:.3f}")

        updated_params = self.param_updater.get_params()
        print(
            f"  动态参数: α={updated_params['alpha']:.3f}, λ={updated_params['lambda']:.3f}, K={updated_params['K_base']}")

    def train(self):
        """完整训练流程"""
        self.setup()

        print(f"\n开始训练 {self.config.epochs} 个epochs...")

        for epoch in range(self.config.epochs):
            epoch_losses = self.train_single_epoch(epoch)

        print("\n训练完成!")
        return self._generate_final_results()

    def _generate_final_results(self):
        """生成最终结果"""
        print("生成最终嵌入...")
        self.model.eval()
        self.model.reset_state()
        new_embeddings = {}

        with torch.no_grad():
            for t in range(self.config.num_timesteps):
                node_features, edge_index, timestep = self.dataset[t]
                embeddings, _ = self.model(node_features.to(self.device), edge_index.to(self.device))
                new_embeddings[str(t)] = embeddings.cpu().numpy()

        return self._save_results(new_embeddings)

    def _save_results(self, new_embeddings):
        """保存训练结果"""
        print(" 保存结果...")

        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.model_dir, exist_ok=True)

        # 保存嵌入
        embedding_path = os.path.join(self.config.output_dir, 'new_temporal_embeddings.pkl')
        with open(embedding_path, 'wb') as f:
            pickle.dump(new_embeddings, f)

        # 保存模型
        model_path = os.path.join(self.config.model_dir, 'enhanced_evolvegcn_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'sharding_state_dict': self.sharding_module.state_dict(),
            'history_states': self.history_states,
            'hyperparams': self.param_updater.get_params(),
            'config': self.config.to_dict()
        }, model_path)

        # 生成最终分片结果
        sharding_results = self._generate_sharding_results(new_embeddings)
        shard_path = os.path.join(self.config.output_dir, 'sharding_results.pkl')
        with open(shard_path, 'wb') as f:
            pickle.dump(sharding_results, f)

        print("所有任务完成!")
        print(f"输出文件:")
        print(f"  - 嵌入结果: {embedding_path}")
        print(f"  - 训练模型: {model_path}")
        print(f"  - 分片结果: {shard_path}")

        # 打印分片结果统计
        for key, value in sharding_results.items():
            print(f"  - {key}: {len(value)} 节点 {value}")

        return new_embeddings, sharding_results, self.history_states

    def _generate_sharding_results(self, new_embeddings):
        """生成分片结果"""
        final_embeddings = torch.tensor(
            new_embeddings[str(self.config.num_timesteps - 1)],
            dtype=torch.float32, device=self.device
        )
        final_assignment, _, _, final_shards = self.sharding_module(final_embeddings, self.history_states)
        hard_assignment = torch.argmax(final_assignment, dim=1)

        sharding_results = {}
        for s in range(final_shards):
            shard_nodes = (hard_assignment == s).nonzero(as_tuple=True)[0].cpu().numpy()
            sharding_results[f'shard_{s}'] = shard_nodes.tolist()

        return sharding_results


def main():
    """主函数"""
    try:
        # 创建训练器并开始训练
        trainer = ShardingTrainer(default_config)

        # 设置随机种子以确保可重复性
        set_random_seed(42)
        embeddings, sharding, history = trainer.train()

        print("训练成功完成！")

        # 输出最终统计
        print(f"\n最终统计:")
        print(f"  生成嵌入: {len(embeddings)} 个时间步")
        print(f"  分片结果: {len(sharding)} 个分片")
        print(f"  历史状态: {len(history)} 个记录")

        return embeddings, sharding, history

    except Exception as e:
        print(f"[ERROR] 训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    main()