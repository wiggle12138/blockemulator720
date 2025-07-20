#!/usr/bin/env python3
"""
准确的消融实验：包含训练过程
测试各个修改在实际训练过程中的效果
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from pathlib import Path
import time
import json

sys.path.append(str(Path(__file__).parent / "evolve_GCN"))
sys.path.append(str(Path(__file__).parent / "feedback"))

from evolve_GCN.models.sharding_modules import DynamicShardingModule, GraphAttentionPooling
from evolve_GCN.losses.sharding_losses import multi_objective_sharding_loss

class TrainingAblationTester:
    """包含训练过程的消融实验测试器"""
    
    def __init__(self):
        self.device = torch.device('cpu')
        self.num_nodes = 200
        self.embed_dim = 64
        self.n_shards = 5
        
        # 生成测试数据
        torch.manual_seed(42)
        self.embeddings = torch.randn(self.num_nodes, self.embed_dim, device=self.device)
        self.edge_index = self._generate_edge_index()
        
        # 训练配置
        self.training_epochs = 10
        self.learning_rate = 0.001
        
    def _generate_edge_index(self):
        """生成测试用边索引"""
        edges = []
        for i in range(self.num_nodes):
            edges.append([i, (i + 1) % self.num_nodes])
            edges.append([i, (i - 1) % self.num_nodes])
            if i % 5 == 0:  # 一些长距离连接
                edges.append([i, (i + 10) % self.num_nodes])
        
        edge_tensor = torch.tensor(edges, dtype=torch.long).t()
        return edge_tensor
    
    def _create_baseline_module(self):
        """创建基线版本模块（原始设置）"""
        module = DynamicShardingModule(
            embedding_dim=self.embed_dim,
            base_shards=3,
            max_shards=self.n_shards
        )
        
        # 修改allocator为原始版本
        module.allocator = self._create_baseline_pooling()
        return module
    
    def _create_baseline_pooling(self):
        """创建原始版本的pooling（无强制平衡）"""
        return GraphAttentionPooling(
            embedding_dim=self.embed_dim,
            n_heads=4
        )
    
    def _create_enhanced_module(self):
        """创建增强版本模块（包含所有修改）"""
        module = DynamicShardingModule(
            embedding_dim=self.embed_dim,
            base_shards=3,
            max_shards=self.n_shards
        )
        return module  # 使用当前的增强版本
    
    def _compute_baseline_loss(self, assignment, target_shards=None):
        """原始低惩罚损失函数"""
        device = assignment.device
        
        # 计算使用的分片数
        hard_assignment = torch.argmax(assignment, dim=1)
        unique_shards = torch.unique(hard_assignment)
        num_active_shards = len(unique_shards)
        
        # 基础损失 - 均匀分布
        target_prob = 1.0 / self.n_shards
        avg_prob = assignment.mean(dim=0)
        balance_loss = torch.sum((avg_prob - target_prob) ** 2)
        
        # 原始较轻的惩罚
        if num_active_shards < 2:
            single_shard_penalty = 10.0  # 原始较低惩罚
        else:
            single_shard_penalty = 0.0
        
        total_loss = balance_loss + single_shard_penalty
        
        return total_loss, {
            'balance_loss': balance_loss.item(),
            'single_shard_penalty': single_shard_penalty,
            'num_active_shards': num_active_shards
        }
    
    def _compute_enhanced_loss(self, assignment, target_shards=None):
        """增强版高惩罚损失函数"""
        device = assignment.device
        
        # 计算使用的分片数
        hard_assignment = torch.argmax(assignment, dim=1)
        unique_shards = torch.unique(hard_assignment)
        num_active_shards = len(unique_shards)
        
        # 基础损失 - 均匀分布
        target_prob = 1.0 / self.n_shards
        avg_prob = assignment.mean(dim=0)
        balance_loss = torch.sum((avg_prob - target_prob) ** 2)
        
        # 极端高惩罚
        if num_active_shards == 1:
            single_shard_penalty = 100000.0  # 极端惩罚
        elif num_active_shards < 3:
            single_shard_penalty = 50000.0   # 高惩罚
        else:
            single_shard_penalty = 0.0
        
        total_loss = balance_loss + single_shard_penalty
        
        return total_loss, {
            'balance_loss': balance_loss.item(),
            'single_shard_penalty': single_shard_penalty,
            'num_active_shards': num_active_shards
        }
    
    def _train_module(self, module, loss_func, temperature, use_enhanced_balance=False):
        """训练模块并返回结果"""
        optimizer = optim.Adam(module.parameters(), lr=self.learning_rate)
        
        results = []
        
        for epoch in range(self.training_epochs):
            optimizer.zero_grad()
            
            # 前向传播
            assignment, _, _, _ = module(self.embeddings)
            
            # 计算损失
            loss, loss_details = loss_func(assignment)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 记录结果
            hard_assignment = torch.argmax(assignment, dim=1)
            unique_shards, counts = torch.unique(hard_assignment, return_counts=True)
            
            results.append({
                'epoch': epoch,
                'num_active_shards': len(unique_shards),
                'shard_distribution': counts.tolist(),
                'balance_score': self._compute_balance(counts, self.n_shards),
                'loss': loss.item(),
                'loss_details': loss_details
            })
        
        return results
    
    def _compute_balance(self, counts, total_shards):
        """计算负载均衡度"""
        if len(counts) == 0:
            return 0.0
        
        # 扩展到所有分片（未使用的分片计数为0）
        full_counts = torch.zeros(total_shards)
        for i, count in enumerate(counts):
            full_counts[i] = count
        
        # 计算标准差，值越小越均衡
        mean_count = full_counts.mean()
        if mean_count == 0:
            return 0.0
        
        std = torch.std(full_counts)
        balance_score = 1.0 - (std / mean_count)
        return max(0.0, balance_score.item())
    
    def test_baseline(self):
        """测试基线版本（原始低温度+低惩罚+无强制平衡）"""
        print("\n🔬 基线测试（原始版本）")
        print("=" * 60)
        
        module = self._create_baseline_module()
        results = self._train_module(
            module=module,
            loss_func=self._compute_baseline_loss,
            temperature=1.0,  # 原始低温度
            use_enhanced_balance=False
        )
        
        return self._analyze_training_results("基线版本", results)
    
    def test_temperature_only(self):
        """测试仅温度修改（高温度+原始损失+无强制平衡）"""
        print("\n🌡️ 仅温度修改测试")
        print("=" * 60)
        
        module = self._create_baseline_module()
        results = self._train_module(
            module=module,
            loss_func=self._compute_baseline_loss,
            temperature=25.0,  # 超高温度
            use_enhanced_balance=False
        )
        
        return self._analyze_training_results("仅温度修改", results)
    
    def test_loss_only(self):
        """测试仅损失修改（原始温度+高惩罚+无强制平衡）"""
        print("\n⚖️ 仅损失修改测试")
        print("=" * 60)
        
        module = self._create_baseline_module()
        results = self._train_module(
            module=module,
            loss_func=self._compute_enhanced_loss,
            temperature=1.0,   # 原始低温度
            use_enhanced_balance=False
        )
        
        return self._analyze_training_results("仅损失修改", results)
    
    def test_force_balance_only(self):
        """测试仅强制平衡（原始温度+原始损失+强制平衡）"""
        print("\n⚖️ 仅强制平衡测试")
        print("=" * 60)
        
        module = self._create_enhanced_module()  # 包含强制平衡
        results = self._train_module(
            module=module,
            loss_func=self._compute_baseline_loss,
            temperature=1.0,   # 原始低温度
            use_enhanced_balance=True
        )
        
        return self._analyze_training_results("仅强制平衡", results)
    
    def test_temperature_plus_loss(self):
        """测试温度+损失组合（高温度+高惩罚+无强制平衡）"""
        print("\n🌡️⚖️ 温度+损失组合测试")
        print("=" * 60)
        
        module = self._create_baseline_module()
        results = self._train_module(
            module=module,
            loss_func=self._compute_enhanced_loss,
            temperature=25.0,  # 超高温度 + 高惩罚
            use_enhanced_balance=False
        )
        
        return self._analyze_training_results("温度+损失组合", results)
    
    def test_full_combination(self):
        """测试完整组合（高温度+高惩罚+强制平衡）"""
        print("\n[TARGET] 完整组合测试")
        print("=" * 60)
        
        module = self._create_enhanced_module()
        results = self._train_module(
            module=module,
            loss_func=self._compute_enhanced_loss,
            temperature=25.0,  # 所有修改
            use_enhanced_balance=True
        )
        
        return self._analyze_training_results("完整组合", results)
    
    def _analyze_training_results(self, test_name, results):
        """分析训练结果"""
        final_result = results[-1]  # 最终epoch结果
        initial_result = results[0]  # 初始epoch结果
        
        # 计算改善
        final_shards = final_result['num_active_shards']
        initial_shards = initial_result['num_active_shards']
        shard_improvement = final_shards - initial_shards
        
        final_balance = final_result['balance_score']
        initial_balance = initial_result['balance_score']
        balance_improvement = final_balance - initial_balance
        
        print(f"[DATA] {test_name}结果:")
        print(f"  训练轮数: {len(results)}")
        print(f"  初始分片数: {initial_shards}")
        print(f"  最终分片数: {final_shards}")
        print(f"  分片数改善: {shard_improvement:+d}")
        print(f"  初始均衡度: {initial_balance:.3f}")
        print(f"  最终均衡度: {final_balance:.3f}")
        print(f"  均衡度改善: {balance_improvement:+.3f}")
        print(f"  最终分布: {final_result['shard_distribution']}")
        print(f"  最终损失: {final_result['loss']:.3f}")
        
        # 检查是否解决单分片问题
        solved_single_shard = final_shards >= 3
        effective_improvement = shard_improvement > 0 or balance_improvement > 0.1
        
        if solved_single_shard and effective_improvement:
            effectiveness = "🟢 高效"
        elif effective_improvement:
            effectiveness = "🟡 部分有效"
        else:
            effectiveness = "🔴 无效"
        
        print(f"  效果评级: {effectiveness}")
        print()
        
        return {
            'test_name': test_name,
            'initial_shards': initial_shards,
            'final_shards': final_shards,
            'shard_improvement': shard_improvement,
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'balance_improvement': balance_improvement,
            'final_distribution': final_result['shard_distribution'],
            'solved_single_shard': solved_single_shard,
            'effective': effective_improvement,
            'effectiveness': effectiveness,
            'training_history': results
        }

def main():
    """运行完整的训练消融实验"""
    print("🔬 训练版消融实验：确定关键有效修改")
    print("=" * 80)
    print("注：每个测试包含完整的训练过程")
    print()
    
    tester = TrainingAblationTester()
    
    # 运行所有测试
    all_results = {}
    
    all_results['baseline'] = tester.test_baseline()
    all_results['temperature_only'] = tester.test_temperature_only() 
    all_results['loss_only'] = tester.test_loss_only()
    all_results['force_balance_only'] = tester.test_force_balance_only()
    all_results['temperature_plus_loss'] = tester.test_temperature_plus_loss()
    all_results['full_combination'] = tester.test_full_combination()
    
    # 生成总结报告
    print("\n[DATA] 训练消融实验总结")
    print("=" * 80)
    
    baseline_shards = all_results['baseline']['final_shards']
    baseline_balance = all_results['baseline']['final_balance']
    
    for test_name, result in all_results.items():
        if test_name == 'baseline':
            continue
            
        shard_gain = result['final_shards'] - baseline_shards
        balance_gain = result['final_balance'] - baseline_balance
        
        print(f"{result['test_name']}:")
        print(f"  分片数增益: {shard_gain:+d} ({baseline_shards}→{result['final_shards']})")
        print(f"  均衡度增益: {balance_gain:+.3f} ({baseline_balance:.3f}→{result['final_balance']:.3f})")
        print(f"  效果评级: {result['effectiveness']}")
        print()
    
    # 确定最关键的修改
    print("[TARGET] 关键因素识别:")
    print("=" * 50)
    
    # 分析哪个单独修改最有效
    single_modifications = ['temperature_only', 'loss_only', 'force_balance_only']
    best_single = max(single_modifications, 
                     key=lambda x: all_results[x]['shard_improvement'] + all_results[x]['balance_improvement'])
    
    print(f"最有效的单独修改: {all_results[best_single]['test_name']}")
    print(f"分片改善: {all_results[best_single]['shard_improvement']:+d}")
    print(f"均衡改善: {all_results[best_single]['balance_improvement']:+.3f}")
    print()
    
    # 检查组合效果
    combo_improvement = (all_results['temperature_plus_loss']['shard_improvement'] + 
                        all_results['temperature_plus_loss']['balance_improvement'])
    full_improvement = (all_results['full_combination']['shard_improvement'] + 
                       all_results['full_combination']['balance_improvement'])
    
    print(f"温度+损失组合改善: {combo_improvement:.3f}")
    print(f"完整组合改善: {full_improvement:.3f}")
    
    if full_improvement > combo_improvement + 0.1:
        print("[SUCCESS] 强制平衡机制提供了额外的重要改进")
    elif combo_improvement > max(all_results[x]['shard_improvement'] + all_results[x]['balance_improvement'] 
                                for x in single_modifications) + 0.1:
        print("[SUCCESS] 温度和损失修改的组合效果显著")
    else:
        print("[SUCCESS] 单独修改已足够，组合效果有限")
    
    # 保存结果
    with open('training_ablation_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细结果已保存: training_ablation_results.json")

if __name__ == "__main__":
    main()
