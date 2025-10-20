"""
分片相关模块集合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

# 修复相对导入问题
try:
    from .temporal_conv import TemporalConvNet
except ImportError:
    try:
        from temporal_conv import TemporalConvNet
    except ImportError:
        import sys
        import importlib.util
        from pathlib import Path
        
        # 使用绝对路径导入
        current_dir = Path(__file__).parent
        temporal_conv_path = current_dir / "temporal_conv.py"
        
        if temporal_conv_path.exists():
            spec = importlib.util.spec_from_file_location("temporal_conv", temporal_conv_path)
            if spec and spec.loader:
                temporal_conv_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(temporal_conv_module)
                TemporalConvNet = getattr(temporal_conv_module, 'TemporalConvNet', None)
        
        if 'TemporalConvNet' not in locals() or TemporalConvNet is None:
            # 如果还是无法导入，创建一个简单的替代实现
            class TemporalConvNet(nn.Module):
                def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Conv1d(num_inputs, num_channels[-1], kernel_size, padding=kernel_size//2),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    )
                
                def forward(self, x):
                    return self.network(x)


class GraphAttentionPooling(nn.Module):
    """基于注意力权重的节点分配模块"""

    def __init__(self, embedding_dim, n_heads=8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads

        # 多头注意力
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.1
        )

        # 分片原型生成器
        self.prototype_generator = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, embeddings, n_shards, temperature=1.0):
        """
        Args:
            embeddings: [num_nodes, embedding_dim]
            n_shards: 分片数量
            temperature: 温度参数
        Returns:
            assignment_matrix: [num_nodes, n_shards]
            enhanced_embeddings: 增强的嵌入
            attention_weights: 注意力权重
        """
        # 确保数据类型一致
        embeddings = embeddings.float()

        # 添加batch维度
        embeddings_batch = embeddings.unsqueeze(0)  # [1, num_nodes, embedding_dim]

        # 自注意力增强嵌入
        enhanced_embeddings, attention_weights = self.multihead_attention(
            embeddings_batch, embeddings_batch, embeddings_batch
        )
        enhanced_embeddings = enhanced_embeddings.squeeze(0)  # [num_nodes, embedding_dim]

        # 生成分片原型 - 强制均匀分布初始化
        with torch.no_grad():
            embeddings_np = enhanced_embeddings.detach().cpu().numpy()
            
            # 尝试多次K-means，选择最均匀的结果
            best_centers = None
            best_balance = float('inf')
            
            for attempt in range(5):  # 多次尝试
                try:
                    kmeans = KMeans(n_clusters=n_shards, random_state=42+attempt, n_init=20)
                    labels = kmeans.fit_predict(embeddings_np)
                    
                    # 计算分布均匀性
                    unique, counts = np.unique(labels, return_counts=True)
                    if len(unique) >= max(2, n_shards // 2):  # 至少要有一半分片有数据
                        balance_score = np.std(counts) / (np.mean(counts) + 1e-8)
                        if balance_score < best_balance:
                            best_balance = balance_score
                            best_centers = kmeans.cluster_centers_
                except:
                    continue
            
            if best_centers is None:
                # 如果K-means失败，使用确定性均匀初始化
                print(f"    [INIT_WARNING] K-means失败，使用均匀初始化")
                indices = torch.randperm(embeddings_np.shape[0])[:n_shards]
                best_centers = embeddings_np[indices]
            
            initial_prototypes = torch.tensor(
                best_centers, dtype=torch.float32, device=embeddings.device
            )
            
            print(f"    [INIT_DEBUG] 初始化平衡度: {best_balance:.3f}, 使用 {len(initial_prototypes)} 个原型")

        # 通过原型生成器精细化
        prototypes = self.prototype_generator(initial_prototypes)

        # 计算相似度矩阵
        similarity = torch.mm(
            F.normalize(enhanced_embeddings, p=2, dim=1),
            F.normalize(prototypes, p=2, dim=1).T
        )

        # 自适应温度：根据相似度分布调整温度 - 极端超高温度版本
        # 如果相似度差异很大，需要更高的温度来避免极化
        similarity_std = torch.std(similarity)
        # 极端超高温度设置 - 强制极度均匀分布
        base_temp = max(temperature, 25.0)  # 极端超高基础温度
        adaptive_temp = base_temp + similarity_std.item() * 10.0  # 极端提高温度
        
        # 应用极端超高温度参数并转换为概率分布 - 强制极度均匀分布
        assignment_matrix = F.softmax(similarity / adaptive_temp, dim=1)
        
        # ========== 极端强制均匀分布机制 ==========
        # 1. 检查分片分配情况
        shard_probs = torch.sum(assignment_matrix, dim=0)
        num_nodes = assignment_matrix.size(0)
        expected_nodes_per_shard = num_nodes / n_shards
        
        # 2. 识别分配不足的分片（极端严格检查）
        under_allocated = shard_probs < (expected_nodes_per_shard * 0.8)  # 极端严格：少于80%就干预
        
        # 3. 极端强制重新分配
        if torch.sum(under_allocated) > 0:
            print(f"    [EXTREME_FORCED_BALANCE] 极端强制调整 {torch.sum(under_allocated).item()} 个分片")
            
            # 极端强制调整：给分配不足的分片大量强制分配节点
            for under_idx in torch.where(under_allocated)[0]:
                # 从相似度最高的节点中极端强制分配
                shard_similarities = similarity[:, under_idx]
                top_candidates = torch.topk(shard_similarities, min(30, num_nodes))[1]
                
                # 极端强制提升这些节点对该分片的概率
                for candidate in top_candidates[:15]:  # 调整更多节点
                    assignment_matrix[candidate, under_idx] += 2.0  # 极端强化提升
                        
                print(f"    [EXTREME_FORCED_BALANCE] 为分片 {under_idx} 极端强制分配节点")
        
        # 4. 重新归一化
        assignment_matrix = F.normalize(assignment_matrix, p=1, dim=1)
        
        # 5. 最终检查分配质量 - 极端均匀化
        final_shard_probs = torch.sum(assignment_matrix, dim=0)
        min_prob = torch.min(final_shard_probs)
        max_prob = torch.max(final_shard_probs)
        
        # 极端强制均匀化机制
        if max_prob > min_prob * 3:  # 极端低阈值，更激进
            print(f"    [EXTREME_FORCE_UNIFORM] 极端强制均匀化: max={max_prob:.2f}, min={min_prob:.2f}")
            # 极端强制均匀化
            uniform_assignment = torch.ones_like(assignment_matrix) / n_shards
            assignment_matrix = 0.3 * assignment_matrix + 0.7 * uniform_assignment  # 极端强的均匀化
        
        # 调试信息：检查分片分配
        hard_assignment = torch.argmax(assignment_matrix, dim=1)
        unique_shards, counts = torch.unique(hard_assignment, return_counts=True)
        print(f"    [ALLOCATOR_DEBUG] 分配器输出: {len(unique_shards)}/{n_shards} 个分片, "
              f"分布: {counts.tolist()}, 温度: {adaptive_temp:.3f}")

        return assignment_matrix, enhanced_embeddings, attention_weights.squeeze(0)


class DynamicShardingModule(nn.Module):
    """动态分片决策模块 - 核心实现，支持空分片处理"""

    def __init__(self, embedding_dim, base_shards=3, max_shards=10):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.base_shards = base_shards
        self.max_shards = max_shards

        # 时间卷积预测分片数
        self.predictor = TemporalConvNet(input_dim=3, max_shards=max_shards)

        # 基于注意力权重的节点分配
        self.allocator = GraphAttentionPooling(embedding_dim)

        # 反馈信号融合参数
        self.feedback_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, Z, history_states=None, feedback_signal=None):
        """
        动态分片决策前向传播 - 包含空分片处理

        Args:
            Z: 当前时刻节点嵌入 [num_nodes, embedding_dim]
            history_states: 历史分片状态 [seq_len, 3] (负载均衡度, 跨片交易率, 安全阈值)
            feedback_signal: 第四步反馈信号 [num_nodes, n_shards]
        Returns:
            S_t: 分片-节点关联矩阵 [num_nodes, n_shards]
            enhanced_embeddings: 增强嵌入
            attention_weights: 注意力权重
            K_t: 预测的分片数
        """
        # 确保输入数据类型一致
        Z = Z.float()
        num_nodes = Z.size(0)
        
        #  缓存嵌入用于后续的反馈处理
        self._cached_embeddings = Z.detach()

        #  改进的分片数预测：考虑最小分片大小约束
        K_t = self._predict_optimal_shard_count(history_states, num_nodes)

        # 动态调节温度系数
        tau_t = self._compute_temperature(history_states)

        # 生成分片-节点关联矩阵
        S_t, enhanced_embeddings, attention_weights = self.allocator(Z, K_t, tau_t)

        # 注入第四步反馈信号
        if feedback_signal is not None:
            S_t = self._apply_feedback(S_t, feedback_signal)

        return S_t, enhanced_embeddings, attention_weights, K_t

    def _predict_optimal_shard_count(self, history_states, num_nodes):
        """预测最优分片数"""
        if history_states is not None and len(history_states) >= 3:
            history_tensor = torch.stack(history_states[-3:]).float()
            predicted_shards = int(self.predictor(history_tensor).item())

            # 限制在合理范围内
            predicted_shards = min(predicted_shards, self.max_shards)
            predicted_shards = max(predicted_shards, 2)

            # 渐进调整：限制分片数变化幅度
            if hasattr(self, 'prev_shard_count'):
                max_change = max(1, self.prev_shard_count // 4)  # 最多变化25%
                predicted_shards = max(
                    min(predicted_shards, self.prev_shard_count + max_change),
                    self.prev_shard_count - max_change
                )

            K_t = predicted_shards
            self.prev_shard_count = K_t
        else:
            K_t = self.base_shards
            self.prev_shard_count = K_t

        return K_t

    def _compute_temperature(self, history_states):
        """计算温度系数 - 激进鼓励分散版本"""
        if history_states is not None and len(history_states) > 0:
            beta_t = history_states[-1][0].item()  # 负载均衡度
            # 适应性温度策略：根据历史表现调整
            base_temp = 5.0
            if beta_t < 0.3:  # 负载均衡度较差时，适度提高温度
                tau_t = base_temp * 1.5
            else:
                tau_t = base_temp
        else:
            tau_t = 5.0  # 标准初始温度
        
        print(f"    [TEMPERATURE_DEBUG] 当前温度: {tau_t:.3f}")
        return tau_t

    def _apply_feedback(self, assignment_matrix, feedback_signal):
        """应用第四步反馈信号 - 自适应增强版本"""
        if feedback_signal is None:
            return assignment_matrix
            
        # 确保反馈信号在正确设备上
        feedback_signal = feedback_signal.float().to(assignment_matrix.device)
        
        # 修正的负载均衡度计算 - 考虑空分片
        shard_sizes = torch.sum(assignment_matrix, dim=0)
        
        # 计算空分片比例作为惩罚
        empty_shards = torch.sum(shard_sizes == 0).float()
        total_shards = torch.tensor(len(shard_sizes), dtype=torch.float32, device=shard_sizes.device)
        empty_ratio = empty_shards / total_shards
        
        # 只计算非空分片的均衡度
        non_empty_sizes = shard_sizes[shard_sizes > 0]
        if len(non_empty_sizes) <= 1:
            # 如果只有0或1个非空分片，这是极度不平衡的状态
            # 均衡度应该非常低，反映真实的不平衡程度
            if len(non_empty_sizes) == 1:
                # 只有1个分片：均衡度 = 1/总分片数 (理想情况下应该有多个分片)
                base_balance = 1.0 / total_shards.item()
            else:
                # 没有非空分片：完全错误的状态
                base_balance = 0.0
        else:
            # 计算非空分片间的均衡度
            base_balance = 1.0 - torch.std(non_empty_sizes) / (torch.mean(non_empty_sizes) + 1e-8)
        
        # 最终均衡度：如果有空分片，进一步降低
        if empty_ratio > 0:
            shard_balance = base_balance * (1.0 - empty_ratio)
        else:
            shard_balance = base_balance
        
        # 调试信息：显示详细的平衡度计算过程
        print(f"    [FEEDBACK_BALANCE_DEBUG] 中间分配-非空分片: {len(non_empty_sizes)}, 空分片: {empty_shards.item():.0f}, "
              f"基础均衡: {base_balance:.3f}, 最终均衡: {shard_balance:.3f}")
        
        adaptive_alpha = torch.clamp(self.feedback_alpha * (0.5 + 0.5 * shard_balance), 0.0, 0.5)
        
        # 检查维度匹配
        num_nodes, current_shards = assignment_matrix.shape
        feedback_nodes, feedback_shards = feedback_signal.shape
        
        print(f"    [FEEDBACK_DEBUG] Assignment: {assignment_matrix.shape}, Feedback: {feedback_signal.shape}")
        print(f"    [FEEDBACK_ADAPTIVE] 分片平衡度: {shard_balance.item():.3f}, 自适应系数: {adaptive_alpha.item():.3f}")
        
        # 处理节点数不匹配
        if feedback_nodes != num_nodes:
            print(f"    [FEEDBACK_ADJUST] 节点数不匹配 {feedback_nodes} vs {num_nodes}, 使用前{min(feedback_nodes, num_nodes)}个")
            min_nodes = min(feedback_nodes, num_nodes)
            feedback_signal = feedback_signal[:min_nodes]
            assignment_matrix = assignment_matrix[:min_nodes]
            
        # 处理分片数不匹配 - 智能维度调整策略
        if feedback_shards != current_shards:
            print(f"    [FEEDBACK_ADJUST] 分片数不匹配 {feedback_shards} vs {current_shards}")
            
            if feedback_shards > current_shards:
                # 反馈分片数多于当前分片数：智能选择最佳分片
                #  改进：综合考虑重要性和多样性
                shard_importance = torch.sum(feedback_signal, dim=0)  # 重要性
                shard_entropy = -torch.sum(F.softmax(feedback_signal, dim=1) * 
                                        torch.log(F.softmax(feedback_signal, dim=1) + 1e-8), dim=0)  # 多样性
                
                # 综合评分
                combined_scores = shard_importance + 0.3 * shard_entropy
                top_shards = torch.topk(combined_scores, current_shards)[1]
                feedback_adjusted = feedback_signal[:, top_shards]
                print(f"    [FEEDBACK_ADJUST] 智能选择 {current_shards} 个最佳分片: {top_shards.tolist()}")
                
            else:
                # 反馈分片数少于当前分片数：基于相似度的智能扩展
                if hasattr(self, '_cached_embeddings') and self._cached_embeddings is not None:
                    embeddings = self._cached_embeddings[:num_nodes].to(assignment_matrix.device)  # 确保维度和设备匹配
                    
                    # 计算当前分片和反馈分片的中心
                    current_centers = []
                    for i in range(current_shards):
                        shard_nodes = torch.where(torch.argmax(assignment_matrix, dim=1) == i)[0]
                        if len(shard_nodes) > 0:
                            center = torch.mean(embeddings[shard_nodes], dim=0)
                        else:
                            center = torch.randn(embeddings.size(1), device=embeddings.device)
                        current_centers.append(center)
                    current_centers = torch.stack(current_centers)
                    
                    feedback_centers = []
                    feedback_probs = F.softmax(feedback_signal, dim=1)
                    for i in range(feedback_shards):
                        shard_nodes = torch.where(torch.argmax(feedback_probs, dim=1) == i)[0]
                        if len(shard_nodes) > 0:
                            center = torch.mean(embeddings[shard_nodes], dim=0)
                        else:
                            center = torch.randn(embeddings.size(1), device=embeddings.device)
                        feedback_centers.append(center)
                    feedback_centers = torch.stack(feedback_centers)
                    
                    # 基于相似度映射扩展
                    similarity_matrix = F.cosine_similarity(
                        current_centers.unsqueeze(1), 
                        feedback_centers.unsqueeze(0), 
                        dim=2
                    )
                    
                    feedback_adjusted = torch.zeros(num_nodes, current_shards, device=assignment_matrix.device)
                    for i in range(current_shards):
                        weights = F.softmax(similarity_matrix[i] * 3, dim=0)  # 温度参数增强区分度
                        for j in range(feedback_shards):
                            feedback_adjusted[:, i] += weights[j] * feedback_probs[:, j]
                    
                    print(f"    [FEEDBACK_ADJUST] 基于嵌入相似度扩展分片数: {feedback_shards} -> {current_shards}")
                else:
                    # 如果没有嵌入信息，使用改进的插值方法
                    feedback_adjusted = F.interpolate(
                        feedback_signal.T.unsqueeze(0),  # [1, feedback_shards, num_nodes]
                        size=current_shards,
                        mode='linear',
                        align_corners=False
                    ).squeeze(0).T  # [num_nodes, current_shards]
                    # 确保插值结果在正确设备上
                    feedback_adjusted = feedback_adjusted.to(assignment_matrix.device)
                    print(f"    [FEEDBACK_ADJUST] 线性插值扩展分片数: {feedback_shards} -> {current_shards}")
        else:
            feedback_adjusted = feedback_signal.to(assignment_matrix.device)
            
        # 确保维度完全匹配
        if feedback_adjusted.shape != assignment_matrix.shape:
            print(f"    [FEEDBACK_ERROR] 维度仍不匹配: {feedback_adjusted.shape} vs {assignment_matrix.shape}")
            return assignment_matrix
            
        # 归一化反馈信号
        feedback_adjusted = F.softmax(feedback_adjusted, dim=1)
        
        # 常规融合
        result = (1 - adaptive_alpha) * assignment_matrix + adaptive_alpha * feedback_adjusted
        
        # 重新归一化确保是有效的概率分布
        result = F.softmax(result * 2, dim=1)  # 温度参数增强确定性
        
        # 评估反馈效果
        feedback_effect = torch.norm(result - assignment_matrix).item()
        print(f"    [FEEDBACK_APPLY] 反馈融合完成，影响强度: {feedback_effect:.4f}")
        return result