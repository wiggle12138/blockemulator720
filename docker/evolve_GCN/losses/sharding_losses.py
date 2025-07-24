"""
分片相关损失函数
"""
import torch
import torch.nn.functional as F


def multi_objective_sharding_loss(shard_assignment, embeddings, edge_index,
                                  prev_assignment=None, security_scores=None,
                                  a=1.0, b=1.0, c=1.5, d=0.5):
    """
    多目标分片损失函数
    L_shard = a*L_balance + b*L_cross + c*L_security + d*L_migrate

    Args:
        shard_assignment: 分片分配矩阵 [num_nodes, n_shards]
        embeddings: 节点嵌入 [num_nodes, embedding_dim]
        edge_index: 边索引 [2, num_edges]
        prev_assignment: 前一轮分配矩阵
        security_scores: 安全评分
        a, b, c, d: 损失权重

    Returns:
        total_loss: 总损失
        loss_components: 损失组件字典
    """
    num_nodes, n_shards = shard_assignment.shape
    device = embeddings.device

    # 确保所有张量数据类型一致
    shard_assignment = shard_assignment.float()
    embeddings = embeddings.float()

    # 1. 负载均衡损失 - 极端强化版本 (极端惩罚机制)
    shard_counts = torch.sum(shard_assignment, dim=0)
    
    # 检测分片使用情况
    non_empty_shards = torch.sum(shard_counts > 0.1)  # 计算非空分片数
    
    if non_empty_shards <= 1:
        # 极端单一分片惩罚 - 极端强化版本
        L_balance = torch.tensor(100000.0, dtype=torch.float32, device=device)
        print(f"    [EXTREME_BALANCE_PENALTY] 极端惩罚单一分片: {L_balance.item()}")
    elif non_empty_shards <= 2:
        # 双分片也给予极端惩罚
        L_balance = torch.tensor(80000.0, dtype=torch.float32, device=device)
        print(f"    [EXTREME_BALANCE_PENALTY] 极端惩罚双分片: {L_balance.item()}")
    elif non_empty_shards < n_shards * 0.7:  # 提高阈值，更严格
        # 空分片过多时的极端强化惩罚
        penalty_factor = 60000.0 * (1.0 - non_empty_shards / n_shards)
        L_balance = torch.tensor(penalty_factor, dtype=torch.float32, device=device)
        print(f"    [EXTREME_BALANCE_PENALTY] 极端惩罚空分片: {L_balance.item()}, 非空分片: {non_empty_shards.item()}/{n_shards}")
    else:
        # 正常的负载均衡计算
        mu = torch.tensor(num_nodes / n_shards, dtype=torch.float32, device=device)
        sigma = torch.std(shard_counts) + 1e-8
        base_balance = torch.sum(((shard_counts - mu) / sigma) ** 2)
        
        # 轻微的多样性奖励
        diversity_bonus = (non_empty_shards / n_shards)
        L_balance = base_balance * (2.0 - diversity_bonus)  # 分片越多，损失越小
        
        print(f"    [BALANCE_DEBUG] 基础损失: {base_balance.item():.2f}, 多样性奖励: {diversity_bonus:.2f}, 最终: {L_balance.item():.2f}")

    # 2. 跨片交易损失
    L_cross = torch.tensor(0.0, dtype=torch.float32, device=device)
    if edge_index is not None:
        hard_assignment = torch.argmax(shard_assignment, dim=1)
        src_shards = hard_assignment[edge_index[0]]
        dst_shards = hard_assignment[edge_index[1]]
        cross_indicator = (src_shards != dst_shards).float()
        L_cross = torch.mean(cross_indicator)

    # 3. 安全约束损失
    L_security = torch.tensor(0.0, dtype=torch.float32, device=device)
    if security_scores is not None:
        security_scores = security_scores.float()
        rho_th = 0.3  # 安全阈值
        min_rho = float('inf')
        for k in range(n_shards):
            shard_weights = shard_assignment[:, k]
            if torch.sum(shard_weights) > 1e-8:
                rho_k = torch.sum(shard_weights * security_scores) / torch.sum(shard_weights)
                min_rho = min(min_rho, rho_k.item())

        if min_rho != float('inf'):
            L_security = torch.relu(torch.tensor(rho_th - min_rho, dtype=torch.float32, device=device))

    # 4. 迁移成本损失 - 处理维度不匹配问题
    L_migrate = torch.tensor(0.0, dtype=torch.float32, device=device)
    if prev_assignment is not None:
        prev_assignment = prev_assignment.float()

        # 处理分片数变化的情况
        if prev_assignment.size(1) != shard_assignment.size(1):
            # 当分片数发生变化时，计算节点级别的变化
            prev_hard = torch.argmax(prev_assignment, dim=1)
            curr_hard = torch.argmax(shard_assignment, dim=1)

            # 计算节点分片变化的比例
            changed_nodes = (prev_hard != curr_hard).float()
            L_migrate = torch.mean(changed_nodes)
        else:
            # 分片数相同时，使用原来的计算方法
            diff_matrix = shard_assignment * (1 - prev_assignment)
            L_migrate = torch.norm(diff_matrix, p='fro') ** 2

    # 组合损失
    total_loss = a * L_balance + b * L_cross + c * L_security + d * L_migrate

    return total_loss, {
        'balance': L_balance.item(),
        'cross': L_cross.item(),
        'security': L_security.item(),
        'migrate': L_migrate.item()
    }


def temporal_consistency_loss(embeddings_t, embeddings_t_minus_1, lambda_contrast=0.1):
    """
    时序一致性损失 + 对比损失正则项

    Args:
        embeddings_t: 当前时刻嵌入
        embeddings_t_minus_1: 前一时刻嵌入
        lambda_contrast: 对比损失权重

    Returns:
        loss: 总损失
    """
    # 确保数据类型一致
    embeddings_t = embeddings_t.float()
    embeddings_t_minus_1 = embeddings_t_minus_1.float()

    mse_loss = F.mse_loss(embeddings_t, embeddings_t_minus_1)

    # 添加对比损失正则项（简化版）
    normalized_t = F.normalize(embeddings_t, p=2, dim=1)
    normalized_t_1 = F.normalize(embeddings_t_minus_1, p=2, dim=1)
    contrast_loss = 1 - F.cosine_similarity(normalized_t, normalized_t_1).mean()

    return mse_loss + lambda_contrast * contrast_loss