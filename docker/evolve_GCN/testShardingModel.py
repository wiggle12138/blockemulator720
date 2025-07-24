import os
import pickle
import sys
import torch.utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import EvolveGCNO
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EvolveGCNWrapper(nn.Module):
    """
    EvolveGCN-O封装器，用于处理时序节点嵌入
    主要功能：
    1. 输入处理：接收节点特征和邻接矩阵
    2. 特征投影：将输入特征映射到模型维度
    3. 时序处理：使用EvolveGCN-O处理时序数据
    4. 状态管理：维护LSTM隐藏状态用于时序一致性
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # 特征投影层（处理输入维度不匹配）
        self.feature_projector = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        # EvolveGCN-O核心模型
        self.evolve_gcn = EvolveGCNO(hidden_dim)
        
        # LSTM隐藏状态初始化
        # self.hidden_state = None
        self.hidden_dim = hidden_dim

    def reset_state(self):
        """重置LSTM隐藏状态，用于新序列开始"""
        # 换了时间步就要重置,reinitialize_weight和reset_parameters
        # 初始化用reset，重置weight用reinit_weight
        # self.hidden_state = None
        self.evolve_gcn.reinitialize_weight()
        # self.evolve_gcn.reset_parameters()

    def forward(self, x, edge_index):
        """
        前向传播
        参数:
            x: 节点特征 [num_nodes, feature_dim]
            edge_index: 边索引 [2, num_edges]
        返回:
            node_embeddings: 新节点嵌入 [num_nodes, hidden_dim]
        """
        # 特征维度匹配
        x = self.feature_projector(x)
        
        # # 处理隐藏状态（确保时序一致性）
        # if self.hidden_state is not None:
        #     # 分离隐藏状态（保留序列内梯度，断开序列间梯度）
        #     hidden, cell = self.evolve_gcn.recurrent_layer.hidden_state
        #     self.evolve_gcn.recurrent_layer.hidden_state = (hidden.detach(), cell.detach())
        
        # 通过EvolveGCN获取节点嵌入（不传递edge_weight）
        node_embeddings = self.evolve_gcn(x, edge_index)
        
        # 更新隐藏状态
        # self.hidden_state = self.evolve_gcn.recurrent_layer.hidden_state
        
        return node_embeddings

class BlockchainDataset(torch.utils.data.Dataset):
    """区块链时序数据集 - 生成多时间步数据"""

    def __init__(self, embedding_path, num_timesteps=10, noise_level=0.02):
        # 加载预训练嵌入
        with open(embedding_path, 'rb') as f:
            self.pretrained_embeddings = pickle.load(f)

        print(f"[SUCCESS] 加载预训练嵌入: {len(self.pretrained_embeddings)} 个节点")

        # 获取节点ID和基础嵌入
        self.node_ids = sorted([str(k) for k in self.pretrained_embeddings.keys()])
        self.num_nodes = len(self.node_ids)
        self.num_timesteps = num_timesteps
        self.noise_level = noise_level

        # 处理基础嵌入
        base_embeddings = []
        for node_id in self.node_ids:
            node_data = self.pretrained_embeddings[node_id]
            if isinstance(node_data, dict):
                embedding = node_data[0] if 0 in node_data else list(node_data.values())[0]
            else:
                embedding = node_data

            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            if embedding.ndim > 1:
                embedding = embedding.flatten()

            base_embeddings.append(embedding)

        # [CONFIG] 确保数据类型一致
        self.base_embeddings = np.stack(base_embeddings).astype(np.float32)
        self.embedding_dim = self.base_embeddings.shape[1]

        # 生成时序嵌入
        self._generate_temporal_embeddings()

        # 加载图结构
        self.edge_index = self._load_edge_index()

        print(f"[SUCCESS] 数据集创建完成: {self.num_nodes} 节点, {self.embedding_dim} 维, {num_timesteps} 时间步")

    def _generate_temporal_embeddings(self):
        """生成时序嵌入数据"""
        self.temporal_embeddings = {}

        for t in range(self.num_timesteps):
            # [CONFIG] 修复：正确处理标量运算
            # 添加时间相关变化
            noise = np.random.normal(0, self.noise_level, self.base_embeddings.shape).astype(np.float32)

            # trend和drift是标量，需要广播到数组形状
            trend_scalar = np.sin(t * 2 * np.pi / self.num_timesteps) * 0.05
            drift_scalar = (t / self.num_timesteps) * 0.02

            # 将标量广播到与base_embeddings相同的形状
            trend = np.full_like(self.base_embeddings, trend_scalar, dtype=np.float32)
            drift = np.full_like(self.base_embeddings, drift_scalar, dtype=np.float32)

            temporal_features = self.base_embeddings + noise + trend + drift
            self.temporal_embeddings[t] = temporal_features.astype(np.float32)

    def _load_edge_index(self):
        """加载图结构"""
        try:
            data = torch.load('step1_adjacency_raw.pt', map_location='cpu')
            edge_index = data['edge_index']
            print(f"[SUCCESS] 加载图结构: {edge_index.shape[1]} 条边")
            return edge_index
        except:
            print("[WARNING] 未找到图结构文件，使用全连接图")
            return self._build_complete_graph()

    def _build_complete_graph(self):
        """构建全连接图"""
        nodes = torch.arange(self.num_nodes)
        src = nodes.repeat(self.num_nodes)
        dst = nodes.repeat_interleave(self.num_nodes)
        return torch.stack([src, dst], dim=0)

    def __len__(self):
        return self.num_timesteps

    def __getitem__(self, idx):
        # [CONFIG] 确保返回数据类型一致
        node_features = torch.tensor(self.temporal_embeddings[idx], dtype=torch.float32)
        return node_features, self.edge_index.clone(), idx

class DynamicShardingModule(nn.Module):
    """
    基础分片决策模块
    功能：
    1. 动态预测分片数量
    2. 多目标损失函数（负载均衡、跨片交易、安全约束、迁移成本）
    3. 接收反馈信号并动态调整损失权重
    """
    def __init__(self, embedding_dim, max_shards=10, init_shards=3):
        """
        参数:
            embedding_dim: 节点嵌入维度
            n_shards: 分片数量
        """
        super().__init__()
        # 分片原型向量（可动态调整数量）
        self.shard_prototypes = nn.Parameter(torch.randn(max_shards, embedding_dim))
        self.current_shards = init_shards
        self.max_shards = max_shards

        # 动态分片数预测器（时间卷积网络）
        self.shard_predictor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, 1)
        )

        # 反馈融合参数
        self.feedback_alpha = nn.Parameter(torch.tensor(0.3))
        self.history_states = []  # 存储[负载均衡度, 跨片交易率, 安全阈值]
        self.cross_rate_increase_streak = 0  # 跨片率连续增长计数

        # 损失权重（动态可调）
        self.loss_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.2, 0.1]))  # α,β,γ,δ

        # 安全评分阈值
        self.security_threshold = 0.5

    def predict_shard_count(self, history_window=3):
        """预测最优分片数 K_t"""
        if len(self.history_states) < history_window:
            return self.current_shards

        # 构建时序特征 [序列长度, 特征维度]
        history = torch.stack(self.history_states[-3:]).unsqueeze(0)  # [1, 3, 3]

        # 时间卷积预测
        K_t = self.shard_predictor(history).squeeze()
        K_t = torch.clamp(K_t, min=2, max=self.shard_prototypes.size(0)).int().item()
        return K_t

    def forward(self, node_embeddings, prev_assignment=None, feedback=None):
        """
        分配节点到分片
        参数:
            node_embeddings: 节点嵌入 [num_nodes, embedding_dim]
            prev_assignment: 上一时刻分配矩阵 [num_nodes, n_shards]
            feedback: 第四步反馈的调整矩阵 [num_nodes, n_shards]
        返回:
            shard_assignment: 节点-分片关联矩阵 [num_nodes, n_shards]
        """
        # 预测当前分片数
        K_t = self.predict_shard_count()
        self.current_shards = K_t

        # 获取当前分片原型
        prototypes = self.shard_prototypes[:K_t]  # [K_t, emb_dim]

        # 计算节点到各分片的相似度
        similarity = F.cosine_similarity(
            node_embeddings.unsqueeze(1),  # [num_nodes, 1, emb_dim]
            prototypes.unsqueeze(0), # [1, n_shards, emb_dim]
            dim=2
        )
        assignment = F.softmax(similarity, dim=1)

        # 融合反馈信号（文档要求：S_t = (1-α)S_t + α feedback）
        if feedback is not None and feedback.shape[1] >= K_t:
            α = torch.sigmoid(self.feedback_alpha)
            assignment = (1 - α) * assignment + α * feedback[:, :K_t]

        return assignment

    def compute_loss(self, assignment, edge_index, security_scores, prev_assignment):
        """
        计算多目标分片损失
        实现文档定义的四元损失函数：
        L_shard = αL_balance + βL_cross + γL_security + δL_migrate
        """
        num_nodes = assignment.shape[0]
        K_t = self.current_shards

        # 1. 负载均衡损失 L_balance
        shard_counts = assignment.sum(dim=0)  # 各分片节点数 [K_t]
        ideal_count = num_nodes / K_t
        L_balance = torch.sum((shard_counts - ideal_count) ** 2) / num_nodes

        # 2. 跨片交易损失 L_cross
        src, dst = edge_index
        # 获取源节点和目标节点的分片标签
        src_shard = torch.argmax(assignment[src], dim=1)  # 源节点分片
        dst_shard = torch.argmax(assignment[dst], dim=1)  # 目标节点分片
        cross_mask = (src_shard != dst_shard).float()     # 跨片交易标志
        L_cross = torch.mean(cross_mask)                  # 跨片交易比例

        # 3. 安全约束损失 L_security
        # 计算各分片平均安全分
        shard_security = torch.einsum('nk,n->k', assignment, security_scores) / (shard_counts + 1e-8)
        # 确保所有分片安全分>阈值（文档设置0.5）
        min_security = torch.min(shard_security)
        L_security = F.relu(self.security_threshold - min_security)

        # 4. 迁移成本损失 L_migrate
        if prev_assignment is not None and prev_assignment.shape[1] >= K_t:
            # 计算节点迁移量（分配矩阵变化量）
            migration_cost = torch.norm(assignment - prev_assignment[:, :K_t], p=1, dim=1)
            L_migrate = torch.mean(migration_cost)
        else:
            L_migrate = torch.tensor(0.0)

        # 加权总损失（动态权重）
        α, β, γ, δ = torch.softmax(self.loss_weights, dim=0)
        total_loss = (α * L_balance + β * L_cross + γ * L_security + δ * L_migrate)

        return total_loss, {
            "L_balance": L_balance,
            "L_cross": L_cross,
            "L_security": L_security,
            "L_migrate": L_migrate
        }

    def update_history(self, metrics):
        """
        更新历史状态（文档要求）
        metrics: dict包含负载均衡度、跨片交易率、安全阈值
        """
        # 更新跨片率连续增长计数
        if len(self.history_states) > 0:
            last_cross_rate = self.history_states[-1][1].item()
            if metrics["cross_rate"] > last_cross_rate:
                self.cross_rate_increase_streak += 1
            else:
                self.cross_rate_increase_streak = 0

        state = torch.tensor([
            metrics.get("balance", 0.0),
            metrics.get("cross_rate", 0.0),
            metrics.get("security", 1.0)
        ])
        self.history_states.append(state)

        # 保持最近10个状态
        if len(self.history_states) > 10:
            self.history_states.pop(0)

    def update_weights_from_feedback(self, metrics):
        """
        根据第四步反馈动态调整损失权重
        文档要求：当负载不均衡时增加α，跨片交易率高时增加β等
        """
        new_weights = self.loss_weights.detach().clone()

        # 规则1：负载不均衡时增强平衡权重
        if metrics["balance"] > 0.15:  # 负载均衡度阈值
            new_weights[0] *= 1.2  # 增加α权重

        # 规则2：跨片交易率高时降低跨片权重
        if metrics["cross_rate"] > 0.3:
            new_weights[1] *= 0.8  # 降低β权重

        # 规则3：安全评分低时增强安全权重
        if metrics["security"] < self.security_threshold:
            new_weights[2] *= 1.5

        # 规则4：跨片率连续增长时增加基础分片数
        if self.cross_rate_increase_streak >= 3:
            self.current_shards = min(self.current_shards + 1, self.max_shards)
            self.cross_rate_increase_streak = 0  # 重置计数

        # 应用新权重（带约束）
        new_weights = torch.clamp(new_weights, 0.05, 1.0)
        self.loss_weights.data = new_weights


def process_data_loader(loader, model, device):
    """
    处理数据加载器，生成新的节点嵌入
    参数:
        loader: 数据加载器
        model: 训练好的EvolveGCN模型
        device: 计算设备
    返回:
        new_embeddings: 新节点嵌入字典 {时间步: 嵌入矩阵}
    """
    model.eval()
    model.reset_state()
    new_embeddings = {}
    
    with torch.no_grad():
        # 修改：正确传递时间步信息
        for i, (node_features, edge_index, node_mask, timestep) in enumerate(loader):
            # 移动到设备
            node_features = node_features.to(device)
            edge_index = edge_index.to(device)
            
            # 获取新嵌入（不传递edge_weight）
            embeddings = model(node_features, edge_index)
            
            # 转换为numpy并保存
            # 使用实际的时间步作为键
            new_embeddings[timestep] = embeddings.cpu().numpy()
    
    return new_embeddings

def evaluate_shard_performance(assignment, edge_index, security_scores):
    """
    评估分片性能指标
    返回:
        metrics: 包含负载均衡度、跨片交易率、安全阈值
    """
    # 1. 负载均衡度
    shard_counts = assignment.sum(dim=0)
    ideal_count = assignment.shape[0] / assignment.shape[1]
    balance = torch.std(shard_counts / ideal_count).item()

    # 2. 跨片交易率
    src, dst = edge_index
    src_shard = torch.argmax(assignment[src], dim=1)
    dst_shard = torch.argmax(assignment[dst], dim=1)
    cross_rate = (src_shard != dst_shard).float().mean().item()

    # 3. 安全阈值
    security = torch.min(torch.einsum('nk,n->k', assignment, security_scores) / shard_counts).item()

    return {
        "balance": balance,
        "cross_rate": cross_rate,
        "security": security
    }

def generate_feedback_matrix(metrics, assignment):
    """
    生成反馈矩阵（供下一时间步使用）
    文档要求：反馈矩阵注入分片决策
    """
    adjustment = torch.ones_like(assignment)

    # 负载不均衡时增强小分片的权重
    if metrics['balance'] > 0.1:
        shard_counts = assignment.sum(dim=0)
        min_idx = torch.argmin(shard_counts)
        adjustment[:, min_idx] *= 1.3

    # 安全评分低时增强高安全节点的权重
    if metrics['security'] < 0.5:
        security_scores = assignment @ torch.arange(assignment.shape[1]).float().to(assignment.device)
        high_security_nodes = security_scores > security_scores.median()
        adjustment[high_security_nodes] *= 1.2

    return adjustment * assignment

def get_security_scores(num_nodes):
    """
    模拟获取节点安全评分
    实际应用中应从安全模块获取
    """
    # 生成随机安全评分（0-1之间）
    return torch.rand(num_nodes).to(device)

def main(lr=0.005, epochs=10, max_shards=10, init_shards=3):
    """主训练函数"""
    # 创建数据集
    dataset = BlockchainDataset("./data/temporal_embeddings.pkl")
    
    # 创建数据加载器 (batch_size=1 因为每个时间步结构不同)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 确定输入维度
    # sample_features, _, _, _ = dataset[0]
    # input_dim = sample_features.shape[1]
    # print("wrs",input_dim)
    hidden_dim = 64  # 嵌入维度
    input_dim = dataset.embedding_dim
    
    # 初始化模型
    model = EvolveGCNWrapper(input_dim, hidden_dim).to(device)

    # 初始化分片决策模块
    sharding_module = DynamicShardingModule(hidden_dim, max_shards, init_shards).to(device)

    # 联合优化器
    optimizer = optim.Adam(
        list(model.parameters()) + list(sharding_module.parameters()),
        lr=lr
    )


    # 训练循环
    for epoch in range(epochs):
        model.train()
        model.reset_state()  # 每个epoch重置状态
        sharding_module.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_shard_loss = 0.0
        prev_assignment = None  # 保存上一时刻分配
        feedback_buffer = None  # 反馈矩阵缓冲区

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", file=sys.stdout)
        
        for node_features, edge_index, _, timestep in train_bar:
            # 准备数
            node_features = node_features.to(device)
            edge_index = edge_index.to(device)
            
            # 检查维度（如果需要）
            if edge_index.dim() > 2:
                edge_index = edge_index.squeeze(0)
            
            # 梯度清零
            optimizer.zero_grad()

            # 1. EvolveGCN前向传播
            new_embeddings = model(node_features, edge_index)

            # 2. 获取安全评分（模拟）
            security_scores = get_security_scores(new_embeddings.shape[0])

            # 3. 分片决策（传入上一时刻分配和反馈）
            assignment = sharding_module(
                new_embeddings,
                prev_assignment,
                feedback_buffer
            )

            # 4. 计算分片损失
            loss_shard, loss_components = sharding_module.compute_loss(
                assignment, edge_index, security_scores, prev_assignment
            )

            # 5. 计算重构损失
            loss_recon = F.mse_loss(new_embeddings, node_features)

            # 6. 总损失
            total_loss = loss_recon + loss_shard

            # 7. 反向传播
            total_loss.backward()
            optimizer.step()

            # 8. 评估性能指标
            metrics = evaluate_shard_performance(assignment, edge_index, security_scores)

            # 9. 更新历史状态
            sharding_module.update_history(metrics)

            # 10. 生成反馈矩阵（供下一时间步使用）
            feedback_buffer = generate_feedback_matrix(metrics, assignment)

            # 保存当前分配
            prev_assignment = assignment.detach()

            # 更新进度
            epoch_loss += total_loss.item()
            epoch_recon_loss += loss_recon.item()
            epoch_shard_loss += loss_shard.item()

            train_bar.set_postfix(
                total_loss=f"{total_loss.item():.4f}",
                recon_loss=f"{loss_recon.item():.4f}",
                shard_loss=f"{loss_shard.item():.4f}",
                balance=f"{metrics['balance']:.4f}",
                cross_rate=f"{metrics['cross_rate']:.4f}"
            )

        # 每轮结束后动态调整损失权重
        sharding_module.update_weights_from_feedback(metrics)

        print(f"Epoch {epoch+1}/{epochs} | Avg Total Loss: {epoch_loss/len(dataset):.4f}")
        print(f"  Recon Loss: {epoch_recon_loss/len(dataset):.4f} | Shard Loss: {epoch_shard_loss/len(dataset):.4f}")
        print(f"  Current Shards: {sharding_module.current_shards} | Loss Weights: {sharding_module.loss_weights.detach().cpu().numpy()}")

    # 保存最终模型
    save_path = "trained_models"
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "evolvegcn_model.pth"))
    torch.save(sharding_module.state_dict(), os.path.join(save_path, "sharding_module.pth"))
    # 生成新嵌入
    print("生成新节点嵌入...")
    new_embeddings = process_data_loader(train_loader, model, device)
    
    # 保存新嵌入
    with open("new_temporal_embeddings.pkl", "wb") as f:
        pickle.dump(new_embeddings, f)
    
    # ====== 分片决策流程 ======
    print("执行分片决策...")
    sharding_results = {}
    prev_assignment = None
    
    for timestep, embeddings in new_embeddings.items():
        # 转换为PyTorch张量
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
        security_scores = get_security_scores(embeddings_tensor.shape[0])

        # 获得分片关联矩阵
        with torch.no_grad():
            # 获取分片关联矩阵
            assignment = sharding_module(embeddings_tensor, prev_assignment)
            shard_labels = torch.argmax(assignment, dim=1)
            assignment_matrix = F.one_hot(shard_labels, num_classes=sharding_module.current_shards).float()

            # 评估性能指标
            metrics = evaluate_shard_performance(assignment, None, security_scores)

            # 生成下一时间步的反馈
            feedback_buffer = generate_feedback_matrix(metrics, assignment)
            prev_assignment = assignment
        
        # 存储结果
        sharding_results[timestep] = {
            'embeddings': embeddings,
            'shard_assignment': assignment_matrix.cpu().numpy(),
            'shard_labels': shard_labels.cpu().numpy(),
            'shard_count': sharding_module.current_shards,
            'metrics': metrics
        }
    
    # 保存分片结果
    with open("sharding_results.pkl", "wb") as f:
        pickle.dump(sharding_results, f)

        # 打印分片结果统计
    for key, value in sharding_results.items():
        print(f"  - {key}: {len(value)} 节点 {value}")
    
    print("分片决策完成! 结果已保存到 sharding_results.pkl")
    print(f"最终分片配置: 最大分片数={max_shards}, 初始分片数={init_shards}")

if __name__ == "__main__":
    # 可配置参数
    config = {
        'lr': 0.001,          # 学习率
        'epochs': 10,          # 训练轮数
        'max_shards': 10,       # 最大分片数量
        'init_shards': 3,       # 初始分片数量
    }
    
    # 运行主函数
    main(
        lr=config['lr'],
        epochs=config['epochs'],
        max_shards=config['max_shards'],
        init_shards=config['init_shards']
    )