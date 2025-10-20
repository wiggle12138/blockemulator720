"""
EvolveGCN包装器 - 实现参数演化和衔接机制
"""
import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import EvolveGCNO


class EvolveGCNWrapper(nn.Module):
    """增强的EvolveGCN包装器 - 实现参数演化和衔接机制"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        #  修复：特征投影层 - 确保输出维度正确
        self.feature_projector = nn.Linear(input_dim, hidden_dim, dtype=torch.float32)

        # EvolveGCN-O核心模型
        self.evolve_gcn = EvolveGCNO(hidden_dim)

        # 增量信号处理器
        self.delta_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim, dtype=torch.float32),
            nn.Tanh()
        )

        #  修复：反馈信号融合层 - 基于投影后的hidden_dim
        feedback_dim = 11  # 4+6+1维反馈向量
        self.feedback_fusion = nn.Sequential(
            nn.Linear(hidden_dim + feedback_dim, hidden_dim, dtype=torch.float32),  # 修复：使用hidden_dim而不是input_dim
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),  # 修复：输出也是hidden_dim
            nn.LayerNorm(hidden_dim, dtype=torch.float32)  # 修复：LayerNorm维度
        )   
    
        # 反馈历史追踪
        self.feedback_history = []
        self.max_feedback_history = 10

        # LSTM状态更新器
        self.lstm_updater = nn.LSTM(
            input_size=hidden_dim + 3,  # 嵌入 + 性能指标
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            dtype=torch.float32
        )

        # 状态管理
        self.prev_embeddings = None
        self.lstm_hidden = None
        
        #  缓存第四步反馈，避免重复加载
        self._cached_feedback = None
        self._feedback_loaded = False

        print(f" EvolveGCNWrapper 初始化:")
        print(f"  输入维度: {input_dim} -> 投影到: {hidden_dim}")
        print(f"  反馈融合: {hidden_dim + feedback_dim} -> {hidden_dim}")

    def reset_state(self, device=None):
        """重置所有状态"""
        if device is None:
            device = next(self.parameters()).device

        self.evolve_gcn.reinitialize_weight()
        self.prev_embeddings = None
        self.lstm_hidden = None
        
        #  重置反馈缓存
        self._cached_feedback = None
        self._feedback_loaded = False

    def compute_delta_signal(self, current_embeddings):
        """计算增量信号 ΔZ_t = ||Z_t - Z_{t-1}||_2"""
        current_embeddings = current_embeddings.float()

        if self.prev_embeddings is None:
            delta = torch.zeros_like(current_embeddings)
        else:
            prev_embeddings = self.prev_embeddings.float()
            delta_norm = torch.norm(current_embeddings - prev_embeddings, p=2, dim=1, keepdim=True)
            delta = delta_norm.expand(-1, current_embeddings.size(1))

        processed_delta = self.delta_processor(delta)
        return processed_delta

    def forward(self, x, edge_index, performance_feedback=None):
        """
        前向传播 - 实现参数演化和衔接机制
        """
        # 确保输入数据类型一致
        x = x.float()
        
        #  减少调试信息输出频率
        debug_interval = getattr(self, '_debug_counter', 0)
        self._debug_counter = debug_interval + 1
        
        if debug_interval % 10 == 0:  # 每10次调用输出一次调试信息
            print(f" 调试信息 (第{debug_interval+1}次):")
            print(f"  原始输入维度: {x.shape}")

        # 1.  特征投影 - 先投影到hidden_dim
        x = self.feature_projector(x)
        
        if debug_interval % 10 == 0:
            print(f"  投影后维度: {x.shape}")

        # 2.  优化：使用缓存机制加载第四步反馈，避免重复加载
        if performance_feedback is None:
            if not self._feedback_loaded:
                self._cached_feedback = self._load_step4_feedback()
                self._feedback_loaded = True
                if self._cached_feedback is not None:
                    print(f" 首次加载第四步反馈: {self._cached_feedback.shape}")
            performance_feedback = self._cached_feedback

        # 3.  处理性能反馈 - 在投影后进行融合
        if performance_feedback is not None:
            # 确保设备一致性
            performance_feedback = performance_feedback.to(x.device)

            # 处理反馈向量维度（确保11维）
            performance_feedback = self._process_feedback_vector(performance_feedback)
            performance_feedback = performance_feedback.float()
            
            if debug_interval % 10 == 0:  #  减少调试输出频率
                print(f"  反馈向量维度: {performance_feedback.shape}")

            #  分析反馈内容
            if performance_feedback.size(0) >= 11:
                core_metrics = performance_feedback[:4]  # 核心4维
                feature_qualities = performance_feedback[4:10]  # 6类特征质量
                combined_score = performance_feedback[10]  # 综合分数

            if debug_interval % 10 == 0:
                print(f"  反馈分析 (第{debug_interval+1}次):")
                print(f"  负载均衡: {core_metrics[0]:.3f}")
                print(f"  跨片率: {core_metrics[1]:.3f}")
                print(f"  安全性: {core_metrics[2]:.3f}")
                print(f"  特征质量: {core_metrics[3]:.3f}")
                print(f"  综合分数: {combined_score:.3f}")

            #  修复：融合反馈信号 - 使用投影后的x
            feedback_expanded = performance_feedback.unsqueeze(0).expand(x.size(0), -1)
            x_with_feedback = torch.cat([x, feedback_expanded], dim=1)
            
            if debug_interval % 10 == 0:
                print(f"  融合前维度: x={x.shape}, feedback_expanded={feedback_expanded.shape}")
                print(f"  拼接后维度: {x_with_feedback.shape}")
            
            x = self.feedback_fusion(x_with_feedback)
            
            if debug_interval % 10 == 0:
                print(f"  融合后维度: {x.shape}")

        # 4. EvolveGCN处理（参数演化: W_t = GRU(W_{t-1}, ΔZ_t)）
        embeddings = self.evolve_gcn(x, edge_index)
        
        if debug_interval % 10 == 0:
            print(f"  EvolveGCN输出维度: {embeddings.shape}")

        # 5. 计算增量信号
        delta_signal = self.compute_delta_signal(embeddings)

        # 6. LSTM状态更新时使用兼容的3维反馈
        if performance_feedback is not None:
            # 构建LSTM输入：[嵌入均值, 核心前3维反馈用于兼容]
            global_embedding = torch.mean(embeddings, dim=0)  # [hidden_dim]
            
            # 取前3维核心反馈用于LSTM，保持向后兼容
            if performance_feedback.size(0) >= 3:
                core_feedback = performance_feedback[:3]  # [3] 负载均衡, 跨片率, 安全性
            else:
                core_feedback = torch.zeros(3, device=x.device, dtype=torch.float32)
            
            lstm_input = torch.cat([global_embedding, core_feedback]).unsqueeze(0).unsqueeze(0)
            
            # 确保LSTM输入数据类型和设备一致
            lstm_input = lstm_input.float().to(x.device)
    
            # LSTM更新
            if self.lstm_hidden is not None:
                # 确保隐藏状态在正确设备上
                h, c = self.lstm_hidden
                self.lstm_hidden = (h.to(x.device), c.to(x.device))
                
            lstm_output, self.lstm_hidden = self.lstm_updater(lstm_input, self.lstm_hidden)
    
        # 7. 保存当前嵌入用于下一时间步
        self.prev_embeddings = embeddings.detach().clone()
    
        return embeddings, delta_signal

    def _load_step4_feedback(self):
        """从第四步加载性能反馈"""
        #  减少调试信息输出频率
        debug_interval = getattr(self, '_debug_counter', 0)
        
        try:
            import pickle
            from pathlib import Path

            feedback_file = Path("step3_performance_feedback.pkl")
            if feedback_file.exists():
                with open(feedback_file, "rb") as f:
                    feedback_data = pickle.load(f)

                # 按照feedback2.py的实际格式提取
                if 'temporal_performance' in feedback_data:
                    temporal_perf = feedback_data['temporal_performance']

                    # 构建11维反馈向量：4维核心 + 6维特征质量 + 1维综合分数
                    feedback_vector = (
                        temporal_perf['performance_vector'] +      # [4] 负载均衡, 跨片率, 安全性, 特征质量
                        temporal_perf['feature_qualities'] +       # [6] 6类特征质量
                        [temporal_perf['combined_score']]          # [1] 综合分数
                    )

                    if debug_interval % 10 == 0:  #  只在调试间隔时输出详细信息
                        print(f" 加载第四步反馈: {len(feedback_vector)}维")
                        print(f"  核心指标: {temporal_perf['performance_vector']}")
                        print(f"  特征质量: {[f'{x:.3f}' for x in temporal_perf['feature_qualities']]}")
                        print(f"  综合分数: {temporal_perf['combined_score']:.3f}")

                    # 确保返回的张量在正确设备上
                    device = next(self.parameters()).device
                    return torch.tensor(feedback_vector, dtype=torch.float32, device=device)
                else:
                    if debug_interval % 10 == 0:
                        print("[WARNING] 反馈数据格式不正确，缺少temporal_performance")
                    return None

            else:
                if debug_interval % 10 == 0:
                    print("[WARNING] 第四步反馈文件不存在")
                return None

        except Exception as e:
            if debug_interval % 10 == 0:
                print(f"[WARNING] 加载第四步反馈失败: {e}")
            return None

    def _process_feedback_vector(self, feedback_vector):
        """处理反馈向量，确保维度正确"""
        if isinstance(feedback_vector, (list, tuple)):
            feedback_vector = torch.tensor(feedback_vector, dtype=torch.float32)

        # 确保反馈向量在正确设备上
        device = next(self.parameters()).device
        feedback_vector = feedback_vector.to(device)

        # 确保是11维向量
        if feedback_vector.size(0) != 11:
            print(f"[WARNING] 反馈向量维度异常: {feedback_vector.size(0)}，期望11维")
            # 填充或截断到11维
            if feedback_vector.size(0) < 11:
                padding_size = 11 - feedback_vector.size(0)
                padding = torch.zeros(padding_size, device=device, dtype=torch.float32)
                feedback_vector = torch.cat([feedback_vector, padding])
            else:
                feedback_vector = feedback_vector[:11]

        return feedback_vector