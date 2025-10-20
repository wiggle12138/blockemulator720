"""
训练配置文件
"""


class TrainingConfig:
    """训练配置类"""

    def __init__(self):
        # 基础训练参数
        self.lr = 0.001
        self.epochs = 30
        self.num_timesteps = 10
        self.base_shards = 3
        self.hidden_dim = 128

        # 数据集参数
        self.embedding_path = "../muti_scale/temporal_embeddings.pkl"
        self.edge_index_path = "../partition/feature/step1_adjacency_raw.pt"
        self.noise_level = 0.02

        # 模型参数
        self.max_shards = 10
        self.n_heads = 8

        # 优化器参数
        self.weight_decay = 1e-5
        self.shard_lr = 0.001

        # 损失函数权重
        self.balance_weight = 0.5
        self.cross_weight = 1.5
        self.security_weight = 0.5
        self.migrate_weight = 0.5

        # 其他参数
        self.max_grad_norm = 1.0
        self.history_length = 10
        self.print_freq = 5

        # 空分片处理参数
        self.min_shard_size = 5          # 最小分片大小
        self.max_empty_ratio = 0.2       # 最大空分片比例
        self.shard_merge_threshold = 3   # 分片合并阈值
        self.enable_empty_shard_handling = True  # 启用空分片处理

        # 输出路径
        self.output_dir = "./outputs"
        self.model_dir = "./trained_models"

    def to_dict(self):
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def print_config(self):
        """打印配置信息"""
        print(" 训练配置:")
        for key, value in self.to_dict().items():
            print(f"  {key}: {value}")


# 默认配置
default_config = TrainingConfig()