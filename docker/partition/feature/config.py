"""
特征提取配置文件
"""

class FeatureDimensions:
    # === 全面数值特征配置 ===
    # 硬件规格特征
    HARDWARE_CPU_DIM = 4          # CPU: 核心数、频率、缓存、架构
    HARDWARE_MEMORY_DIM = 3       # 内存: 容量、带宽、类型编码
    HARDWARE_STORAGE_DIM = 3      # 存储: 容量、读写速度、类型编码
    HARDWARE_NETWORK_DIM = 3      # 网络: 上游/下游带宽、延迟
    OPERATIONAL_STATUS_DIM = 4    # 运营: 在线时间、资格、CPU/内存使用率

    # 链上行为特征
    TRANSACTION_CAPABILITY_DIM = 6  # 交易能力: TPS、延迟、资源消耗等
    CROSS_SHARD_DIM = 2            # 跨分片: 节点间/分片间交易量
    BLOCK_GENERATION_DIM = 2       # 区块生成: 间隔、标准差
    ECONOMIC_DIM = 1               # 经济: 费用贡献率
    SMART_CONTRACT_DIM = 1         # 智能合约: 调用频率
    TRANSACTION_TYPES_DIM = 2      # 交易类型: 普通/合约交易比例
    CONSENSUS_DIM = 3              # 共识: 参与率、奖励、成功率

    # 网络拓扑特征
    GEO_LOCATION_DIM = 3          # 地理: 区域、时区、数据中心编码
    CONNECTIONS_DIM = 4           # 连接: 分片内外连接、权重度、活跃连接
    HIERARCHY_DIM = 2             # 层次: 深度、连接密度
    CENTRALITY_DIM = 4            # 中心性: 特征向量、接近、介数、影响力
    SHARD_ALLOCATION_DIM = 7      # 分片分配: 优先级、适应性、偏好统计

    # 动态属性特征
    COMPUTE_DIM = 3               # 计算: CPU/内存使用、资源波动
    STORAGE_DYNAMIC_DIM = 2       # 存储: 可用空间、利用率
    NETWORK_DYNAMIC_DIM = 3       # 网络: 延迟波动、平均延迟、带宽使用
    TRANSACTIONS_DYNAMIC_DIM = 3  # 交易: 频率、处理延迟、质押变化
    REPUTATION_DIM = 2            # 声誉: 在线时间、声誉分数

    # 异构类型特征
    NODE_TYPE_DIM = 5             # 节点类型: one-hot编码
    FUNCTION_TAGS_DIM = 5         # 功能标签: 数量及类型分布
    SUPPORTED_FUNCS_DIM = 3       # 支持功能: 数量、优先级统计
    APPLICATION_DIM = 4           # 应用: 状态、负载指标

    # 计算总维度
    NUMERIC_DIM = (HARDWARE_CPU_DIM + HARDWARE_MEMORY_DIM + HARDWARE_STORAGE_DIM +
                   HARDWARE_NETWORK_DIM + OPERATIONAL_STATUS_DIM +
                   TRANSACTION_CAPABILITY_DIM + CROSS_SHARD_DIM + BLOCK_GENERATION_DIM +
                   ECONOMIC_DIM + SMART_CONTRACT_DIM + TRANSACTION_TYPES_DIM + CONSENSUS_DIM +
                   GEO_LOCATION_DIM + CONNECTIONS_DIM + HIERARCHY_DIM + CENTRALITY_DIM +
                   SHARD_ALLOCATION_DIM +
                   COMPUTE_DIM + STORAGE_DYNAMIC_DIM + NETWORK_DYNAMIC_DIM +
                   TRANSACTIONS_DYNAMIC_DIM + REPUTATION_DIM +
                   NODE_TYPE_DIM + FUNCTION_TAGS_DIM + SUPPORTED_FUNCS_DIM + APPLICATION_DIM)

    # === 其他特征配置 (保持不变) ===
    CATEGORICAL_DIM = 15          # 分类特征维度 (减少，因为很多已包含在数值特征中)

    # 序列和图特征配置
    MAX_SEQUENCE_LENGTH = 50
    SEQUENCE_FEATURE_DIM = 32
    MAX_NEIGHBORS = 20
    NEIGHBOR_FEATURE_DIM = 10
    GRAPH_FEATURE_DIM = 10

    # 总的经典特征维度
    CLASSIC_RAW_DIM = NUMERIC_DIM + CATEGORICAL_DIM + SEQUENCE_FEATURE_DIM + GRAPH_FEATURE_DIM  # 约141维
    CLASSIC_DIM = 128  # 投影后维度 (增加以容纳更多信息)

    # 图特征和融合特征维度
    GRAPH_OUTPUT_DIM = 96   # 稍微增加
    FUSED_DIM = 256         # 增加融合特征维度

# 编码映射配置
class EncodingMaps:
    """特征编码映射"""

    # CPU架构映射
    CPU_ARCHITECTURE = {
        'x64': 4.0, 'x86': 3.0, 'ARM': 2.0, 'RISC-V': 1.0, 'unknown': 0.0
    }

    # 内存类型映射
    MEMORY_TYPE = {
        'DDR5': 5.0, 'DDR4': 4.0, 'DDR3': 3.0, 'DDR2': 2.0, 'unknown': 0.0
    }

    # 存储类型映射
    STORAGE_TYPE = {
        'NVMe': 4.0, 'SSD': 3.0, 'HDD': 2.0, 'eMMC': 1.0, 'unknown': 0.0
    }

    # 地理区域映射
    REGION = {
        'Asia': 1.0, 'Europe': 2.0, 'North_America': 3.0,
        'South_America': 4.0, 'Africa': 5.0, 'Oceania': 6.0, 'Antarctica': 7.0, 'unknown': 0.0
    }

    # 时区映射 (UTC偏移)
    TIMEZONE_OFFSET = {
        'UTC-12': -12.0, 'UTC-11': -11.0, 'UTC-10': -10.0, 'UTC-9': -9.0,
        'UTC-8': -8.0, 'UTC-7': -7.0, 'UTC-6': -6.0, 'UTC-5': -5.0,
        'UTC-4': -4.0, 'UTC-3': -3.0, 'UTC-2': -2.0, 'UTC-1': -1.0,
        'UTC+0': 0.0, 'UTC+1': 1.0, 'UTC+2': 2.0, 'UTC+3': 3.0,
        'UTC+4': 4.0, 'UTC+5': 5.0, 'UTC+6': 6.0, 'UTC+7': 7.0,
        'UTC+8': 8.0, 'UTC+9': 9.0, 'UTC+10': 10.0, 'UTC+11': 11.0,
        'UTC+12': 12.0, 'unknown': 0.0
    }

    # 应用状态映射
    APPLICATION_STATE = {
        'active': 4.0, 'standby': 3.0, 'maintenance': 2.0, 'inactive': 1.0, 'unknown': 0.0
    }

    # 节点类型 (与NodeTypes保持一致)
    NODE_TYPES = ['full_node', 'light_node', 'miner', 'validator', 'storage']

# 继承原有配置
class RelationTypes:
    COMPETE = 0      # 竞争关系
    COOPERATE = 3    # 协作关系
    SERVE = 2        # 服务关系
    VALIDATE = 1     # 验证关系
    CONNECT = 4      # 连接关系
    COMMUNICATE = 5  # 通信关系

    NUM_RELATIONS = 3

class NodeTypes:
    TYPES = ['full_node', 'light_node', 'miner', 'validator', 'storage']