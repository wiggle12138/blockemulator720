"""
特征提取配置文件 - 基于committee_evolvegcn.go的40维真实特征结构
"""

class FeatureDimensions:
    # === 基于committee_evolvegcn.go的40维真实特征结构 ===
    
    # 硬件特征 (11维) - Static Features
    HARDWARE_CPU_DIM = 2          # CPU: 核心数、频率
    HARDWARE_MEMORY_DIM = 3       # 内存: 容量、带宽、类型编码
    HARDWARE_STORAGE_DIM = 3      # 存储: 容量、读写速度、类型编码
    HARDWARE_NETWORK_DIM = 3      # 网络: 上游/下游带宽、延迟
    HARDWARE_DIM = HARDWARE_CPU_DIM + HARDWARE_MEMORY_DIM + HARDWARE_STORAGE_DIM + HARDWARE_NETWORK_DIM  # 11维

    # 网络拓扑特征 (5维) - Static Features
    NETWORK_TOPOLOGY_DIM = 5      # 网络拓扑: 层次、连接等

    # 异构类型特征 (2维) - Static Features  
    HETEROGENEOUS_TYPE_DIM = 2    # 异构类型: 节点类型、应用状态

    # 链上行为特征 (15维) - Static Features
    ONCHAIN_BEHAVIOR_DIM = 15     # 链上行为: 交易能力、跨分片、区块生成、交易类型、共识

    # 动态属性特征 (7维) - Dynamic Features
    DYNAMIC_ATTRIBUTES_DIM = 7    # 动态属性: 交易处理、应用状态、其他动态特征

    # === 40维总计算 ===
    REAL_FEATURE_DIM = (HARDWARE_DIM + NETWORK_TOPOLOGY_DIM + HETEROGENEOUS_TYPE_DIM + 
                        ONCHAIN_BEHAVIOR_DIM + DYNAMIC_ATTRIBUTES_DIM)  # 11+5+2+15+7=40维

    # === 其他兼容性配置 - 调整为仅支持40维真实特征 ===
    CATEGORICAL_DIM = 0           # 分类特征已包含在40维中，不再额外添加

    # 序列和图特征配置 - 调整为40维适配
    MAX_SEQUENCE_LENGTH = 50
    SEQUENCE_FEATURE_DIM = 0      # 时序特征直接从40维提取，不再额外生成
    MAX_NEIGHBORS = 20
    NEIGHBOR_FEATURE_DIM = 0      # 邻居特征从40维衍生
    GRAPH_FEATURE_DIM = 0         # 图特征从40维衍生

    # 总的经典特征维度 - 修复为动态配置
    CLASSIC_RAW_DIM = REAL_FEATURE_DIM  # 40维直接输入
    CLASSIC_DIM = 64  # 修复：调整为64维以匹配真实需求

    # 图特征和融合特征维度
    GRAPH_OUTPUT_DIM = 96   # 保持不变
    FUSED_DIM = 256         # 保持不变

    # === 维度映射字典 (用于complete_integrated_sharding_system.py) ===
    @classmethod
    def get_real_feature_dims(cls):
        """返回真实特征维度映射"""
        return {
            'hardware': cls.HARDWARE_DIM,                    # 11
            'network_topology': cls.NETWORK_TOPOLOGY_DIM,   # 5  
            'heterogeneous_type': cls.HETEROGENEOUS_TYPE_DIM, # 2
            'onchain_behavior': cls.ONCHAIN_BEHAVIOR_DIM,    # 15
            'dynamic_attributes': cls.DYNAMIC_ATTRIBUTES_DIM  # 7
        }

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

    # 节点类型映射
    NODE_TYPE = {
        'full_node': 1.0, 'light_node': 2.0, 'miner': 3.0, 'validator': 4.0, 'storage': 5.0, 'unknown': 0.0
    }

    # 节点类型列表 (与NodeTypes保持一致)
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