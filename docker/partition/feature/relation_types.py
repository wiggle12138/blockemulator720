# 为了兼容性，创建一个简单的别名文件
try:
    from .config import RelationTypes
except ImportError:
    from config import RelationTypes
