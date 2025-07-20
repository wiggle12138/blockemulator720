import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, Any
import pandas as pd

# 方法1：直接设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class AdjacencyMatrixAnalyzer:
    """邻接矩阵分析工具"""

    def __init__(self):
        self.data = None
        self.matrix_info = {}

    def load_pt_file(self, pt_file: str) -> Dict[str, Any]:
        """加载.pt文件并显示基本信息"""
        print(f"=== 加载 {pt_file} ===")

        try:
            self.data = torch.load(pt_file, map_location='cpu')
            print("[SUCCESS] 文件加载成功")

            # 显示文件结构
            print("\n📁 文件内容结构:")
            for key, value in self.data.items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: torch.Tensor {value.shape} ({value.dtype})")
                elif isinstance(value, np.ndarray):
                    print(f"  - {key}: numpy.ndarray {value.shape} ({value.dtype})")
                elif isinstance(value, dict):
                    print(f"  - {key}: dict (包含 {len(value)} 个键)")
                else:
                    print(f"  - {key}: {type(value)} - {str(value)[:50]}...")

            return self.data

        except Exception as e:
            print(f"[ERROR] 加载失败: {e}")
            return None

    def analyze_adjacency_matrix(self, matrix_key: str = 'adjacency_matrix'):
        """分析邻接矩阵"""
        if self.data is None:
            print("[ERROR] 请先加载.pt文件")
            return

        # 寻找邻接矩阵
        adj_matrix = None
        possible_keys = [matrix_key, 'adjacency_matrix_dense', 'original_adjacency_matrix']

        for key in possible_keys:
            if key in self.data:
                adj_matrix = self.data[key]
                if isinstance(adj_matrix, torch.Tensor):
                    adj_matrix = adj_matrix.numpy()
                print(f"[SUCCESS] 找到邻接矩阵: {key}")
                break

        if adj_matrix is None:
            print("[ERROR] 未找到邻接矩阵")
            print("可用的键:", list(self.data.keys()))
            return

        print(f"\n=== 邻接矩阵分析 ===")
        print(f"矩阵形状: {adj_matrix.shape}")
        print(f"数据类型: {adj_matrix.dtype}")
        print(f"数值范围: [{adj_matrix.min():.4f}, {adj_matrix.max():.4f}]")
        print(f"非零元素: {np.count_nonzero(adj_matrix)}")
        print(f"稀疏度: {(1 - np.count_nonzero(adj_matrix) / adj_matrix.size) * 100:.2f}%")

        # 检查对称性
        is_symmetric = np.allclose(adj_matrix, adj_matrix.T, atol=1e-6)
        print(f"是否对称: {'是' if is_symmetric else '否'}")

        # 度统计
        if len(adj_matrix.shape) == 2:
            row_sums = adj_matrix.sum(axis=1)  # 出度
            col_sums = adj_matrix.sum(axis=0)  # 入度

            print(f"\n[DATA] 度统计:")
            print(f"平均出度: {row_sums.mean():.2f}")
            print(f"平均入度: {col_sums.mean():.2f}")
            print(f"最大出度: {row_sums.max():.0f}")
            print(f"最大入度: {col_sums.max():.0f}")
            print(f"零出度节点: {(row_sums == 0).sum()}")
            print(f"零入度节点: {(col_sums == 0).sum()}")

        # 保存分析结果
        self.matrix_info = {
            'shape': adj_matrix.shape,
            'dtype': str(adj_matrix.dtype),
            'num_edges': int(np.count_nonzero(adj_matrix)),
            'sparsity': float(1 - np.count_nonzero(adj_matrix) / adj_matrix.size),
            'is_symmetric': bool(is_symmetric),
            'out_degrees': row_sums.tolist() if len(adj_matrix.shape) == 2 else None,
            'in_degrees': col_sums.tolist() if len(adj_matrix.shape) == 2 else None
        }

        return adj_matrix

    def visualize_adjacency_matrix(self, adj_matrix: np.ndarray, max_nodes: int = 100):
        """可视化邻接矩阵"""
        print(f"\n=== 邻接矩阵可视化 ===")

        # 如果矩阵太大，只显示前max_nodes个节点
        if adj_matrix.shape[0] > max_nodes:
            print(f"矩阵太大，只显示前{max_nodes}个节点")
            adj_matrix = adj_matrix[:max_nodes, :max_nodes]

        plt.figure(figsize=(12, 10))

        # 主图：邻接矩阵热图
        plt.subplot(2, 2, 1)
        sns.heatmap(adj_matrix, cmap='Blues', cbar=True, square=True)
        plt.title(f'邻接矩阵热图 ({adj_matrix.shape[0]}×{adj_matrix.shape[1]})')
        plt.xlabel('目标节点')
        plt.ylabel('源节点')

        # 度分布
        if len(adj_matrix.shape) == 2:
            row_sums = adj_matrix.sum(axis=1)
            col_sums = adj_matrix.sum(axis=0)

            plt.subplot(2, 2, 2)
            plt.hist(row_sums, bins=20, alpha=0.7, label='出度')
            plt.hist(col_sums, bins=20, alpha=0.7, label='入度')
            plt.xlabel('度数')
            plt.ylabel('节点数量')
            plt.title('度分布')
            plt.legend()

            # 度相关性
            plt.subplot(2, 2, 3)
            plt.scatter(row_sums, col_sums, alpha=0.6)
            plt.xlabel('出度')
            plt.ylabel('入度')
            plt.title('出度vs入度')

            # 连接模式
            plt.subplot(2, 2, 4)
            # 显示连接的稀疏模式
            rows, cols = np.where(adj_matrix > 0)
            plt.scatter(cols, rows, alpha=0.5, s=1)
            plt.xlabel('目标节点')
            plt.ylabel('源节点')
            plt.title('连接模式')
            plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.savefig('adjacency_matrix_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("[SUCCESS] 可视化已保存为 adjacency_matrix_analysis.png")

    def analyze_edge_data(self):
        """分析边数据"""
        if self.data is None:
            print("[ERROR] 请先加载.pt文件")
            return

        print(f"\n=== 边数据分析 ===")

        # 查找边索引和边类型
        edge_index = None
        edge_type = None

        for key in ['edge_index', 'original_edge_index']:
            if key in self.data:
                edge_index = self.data[key]
                print(f"[SUCCESS] 找到边索引: {key} {edge_index.shape}")
                break

        for key in ['edge_type', 'original_edge_type']:
            if key in self.data:
                edge_type = self.data[key]
                print(f"[SUCCESS] 找到边类型: {key} {edge_type.shape}")
                break

        if edge_index is not None:
            if isinstance(edge_index, torch.Tensor):
                edge_index = edge_index.numpy()

            print(f"边索引形状: {edge_index.shape}")
            print(f"边数量: {edge_index.shape[1] if len(edge_index.shape) > 1 else len(edge_index)}")

            if len(edge_index.shape) == 2 and edge_index.shape[0] == 2:
                sources = edge_index[0]
                targets = edge_index[1]

                print(f"源节点范围: [{sources.min()}, {sources.max()}]")
                print(f"目标节点范围: [{targets.min()}, {targets.max()}]")

                # 检查自环
                self_loops = (sources == targets).sum()
                print(f"自环数量: {self_loops}")

        if edge_type is not None:
            if isinstance(edge_type, torch.Tensor):
                edge_type = edge_type.numpy()

            print(f"\n[DATA] 边类型统计:")
            unique_types, counts = np.unique(edge_type, return_counts=True)
            for utype, count in zip(unique_types, counts):
                print(f"  类型 {utype}: {count} 条边 ({count/len(edge_type)*100:.1f}%)")

    def check_matrix_quality(self, adj_matrix: np.ndarray) -> Dict[str, bool]:
        """检查邻接矩阵质量"""
        print(f"\n=== 邻接矩阵质量检查 ===")

        checks = {}

        # 1. 基本形状检查
        checks['is_square'] = adj_matrix.shape[0] == adj_matrix.shape[1]
        print(f"[SUCCESS] 方形矩阵: {checks['is_square']}")

        # 2. 数值检查
        checks['no_negative'] = (adj_matrix >= 0).all()
        checks['no_nan'] = not np.isnan(adj_matrix).any()
        checks['no_inf'] = not np.isinf(adj_matrix).any()
        print(f"[SUCCESS] 非负数值: {checks['no_negative']}")
        print(f"[SUCCESS] 无NaN值: {checks['no_nan']}")
        print(f"[SUCCESS] 无Inf值: {checks['no_inf']}")

        # 3. 连通性检查
        checks['has_edges'] = np.count_nonzero(adj_matrix) > 0
        checks['not_too_sparse'] = np.count_nonzero(adj_matrix) / adj_matrix.size > 0.001
        checks['not_too_dense'] = np.count_nonzero(adj_matrix) / adj_matrix.size < 0.5
        print(f"[SUCCESS] 有边存在: {checks['has_edges']}")
        print(f"[SUCCESS] 稀疏度合理: {checks['not_too_sparse']} (>0.1%)")
        print(f"[SUCCESS] 不过于密集: {checks['not_too_dense']} (<50%)")

        # 4. 孤立节点检查
        row_sums = adj_matrix.sum(axis=1)
        col_sums = adj_matrix.sum(axis=0)
        isolated_nodes = ((row_sums + col_sums) == 0).sum()
        checks['few_isolated'] = isolated_nodes < adj_matrix.shape[0] * 0.1
        print(f"[SUCCESS] 孤立节点少: {checks['few_isolated']} ({isolated_nodes}个)")

        # 5. 度分布检查
        max_degree = max(row_sums.max(), col_sums.max())
        checks['reasonable_max_degree'] = max_degree < adj_matrix.shape[0] * 0.8
        print(f"[SUCCESS] 最大度合理: {checks['reasonable_max_degree']} (最大度: {max_degree})")

        overall_quality = all(checks.values())
        print(f"\n{'[SUCCESS]' if overall_quality else '[ERROR]'} 总体质量: {'良好' if overall_quality else '需要改进'}")

        return checks

def analyze_pt_files():
    """分析多个.pt文件中的邻接矩阵"""
    analyzer = AdjacencyMatrixAnalyzer()

    # 预期的文件列表
    pt_files = [
        "step1_adjacency_raw.pt",
        "step1_rgcn_layers.pt",
        "step1_large_samples_adjacency_complete.pt",
        # "evolve_gcno_input.pt"
    ]

    results = {}

    for pt_file in pt_files:
        print(f"\n{'='*60}")
        print(f"分析文件: {pt_file}")
        print(f"{'='*60}")

        try:
            # 加载文件
            data = analyzer.load_pt_file(pt_file)
            if data is None:
                continue

            # 分析邻接矩阵
            adj_matrix = analyzer.analyze_adjacency_matrix()
            if adj_matrix is not None:
                # 质量检查
                quality_checks = analyzer.check_matrix_quality(adj_matrix)

                # 边数据分析
                analyzer.analyze_edge_data()

                # 可视化（仅对较小的矩阵）
                if adj_matrix.shape[0] <= 500:
                    analyzer.visualize_adjacency_matrix(adj_matrix)

                results[pt_file] = {
                    'matrix_info': analyzer.matrix_info,
                    'quality_checks': quality_checks,
                    'suitable_for_evolve_gcno': all(quality_checks.values())
                }

        except FileNotFoundError:
            print(f"[WARNING] 文件不存在: {pt_file}")
        except Exception as e:
            print(f"[ERROR] 分析出错: {e}")

    # 生成分析报告
    generate_analysis_report(results)
    return results

def generate_analysis_report(results: Dict):
    """生成分析报告"""
    print(f"\n{'='*80}")
    print(f"[DATA] 邻接矩阵分析报告")
    print(f"{'='*80}")

    if not results:
        print("[ERROR] 没有成功分析的文件")
        return

    for filename, result in results.items():
        print(f"\n📁 {filename}:")
        matrix_info = result.get('matrix_info', {})
        quality_checks = result.get('quality_checks', {})

        print(f"  - 矩阵大小: {matrix_info.get('shape', 'N/A')}")
        print(f"  - 边数量: {matrix_info.get('num_edges', 'N/A')}")
        print(f"  - 稀疏度: {matrix_info.get('sparsity', 0)*100:.2f}%")
        print(f"  - 对称性: {'是' if matrix_info.get('is_symmetric', False) else '否'}")
        print(f"  - 质量评分: {sum(quality_checks.values())}/{len(quality_checks)}")
        print(f"  - 适合EvolveGCNO: {'[SUCCESS]' if result.get('suitable_for_evolve_gcno', False) else '[ERROR]'}")

    # 推荐
    print(f"\n[TARGET] 推荐使用:")
    suitable_files = [f for f, r in results.items() if r.get('suitable_for_evolve_gcno', False)]
    if suitable_files:
        for f in suitable_files:
            print(f"  [SUCCESS] {f}")
    else:
        print("  [WARNING] 所有文件都需要改进")

if __name__ == "__main__":
    # 运行分析
    results = analyze_pt_files()