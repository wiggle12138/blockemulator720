"""
第三步⇄第四步闭环流水线 - 支持多轮迭代，正确处理反馈依赖，包含空分片处理策略
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import shutil
import pickle
import numpy as np

def create_data_links_for_third_step():
    """为第三步创建数据文件链接，解决路径问题"""
    print("\n[INFO] 为第三步创建数据文件链接...")

    evolve_gcn_dir = Path("evolve_GCN")

    # 需要复制的文件
    files_to_copy = [
        ("./muti_scale/temporal_embeddings.pkl", "./evolve_GCN/temporal_embeddings.pkl"),
        ("./partition/feature/step1_adjacency_raw.pt", "./evolve_GCN/step1_adjacency_raw.pt"),
        ("./partition/feature/step1_large_samples.pt", "./evolve_GCN/step1_large_samples.pt")
    ]

    for src, dst in files_to_copy:
        src_path = Path(src)
        dst_path = Path(dst)

        if src_path.exists():
            try:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                if dst_path.exists():
                    dst_path.unlink()
                shutil.copy2(src_path, dst_path)
                print(f"  [COPY] {src} -> {dst}")
            except Exception as e:
                print(f"  [ERROR] 复制失败 {src} -> {dst}: {e}")
        else:
            print(f"  [WARNING] 源文件不存在: {src}")

def setup_step4_feedback_for_step3(iteration):
    """为第三步设置第四步反馈文件"""
    print(f"\n[INFO] 为第三步迭代{iteration}设置第四步反馈...")

    if iteration == 1:
        print("  [INFO] 第一轮迭代，第三步使用默认反馈")
        return True

    # 从第二轮开始，第三步需要上一轮第四步的反馈
    feedback_source = Path("feedback/step3_performance_feedback.pkl")

    # 第三步会在多个位置查找反馈文件
    feedback_targets = [
        Path("evolve_GCN/step3_performance_feedback.pkl"),  # 主要位置
        Path("step3_performance_feedback.pkl"),             # 根目录备用位置
        Path("feedback/step3_performance_feedback.pkl")     # 原始位置
    ]

    if feedback_source.exists():
        # 复制反馈文件到第三步可以找到的位置
        for target in feedback_targets[:2]:  # 只复制到前两个位置
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    target.unlink()
                shutil.copy2(feedback_source, target)
                print(f"  [COPY] 反馈文件: {feedback_source} -> {target}")
            except Exception as e:
                print(f"  [ERROR] 复制反馈文件失败: {e}")

        # 验证反馈文件内容
        try:
            import pickle
            with open(feedback_source, "rb") as f:
                feedback_data = pickle.load(f)

            if 'temporal_performance' in feedback_data:
                temporal_perf = feedback_data['temporal_performance']
                print(f"  [VERIFY] 反馈数据验证:")
                print(f"    性能向量维度: {len(temporal_perf.get('performance_vector', []))}")
                print(f"    特征质量维度: {len(temporal_perf.get('feature_qualities', []))}")
                print(f"    综合分数: {temporal_perf.get('combined_score', 'N/A')}")
                return True
            else:
                print(f"  [WARNING] 反馈文件格式异常")
                return False

        except Exception as e:
            print(f"  [ERROR] 验证反馈文件失败: {e}")
            return False
    else:
        print(f"  [WARNING] 第四步反馈文件不存在: {feedback_source}")
        print(f"  [INFO] 第三步将使用默认反馈")
        return True

def check_files_in_working_dir(description, file_paths, working_dir=None):
    """在指定工作目录中检查文件是否存在"""
    print(f"\n[INFO] {description}:")
    if working_dir:
        print(f"  检查目录: {working_dir}")

    existing = []
    missing = []

    for file_path in file_paths:
        if working_dir:
            full_path = Path(working_dir) / file_path
        else:
            full_path = Path(file_path)

        if full_path.exists():
            size = full_path.stat().st_size / 1024
            print(f"  [OK] {file_path} ({size:.1f} KB)")
            existing.append(file_path)
        else:
            print(f"  [MISSING] {file_path}")
            if working_dir:
                alternatives = [
                    Path(working_dir) / "data" / file_path,
                    Path(working_dir) / Path(file_path).name
                ]
                for alt_path in alternatives:
                    if alt_path.exists():
                        size = alt_path.stat().st_size / 1024
                        print(f"    [FOUND] 替代位置: {alt_path} ({size:.1f} KB)")
                        existing.append(str(alt_path))
                        break
                else:
                    missing.append(file_path)
            else:
                missing.append(file_path)

    return existing, missing

def run_step_with_correct_paths(step_name, command, expected_outputs=None, required_inputs=None, cwd=None):
    """执行步骤并检查正确路径下的文件"""
    print(f"\n{'='*60}")
    print(f"{step_name}")
    print('='*60)
    print(f"命令: {command}")
    if cwd:
        print(f"工作目录: {cwd}")

    # 检查输入文件
    if required_inputs:
        existing_inputs, missing_inputs = check_files_in_working_dir("输入文件检查", required_inputs, cwd)
        if missing_inputs:
            print(f"[WARNING] 缺失输入文件: {missing_inputs}")

    # 执行命令
    start_time = time.time()
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=cwd
        )

        execution_time = time.time() - start_time

        if result.returncode == 0:
            print(f"[SUCCESS] {step_name} 成功 ({execution_time:.1f}s)")
            if result.stdout.strip():
                output_lines = result.stdout.strip().split('\n')
                print("输出 (最后3行):")
                for line in output_lines[-3:]:
                    print(f"  {line}")
        else:
            print(f"[FAILED] {step_name} 失败 ({execution_time:.1f}s)")
            print("错误信息:")
            if result.stderr:
                error_lines = result.stderr.strip().split('\n')
                for line in error_lines[-5:]:
                    print(f"  {line}")
            return False
    except Exception as e:
        print(f"[ERROR] 执行异常: {e}")
        return False

    # 检查输出文件
    if expected_outputs:
        existing_outputs, missing_outputs = check_files_in_working_dir("输出文件检查", expected_outputs)

    return True

def analyze_sharding_results(iteration):
    """分析分片结果"""
    print(f"\n[ANALYSIS] 分析第{iteration}轮分片结果:")

    shard_result_path = Path("evolve_GCN/outputs/sharding_results.pkl")
    if shard_result_path.exists():
        try:
            import pickle
            with open(shard_result_path, "rb") as f:
                shard_data = pickle.load(f)

            print(f"  [INFO] 分片结果类型: {type(shard_data)}")

            if isinstance(shard_data, dict):
                total_nodes = 0
                empty_shards = 0

                print(f"  [INFO] 分片详情:")
                for key, value in shard_data.items():
                    if isinstance(value, (list, tuple)):
                        node_count = len(value)
                        total_nodes += node_count
                        if node_count == 0:
                            empty_shards += 1
                            print(f"    {key}: {node_count} 节点 [EMPTY]")
                        else:
                            print(f"    {key}: {node_count} 节点")

                print(f"  [SUMMARY] 总分片数: {len(shard_data)}, 空分片: {empty_shards}, 总节点: {total_nodes}")

                # 计算负载均衡度
                if len(shard_data) > empty_shards:
                    non_empty_counts = [len(v) for v in shard_data.values() if len(v) > 0]
                    if non_empty_counts:
                        avg_load = sum(non_empty_counts) / len(non_empty_counts)
                        load_variance = sum((x - avg_load) ** 2 for x in non_empty_counts) / len(non_empty_counts)
                        balance_score = 1.0 / (1.0 + load_variance / (avg_load + 1e-6))
                        print(f"  [METRICS] 负载均衡度: {balance_score:.3f}")
                        return balance_score

        except Exception as e:
            print(f"  [ERROR] 分析分片结果失败: {e}")
    else:
        print(f"  [WARNING] 分片结果文件不存在")

    return 0.0

def analyze_feedback_results(iteration):
    """分析反馈结果"""
    print(f"\n[ANALYSIS] 分析第{iteration}轮反馈结果:")

    feedback_path = Path("feedback/step3_performance_feedback.pkl")
    if feedback_path.exists():
        try:
            import pickle
            with open(feedback_path, "rb") as f:
                feedback_data = pickle.load(f)

            print(f"  [INFO] 反馈数据类型: {type(feedback_data)}")

            if isinstance(feedback_data, dict):
                if 'temporal_performance' in feedback_data:
                    temporal_perf = feedback_data['temporal_performance']
                    print(f"  [METRICS] 性能指标:")

                    # 显示性能向量详情
                    perf_vector = temporal_perf.get('performance_vector', [])
                    if len(perf_vector) >= 4:
                        print(f"    负载均衡: {perf_vector[0]:.3f}")
                        print(f"    跨片交易率: {perf_vector[1]:.3f}")
                        print(f"    安全性: {perf_vector[2]:.3f}")
                        print(f"    特征质量: {perf_vector[3]:.3f}")

                    # 显示特征质量详情
                    feature_qualities = temporal_perf.get('feature_qualities', [])
                    if feature_qualities:
                        print(f"    6类特征质量: {[f'{x:.3f}' for x in feature_qualities]}")

                    combined_score = temporal_perf.get('combined_score', 0)
                    print(f"    综合分数: {combined_score:.3f}")

                    return combined_score

        except Exception as e:
            print(f"  [ERROR] 分析反馈结果失败: {e}")
    else:
        print(f"  [WARNING] 反馈结果文件不存在")

    return 0.0

def check_convergence(iteration_results, min_iterations=3):
    """检查是否收敛"""
    if len(iteration_results) < min_iterations:
        return False, "insufficient_iterations"

    # 检查最近3轮的性能趋势
    recent_scores = [r['combined_score'] for r in iteration_results[-3:]]

    # 如果分数变化很小，认为收敛
    score_variance = sum((x - sum(recent_scores)/len(recent_scores))**2 for x in recent_scores) / len(recent_scores)
    if score_variance < 0.001:
        return True, "performance_stable"

    # 如果分数持续改善但改善幅度很小
    if len(recent_scores) >= 2:
        improvements = [recent_scores[i+1] - recent_scores[i] for i in range(len(recent_scores)-1)]
        if all(imp >= 0 for imp in improvements) and all(imp < 0.01 for imp in improvements):
            return True, "marginal_improvement"

    return False, "continuing"

def check_prerequisite_files():
    """检查运行第三步⇄第四步闭环所需的前置文件"""
    print("[INFO] 检查前置文件...")

    prerequisite_files = [
        "./partition/feature/step1_large_samples.pt",
        "./partition/feature/step1_adjacency_raw.pt",
        "./muti_scale/temporal_embeddings.pkl"
    ]

    existing_files, missing_files = check_files_in_working_dir("前置文件检查", prerequisite_files)

    if missing_files:
        print(f"\n[ERROR] 缺失关键前置文件: {missing_files}")
        return False

    print("[OK] 所有前置文件检查通过")
    return True

def run_iterative_loop(max_iterations=5, convergence_check=True):
    """运行迭代闭环 - 正确处理第四步反馈依赖"""
    print(f"\n[INFO] 开始第三步⇄第四步闭环迭代")
    print(f"[CONFIG] 最大迭代次数: {max_iterations}")
    print(f"[CONFIG] 收敛检查: {'启用' if convergence_check else '禁用'}")
    print(f"[CONFIG] 反馈机制: 第三步从第2轮开始使用第四步反馈")
    print("=" * 60)

    python_exe = r"E:\Codefield\BlockEmulator\.venv\Scripts\python.exe"
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    iteration_results = []

    for iteration in range(1, max_iterations + 1):
        print(f"\n[ITERATION] 闭环迭代 {iteration}/{max_iterations}")
        print("-" * 40)

        iteration_start_time = time.time()

        # 1. 为第三步准备第四步反馈（从第2轮开始）
        feedback_ready = setup_step4_feedback_for_step3(iteration)
        if not feedback_ready:
            print(f"[WARNING] 第四步反馈准备失败，但继续执行")

        # 2. 第三步：EvolveGCN分片
        step3_inputs = [
            "temporal_embeddings.pkl",
            "step1_adjacency_raw.pt"
        ]

        # 从第2轮开始，第三步还会尝试读取反馈文件
        if iteration > 1:
            step3_inputs.extend([
                "step3_performance_feedback.pkl"  # 可选输入
            ])

        step3_outputs = [
            "./evolve_GCN/outputs/new_temporal_embeddings.pkl",
            "./evolve_GCN/outputs/sharding_results.pkl",
            "./evolve_GCN/trained_models/enhanced_evolvegcn_model.pth"
        ]

        print(f"[STEP3] 执行第三步：EvolveGCN分片 (迭代{iteration})")
        if iteration > 1:
            print(f"        本轮将使用上一轮第四步的反馈进行优化")

        step3_success = run_step_with_correct_paths(
            f"[STEP3] 第三步：EvolveGCN分片 (迭代{iteration})",
            f'"{python_exe}" train.py',
            expected_outputs=step3_outputs,
            required_inputs=step3_inputs,
            cwd="./evolve_GCN"
        )

        if not step3_success:
            print(f"[FAILED] 第三步迭代{iteration}失败，停止闭环")
            break

        # 3. 分析第三步结果
        balance_score = analyze_sharding_results(iteration)

        # 4. 第四步：反馈评估
        step4_inputs = [
            "./partition/feature/step1_large_samples.pt"
        ]

        # 查找第三步输出文件
        possible_shard_files = [
            "./evolve_GCN/outputs/sharding_results.pkl",
            "./evolve_GCN/output/sharding_results.pkl"
        ]

        found_shard_file = None
        for shard_file in possible_shard_files:
            if os.path.exists(shard_file):
                step4_inputs.append(shard_file)
                found_shard_file = shard_file
                break

        step4_outputs = [
            "./feedback/step3_performance_feedback.pkl",
            "./feedback/stable_feature_config.pkl"
        ]

        print(f"\n[STEP4] 执行第四步：反馈评估 (迭代{iteration})")
        print(f"        生成的反馈将用于下一轮第三步优化")

        step4_success = run_step_with_correct_paths(
            f"[STEP4] 第四步：反馈评估 (迭代{iteration})",
            f'"{python_exe}" feedback/feedback2.py',
            expected_outputs=step4_outputs,
            required_inputs=step4_inputs
        )

        if not step4_success:
            print(f"[FAILED] 第四步迭代{iteration}失败，停止闭环")
            break

        # 5. 分析第四步结果
        combined_score = analyze_feedback_results(iteration)

        iteration_time = time.time() - iteration_start_time

        # 6. 记录迭代结果
        iteration_result = {
            'iteration': iteration,
            'balance_score': balance_score,
            'combined_score': combined_score,
            'step3_success': step3_success,
            'step4_success': step4_success,
            'execution_time': iteration_time,
            'found_shard_file': found_shard_file is not None,
            'used_feedback': iteration > 1  # 标记是否使用了反馈
        }
        iteration_results.append(iteration_result)

        print(f"\n[RESULT] 迭代{iteration}完成:")
        print(f"  执行时间: {iteration_time:.1f}s")
        print(f"  负载均衡度: {balance_score:.3f}")
        print(f"  综合分数: {combined_score:.3f}")
        print(f"  使用反馈: {'是' if iteration > 1 else '否 (首轮)'}")

        # 7. 收敛检查
        if convergence_check and iteration >= 3:
            converged, reason = check_convergence(iteration_results)
            if converged:
                print(f"\n[CONVERGENCE] 迭代在第{iteration}轮收敛 (原因: {reason})")
                print("[INFO] 提前结束迭代")
                break
            else:
                print(f"[CONVERGENCE] 未收敛，继续迭代 (状态: {reason})")

        # 8. 迭代间隔（给系统一点时间）
        if iteration < max_iterations:
            print(f"[INFO] 准备下一轮迭代...")
            time.sleep(1)  # 短暂暂停

    return iteration_results

def print_final_summary(iteration_results):
    """打印最终汇总结果"""
    print(f"\n[SUMMARY] 闭环迭代完成")
    print("=" * 60)

    if not iteration_results:
        print("[ERROR] 没有成功的迭代")
        return

    print(f"[INFO] 总迭代次数: {len(iteration_results)}")

    # 性能趋势
    balance_scores = [r['balance_score'] for r in iteration_results]
    combined_scores = [r['combined_score'] for r in iteration_results]

    print(f"[METRICS] 性能趋势:")
    print(f"  负载均衡度: {balance_scores[0]:.3f} -> {balance_scores[-1]:.3f}")
    print(f"  综合分数: {combined_scores[0]:.3f} -> {combined_scores[-1]:.3f}")

    if len(iteration_results) > 1:
        balance_improvement = balance_scores[-1] - balance_scores[0]
        score_improvement = combined_scores[-1] - combined_scores[0]
        print(f"[IMPROVEMENT] 改善幅度:")
        print(f"  负载均衡: {balance_improvement:+.3f}")
        print(f"  综合分数: {score_improvement:+.3f}")

    # 反馈效果分析
    feedback_iterations = [r for r in iteration_results if r.get('used_feedback', False)]
    if feedback_iterations:
        print(f"[FEEDBACK] 反馈效果:")
        print(f"  使用反馈的轮次: {len(feedback_iterations)}")
        if len(feedback_iterations) >= 2:
            feedback_scores = [r['combined_score'] for r in feedback_iterations]
            feedback_improvement = feedback_scores[-1] - feedback_scores[0]
            print(f"  反馈期间改善: {feedback_improvement:+.3f}")

    # 执行统计
    total_time = sum(r['execution_time'] for r in iteration_results)
    avg_time = total_time / len(iteration_results)
    print(f"[TIMING] 执行时间:")
    print(f"  总时间: {total_time:.1f}s")
    print(f"  平均单轮: {avg_time:.1f}s")

    # 成功率
    success_rate = sum(1 for r in iteration_results if r['step3_success'] and r['step4_success']) / len(iteration_results)
    print(f"[SUCCESS] 成功率: {success_rate*100:.1f}%")

def main():
    """主执行函数 - 支持配置迭代次数"""
    print("第三步⇄第四步闭环流水线 - 多轮迭代版 (支持反馈依赖)")
    print("=" * 60)

    # 配置参数 - 可以在这里调整
    MAX_ITERATIONS = 5        # 最大迭代次数
    ENABLE_CONVERGENCE = True # 是否启用收敛检查

    python_exe = r"E:\Codefield\BlockEmulator\.venv\Scripts\python.exe"
    if not os.path.exists(python_exe):
        python_exe = sys.executable
        print(f"[WARNING] 使用系统Python: {python_exe}")
    else:
        print(f"[INFO] 使用虚拟环境Python: {python_exe}")

    print(f"[INFO] 工作目录: {os.getcwd()}")
    print(f"[INFO] 反馈机制: 第三步在第2轮及后续迭代中使用第四步反馈")

    # 检查前置文件
    if not check_prerequisite_files():
        print("\n[ERROR] 前置文件检查失败，无法启动闭环流水线")
        return

    # 创建数据链接
    create_data_links_for_third_step()

    # 运行迭代闭环
    iteration_results = run_iterative_loop(
        max_iterations=MAX_ITERATIONS,
        convergence_check=ENABLE_CONVERGENCE
    )

    # 打印最终汇总
    print_final_summary(iteration_results)

if __name__ == "__main__":
    main()