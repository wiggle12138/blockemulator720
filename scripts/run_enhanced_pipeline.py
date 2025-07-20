#!/usr/bin/env python3
"""
第三步-第四步完整运行脚本
使用优化后的空分片处理和反馈机制
"""

import os
import sys
import subprocess
import pickle
import torch
from pathlib import Path

def setup_environment():
    """设置运行环境"""
    print("[CONFIG] 设置运行环境...")
    
    # 检查关键目录
    required_dirs = [
        "evolve_GCN", 
        "feedback", 
        "muti_scale",
        "partition/feature"
    ]
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"[ERROR] 缺少必要目录: {dir_path}")
            return False
    
    print("[SUCCESS] 环境检查通过")
    return True

def check_data_files():
    """检查数据文件是否存在"""
    print("\n[DATA] 检查数据文件...")
    
    required_files = [
        "muti_scale/temporal_embeddings.pkl",
        "partition/feature/step1_adjacency_raw.pt",
        "evolve_GCN/temporal_embeddings.pkl",  # 备用位置
        "evolve_GCN/step1_adjacency_raw.pt"    # 备用位置
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("[WARNING]  部分数据文件缺失，将创建链接...")
        return create_data_links()
    else:
        print("[SUCCESS] 所有数据文件就绪")
        return True

def create_data_links():
    """创建数据文件链接"""
    print("🔗 创建数据文件链接...")
    
    # 源文件到目标文件的映射
    file_mappings = [
        ("muti_scale/temporal_embeddings.pkl", "evolve_GCN/temporal_embeddings.pkl"),
        ("partition/feature/step1_adjacency_raw.pt", "evolve_GCN/step1_adjacency_raw.pt"),
        ("muti_scale/temporal_embeddings.pkl", "evolve_GCN/data/temporal_embeddings.pkl"),
    ]
    
    for src, dst in file_mappings:
        src_path = Path(src)
        dst_path = Path(dst)
        
        if src_path.exists() and not dst_path.exists():
            # 创建目标目录
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                # 创建硬链接或复制文件
                import shutil
                shutil.copy2(src_path, dst_path)
                print(f"   [SUCCESS] {src} -> {dst}")
            except Exception as e:
                print(f"   [ERROR] 复制失败 {src} -> {dst}: {e}")
                return False
    
    return True

def run_step3_enhanced(max_iterations=1):
    """运行增强版第三步"""
    print(f"\n[START] 运行第三步 (增强版) - {max_iterations} 轮迭代...")
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n=== 第三步 - 迭代 {iteration} ===")
        
        try:
            # 切换到evolve_GCN目录
            os.chdir("evolve_GCN")
            
            # 运行训练
            result = subprocess.run([
                sys.executable, "train.py"
            ], capture_output=True, text=True, timeout=1800)  # 30分钟超时
            
            if result.returncode == 0:
                print("   [SUCCESS] 第三步训练完成")
                print("   [DATA] 训练输出:")
                # 显示最后几行输出
                output_lines = result.stdout.split('\n')[-10:]
                for line in output_lines:
                    if line.strip():
                        print(f"     {line}")
                
                # 检查输出文件
                output_files = [
                    "outputs/new_temporal_embeddings.pkl",
                    "outputs/sharding_results.pkl",
                    "trained_models/enhanced_evolvegcn_model.pth"
                ]
                
                for output_file in output_files:
                    if Path(output_file).exists():
                        print(f"   [SUCCESS] 生成输出: {output_file}")
                    else:
                        print(f"   [WARNING]  缺少输出: {output_file}")
                
            else:
                print("   [ERROR] 第三步训练失败")
                print("   错误输出:", result.stderr[-500:])  # 显示最后500字符
                
            # 返回上级目录
            os.chdir("..")
            
            # 如果成功，运行第四步
            if result.returncode == 0:
                run_step4_feedback()
            
        except subprocess.TimeoutExpired:
            print("   ⏰ 第三步训练超时")
            os.chdir("..")
        except Exception as e:
            print(f"   [ERROR] 第三步运行异常: {e}")
            os.chdir("..")

def run_step4_feedback():
    """运行第四步反馈"""
    print(f"\n🔄 运行第四步反馈...")
    
    try:
        # 切换到feedback目录
        os.chdir("feedback")
        
        # 运行统一反馈引擎
        result = subprocess.run([
            sys.executable, "run_step4_unified_feedback.py"
        ], capture_output=True, text=True, timeout=600)  # 10分钟超时
        
        if result.returncode == 0:
            print("   [SUCCESS] 第四步反馈完成")
            
            # 检查输出文件
            output_files = [
                "step4_feedback_result.pkl",
                "step4_readable_result.json"
            ]
            
            for output_file in output_files:
                if Path(output_file).exists():
                    print(f"   [SUCCESS] 生成反馈: {output_file}")
                else:
                    print(f"   [WARNING]  缺少反馈: {output_file}")
            
            # 显示部分输出
            output_lines = result.stdout.split('\n')[-5:]
            for line in output_lines:
                if line.strip():
                    print(f"     {line}")
                    
        else:
            print("   [ERROR] 第四步反馈失败")
            print("   错误输出:", result.stderr[-300:])
            
        # 返回上级目录
        os.chdir("..")
        
    except Exception as e:
        print(f"   [ERROR] 第四步运行异常: {e}")
        os.chdir("..")

def analyze_results():
    """分析运行结果"""
    print(f"\n📈 分析运行结果...")
    
    # 检查第三步输出
    step3_outputs = [
        "evolve_GCN/outputs/sharding_results.pkl",
        "evolve_GCN/outputs/new_temporal_embeddings.pkl"
    ]
    
    for output_path in step3_outputs:
        if Path(output_path).exists():
            try:
                with open(output_path, 'rb') as f:
                    data = pickle.load(f)
                    
                if 'sharding_results' in output_path:
                    print(f"   [DATA] 分片结果: {len(data)} 个分片")
                    for shard_name, nodes in data.items():
                        print(f"     - {shard_name}: {len(nodes)} 个节点")
                        
                elif 'temporal_embeddings' in output_path:
                    print(f"   🧠 嵌入结果: {len(data)} 个时间步")
                    
            except Exception as e:
                print(f"   [WARNING]  分析 {output_path} 失败: {e}")
    
    # 检查第四步输出
    step4_outputs = [
        "feedback/step4_feedback_result.pkl",
        "feedback/step4_readable_result.json"
    ]
    
    for output_path in step4_outputs:
        if Path(output_path).exists():
            try:
                if output_path.endswith('.json'):
                    import json
                    with open(output_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    print(f"   🔄 反馈结果: 综合评分 {data.get('overall_score', 'N/A')}")
                else:
                    with open(output_path, 'rb') as f:
                        data = pickle.load(f)
                    print(f"   🔄 反馈数据: {type(data)}")
                    
            except Exception as e:
                print(f"   [WARNING]  分析 {output_path} 失败: {e}")

def main():
    """主函数"""
    print("=" * 70)
    print("[START] 第三步-第四步完整运行流水线")
    print("   支持空分片处理和智能反馈融合")
    print("=" * 70)
    
    # 环境检查
    if not setup_environment():
        print("[ERROR] 环境检查失败，退出")
        return
    
    # 数据文件检查
    if not check_data_files():
        print("[ERROR] 数据文件检查失败，退出")
        return
    
    # 运行主流程
    try:
        # 运行第三步（包含第四步反馈）
        run_step3_enhanced(max_iterations=1)
        
        # 分析结果
        analyze_results()
        
        print("\n" + "=" * 70)
        print("🎉 运行完成！")
        print("主要输出文件:")
        print("  - evolve_GCN/outputs/sharding_results.pkl (分片结果)")
        print("  - evolve_GCN/outputs/new_temporal_embeddings.pkl (新嵌入)")
        print("  - feedback/step4_feedback_result.pkl (第四步反馈)")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中断")
    except Exception as e:
        print(f"\n[ERROR] 运行异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
