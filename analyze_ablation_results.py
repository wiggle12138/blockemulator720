#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Ablation Study Results Analysis
This script analyzes real experimental data from three ablation variants:
1. ours_full - Complete implementation with all features
2. ours_no_contrastive - Without contrastive learning component  
3. ours_no_layering - Without layered processing component

Author: BlockEmulator Analysis Suite
Date: 2024-12-31
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path

# Configuration
RESULTS_BASE_DIR = "results"
VARIANTS = {
    "ours_full": "完整实现 (Full Implementation)",
    "ours_no_contrastive": "无对比学习 (No Contrastive Learning)",
    "ours_no_layering": "无分层处理 (No Layered Processing)"
}

# Colors for visualization
COLORS = {
    "ours_full": "#2E8B57",        # Sea Green - Best performance
    "ours_no_contrastive": "#FF6347",  # Tomato - Ablation 1
    "ours_no_layering": "#4169E1"      # Royal Blue - Ablation 2
}

class AblationAnalyzer:
    def __init__(self):
        """Initialize the ablation analyzer"""
        self.data = {}
        self.metrics = {}
        self.comparison_results = {}
        
    def load_data(self):
        """Load experimental data from all three variants"""
        print("=== 开始加载实验数据 ===")
        
        for variant in VARIANTS.keys():
            variant_path = Path(RESULTS_BASE_DIR) / variant / "expTest" / "result" / "supervisor_measureOutput"
            
            print(f"\n加载变体: {variant}")
            print(f"数据路径: {variant_path}")
            
            if not variant_path.exists():
                print(f"⚠️  警告: 路径不存在 - {variant_path}")
                continue
                
            self.data[variant] = {}
            
            # Load each metric file
            metric_files = {
                'tps': 'Average_TPS.csv',
                'latency': 'Transaction_Confirm_Latency.csv', 
                'ctx_ratio': 'CrossTransaction_ratio.csv',
                'tx_details': 'Tx_Details.csv',
                'tx_number': 'Tx_number.csv'
            }
            
            for metric_key, filename in metric_files.items():
                file_path = variant_path / filename
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        self.data[variant][metric_key] = df
                        print(f"  ✅ {metric_key}: {len(df)} 条记录")
                    except Exception as e:
                        print(f"  ❌ 加载失败 {metric_key}: {e}")
                else:
                    print(f"  ⚠️  文件不存在: {filename}")
        
        print(f"\n数据加载完成! 成功加载 {len(self.data)} 个变体的数据")
        
    def calculate_metrics(self):
        """Calculate key performance metrics for each variant"""
        print("\n=== 计算性能指标 ===")
        
        for variant in self.data.keys():
            print(f"\n处理变体: {variant}")
            variant_metrics = {}
            
            # TPS Analysis
            if 'tps' in self.data[variant]:
                tps_df = self.data[variant]['tps']
                # Filter out zero TPS epochs (system idle periods)
                valid_tps = tps_df[tps_df['Avg. TPS of this epoch'] > 0]['Avg. TPS of this epoch']
                
                variant_metrics['tps'] = {
                    'avg': valid_tps.mean(),
                    'max': valid_tps.max(), 
                    'min': valid_tps.min(),
                    'std': valid_tps.std(),
                    'valid_epochs': len(valid_tps),
                    'total_epochs': len(tps_df)
                }
                print(f"  TPS指标: 平均={variant_metrics['tps']['avg']:.2f}, 最大={variant_metrics['tps']['max']:.2f}")
            
            # Latency Analysis  
            if 'latency' in self.data[variant]:
                latency_df = self.data[variant]['latency']
                # Calculate average latency per transaction (convert from seconds to milliseconds)
                valid_latency = latency_df[latency_df['Total tx # in this epoch'] > 0]
                
                if len(valid_latency) > 0:
                    avg_latency_per_tx = (valid_latency['Sum of All Tx TCL (sec.)'] * 1000) / valid_latency['Total tx # in this epoch']
                    
                    variant_metrics['latency'] = {
                        'avg_ms': avg_latency_per_tx.mean(),
                        'max_ms': avg_latency_per_tx.max(),
                        'min_ms': avg_latency_per_tx.min(), 
                        'std_ms': avg_latency_per_tx.std(),
                        'valid_epochs': len(valid_latency)
                    }
                    print(f"  延迟指标: 平均={variant_metrics['latency']['avg_ms']:.2f}ms")
            
            # Cross-Transaction Ratio Analysis
            if 'ctx_ratio' in self.data[variant]:
                ctx_df = self.data[variant]['ctx_ratio']
                # Filter out NaN and zero transaction epochs
                valid_ctx = ctx_df[(ctx_df['Total tx # in this epoch'] > 0) & 
                                  (ctx_df['CTX ratio of this epoch'].notna())]
                
                if len(valid_ctx) > 0:
                    variant_metrics['ctx_ratio'] = {
                        'avg': valid_ctx['CTX ratio of this epoch'].mean(),
                        'max': valid_ctx['CTX ratio of this epoch'].max(),
                        'min': valid_ctx['CTX ratio of this epoch'].min(),
                        'std': valid_ctx['CTX ratio of this epoch'].std(),
                        'valid_epochs': len(valid_ctx)
                    }
                    print(f"  跨分片比例: 平均={variant_metrics['ctx_ratio']['avg']:.3f}")
            
            self.metrics[variant] = variant_metrics
            
        print("\n指标计算完成!")
        
    def generate_comparison_analysis(self):
        """Generate detailed comparison analysis between variants"""
        print("\n=== 生成对比分析 ===")
        
        if 'ours_full' not in self.metrics:
            print("❌ 错误: 缺少完整实现数据!")
            return
            
        baseline = self.metrics['ours_full']
        comparisons = {}
        
        for variant in ['ours_no_contrastive', 'ours_no_layering']:
            if variant not in self.metrics:
                print(f"⚠️  跳过变体 {variant} - 数据不完整")
                continue
                
            print(f"\n对比分析: ours_full vs {variant}")
            comparison = {}
            
            # TPS Comparison
            if 'tps' in baseline and 'tps' in self.metrics[variant]:
                tps_improvement = ((baseline['tps']['avg'] - self.metrics[variant]['tps']['avg']) / 
                                 self.metrics[variant]['tps']['avg'] * 100)
                comparison['tps_improvement_pct'] = tps_improvement
                print(f"  TPS提升: {tps_improvement:.1f}%")
            
            # Latency Comparison (lower is better)
            if 'latency' in baseline and 'latency' in self.metrics[variant]:
                latency_reduction = ((self.metrics[variant]['latency']['avg_ms'] - baseline['latency']['avg_ms']) / 
                                   self.metrics[variant]['latency']['avg_ms'] * 100)
                comparison['latency_reduction_pct'] = latency_reduction
                print(f"  延迟降低: {latency_reduction:.1f}%")
            
            # CTX Ratio Comparison
            if 'ctx_ratio' in baseline and 'ctx_ratio' in self.metrics[variant]:
                ctx_change = ((baseline['ctx_ratio']['avg'] - self.metrics[variant]['ctx_ratio']['avg']) / 
                             self.metrics[variant]['ctx_ratio']['avg'] * 100)
                comparison['ctx_ratio_change_pct'] = ctx_change
                print(f"  跨分片比例变化: {ctx_change:.1f}%")
            
            comparisons[variant] = comparison
            
        self.comparison_results = comparisons
        print("\n对比分析完成!")
        
    def create_static_performance_chart(self):
        """Create static bar chart comparing average performance metrics"""
        print("\n生成静态性能对比图表...")
        
        # Prepare data for plotting
        variants = []
        tps_values = []
        latency_values = []
        ctx_ratio_values = []
        
        for variant in VARIANTS.keys():
            if variant in self.metrics:
                variants.append(VARIANTS[variant])
                
                # TPS data
                tps_values.append(self.metrics[variant].get('tps', {}).get('avg', 0))
                
                # Latency data (convert to ms)
                latency_values.append(self.metrics[variant].get('latency', {}).get('avg_ms', 0))
                
                # CTX ratio data
                ctx_ratio_values.append(self.metrics[variant].get('ctx_ratio', {}).get('avg', 0))
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Colors for bars
        bar_colors = [COLORS[variant] for variant in VARIANTS.keys() if variant in self.metrics]
        
        # TPS Comparison
        bars1 = ax1.bar(variants, tps_values, color=bar_colors, alpha=0.8)
        ax1.set_title('平均 TPS (Transactions Per Second)', fontsize=14, weight='bold')
        ax1.set_ylabel('TPS', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, tps_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # Latency Comparison
        bars2 = ax2.bar(variants, latency_values, color=bar_colors, alpha=0.8)
        ax2.set_title('平均交易确认延迟 (Transaction Latency)', fontsize=14, weight='bold')
        ax2.set_ylabel('延迟 (毫秒)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars2, latency_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}ms', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # CTX Ratio Comparison
        bars3 = ax3.bar(variants, ctx_ratio_values, color=bar_colors, alpha=0.8)
        ax3.set_title('跨分片交易比例 (Cross-Shard TX Ratio)', fontsize=14, weight='bold')
        ax3.set_ylabel('比例', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars3, ctx_ratio_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        plt.tight_layout()
        
        # Save chart
        static_chart_path = "ablation_static_performance_comparison.png"
        plt.savefig(static_chart_path, dpi=300, bbox_inches='tight')
        print(f"✅ 静态对比图表已保存: {static_chart_path}")
        
        plt.show()
        
    def create_dynamic_evolution_chart(self):
        """Create line chart showing performance evolution over epochs"""
        print("\n生成动态演化图表...")
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot TPS evolution
        for variant in VARIANTS.keys():
            if variant in self.data and 'tps' in self.data[variant]:
                tps_df = self.data[variant]['tps']
                # Filter valid epochs
                valid_epochs = tps_df[tps_df['Avg. TPS of this epoch'] > 0]
                
                ax1.plot(valid_epochs['EpochID'], valid_epochs['Avg. TPS of this epoch'], 
                        marker='o', linewidth=2, label=VARIANTS[variant], color=COLORS[variant])
        
        ax1.set_title('TPS 随时间演化 (TPS Evolution Over Time)', fontsize=14, weight='bold')
        ax1.set_xlabel('Epoch ID', fontsize=12)
        ax1.set_ylabel('TPS', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot Latency evolution
        for variant in VARIANTS.keys():
            if variant in self.data and 'latency' in self.data[variant]:
                latency_df = self.data[variant]['latency']
                valid_epochs = latency_df[latency_df['Total tx # in this epoch'] > 0]
                
                if len(valid_epochs) > 0:
                    # Calculate per-transaction latency in milliseconds
                    per_tx_latency = (valid_epochs['Sum of All Tx TCL (sec.)'] * 1000) / valid_epochs['Total tx # in this epoch']
                    
                    ax2.plot(valid_epochs['EpochID'], per_tx_latency,
                            marker='s', linewidth=2, label=VARIANTS[variant], color=COLORS[variant])
        
        ax2.set_title('交易确认延迟随时间演化 (Latency Evolution Over Time)', fontsize=14, weight='bold')
        ax2.set_xlabel('Epoch ID', fontsize=12) 
        ax2.set_ylabel('延迟 (毫秒)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot CTX Ratio evolution
        for variant in VARIANTS.keys():
            if variant in self.data and 'ctx_ratio' in self.data[variant]:
                ctx_df = self.data[variant]['ctx_ratio']
                valid_epochs = ctx_df[(ctx_df['Total tx # in this epoch'] > 0) & 
                                     (ctx_df['CTX ratio of this epoch'].notna())]
                
                ax3.plot(valid_epochs['EpochID'], valid_epochs['CTX ratio of this epoch'],
                        marker='^', linewidth=2, label=VARIANTS[variant], color=COLORS[variant])
        
        ax3.set_title('跨分片交易比例随时间演化 (CTX Ratio Evolution Over Time)', fontsize=14, weight='bold')
        ax3.set_xlabel('Epoch ID', fontsize=12)
        ax3.set_ylabel('跨分片比例', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        dynamic_chart_path = "ablation_dynamic_evolution_comparison.png"
        plt.savefig(dynamic_chart_path, dpi=300, bbox_inches='tight')
        print(f"✅ 动态演化图表已保存: {dynamic_chart_path}")
        
        plt.show()
        
    def export_csv_data(self):
        """Export all analysis data to CSV for academic publication"""
        print("\n=== 导出 CSV 数据用于学术发表 ===")
        
        # 1. Summary Statistics CSV
        summary_data = []
        for variant, metrics in self.metrics.items():
            row = {
                'Variant': variant,
                'Variant_Name': VARIANTS[variant],
                'Avg_TPS': metrics.get('tps', {}).get('avg', 0),
                'Max_TPS': metrics.get('tps', {}).get('max', 0),
                'Avg_Latency_ms': metrics.get('latency', {}).get('avg_ms', 0),
                'Avg_CTX_Ratio': metrics.get('ctx_ratio', {}).get('avg', 0),
                'Valid_Epochs': metrics.get('tps', {}).get('valid_epochs', 0)
            }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = "ablation_summary_statistics.csv"
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 汇总统计数据: {summary_csv_path}")
        
        # 2. Comparison Analysis CSV
        comparison_data = []
        baseline_variant = 'ours_full'
        
        if baseline_variant in self.metrics:
            for variant in ['ours_no_contrastive', 'ours_no_layering']:
                if variant in self.comparison_results:
                    row = {
                        'Comparison': f'{baseline_variant}_vs_{variant}',
                        'Baseline': VARIANTS[baseline_variant],
                        'Ablation': VARIANTS[variant],
                        'TPS_Improvement_Percent': self.comparison_results[variant].get('tps_improvement_pct', 0),
                        'Latency_Reduction_Percent': self.comparison_results[variant].get('latency_reduction_pct', 0),
                        'CTX_Ratio_Change_Percent': self.comparison_results[variant].get('ctx_ratio_change_pct', 0)
                    }
                    comparison_data.append(row)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_csv_path = "ablation_comparison_analysis.csv"
            comparison_df.to_csv(comparison_csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ 对比分析数据: {comparison_csv_path}")
        
        # 3. Detailed Evolution Data CSV
        evolution_data = []
        
        for variant in VARIANTS.keys():
            if variant in self.data:
                # TPS evolution
                if 'tps' in self.data[variant]:
                    tps_df = self.data[variant]['tps']
                    for _, row in tps_df.iterrows():
                        evolution_data.append({
                            'Variant': variant,
                            'Variant_Name': VARIANTS[variant],
                            'Epoch_ID': row['EpochID'],
                            'Metric_Type': 'TPS',
                            'Value': row['Avg. TPS of this epoch'],
                            'Unit': 'transactions_per_second'
                        })
                
                # Latency evolution
                if 'latency' in self.data[variant]:
                    latency_df = self.data[variant]['latency']
                    for _, row in latency_df.iterrows():
                        if row['Total tx # in this epoch'] > 0:
                            per_tx_latency = (row['Sum of All Tx TCL (sec.)'] * 1000) / row['Total tx # in this epoch']
                            evolution_data.append({
                                'Variant': variant,
                                'Variant_Name': VARIANTS[variant],
                                'Epoch_ID': row['EpochID'],
                                'Metric_Type': 'Latency',
                                'Value': per_tx_latency,
                                'Unit': 'milliseconds'
                            })
                
                # CTX Ratio evolution
                if 'ctx_ratio' in self.data[variant]:
                    ctx_df = self.data[variant]['ctx_ratio']
                    for _, row in ctx_df.iterrows():
                        if pd.notna(row['CTX ratio of this epoch']):
                            evolution_data.append({
                                'Variant': variant,
                                'Variant_Name': VARIANTS[variant],
                                'Epoch_ID': row['EpochID'],
                                'Metric_Type': 'CTX_Ratio',
                                'Value': row['CTX ratio of this epoch'],
                                'Unit': 'ratio'
                            })
        
        if evolution_data:
            evolution_df = pd.DataFrame(evolution_data)
            evolution_csv_path = "ablation_detailed_evolution.csv"
            evolution_df.to_csv(evolution_csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ 详细演化数据: {evolution_csv_path}")
        
        # 4. Raw data aggregation for plotting
        plotting_data = {}
        
        # Static chart data
        static_chart_data = []
        for variant in VARIANTS.keys():
            if variant in self.metrics:
                static_chart_data.append({
                    'Variant': variant,
                    'Variant_Name': VARIANTS[variant],
                    'Avg_TPS': self.metrics[variant].get('tps', {}).get('avg', 0),
                    'Avg_Latency_ms': self.metrics[variant].get('latency', {}).get('avg_ms', 0),
                    'Avg_CTX_Ratio': self.metrics[variant].get('ctx_ratio', {}).get('avg', 0)
                })
        
        static_df = pd.DataFrame(static_chart_data)
        static_plotting_csv_path = "ablation_static_chart_data.csv"
        static_df.to_csv(static_plotting_csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 静态图表绘制数据: {static_plotting_csv_path}")
        
        print(f"\n📊 所有 CSV 数据文件已导出完成!")
        print(f"   - 汇总统计: ablation_summary_statistics.csv")
        print(f"   - 对比分析: ablation_comparison_analysis.csv") 
        print(f"   - 详细演化: ablation_detailed_evolution.csv")
        print(f"   - 静态图表: ablation_static_chart_data.csv")
        
    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        print("\n=== 生成分析报告 ===")
        
        report = []
        report.append("# BlockEmulator Ablation Study Analysis Report")
        report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("## 执行摘要 (Executive Summary)")
        if 'ours_full' in self.metrics:
            baseline = self.metrics['ours_full']
            report.append(f"- **完整实现平均 TPS**: {baseline.get('tps', {}).get('avg', 0):.2f}")
            report.append(f"- **完整实现平均延迟**: {baseline.get('latency', {}).get('avg_ms', 0):.2f} ms")
            report.append(f"- **完整实现跨分片比例**: {baseline.get('ctx_ratio', {}).get('avg', 0):.3f}")
        report.append("")
        
        # Variant Comparison
        report.append("## 变体对比分析 (Variant Comparison)")
        for variant, comparison in self.comparison_results.items():
            report.append(f"### {VARIANTS[variant]}")
            if 'tps_improvement_pct' in comparison:
                report.append(f"- TPS 提升: **{comparison['tps_improvement_pct']:.1f}%**")
            if 'latency_reduction_pct' in comparison:
                report.append(f"- 延迟降低: **{comparison['latency_reduction_pct']:.1f}%**")
            if 'ctx_ratio_change_pct' in comparison:
                report.append(f"- 跨分片比例变化: **{comparison['ctx_ratio_change_pct']:.1f}%**")
            report.append("")
        
        # Key Findings
        report.append("## 关键发现 (Key Findings)")
        
        # Determine best performer
        if len(self.metrics) >= 2:
            tps_rankings = sorted(self.metrics.items(), 
                                key=lambda x: x[1].get('tps', {}).get('avg', 0), reverse=True)
            
            if tps_rankings:
                best_variant = tps_rankings[0][0]
                report.append(f"1. **性能最佳变体**: {VARIANTS[best_variant]} (平均 TPS: {tps_rankings[0][1].get('tps', {}).get('avg', 0):.2f})")
                
                if best_variant == 'ours_full':
                    report.append("   - 完整实现展现出最优性能，证明所有组件的协同效应")
                    
                    # Analyze which component contributes most
                    if 'ours_no_contrastive' in self.comparison_results and 'ours_no_layering' in self.comparison_results:
                        contrastive_impact = self.comparison_results['ours_no_contrastive'].get('tps_improvement_pct', 0)
                        layering_impact = self.comparison_results['ours_no_layering'].get('tps_improvement_pct', 0)
                        
                        if contrastive_impact > layering_impact:
                            report.append("   - 对比学习组件对性能提升贡献更大")
                        else:
                            report.append("   - 分层处理组件对性能提升贡献更大")
        
        report.append("")
        
        # Technical Insights
        report.append("## 技术洞察 (Technical Insights)")
        report.append("1. **系统负载特性**: 从跨分片交易比例可以看出系统的分片效果")
        report.append("2. **性能稳定性**: 从不同 epoch 的性能变化可以评估系统稳定性")
        report.append("3. **优化方向**: 基于 ablation 结果确定后续优化的重点组件")
        report.append("")
        
        # Save report
        report_path = "ablation_analysis_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"✅ 分析报告已保存: {report_path}")
        
        # Also save as JSON for programmatic access
        report_data = {
            'generation_time': datetime.now().isoformat(),
            'metrics': self.metrics,
            'comparisons': self.comparison_results,
            'variants': VARIANTS
        }
        
        json_report_path = "ablation_analysis_report.json"
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON 报告已保存: {json_report_path}")
        
    def run_complete_analysis(self):
        """Run the complete ablation analysis pipeline"""
        print("🚀 开始完整的 Ablation Study 分析")
        print("="*60)
        
        try:
            # Step 1: Load data
            self.load_data()
            
            if not self.data:
                print("❌ 错误: 无法加载任何实验数据!")
                return
            
            # Step 2: Calculate metrics
            self.calculate_metrics()
            
            # Step 3: Generate comparisons
            self.generate_comparison_analysis()
            
            # Step 4: Create visualizations
            self.create_static_performance_chart()
            self.create_dynamic_evolution_chart()
            
            # Step 5: Export CSV data
            self.export_csv_data()
            
            # Step 6: Generate comprehensive report
            self.generate_analysis_report()
            
            print("\n" + "="*60)
            print("🎉 Ablation Study 分析完成!")
            print("\n📁 生成的文件:")
            print("   📊 ablation_static_performance_comparison.png - 静态性能对比图")
            print("   📈 ablation_dynamic_evolution_comparison.png - 动态演化图")
            print("   📋 ablation_summary_statistics.csv - 汇总统计数据")
            print("   📋 ablation_comparison_analysis.csv - 对比分析数据")
            print("   📋 ablation_detailed_evolution.csv - 详细演化数据")
            print("   📋 ablation_static_chart_data.csv - 静态图表数据")
            print("   📄 ablation_analysis_report.md - 分析报告")
            print("   📄 ablation_analysis_report.json - JSON 格式报告")
            print("\n✨ 所有数据和图表已准备好用于学术发表!")
            
        except Exception as e:
            print(f"❌ 分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution function"""
    print("BlockEmulator Ablation Study Results Analyzer")
    print("============================================")
    print("这个脚本将分析三个变体的实验结果:")
    print("1. ours_full - 完整实现")  
    print("2. ours_no_contrastive - 无对比学习")
    print("3. ours_no_layering - 无分层处理")
    print()
    
    # Initialize and run analysis
    analyzer = AblationAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
