import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def extract_final_metrics_from_csv(csv_path):
    """Extract the final epoch metrics from a results CSV file"""
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    
    # Get the last row (final epoch)
    final_row = df.iloc[-1]
    
    # Extract relevant metrics
    metrics = {
        'map50': final_row.get('metrics/mAP50(B)', None),
        'map5095': final_row.get('metrics/mAP50-95(B)', None),
        'precision': final_row.get('metrics/precision(B)', None),
        'recall': final_row.get('metrics/recall(B)', None)
    }
    
    return metrics

def create_combined_results_dataframe():
    """Create a combined dataframe with all model results"""
    results = []
    
    # YOLO baseline results
    yolo_metrics = extract_final_metrics_from_csv('yolov12_results.csv')
    if yolo_metrics:
        results.append({
            'model': 'YOLOv12L',
            'map50': yolo_metrics['map50'],
            'map5095': yolo_metrics['map5095'],
            'precision': yolo_metrics['precision'],
            'recall': yolo_metrics['recall'],
            'type': 'Object Detection'
        })
    
    # RT-DETR results
    rtdetr_metrics = extract_final_metrics_from_csv('rt_detr_results.csv')
    if rtdetr_metrics:
        results.append({
            'model': 'RT-DETR-L',
            'map50': rtdetr_metrics['map50'],
            'map5095': rtdetr_metrics['map5095'],
            'precision': rtdetr_metrics['precision'],
            'recall': rtdetr_metrics['recall'],
            'type': 'Object Detection'
        })
    
    # YOLO+CBAM results (from best performing CBAM experiment)
    cbam_metrics = extract_final_metrics_from_csv('experiments/cbam/yolo12_cbam7/results.csv')
    if cbam_metrics:
        results.append({
            'model': 'YOLOv12L+CBAM',
            'map50': cbam_metrics['map50'],
            'map5095': cbam_metrics['map5095'],
            'precision': cbam_metrics['precision'],
            'recall': cbam_metrics['recall'],
            'type': 'Object Detection + Attention'
        })
    

    
    return pd.DataFrame(results)

def plot_model_comparison(df, output_dir='plots/comparison'):
    """Create comprehensive comparison plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. mAP Comparison Bar Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # mAP@0.5
    ax1.bar(df['model'], df['map50'], alpha=0.8)
    ax1.set_title('mAP@0.5 Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('mAP@0.5', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(df['map50']):
        if pd.notna(v):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # mAP@0.5:0.95
    ax2.bar(df['model'], df['map5095'], alpha=0.8, color='orange')
    ax2.set_title('mAP@0.5:0.95 Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('mAP@0.5:0.95', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(df['map5095']):
        if pd.notna(v):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/map_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Precision vs Recall Scatter Plot
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'purple']
    markers = ['o', 's', '^', 'D']
    
    for i, (_, row) in enumerate(df.iterrows()):
        if pd.notna(row['precision']) and pd.notna(row['recall']):
            plt.scatter(row['recall'], row['precision'], 
                       s=200, c=colors[i % len(colors)], 
                       marker=markers[i % len(markers)], 
                       alpha=0.8, edgecolors='black', linewidth=2,
                       label=row['model'])
            
            # Add model name near the point
            plt.annotate(row['model'], 
                        (row['recall'], row['precision']),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision vs Recall Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/precision_recall_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Comprehensive Metrics Radar Chart
    create_radar_chart(df, f'{output_dir}/radar_comparison.png')
    
    # 4. Performance Improvement Analysis
    create_improvement_analysis(df, f'{output_dir}/improvement_analysis.png')

def create_radar_chart(df, output_path):
    """Create a radar chart comparing all metrics"""
    # Normalize metrics to 0-1 scale for fair comparison
    metrics_cols = ['map50', 'map5095', 'precision', 'recall']
    df_norm = df.copy()
    
    for col in metrics_cols:
        if df[col].notna().any():
            max_val = df[col].max()
            min_val = df[col].min()
            if max_val > min_val:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 1.0
    
    # Set up the radar chart
    categories = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    colors = ['blue', 'red', 'green', 'purple']
    
    for i, (_, row) in enumerate(df_norm.iterrows()):
        values = [row[col] for col in metrics_cols]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, 
                label=row['model'], color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Comparison (Normalized)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_improvement_analysis(df, output_path):
    """Create plots showing CBAM improvements over baseline"""
    # Find YOLO baseline and CBAM results for comparison
    yolo_baseline = df[df['model'] == 'YOLOv12L'].iloc[0] if len(df[df['model'] == 'YOLOv12L']) > 0 else None
    yolo_cbam = df[df['model'].str.contains('CBAM')].iloc[0] if len(df[df['model'].str.contains('CBAM')]) > 0 else None
    
    if yolo_baseline is None or yolo_cbam is None:
        print("Could not find both baseline and CBAM results for comparison")
        return
    
    # Calculate improvements
    metrics = ['map50', 'map5095', 'precision', 'recall']
    improvements = {}
    
    for metric in metrics:
        if pd.notna(yolo_baseline[metric]) and pd.notna(yolo_cbam[metric]):
            baseline_val = yolo_baseline[metric]
            cbam_val = yolo_cbam[metric]
            improvement = ((cbam_val - baseline_val) / baseline_val) * 100
            improvements[metric] = improvement
    
    # Create improvement bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metric_names = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
    improvement_values = [improvements.get(m, 0) for m in metrics]
    
    colors = ['green' if v >= 0 else 'red' for v in improvement_values]
    bars = ax.bar(metric_names, improvement_values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, improvement_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1),
                f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('YOLO+CBAM Performance Improvement over Baseline YOLO', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("YOLO+CBAM PERFORMANCE ANALYSIS")
    print("="*60)
    print(f"Baseline YOLO mAP@0.5: {yolo_baseline['map50']:.3f}")
    print(f"YOLO+CBAM mAP@0.5: {yolo_cbam['map50']:.3f}")
    print(f"Improvement: {improvements.get('map50', 0):.1f}%")
    print()
    print(f"Baseline YOLO mAP@0.5:0.95: {yolo_baseline['map5095']:.3f}")
    print(f"YOLO+CBAM mAP@0.5:0.95: {yolo_cbam['map5095']:.3f}")
    print(f"Improvement: {improvements.get('map5095', 0):.1f}%")
    print("="*60)

def main():
    """Main function to create all plots"""
    print("Creating combined model comparison plots...")
    
    # Create results dataframe
    df = create_combined_results_dataframe()
    
    if df.empty:
        print("No results found! Make sure the CSV files exist.")
        return
    
    print("Found results for the following models:")
    for model in df['model'].values:
        print(f"  - {model}")
    
    # Create comparison plots
    plot_model_comparison(df)
    
    print(f"\nPlots saved to: plots/comparison/")
    print("Generated plots:")
    print("  - map_comparison.png: mAP comparison bar charts")
    print("  - precision_recall_scatter.png: Precision vs Recall scatter plot")
    print("  - radar_comparison.png: Normalized metrics radar chart")
    print("  - improvement_analysis.png: CBAM improvement analysis")

if __name__ == "__main__":
    main()