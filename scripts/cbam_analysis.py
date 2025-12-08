import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_cbam_analysis_plots():
    """Create comprehensive CBAM analysis plots"""
    
    # Create output directory
    output_dir = 'plots/cbam_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Load CBAM results (using cbam7 as it has better performance)
    cbam_df = pd.read_csv('experiments/cbam/yolo12_cbam7/results.csv')
    
    # Load baseline YOLO and RT-DETR for comparison
    yolo_df = pd.read_csv('yolov12_results.csv')
    rtdetr_df = pd.read_csv('rt_detr_results.csv')
    
    print("Creating CBAM analysis plots...")
    print(f"CBAM epochs: {len(cbam_df)}")
    print(f"YOLO epochs: {len(yolo_df)}")
    print(f"RT-DETR epochs: {len(rtdetr_df)}")
    
    # 1. Learning Curves Comparison
    create_learning_curves(cbam_df, yolo_df, rtdetr_df, f'{output_dir}/learning_curves.png')
    
    # 2. Loss Progression Comparison
    create_loss_comparison(cbam_df, yolo_df, rtdetr_df, f'{output_dir}/loss_comparison.png')
    
    # 3. Metrics Progression
    create_metrics_progression(cbam_df, yolo_df, rtdetr_df, f'{output_dir}/metrics_progression.png')
    
    # 4. CBAM-specific detailed analysis
    create_cbam_detailed_analysis(cbam_df, f'{output_dir}/cbam_detailed.png')
    
    # 5. Final performance comparison
    create_final_comparison(cbam_df, yolo_df, rtdetr_df, f'{output_dir}/final_performance.png')
    
    # Print summary
    print_performance_summary(cbam_df, yolo_df, rtdetr_df)

def create_learning_curves(cbam_df, yolo_df, rtdetr_df, output_path):
    """Create learning curves comparing all three models"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # mAP@0.5 progression
    ax1.plot(cbam_df['epoch'], cbam_df['metrics/mAP50(B)'], 'g-', linewidth=2, label='YOLOv12L+CBAM', marker='o', markersize=4)
    ax1.plot(yolo_df['epoch'], yolo_df['metrics/mAP50(B)'], 'b-', linewidth=2, label='YOLOv12L', marker='s', markersize=4)
    ax1.plot(rtdetr_df['epoch'], rtdetr_df['metrics/mAP50(B)'], 'r-', linewidth=2, label='RT-DETR-L', marker='^', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('mAP@0.5')
    ax1.set_title('mAP@0.5 Learning Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # mAP@0.5:0.95 progression
    ax2.plot(cbam_df['epoch'], cbam_df['metrics/mAP50-95(B)'], 'g-', linewidth=2, label='YOLOv12L+CBAM', marker='o', markersize=4)
    ax2.plot(yolo_df['epoch'], yolo_df['metrics/mAP50-95(B)'], 'b-', linewidth=2, label='YOLOv12L', marker='s', markersize=4)
    ax2.plot(rtdetr_df['epoch'], rtdetr_df['metrics/mAP50-95(B)'], 'r-', linewidth=2, label='RT-DETR-L', marker='^', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP@0.5:0.95')
    ax2.set_title('mAP@0.5:0.95 Learning Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Precision progression
    ax3.plot(cbam_df['epoch'], cbam_df['metrics/precision(B)'], 'g-', linewidth=2, label='YOLOv12L+CBAM', marker='o', markersize=4)
    ax3.plot(yolo_df['epoch'], yolo_df['metrics/precision(B)'], 'b-', linewidth=2, label='YOLOv12L', marker='s', markersize=4)
    ax3.plot(rtdetr_df['epoch'], rtdetr_df['metrics/precision(B)'], 'r-', linewidth=2, label='RT-DETR-L', marker='^', markersize=4)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision Learning Curves')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Recall progression
    ax4.plot(cbam_df['epoch'], cbam_df['metrics/recall(B)'], 'g-', linewidth=2, label='YOLOv12L+CBAM', marker='o', markersize=4)
    ax4.plot(yolo_df['epoch'], yolo_df['metrics/recall(B)'], 'b-', linewidth=2, label='YOLOv12L', marker='s', markersize=4)
    ax4.plot(rtdetr_df['epoch'], rtdetr_df['metrics/recall(B)'], 'r-', linewidth=2, label='RT-DETR-L', marker='^', markersize=4)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Recall')
    ax4.set_title('Recall Learning Curves')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_loss_comparison(cbam_df, yolo_df, rtdetr_df, output_path):
    """Create loss comparison plots"""
    
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training losses
    ax1.plot(cbam_df['epoch'], cbam_df['train/box_loss'], 'g-', linewidth=2, label='YOLOv12L+CBAM')
    ax1.plot(yolo_df['epoch'], yolo_df['train/box_loss'], 'b-', linewidth=2, label='YOLOv12L')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Box Loss')
    ax1.set_title('Training Box Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(cbam_df['epoch'], cbam_df['train/cls_loss'], 'g-', linewidth=2, label='YOLOv12L+CBAM')
    ax2.plot(yolo_df['epoch'], yolo_df['train/cls_loss'], 'b-', linewidth=2, label='YOLOv12L')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Class Loss')
    ax2.set_title('Training Classification Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(cbam_df['epoch'], cbam_df['train/dfl_loss'], 'g-', linewidth=2, label='YOLOv12L+CBAM')
    ax3.plot(yolo_df['epoch'], yolo_df['train/dfl_loss'], 'b-', linewidth=2, label='YOLOv12L')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Training DFL Loss')
    ax3.set_title('Training DFL Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Validation losses
    ax4.plot(cbam_df['epoch'], cbam_df['val/box_loss'], 'g-', linewidth=2, label='YOLOv12L+CBAM')
    ax4.plot(yolo_df['epoch'], yolo_df['val/box_loss'], 'b-', linewidth=2, label='YOLOv12L')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Box Loss')
    ax4.set_title('Validation Box Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5.plot(cbam_df['epoch'], cbam_df['val/cls_loss'], 'g-', linewidth=2, label='YOLOv12L+CBAM')
    ax5.plot(yolo_df['epoch'], yolo_df['val/cls_loss'], 'b-', linewidth=2, label='YOLOv12L')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Validation Class Loss')
    ax5.set_title('Validation Classification Loss')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6.plot(cbam_df['epoch'], cbam_df['val/dfl_loss'], 'g-', linewidth=2, label='YOLOv12L+CBAM')
    ax6.plot(yolo_df['epoch'], yolo_df['val/dfl_loss'], 'b-', linewidth=2, label='YOLOv12L')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Validation DFL Loss')
    ax6.set_title('Validation DFL Loss')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_metrics_progression(cbam_df, yolo_df, rtdetr_df, output_path):
    """Create detailed metrics progression over epochs"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Normalize epoch to percentage of training completed
    cbam_progress = cbam_df['epoch'] / cbam_df['epoch'].max() * 100
    yolo_progress = yolo_df['epoch'] / yolo_df['epoch'].max() * 100
    rtdetr_progress = rtdetr_df['epoch'] / rtdetr_df['epoch'].max() * 100
    
    # Plot multiple metrics for comparison
    ax.plot(cbam_progress, cbam_df['metrics/mAP50(B)'], 'g-', linewidth=2.5, label='CBAM mAP@0.5', alpha=0.8)
    ax.plot(cbam_progress, cbam_df['metrics/mAP50-95(B)'], 'g--', linewidth=2.5, label='CBAM mAP@0.5:0.95', alpha=0.8)
    
    ax.plot(yolo_progress, yolo_df['metrics/mAP50(B)'], 'b-', linewidth=2.5, label='YOLO mAP@0.5', alpha=0.8)
    ax.plot(yolo_progress, yolo_df['metrics/mAP50-95(B)'], 'b--', linewidth=2.5, label='YOLO mAP@0.5:0.95', alpha=0.8)
    
    ax.plot(rtdetr_progress, rtdetr_df['metrics/mAP50(B)'], 'r-', linewidth=2.5, label='RT-DETR mAP@0.5', alpha=0.8)
    ax.plot(rtdetr_progress, rtdetr_df['metrics/mAP50-95(B)'], 'r--', linewidth=2.5, label='RT-DETR mAP@0.5:0.95', alpha=0.8)
    
    ax.set_xlabel('Training Progress (%)', fontsize=12)
    ax.set_ylabel('Performance Metric', fontsize=12)
    ax.set_title('Model Performance Progression', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_cbam_detailed_analysis(cbam_df, output_path):
    """Create detailed CBAM-specific analysis"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Training time per epoch
    if 'time' in cbam_df.columns:
        epoch_times = cbam_df['time'].diff().fillna(cbam_df['time'].iloc[0])
        ax1.plot(cbam_df['epoch'], epoch_times, 'g-', linewidth=2, marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Time per Epoch (seconds)')
        ax1.set_title('CBAM Training Time per Epoch')
        ax1.grid(True, alpha=0.3)
    
    # Learning rate schedule
    ax2.plot(cbam_df['epoch'], cbam_df['lr/pg0'], 'purple', linewidth=2, label='Learning Rate')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('CBAM Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Training vs Validation loss comparison
    ax3.plot(cbam_df['epoch'], cbam_df['train/box_loss'], 'b-', linewidth=2, label='Training Box Loss')
    ax3.plot(cbam_df['epoch'], cbam_df['val/box_loss'], 'r-', linewidth=2, label='Validation Box Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Box Loss')
    ax3.set_title('CBAM Training vs Validation Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Precision-Recall curve over epochs
    ax4.scatter(cbam_df['metrics/recall(B)'], cbam_df['metrics/precision(B)'], 
               c=cbam_df['epoch'], cmap='viridis', s=60, alpha=0.8)
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('CBAM Precision-Recall Evolution')
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Epoch')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_final_comparison(cbam_df, yolo_df, rtdetr_df, output_path):
    """Create final performance comparison"""
    
    # Get final epoch results
    cbam_final = cbam_df.iloc[-1]
    yolo_final = yolo_df.iloc[-1]
    rtdetr_final = rtdetr_df.iloc[-1]
    
    models = ['YOLOv12L+CBAM', 'YOLOv12L', 'RT-DETR-L']
    map50_values = [cbam_final['metrics/mAP50(B)'], yolo_final['metrics/mAP50(B)'], rtdetr_final['metrics/mAP50(B)']]
    map5095_values = [cbam_final['metrics/mAP50-95(B)'], yolo_final['metrics/mAP50-95(B)'], rtdetr_final['metrics/mAP50-95(B)']]
    precision_values = [cbam_final['metrics/precision(B)'], yolo_final['metrics/precision(B)'], rtdetr_final['metrics/precision(B)']]
    recall_values = [cbam_final['metrics/recall(B)'], yolo_final['metrics/recall(B)'], rtdetr_final['metrics/recall(B)']]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['green', 'blue', 'red']
    
    # mAP@0.5 comparison
    bars1 = ax1.bar(models, map50_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('mAP@0.5')
    ax1.set_title('Final mAP@0.5 Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars1, map50_values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # mAP@0.5:0.95 comparison  
    bars2 = ax2.bar(models, map5095_values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('mAP@0.5:0.95')
    ax2.set_title('Final mAP@0.5:0.95 Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars2, map5095_values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Precision comparison
    bars3 = ax3.bar(models, precision_values, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Precision')
    ax3.set_title('Final Precision Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars3, precision_values):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Recall comparison
    bars4 = ax4.bar(models, recall_values, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Recall')
    ax4.set_title('Final Recall Comparison')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars4, recall_values):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_performance_summary(cbam_df, yolo_df, rtdetr_df):
    """Print detailed performance summary"""
    
    cbam_final = cbam_df.iloc[-1]
    yolo_final = yolo_df.iloc[-1]
    rtdetr_final = rtdetr_df.iloc[-1]
    
    print("\n" + "="*80)
    print("COMPREHENSIVE CBAM ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nFINAL PERFORMANCE (Epoch 20):")
    print("-" * 50)
    print(f"{'Model':<20} {'mAP@0.5':<10} {'mAP@0.5:0.95':<12} {'Precision':<10} {'Recall':<10}")
    print("-" * 50)
    print(f"{'YOLOv12L+CBAM':<20} {cbam_final['metrics/mAP50(B)']:<10.3f} {cbam_final['metrics/mAP50-95(B)']:<12.3f} {cbam_final['metrics/precision(B)']:<10.3f} {cbam_final['metrics/recall(B)']:<10.3f}")
    print(f"{'YOLOv12L':<20} {yolo_final['metrics/mAP50(B)']:<10.3f} {yolo_final['metrics/mAP50-95(B)']:<12.3f} {yolo_final['metrics/precision(B)']:<10.3f} {yolo_final['metrics/recall(B)']:<10.3f}")
    print(f"{'RT-DETR-L':<20} {rtdetr_final['metrics/mAP50(B)']:<10.3f} {rtdetr_final['metrics/mAP50-95(B)']:<12.3f} {rtdetr_final['metrics/precision(B)']:<10.3f} {rtdetr_final['metrics/recall(B)']:<10.3f}")
    
    print(f"\nCBAM vs YOLO IMPROVEMENTS:")
    print("-" * 30)
    map50_improvement = ((cbam_final['metrics/mAP50(B)'] - yolo_final['metrics/mAP50(B)']) / yolo_final['metrics/mAP50(B)']) * 100
    map5095_improvement = ((cbam_final['metrics/mAP50-95(B)'] - yolo_final['metrics/mAP50-95(B)']) / yolo_final['metrics/mAP50-95(B)']) * 100
    precision_improvement = ((cbam_final['metrics/precision(B)'] - yolo_final['metrics/precision(B)']) / yolo_final['metrics/precision(B)']) * 100
    recall_improvement = ((cbam_final['metrics/recall(B)'] - yolo_final['metrics/recall(B)']) / yolo_final['metrics/recall(B)']) * 100
    
    print(f"mAP@0.5: {map50_improvement:+.1f}%")
    print(f"mAP@0.5:0.95: {map5095_improvement:+.1f}%")
    print(f"Precision: {precision_improvement:+.1f}%")
    print(f"Recall: {recall_improvement:+.1f}%")
    
    if 'time' in cbam_df.columns and 'time' in yolo_df.columns:
        cbam_total_time = cbam_final['time']
        yolo_total_time = yolo_final['time']
        print(f"\nTRAINING TIME:")
        print("-" * 20)
        print(f"CBAM Total Time: {cbam_total_time:.1f} seconds ({cbam_total_time/60:.1f} minutes)")
        print(f"YOLO Total Time: {yolo_total_time:.1f} seconds ({yolo_total_time/60:.1f} minutes)")
        print(f"Time Overhead: {((cbam_total_time - yolo_total_time) / yolo_total_time * 100):+.1f}%")
    
    print("\n" + "="*80)
    print("PLOTS GENERATED IN: plots/cbam_analysis/")
    print("- learning_curves.png: Learning curves comparison")
    print("- loss_comparison.png: Training and validation losses")
    print("- metrics_progression.png: Performance over training progress")
    print("- cbam_detailed.png: CBAM-specific analysis")
    print("- final_performance.png: Final performance comparison")
    print("="*80)

if __name__ == "__main__":
    create_cbam_analysis_plots()