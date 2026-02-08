#Presented by KeJi
#Date : 2026-01-30

"""
热力图绘制脚本2
读取compare.xlsx的DstateCompare sheet，展示D-state×Inner Embedding Dimension对准确率的影响
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import argparse

# 学术论文风格配置（字体稍大）
ACADEMIC_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'axes.linewidth': 1.2,
}

# NATURE配色方案用于colorbar
CMAP_NAME = 'YlOrRd'  # 黄-橙-红渐变，适合展示准确率

def Load_Data(filepath, sheet_name='DstateCompare'):
    """加载compare.xlsx的DstateCompare sheet"""
    df = pd.read_excel(filepath, sheet_name=sheet_name, index_col=0)
    return df

def Draw_Heatmap(df, output_path, title='D-state vs Inner Embedding Dimension - Accuracy', 
                 cmap=CMAP_NAME, annot=True, fmt='.2f'):
    """绘制热力图"""
    plt.rcParams.update(ACADEMIC_STYLE)
    
    # 计算合适的图像尺寸
    n_rows, n_cols = df.shape
    fig_width = max(8, n_cols * 1.2)
    fig_height = max(6, n_rows * 0.8)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # 绘制热力图
    heatmap = sns.heatmap(
        df,
        ax=ax,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        linewidths=0.5,
        linecolor='black',
        cbar_kws={
            'label': 'Accuracy (%)',
            'shrink': 0.8
        },
        annot_kws={
            'size': 12,
            'weight': 'bold'
        }
    )
    
    # 设置标签
    ax.set_xlabel('Inner Embedding Dimension', fontweight='bold')
    ax.set_ylabel('D-state', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    
    # 旋转x轴标签
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    # 设置colorbar标签字体
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存PNG和PDF
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"保存: {output_path}, {pdf_path}")

def main():
    parser = argparse.ArgumentParser(description='绘制D-state vs Inner Embedding Dimension热力图')
    parser.add_argument('-i', '--input', type=str, 
                        default='compare.xlsx',
                        help='输入文件路径 (默认: compare.xlsx)')
    parser.add_argument('-s', '--sheet', type=str,
                        default='DstateCompare',
                        help='Sheet名称 (默认: DstateCompare)')
    parser.add_argument('-o', '--output', type=str,
                        default='Img/dstate_compare_heatmap.png',
                        help='输出文件路径 (默认: Img/dstate_compare_heatmap.png)')
    parser.add_argument('-t', '--title', type=str,
                        default='D-state vs Inner Embedding Dimension - Accuracy',
                        help='图表标题')
    parser.add_argument('--cmap', type=str,
                        default='YlOrRd',
                        help='颜色映射 (默认: YlOrRd)')
    parser.add_argument('--no-annot', action='store_true',
                        help='不显示数值标注')
    
    args = parser.parse_args()
    
    # 确定输入文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.input):
        input_path = os.path.join(script_dir, args.input)
    else:
        input_path = args.input
    
    # 确定输出路径
    if not os.path.isabs(args.output):
        output_path = os.path.join(script_dir, args.output)
    else:
        output_path = args.output
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 加载数据
    print(f"加载数据: {input_path}, sheet: {args.sheet}")
    df = Load_Data(input_path, args.sheet)
    print(f"数据形状: {df.shape}")
    print(f"行索引(D-state): {df.index.tolist()}")
    print(f"列名(Inner Embedding Dimension): {df.columns.tolist()}")
    
    # 绘制热力图
    Draw_Heatmap(
        df, 
        output_path, 
        title=args.title,
        cmap=args.cmap,
        annot=not args.no_annot
    )
    
    print("完成!")

if __name__ == '__main__':
    main()
