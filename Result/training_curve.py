#Presented by KeJi
#Date: 2026-01-22

"""
训练曲线绘图脚本
从result.xlsx的Summary sheet读取数据，绘制训练曲线
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# NATURE配色方案
NATURE_COLORS = {
    'Edge Vim Max Model': '#CC247C',      # Magenta
    'Vim Small': '#E95351',               # Red
    'Vim tiny': '#F7A24F',                # Orange
    'Edge Vim Random Model 1': '#4EA660',  # Green
    'Edge Vim Random Model 2': '#5C9AD4',  # Blue
    'Edge Vim Min Model': '#AA77E9',       # Purple
}

# 学术风格配置（字号优化，适配双栏论文）
ACADEMIC_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
}

# 线条标记样式
LINE_MARKERS = {
    'Edge Vim Max Model': 'o',
    'Vim Small': 's',
    'Vim tiny': '^',
    'Edge Vim Min Model': 'D',
}

SCATTER_MARKERS = {
    'Edge Vim Random Model 1': 'P',  # plus filled
    'Edge Vim Random Model 2': 'X',  # x filled
}


def Load_Data(xlsx_path: str) -> pd.DataFrame:
    """
    从xlsx文件的Summary sheet加载数据
    
    Args:
        xlsx_path: xlsx文件路径
        
    Returns:
        DataFrame包含所有模型的训练数据
    """
    df = pd.read_excel(xlsx_path, sheet_name='Summary', engine='openpyxl')
    return df


def Draw_Training_Curve(
    df: pd.DataFrame,
    output_path: str = None,
    title: str = 'Training Accuracy Curves',
    show_legend: bool = True,
    figsize: tuple = (10, 6),
    dpi: int = 300,
    ylim: tuple = None,
    annotate_final: bool = True
):
    """
    绘制训练曲线图
    
    Args:
        df: 包含训练数据的DataFrame
        output_path: 输出文件路径
        title: 图表标题
        show_legend: 是否显示图例
        figsize: 图表尺寸
        dpi: 输出分辨率
        ylim: y轴范围 (min, max)，用于放大特定准确率区间
        annotate_final: 是否在最后一个epoch标注EdgeVim Max Model和Vim Small
    """
    plt.rcParams.update(ACADEMIC_STYLE)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = df['epoch'].values
    
    # 绘制顺序：Max → Random1 → Random2 → Min → Small → tiny
    plot_order = [
        ('Edge Vim Max Model', 'line'),
        ('Edge Vim Random Model 1', 'scatter'),
        ('Edge Vim Random Model 2', 'scatter'),
        ('Edge Vim Min Model', 'line'),
        ('Vim Small', 'line'),
        ('Vim tiny', 'line'),
    ]
    
    # 按照指定顺序绘制
    for model, plot_type in plot_order:
        if model not in df.columns:
            continue
        data = df[model].values
        # 过滤掉NaN值
        valid_mask = ~np.isnan(data)
        valid_epochs = epochs[valid_mask]
        valid_data = data[valid_mask]
        
        if len(valid_data) == 0:
            continue
        
        if plot_type == 'scatter':
            ax.scatter(valid_epochs, valid_data,
                      color=NATURE_COLORS[model],
                      marker=SCATTER_MARKERS[model],
                      s=100,
                      edgecolor='black',
                      linewidth=0.5,
                      label=model,
                      zorder=10)
        else:
            # 根据数据点数量决定marker间隔，但始终使用带marker的plot确保图例正确
            if len(valid_data) <= 20:
                markevery = 1  # 每个点都标记
            else:
                markevery = 50  # 每隔50个点标记
            
            ax.plot(valid_epochs, valid_data,
                   color=NATURE_COLORS[model],
                   marker=LINE_MARKERS[model],
                   markersize=6,
                   markevery=markevery,
                   linewidth=2,
                   label=model,
                   markeredgecolor='black',
                   markeredgewidth=0.5)
    
    # 在epoch 299处为Edge Vim Max Model和Vim Small添加标记点
    if annotate_final:
        final_epoch = df['epoch'].max()
        final_row = df[df['epoch'] == final_epoch].iloc[0]
        
        if 'Edge Vim Max Model' in df.columns and 'Vim Small' in df.columns:
            max_acc = final_row['Edge Vim Max Model']
            small_acc = final_row['Vim Small']
            
            if not np.isnan(max_acc) and not np.isnan(small_acc):
                # 在最后一点处添加标记点
                ax.scatter([final_epoch], [max_acc], color=NATURE_COLORS['Edge Vim Max Model'],
                          marker=LINE_MARKERS['Edge Vim Max Model'], s=60, edgecolor='black',
                          linewidth=0.5, zorder=15)
                ax.scatter([final_epoch], [small_acc], color=NATURE_COLORS['Vim Small'],
                          marker=LINE_MARKERS['Vim Small'], s=60, edgecolor='black',
                          linewidth=0.5, zorder=15)
    
    # 设置坐标轴
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Top-1 Accuracy (%)', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    
    # 设置y轴范围
    if ylim:
        ax.set_ylim(ylim)
    
    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    # 设置图例
    if show_legend:
        legend_loc = 'upper left' if ylim else 'lower right'
        ax.legend(loc=legend_loc, frameon=True, fancybox=True,
                 framealpha=0.9, edgecolor='black')
    
    # 设置边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"图表已保存至: {output_path}")
        
        # 如果是PNG，同时保存PDF
        if output_path.endswith('.png'):
            pdf_path = output_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"PDF版本已保存至: {pdf_path}")
    else:
        plt.show()
    
    plt.close()


def Draw_Zoomed_Curve(
    df: pd.DataFrame,
    output_path: str = None,
    epoch_range: tuple = (200, 300),
    ylim: tuple = None,
    title: str = 'Training Accuracy Curves (Zoomed)',
    figsize: tuple = (10, 6),
    dpi: int = 300
):
    """
    绘制放大后的训练曲线图（聚焦特定epoch范围或y轴范围）
    
    Args:
        df: 包含训练数据的DataFrame
        output_path: 输出文件路径
        epoch_range: epoch范围 (start, end)
        ylim: y轴范围 (min, max)
        title: 图表标题
        figsize: 图表尺寸
        dpi: 输出分辨率
    """
    plt.rcParams.update(ACADEMIC_STYLE)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 筛选epoch范围
    mask = (df['epoch'] >= epoch_range[0]) & (df['epoch'] <= epoch_range[1])
    df_filtered = df[mask]
    epochs = df_filtered['epoch'].values
    
    # 绘制顺序：Max → Random1 → Random2 → Min → Small → tiny
    plot_order = [
        ('Edge Vim Max Model', 'line'),
        ('Edge Vim Random Model 1', 'scatter'),
        ('Edge Vim Random Model 2', 'scatter'),
        ('Edge Vim Min Model', 'line'),
        ('Vim Small', 'line'),
        ('Vim tiny', 'line'),
    ]
    
    # 按照指定顺序绘制
    for model, plot_type in plot_order:
        if model not in df_filtered.columns:
            continue
        data = df_filtered[model].values
        valid_mask = ~np.isnan(data)
        valid_epochs = epochs[valid_mask]
        valid_data = data[valid_mask]
        
        if len(valid_data) == 0:
            continue
        
        if plot_type == 'scatter':
            ax.scatter(valid_epochs, valid_data,
                      color=NATURE_COLORS[model],
                      marker=SCATTER_MARKERS[model],
                      s=100,
                      edgecolor='black',
                      linewidth=0.5,
                      label=model,
                      zorder=10)
        else:
            ax.plot(valid_epochs, valid_data,
                   color=NATURE_COLORS[model],
                   marker=LINE_MARKERS[model],
                   markersize=6,
                   linewidth=2,
                   label=model,
                   markeredgecolor='black',
                   markeredgewidth=0.5)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Top-1 Accuracy (%)', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    
    # 设置y轴范围
    if ylim:
        ax.set_ylim(ylim)
    
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    legend_loc = 'upper left' if ylim else 'lower right'
    ax.legend(loc=legend_loc, frameon=True, fancybox=True,
             framealpha=0.9, edgecolor='black')
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')
    
    plt.tight_layout()
    
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"图表已保存至: {output_path}")
        
        if output_path.endswith('.png'):
            pdf_path = output_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"PDF版本已保存至: {pdf_path}")
    else:
        plt.show()
    
    plt.close()


def Main():
    """主函数"""
    parser = argparse.ArgumentParser(description='绘制训练曲线图')
    parser.add_argument('--input', '-i', type=str,
                       default='Log/result.xlsx',
                       help='输入xlsx文件路径')
    parser.add_argument('--output', '-o', type=str,
                       default='Img/training_curve.png',
                       help='输出图片路径')
    parser.add_argument('--title', '-t', type=str,
                       default='Training Accuracy on ImageNet-1K',
                       help='图表标题')
    parser.add_argument('--zoomed', '-z', action='store_true',
                       help='绘制放大版本(epoch 200-300)')
    parser.add_argument('--epoch-start', type=int, default=200,
                       help='放大版本起始epoch')
    parser.add_argument('--epoch-end', type=int, default=300,
                       help='放大版本结束epoch')
    parser.add_argument('--ylim', type=str, default=None,
                       help='y轴范围，格式: min,max (如70,80)')
    parser.add_argument('--figsize', type=str, default='10,6',
                       help='图表尺寸，格式: width,height')
    parser.add_argument('--dpi', type=int, default=300,
                       help='输出DPI')
    parser.add_argument('--all', '-a', action='store_true',
                       help='生成全部图表（完整、epoch放大、y轴放大版本）')
    
    args = parser.parse_args()
    
    # 解析figsize
    figsize = tuple(map(float, args.figsize.split(',')))
    
    # 解析ylim
    ylim = None
    if args.ylim:
        ylim = tuple(map(float, args.ylim.split(',')))
    
    # 加载数据
    print(f"正在加载数据: {args.input}")
    df = Load_Data(args.input)
    print(f"数据加载完成，共 {len(df)} 行")
    
    if args.all:
        # 生成完整曲线
        full_output = args.output
        Draw_Training_Curve(df, full_output, args.title,
                           figsize=figsize, dpi=args.dpi)
        
        # 生成epoch放大版本
        zoomed_output = args.output.replace('.png', '_epoch_zoomed.png')
        Draw_Zoomed_Curve(df, zoomed_output,
                         epoch_range=(args.epoch_start, args.epoch_end),
                         title=f'{args.title} (Epoch {args.epoch_start}-{args.epoch_end})',
                         figsize=figsize, dpi=args.dpi)
        
        # 生成y轴放大版本(70-80)
        ylim_output = args.output.replace('.png', '_acc_zoomed.png')
        Draw_Training_Curve(df, ylim_output,
                           f'{args.title} (Accuracy 70-80%)',
                           figsize=figsize, dpi=args.dpi, ylim=(70, 80))
    elif args.zoomed:
        Draw_Zoomed_Curve(df, args.output,
                         epoch_range=(args.epoch_start, args.epoch_end),
                         ylim=ylim,
                         title=args.title,
                         figsize=figsize, dpi=args.dpi)
    else:
        Draw_Training_Curve(df, args.output, args.title,
                           figsize=figsize, dpi=args.dpi, ylim=ylim)


if __name__ == '__main__':
    Main()
