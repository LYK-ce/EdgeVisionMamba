#Presented by KeJi
#Date: 2026-01-30

"""
Draw EdgeVim vs Vim comparison chart.
- EdgeVim: line chart
- Vim Tiny (7M) & Vim Small (26M): scatter with different markers
- 500 Random sampled EdgeVim models: scatter with same marker

Usage:
    python draw_edgevim_compare.py
    python draw_edgevim_compare.py -o edgevim_compare.pdf
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 常量
DEFAULT_INPUT = "compare.xlsx"
EDGEVIM_SHEET = "EdgeVim"
RANDOM_SHEET = "Random"
DEFAULT_OUTPUT = "Img/edgevim_compare"

# 学术风格配置 - 字体稍微放大
ACADEMIC_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.titlesize': 18,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
}

# NATURE配色方案
COLORS = {
    'EdgeVim': '#CC247C',       # Magenta
    'Vim Tiny': '#4EA660',      # Green
    'Vim Small': '#5C9AD4',     # Blue
    'Random': '#F7A24F',        # Orange (半透明)
}

# 线型和标记
LINE_STYLES = {
    'EdgeVim': '-',
}

MARKERS = {
    'EdgeVim': 'o',
    'Vim Tiny': '^',
    'Vim Small': 's',
    'Random': '.',
}


def Load_Data(xlsx_path: str) -> tuple:
    """
    加载EdgeVim sheet和Random sheet的数据
    
    Returns:
        tuple: (edgevim_df, vim_data, random_df)
    """
    # 读取EdgeVim sheet
    edgevim_df = pd.read_excel(xlsx_path, sheet_name=EDGEVIM_SHEET)
    
    # 解析列名（处理编码问题）
    cols = edgevim_df.columns.tolist()
    params_col = cols[0]  # 第一列是参数量
    edgevim_col = cols[1]  # 第二列是EdgeVim准确率
    vim_col = cols[2] if len(cols) > 2 else None  # 第三列是Vim准确率
    
    # 提取EdgeVim数据（非NaN行）
    edgevim_data = edgevim_df[[params_col, edgevim_col]].dropna()
    edgevim_data.columns = ['Params', 'Accuracy']
    
    # 提取Vim数据（非NaN行）
    vim_data = {}
    if vim_col:
        vim_rows = edgevim_df[[params_col, vim_col]].dropna()
        for _, row in vim_rows.iterrows():
            params = row.iloc[0]
            acc = row.iloc[1]
            if params == 7:
                vim_data['Vim Tiny'] = (params, acc)
            elif params == 26:
                vim_data['Vim Small'] = (params, acc)
    
    # 读取Random sheet
    try:
        random_df = pd.read_excel(xlsx_path, sheet_name=RANDOM_SHEET)
        # 确保列名正确
        if 'Params' not in random_df.columns:
            random_df.columns = ['Params', 'Accuracy']
        # 将参数量转换为M单位
        random_df['Params_M'] = random_df['Params'] / 1e6
    except Exception as e:
        print(f"Warning: Failed to read Random sheet: {e}")
        random_df = None
    
    return edgevim_data, vim_data, random_df


def Draw_Edgevim_Compare(
    edgevim_data: pd.DataFrame,
    vim_data: dict,
    random_df: pd.DataFrame,
    output_path: str,
    title: str = None,
    figsize: tuple = (10, 7),
    dpi: int = 300
):
    """
    绘制EdgeVim vs Vim对比图
    
    Args:
        edgevim_data: EdgeVim模型数据 (Params, Accuracy)
        vim_data: Vim模型数据 {'Vim Tiny': (params, acc), 'Vim Small': (params, acc)}
        random_df: Random sampled模型数据 (Params, Accuracy, Params_M)
        output_path: 输出路径（不含扩展名）
        title: 图表标题
        figsize: 图表尺寸
        dpi: 输出DPI
    """
    plt.rcParams.update(ACADEMIC_STYLE)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 1. 绘制Random sampled模型（最底层，半透明）
    if random_df is not None and len(random_df) > 0:
        ax.scatter(
            random_df['Params_M'],
            random_df['Accuracy'],
            c=COLORS['Random'],
            marker=MARKERS['Random'],
            s=30,
            alpha=0.5,
            label=f'Random Sampled EdgeVim (n={len(random_df)})',
            zorder=1
        )
    
    # 2. 绘制EdgeVim折线图
    ax.plot(
        edgevim_data['Params'],
        edgevim_data['Accuracy'],
        color=COLORS['EdgeVim'],
        linestyle=LINE_STYLES['EdgeVim'],
        linewidth=2.5,
        marker=MARKERS['EdgeVim'],
        markersize=10,
        markeredgecolor='black',
        markeredgewidth=1,
        label='EdgeVim',
        zorder=3
    )
    
    # 3. 绘制Vim模型散点图
    for name, (params, acc) in vim_data.items():
        ax.scatter(
            params,
            acc,
            c=COLORS[name],
            marker=MARKERS[name],
            s=150,
            edgecolors='black',
            linewidths=1.5,
            label=name,
            zorder=4
        )
    
    # 设置轴标签
    ax.set_xlabel('Parameters (M)', fontweight='bold')
    ax.set_ylabel('Top-1 Accuracy (%)', fontweight='bold')
    
    # 设置标题
    if title:
        ax.set_title(title, fontweight='bold', pad=10)
    
    # 设置网格
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # 设置图例
    ax.legend(
        loc='lower right',
        frameon=True,
        fancybox=False,
        edgecolor='black',
    )
    
    # 设置x轴范围
    ax.set_xlim(0, 30)
    
    # 保存图表
    plt.tight_layout()
    
    # 保存PNG和PDF
    for ext in ['png', 'pdf']:
        save_path = f"{output_path}.{ext}"
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Chart saved to: {save_path}")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Draw EdgeVim vs Vim comparison chart')
    parser.add_argument('-i', '--input', type=str, default=DEFAULT_INPUT,
                        help=f'Input xlsx file path (default: {DEFAULT_INPUT})')
    parser.add_argument('-o', '--output', type=str, default=DEFAULT_OUTPUT,
                        help=f'Output file path without extension (default: {DEFAULT_OUTPUT})')
    parser.add_argument('-t', '--title', type=str, default='Model Accuracy of EdgeVim',
                        help='Chart title (default: Model Accuracy of EdgeVim)')
    parser.add_argument('--figsize', type=str, default='10,7',
                        help='Figure size (width,height)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output DPI')
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    
    # 构建完整路径
    xlsx_path = script_dir / args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_path = script_dir / args.output if not Path(args.output).is_absolute() else Path(args.output)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 解析figsize
    figsize = tuple(map(float, args.figsize.split(',')))
    
    print(f"Loading data from: {xlsx_path}")
    edgevim_data, vim_data, random_df = Load_Data(str(xlsx_path))
    
    print(f"EdgeVim data points: {len(edgevim_data)}")
    print(f"Vim models: {list(vim_data.keys())}")
    if random_df is not None:
        print(f"Random sampled models: {len(random_df)}")
    
    Draw_Edgevim_Compare(
        edgevim_data=edgevim_data,
        vim_data=vim_data,
        random_df=random_df,
        output_path=str(output_path),
        title=args.title,
        figsize=figsize,
        dpi=args.dpi
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
