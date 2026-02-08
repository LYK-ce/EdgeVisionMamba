#Presented by KeJi
#Date : 2026-01-29

"""
设备部署基准测试折线图绘制脚本
读取benchmark1.xlsx，生成4张折线图（每台设备一张）
展示不同分辨率下各模型的推理延迟
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    'legend.fontsize': 16,
    'figure.titlesize': 20,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
}

# NATURE配色方案
NATURE_COLORS = [
    '#E95351',  # Red - ResNet50
    '#5C9AD4',  # Blue - ViT
    '#4EA660',  # Green - Vim
]

# 线型和标记样式
LINE_STYLES = ['-', '--', '-.']
MARKERS = ['o', 's', '^']

def Load_Data(filepath):
    """加载benchmark1.xlsx数据"""
    sheets = pd.read_excel(filepath, sheet_name=None)
    data = {}
    
    for sheet_name, df in sheets.items():
        # 第一列为模型名
        df = df.rename(columns={df.columns[0]: 'Model'})
        data[sheet_name] = df
    
    return data

def Draw_Device_Chart(device_name, df, output_path, show_legend=True):
    """绘制单个设备的折线图"""
    plt.rcParams.update(ACADEMIC_STYLE)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 数据准备
    models = df['Model'].tolist()
    # 获取列名（除第一列Model外），作为分辨率标签
    resolutions = df.columns[1:].tolist()
    # 提取分辨率数值用于X轴（如64, 96, 128, 160, 224）
    x_values = []
    x_labels = []
    for res in resolutions:
        # 解析分辨率，如"64*64"提取64
        res_str = str(res)
        if '*' in res_str:
            val = int(res_str.split('*')[0])
        else:
            val = int(res_str)
        x_values.append(val)
        x_labels.append(res_str)
    
    x = np.array(x_values)
    
    # 绘制折线图
    for i, model in enumerate(models):
        values = df.iloc[i, 1:].values.astype(float)
        ax.plot(x, values,
                label=model,
                color=NATURE_COLORS[i],
                linestyle=LINE_STYLES[i],
                marker=MARKERS[i],
                markersize=10,
                linewidth=2.5,
                markeredgecolor='black',
                markeredgewidth=1.0)
        
        # 在数据点旁添加数值标签
        for xi, val in zip(x, values):
            ax.annotate(f'{val:.0f}' if val >= 100 else f'{val:.1f}',
                        xy=(xi, val),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=11)
    
    # 设置坐标轴
    ax.set_xlabel('Input Resolution', fontweight='bold')
    ax.set_ylabel('Inference Latency (ms)', fontweight='bold')
    ax.set_title(f'{device_name}', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    
    # 设置y轴从0开始
    ax.set_ylim(bottom=0)
    
    # 图例
    if show_legend:
        ax.legend(loc='upper left', frameon=True, edgecolor='black')
    
    # 网格线置于底层
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # 保存PNG和PDF
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"保存: {output_path}, {pdf_path}")

def Draw_All_Devices(data, output_dir):
    """生成所有设备的柱状图"""
    device_names = {
        'Laptop1': 'Platform 1',
        'Laptop 2': 'Platform 2',
        'Pi 5': 'Platform 3',
        'RK3588': 'Platform 4'
    }
    
    for sheet_name, df in data.items():
        display_name = device_names.get(sheet_name, sheet_name)
        filename = sheet_name.lower().replace(' ', '_') + '_benchmark.png'
        output_path = os.path.join(output_dir, filename)
        Draw_Device_Chart(display_name, df, output_path)

def main():
    parser = argparse.ArgumentParser(description='绘制设备部署基准测试柱状图')
    parser.add_argument('-i', '--input', type=str, 
                        default='benchmark1.xlsx',
                        help='输入文件路径 (默认: benchmark1.xlsx)')
    parser.add_argument('-o', '--output', type=str,
                        default='Img',
                        help='输出目录 (默认: Img)')
    parser.add_argument('-d', '--device', type=str,
                        help='指定设备 (Laptop1/Laptop 2/Pi 5/RK3588)')
    
    args = parser.parse_args()
    
    # 确定输入文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.input):
        input_path = os.path.join(script_dir, args.input)
    else:
        input_path = args.input
    
    # 确定输出目录
    if not os.path.isabs(args.output):
        output_dir = os.path.join(script_dir, args.output)
    else:
        output_dir = args.output
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print(f"加载数据: {input_path}")
    data = Load_Data(input_path)
    
    if args.device:
        # 绘制指定设备
        if args.device in data:
            device_names = {
                'Laptop1': 'Platform 1',
                'Laptop 2': 'Platform 2',
                'Pi 5': 'Platform 3',
                'RK3588': 'Platform 4'
            }
            display_name = device_names.get(args.device, args.device)
            filename = args.device.lower().replace(' ', '_') + '_benchmark.png'
            output_path = os.path.join(output_dir, filename)
            Draw_Device_Chart(display_name, data[args.device], output_path)
        else:
            print(f"错误: 未找到设备 '{args.device}'")
            print(f"可用设备: {list(data.keys())}")
    else:
        # 绘制所有设备
        Draw_All_Devices(data, output_dir)
    
    print("完成!")

if __name__ == '__main__':
    main()
