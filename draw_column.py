#Presented by KeJi
#Date: 2026-01-19

"""
Academic Column Chart Generator for VisionMamba CPU Optimization Results

Usage:
    python draw_column.py --model vim_5m --configs "Python Original,SIMD" --output vim_5m_compare.png
    python draw_column.py --model vim_10m --all --output vim_10m_all.png
    python draw_column.py --model vim_5m --speedup --baseline "Python Original" -o speedup.png
    python draw_column.py --list-models
    python draw_column.py --list-configs
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path


# Academic style configuration
ACADEMIC_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.titlesize': 14,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
}

# Color palette for different optimization configs (colorblind-friendly)
COLOR_PALETTE = {
    'Python Original': '#1f77b4',        # Blue
    'Loop Optimization': '#ff7f0e',      # Orange
    'Bidirectional Fusion': '#2ca02c',   # Green
    'Loop + Fusion': '#d62728',          # Red
    'SIMD': '#9467bd',                   # Purple
    'SIMD + Loop': '#8c564b',            # Brown
    'SIMD + Fusion': '#e377c2',          # Pink
    'SIMD + Loop + Fusion': '#17becf',   # Cyan
}

# Hatch patterns for black-white printing
HATCH_PATTERNS = {
    'Python Original': '',
    'Loop Optimization': '/',
    'Bidirectional Fusion': '\\',
    'Loop + Fusion': 'x',
    'SIMD': '',
    'SIMD + Loop': '/',
    'SIMD + Fusion': '\\',
    'SIMD + Loop + Fusion': 'x',
}

# All available optimization configurations
ALL_CONFIGS = [
    'Python Original',
    'Loop Optimization',
    'Bidirectional Fusion',
    'Loop + Fusion',
    'SIMD',
    'SIMD + Loop',
    'SIMD + Fusion',
    'SIMD + Loop + Fusion'
]

# Device name mapping for display
DEVICE_NAMES = {
    'Intel(R) Core(TM) Ultra 9 285H 16GB': 'Intel Ultra 9 285H',
    'Intel(R) Core? 5 220H 8GB': 'Intel Core 5 220H',
    'ARM64_Raspberry_Pi_5': 'Raspberry Pi 5',
    'RK3588 8G': 'RK3588'
}


def Parse_Xlsx(file_path: str) -> dict:
    """
    Parse the result_final.xlsx file and return a dictionary of models and their data.
    
    Returns:
        dict: {model_name: {config_name: {device_name: value}}}
    """
    df = pd.read_excel(file_path, header=None)
    
    models_data = {}
    current_model = None
    devices = []
    
    # Known config names for detection
    known_configs = {'Python Original', 'Loop Optimization', 'Bidirectional Fusion',
                     'Loop + Fusion', 'SIMD', 'SIMD + Loop', 'SIMD + Fusion', 'SIMD + Loop + Fusion'}
    
    for idx, row in df.iterrows():
        first_cell = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ''
        
        # Skip empty rows and section headers
        if not first_cell or first_cell == 'nan' or '===' in first_cell:
            continue
        
        # Model name line - detect by ": vim_" pattern
        if ': vim_' in first_cell:
            # Extract model name: "xxx: vim_5m (5.46M)" -> "vim_5m"
            parts = first_cell.split(':')
            if len(parts) > 1:
                model_info = parts[1].strip()
                model_name = model_info.split('(')[0].strip()
                current_model = model_name
                models_data[current_model] = {}
            continue
        
        # Device header line - detect by checking if columns 1-4 contain device names
        second_cell = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ''
        if 'Intel' in second_cell or 'ARM' in second_cell:
            devices = [str(row.iloc[i]).strip() for i in range(1, 5) if pd.notna(row.iloc[i])]
            continue
        
        # Data line - detect by known config names
        if current_model and devices and first_cell in known_configs:
            config_name = first_cell
            models_data[current_model][config_name] = {}
            for i, device in enumerate(devices):
                val = row.iloc[i + 1]
                if pd.notna(val):
                    models_data[current_model][config_name][device] = float(val)
    
    return models_data


def Parse_Csv(file_path: str) -> dict:
    """
    Parse the result_parsed.csv file and return a dictionary of models and their data.
    
    Returns:
        dict: {model_name: {config_name: {device_name: value}}}
    """
    models_data = {}
    current_model = None
    devices = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith('==='):
            continue
        
        if line.startswith('模型:'):
            model_info = line.replace('模型:', '').strip()
            model_name = model_info.split('(')[0].strip()
            current_model = model_name
            models_data[current_model] = {}
            continue
        
        if line.startswith('配置,'):
            devices = line.split(',')[1:]
            continue
        
        if current_model and ',' in line and not line.startswith('配置'):
            parts = line.split(',')
            config_name = parts[0]
            values = [float(v) for v in parts[1:]]
            
            models_data[current_model][config_name] = {}
            for i, device in enumerate(devices):
                models_data[current_model][config_name][device] = values[i]
    
    return models_data


def Draw_Column_Chart(
    data: dict,
    model_name: str,
    configs: list,
    output_path: str,
    title: str = None,
    figsize: tuple = (12, 6),
    dpi: int = 300,
    show_values: bool = False,
    log_scale: bool = False,
    ylabel: str = "Inference Time (ms)"
):
    """
    Draw an academic-style column chart.
    
    Args:
        data: Parsed data dictionary
        model_name: Model to visualize
        configs: List of optimization configurations to include
        output_path: Output file path
        title: Chart title (optional)
        figsize: Figure size
        dpi: Output DPI
        show_values: Whether to show values on bars
        log_scale: Whether to use logarithmic scale
        ylabel: Y-axis label
    """
    plt.rcParams.update(ACADEMIC_STYLE)
    
    if model_name not in data:
        print(f"Error: Model '{model_name}' not found.")
        print(f"Available models: {list(data.keys())}")
        return False
    
    model_data = data[model_name]
    
    # Filter configs
    valid_configs = [c for c in configs if c in model_data]
    if not valid_configs:
        print(f"Error: No valid configurations found.")
        print(f"Available configs: {list(model_data.keys())}")
        return False
    
    # Get devices
    devices = list(model_data[valid_configs[0]].keys())
    device_labels = [DEVICE_NAMES.get(d, d) for d in devices]
    
    n_devices = len(devices)
    n_configs = len(valid_configs)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bar_width = 0.8 / n_configs
    x = np.arange(n_devices)
    
    for i, config in enumerate(valid_configs):
        values = [model_data[config][device] for device in devices]
        offset = (i - n_configs / 2 + 0.5) * bar_width
        
        bars = ax.bar(
            x + offset,
            values,
            bar_width,
            label=config,
            color=COLOR_PALETTE.get(config, '#333333'),
            hatch=HATCH_PATTERNS.get(config, ''),
            edgecolor='black',
            linewidth=0.5
        )
        
        if show_values:
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(
                    f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=6,
                    rotation=90
                )
    
    ax.set_xlabel('Device', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    
    if title:
        ax.set_title(title, fontweight='bold', pad=10)
    else:
        ax.set_title(f'Inference Time Comparison - {model_name}', fontweight='bold', pad=10)
    
    ax.set_xticks(x)
    ax.set_xticklabels(device_labels, rotation=15, ha='right')
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=True,
        fancybox=False,
        edgecolor='black'
    )
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Chart saved to: {output_path}")
    return True


def Draw_Multi_Device_Comparison(
    data: dict,
    model_name: str,
    configs: list,
    output_path: str,
    figsize: tuple = (14, 8),
    dpi: int = 300
):
    """
    Draw a 2x2 subplot comparison for each device.
    """
    plt.rcParams.update(ACADEMIC_STYLE)
    
    if model_name not in data:
        print(f"Error: Model '{model_name}' not found.")
        return False
    
    model_data = data[model_name]
    valid_configs = [c for c in configs if c in model_data]
    devices = list(model_data[valid_configs[0]].keys())
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for idx, device in enumerate(devices):
        ax = axes[idx]
        values = [model_data[config][device] for config in valid_configs]
        colors = [COLOR_PALETTE.get(c, '#333333') for c in valid_configs]
        hatches = [HATCH_PATTERNS.get(c, '') for c in valid_configs]
        
        bars = ax.bar(
            range(len(valid_configs)),
            values,
            color=colors,
            edgecolor='black',
            linewidth=0.5
        )
        
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        
        ax.set_title(DEVICE_NAMES.get(device, device), fontweight='bold')
        ax.set_ylabel('Time (ms)')
        ax.set_xticks(range(len(valid_configs)))
        ax.set_xticklabels(valid_configs, rotation=45, ha='right', fontsize=7)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
    
    fig.suptitle(f'Inference Time by Device - {model_name}', fontweight='bold', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Multi-device chart saved to: {output_path}")
    return True


def Draw_Speedup_Chart(
    data: dict,
    model_name: str,
    baseline_config: str,
    configs: list,
    output_path: str,
    figsize: tuple = (12, 6),
    dpi: int = 300
):
    """
    Draw a speedup comparison chart relative to baseline.
    """
    plt.rcParams.update(ACADEMIC_STYLE)
    
    if model_name not in data:
        print(f"Error: Model '{model_name}' not found.")
        return False
    
    model_data = data[model_name]
    
    if baseline_config not in model_data:
        print(f"Error: Baseline config '{baseline_config}' not found.")
        return False
    
    valid_configs = [c for c in configs if c in model_data and c != baseline_config]
    devices = list(model_data[baseline_config].keys())
    device_labels = [DEVICE_NAMES.get(d, d) for d in devices]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    n_devices = len(devices)
    n_configs = len(valid_configs)
    bar_width = 0.8 / n_configs
    x = np.arange(n_devices)
    
    for i, config in enumerate(valid_configs):
        speedups = []
        for device in devices:
            baseline_val = model_data[baseline_config][device]
            current_val = model_data[config][device]
            speedup = baseline_val / current_val
            speedups.append(speedup)
        
        offset = (i - n_configs / 2 + 0.5) * bar_width
        
        ax.bar(
            x + offset,
            speedups,
            bar_width,
            label=config,
            color=COLOR_PALETTE.get(config, '#333333'),
            hatch=HATCH_PATTERNS.get(config, ''),
            edgecolor='black',
            linewidth=0.5
        )
    
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Baseline (1x)')
    
    ax.set_xlabel('Device', fontweight='bold')
    ax.set_ylabel('Speedup (x)', fontweight='bold')
    ax.set_title(f'Speedup vs {baseline_config} - {model_name}', fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(device_labels, rotation=15, ha='right')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=True,
        fancybox=False,
        edgecolor='black'
    )
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Speedup chart saved to: {output_path}")
    return True


def Main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Academic Column Chart Generator for VisionMamba Optimization Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python draw_column.py --model vim_5m --configs "Python Original,SIMD" -o compare.png
  python draw_column.py --model vim_10m --all -o all_configs.png
  python draw_column.py --model vim_5m --speedup --baseline "Python Original" -o speedup.png
  python draw_column.py --model vim_5m --multi-device --all -o multi.png
  python draw_column.py --list-models
  python draw_column.py --list-configs
        """
    )
    
    parser.add_argument('--model', '-m', type=str, help='Model name to visualize')
    parser.add_argument('--configs', '-c', type=str, help='Comma-separated list of configs')
    parser.add_argument('--all', '-a', action='store_true', help='Use all configurations')
    parser.add_argument('--output', '-o', type=str, default='output.png', help='Output file path')
    parser.add_argument('--title', '-t', type=str, help='Chart title')
    parser.add_argument('--figsize', type=str, default='12,6', help='Figure size (width,height)')
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI')
    parser.add_argument('--show-values', action='store_true', help='Show values on bars')
    parser.add_argument('--log-scale', action='store_true', help='Use logarithmic y-axis')
    parser.add_argument('--speedup', action='store_true', help='Draw speedup chart')
    parser.add_argument('--baseline', type=str, default='Python Original', help='Baseline config for speedup')
    parser.add_argument('--multi-device', action='store_true', help='Draw multi-device subplot')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    parser.add_argument('--list-configs', action='store_true', help='List available configurations')
    parser.add_argument('--data-file', type=str, default='result_final.xlsx', help='Input data file (xlsx or csv)')
    
    args = parser.parse_args()
    
    # Determine data file path
    script_dir = Path(__file__).parent
    data_file = script_dir / args.data_file
    
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        return 1
    
    # Parse data based on file extension
    if str(data_file).endswith('.xlsx'):
        data = Parse_Xlsx(str(data_file))
    else:
        data = Parse_Csv(str(data_file))
    
    # List models
    if args.list_models:
        print("Available models:")
        for model in data.keys():
            print(f"  - {model}")
        return 0
    
    # List configs
    if args.list_configs:
        print("Available configurations:")
        for config in ALL_CONFIGS:
            print(f"  - {config}")
        return 0
    
    # Validate model selection
    if not args.model:
        print("Error: Please specify a model with --model")
        parser.print_help()
        return 1
    
    # Determine configs
    if args.all:
        configs = ALL_CONFIGS
    elif args.configs:
        configs = [c.strip() for c in args.configs.split(',')]
    else:
        print("Error: Please specify configs with --configs or use --all")
        return 1
    
    # Parse figsize
    figsize = tuple(map(float, args.figsize.split(',')))
    
    # Generate chart
    if args.speedup:
        success = Draw_Speedup_Chart(
            data=data,
            model_name=args.model,
            baseline_config=args.baseline,
            configs=configs,
            output_path=args.output,
            figsize=figsize,
            dpi=args.dpi
        )
    elif args.multi_device:
        success = Draw_Multi_Device_Comparison(
            data=data,
            model_name=args.model,
            configs=configs,
            output_path=args.output,
            figsize=figsize,
            dpi=args.dpi
        )
    else:
        success = Draw_Column_Chart(
            data=data,
            model_name=args.model,
            configs=configs,
            output_path=args.output,
            title=args.title,
            figsize=figsize,
            dpi=args.dpi,
            show_values=args.show_values,
            log_scale=args.log_scale
        )
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(Main())
