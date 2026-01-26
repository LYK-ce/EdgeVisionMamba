#Presented by KeJi
#Date: 2026-01-26

"""
性能分析脚本 - 用于配合VTune/perf进行CPU性能分析

功能：
1. 创建标准模型：Vim Tiny (~7M), ResNet-50 (~25.6M), ViT-Small (~22M)
2. 对于Vim支持16种优化配置选择
3. 运行指定次数的推理，供性能分析工具采样

使用方式：
=========

基本用法：
    python analysis.py --model resnet    # 测试ResNet-50
    python analysis.py --model vit       # 测试ViT-Small
    python analysis.py --model vim       # 测试Vim Tiny (默认SIMD优化)
    
Vim优化选项 (--optim参数)：
    python analysis.py --model vim --optim 0   # Python-Original
    python analysis.py --model vim --optim 1   # Python-Fixlen
    python analysis.py --model vim --optim 2   # Python-Fused
    python analysis.py --model vim --optim 3   # Python-Fused-Fixlen
    python analysis.py --model vim --optim 4   # CPP-Original
    python analysis.py --model vim --optim 5   # CPP-Fixlen
    python analysis.py --model vim --optim 6   # CPP-Fused
    python analysis.py --model vim --optim 7   # CPP-Fused-Fixlen
    python analysis.py --model vim --optim 8   # FullCPP-Original
    python analysis.py --model vim --optim 9   # FullCPP-Fixlen
    python analysis.py --model vim --optim 10  # FullCPP-Fused
    python analysis.py --model vim --optim 11  # FullCPP-Fused-Fixlen
    python analysis.py --model vim --optim 12  # SIMD (默认)
    python analysis.py --model vim --optim 13  # SIMD-Fixlen
    python analysis.py --model vim --optim 14  # SIMD-Fused
    python analysis.py --model vim --optim 15  # SIMD-Fused-Fixlen

调整运行次数：
    python analysis.py --model vim --runs 100  # 运行100次推理

配合VTune使用 (Windows)：
========================
# 收集微架构分析
vtune -collect uarch-exploration -- python analysis.py --model vim --runs 50

# 收集热点分析
vtune -collect hotspots -- python analysis.py --model resnet --runs 50

# 查看报告
vtune -report summary -result-dir r000ue

配合perf使用 (Linux)：
=====================
# 基本性能计数器
perf stat -e cycles,instructions,cache-references,cache-misses,\\
L1-dcache-loads,L1-dcache-load-misses \\
python analysis.py --model vim --runs 50

# 详细采样分析
perf record -g python analysis.py --model vim --runs 50
perf report

# 获取IPC和Cache Miss数据
perf stat -e cycles,instructions,L1-dcache-load-misses,LLC-load-misses \\
python analysis.py --model vit --runs 50

预期对比结果：
=============
模型              参数量       IPC预期      L1 Miss预期     Backend Bound预期
Vim Tiny          ~7M          0.5-0.8      10-20%          60-70%
ViT-Small         ~22M         2.0-3.5      < 2%            < 30%
ResNet-50         ~25.6M       2.0-3.0      < 5%            < 30%
"""

import os
import sys
import argparse
import torch
import time
import platform

# 禁用CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['SELECTIVE_SCAN_FORCE_FALLBACK'] = 'TRUE'
os.environ['CAUSAL_CONV1D_FORCE_FALLBACK'] = 'TRUE'

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

parent_dir = os.path.dirname(current_dir)
mamba_dir = os.path.join(parent_dir, 'mamba-1p1p1')
if os.path.exists(mamba_dir):
    sys.path.insert(0, mamba_dir)

modules_dir = os.path.join(mamba_dir, 'mamba_ssm', 'modules')
if os.path.exists(modules_dir):
    sys.path.insert(0, modules_dir)

# ==================== 配置 ====================
IMG_SIZE = 224
NUM_CLASSES = 1000

# Vim 16种优化配置
VIM_OPTIMIZATION_CONFIGS = [
    # Python实现（4种）
    {'name': 'Python-Original', 'use_cpp_scan': False, 'use_fixlen_scan': False, 'use_fused_bidirectional': False},
    {'name': 'Python-Fixlen', 'use_cpp_scan': False, 'use_fixlen_scan': True, 'use_fused_bidirectional': False},
    {'name': 'Python-Fused', 'use_cpp_scan': False, 'use_fixlen_scan': False, 'use_fused_bidirectional': True},
    {'name': 'Python-Fused-Fixlen', 'use_cpp_scan': False, 'use_fixlen_scan': True, 'use_fused_bidirectional': True},
    # C++实现（4种）
    {'name': 'CPP-Original', 'use_cpp_scan': True, 'use_fixlen_scan': False, 'use_fused_bidirectional': False},
    {'name': 'CPP-Fixlen', 'use_cpp_scan': True, 'use_fixlen_scan': True, 'use_fused_bidirectional': False},
    {'name': 'CPP-Fused', 'use_cpp_scan': True, 'use_fixlen_scan': False, 'use_fused_bidirectional': True},
    {'name': 'CPP-Fused-Fixlen', 'use_cpp_scan': True, 'use_fixlen_scan': True, 'use_fused_bidirectional': True},
    # 全C++实现（4种）
    {'name': 'FullCPP-Original', 'use_cpp_scan': True, 'use_fixlen_scan': False, 'use_fused_bidirectional': False, 'use_full_cpp': True},
    {'name': 'FullCPP-Fixlen', 'use_cpp_scan': True, 'use_fixlen_scan': True, 'use_fused_bidirectional': False, 'use_full_cpp': True},
    {'name': 'FullCPP-Fused', 'use_cpp_scan': True, 'use_fixlen_scan': False, 'use_fused_bidirectional': True, 'use_full_cpp': True},
    {'name': 'FullCPP-Fused-Fixlen', 'use_cpp_scan': True, 'use_fixlen_scan': True, 'use_fused_bidirectional': True, 'use_full_cpp': True},
    # SIMD实现（4种）
    {'name': 'SIMD', 'use_simd_scan': True},
    {'name': 'SIMD-Fixlen', 'use_simd_fixlen_scan': True},
    {'name': 'SIMD-Fused', 'use_fused_bidirectional': True, 'use_simd_fused_scan': True},
    {'name': 'SIMD-Fused-Fixlen', 'use_fused_bidirectional': True, 'use_simd_fused_fixlen_scan': True},
]


def Count_Params(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def Create_Resnet50():
    """创建ResNet-50模型 (~25.6M参数)"""
    try:
        from torchvision.models import resnet50
        model = resnet50(weights=None, num_classes=NUM_CLASSES)
        model.eval()
        return model
    except ImportError:
        print("[ERROR] torchvision not available. Install with: pip install torchvision")
        sys.exit(1)


def Create_Vit_Small():
    """创建ViT-Small模型 (~22M参数)"""
    try:
        import timm
        model = timm.create_model(
            'vit_small_patch16_224',
            pretrained=False,
            num_classes=NUM_CLASSES,
        )
        model.eval()
        return model
    except ImportError:
        print("[ERROR] timm not available. Install with: pip install timm")
        sys.exit(1)


def Create_Vim_Tiny(optim_config):
    """创建Vim Tiny模型 (~7M参数)"""
    try:
        from timm.models import create_model
        import models_mamba
        
        model = create_model(
            'vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2',
            pretrained=False,
            num_classes=NUM_CLASSES,
            img_size=IMG_SIZE,
            drop_rate=0.0,
            drop_path_rate=0.1,
            drop_block_rate=None,
            use_cpp_scan=optim_config.get('use_cpp_scan', False),
            use_fixlen_scan=optim_config.get('use_fixlen_scan', False),
            use_fused_bidirectional=optim_config.get('use_fused_bidirectional', False),
            use_full_cpp=optim_config.get('use_full_cpp', False),
            use_simd_scan=optim_config.get('use_simd_scan', False),
            use_simd_fixlen_scan=optim_config.get('use_simd_fixlen_scan', False),
            use_simd_fused_scan=optim_config.get('use_simd_fused_scan', False),
            use_simd_fused_fixlen_scan=optim_config.get('use_simd_fused_fixlen_scan', False),
        )
        model.eval()
        return model
    except Exception as e:
        print(f"[ERROR] Failed to create Vim model: {e}")
        sys.exit(1)


def Run_Inference(model, input_tensor, num_runs, warmup=3):
    """运行推理（供性能分析工具采样）"""
    print(f"[INFO] Warmup: {warmup} runs...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    print(f"[INFO] Running {num_runs} inference iterations...")
    print("[INFO] ===== PROFILING REGION START =====")
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for i in range(num_runs):
            _ = model(input_tensor)
    end_time = time.perf_counter()
    
    print("[INFO] ===== PROFILING REGION END =====")
    
    total_ms = (end_time - start_time) * 1000
    avg_ms = total_ms / num_runs
    print(f"[RESULT] Total: {total_ms:.2f}ms, Avg: {avg_ms:.2f}ms/iter")
    
    return avg_ms


def main():
    parser = argparse.ArgumentParser(
        description='CPU性能分析脚本 - 配合VTune/perf使用',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python analysis.py --model resnet --runs 50      # 测试ResNet-50
  python analysis.py --model vit --runs 50         # 测试ViT-Small
  python analysis.py --model vim --optim 12        # 测试Vim Tiny SIMD优化
  
配合VTune:
  vtune -collect uarch-exploration -- python analysis.py --model vim --runs 50
  
配合perf:
  perf stat -e cycles,instructions,cache-misses python analysis.py --model vim --runs 50
        """
    )
    
    parser.add_argument('--model', type=str, required=True, 
                        choices=['resnet', 'vit', 'vim'],
                        help='模型类型: resnet (ResNet-50), vit (ViT-Small), vim (Vim Tiny)')
    parser.add_argument('--optim', type=int, default=12,
                        help='Vim优化配置索引 (0-15)，默认12=SIMD')
    parser.add_argument('--runs', type=int, default=30,
                        help='推理运行次数，默认30')
    parser.add_argument('--warmup', type=int, default=3,
                        help='预热次数，默认3')
    parser.add_argument('--list-optim', action='store_true',
                        help='列出所有Vim优化配置')
    
    args = parser.parse_args()
    
    # 列出优化配置
    if args.list_optim:
        print("Vim优化配置列表:")
        print("-" * 40)
        for i, cfg in enumerate(VIM_OPTIMIZATION_CONFIGS):
            print(f"  {i:>2}: {cfg['name']}")
        return
    
    # 打印环境信息
    print("=" * 60)
    print("CPU性能分析脚本")
    print("=" * 60)
    print(f"Platform: {platform.machine()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CPU Threads: {torch.get_num_threads()}")
    print(f"Model: {args.model}")
    if args.model == 'vim':
        optim_cfg = VIM_OPTIMIZATION_CONFIGS[args.optim]
        print(f"Vim Optimization: [{args.optim}] {optim_cfg['name']}")
    print(f"Runs: {args.runs}")
    print()
    
    # 创建输入
    input_tensor = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    print(f"Input shape: {input_tensor.shape}")
    
    # 创建模型
    print("[INFO] Creating model...")
    if args.model == 'resnet':
        model = Create_Resnet50()
        model_name = "ResNet-50"
    elif args.model == 'vit':
        model = Create_Vit_Small()
        model_name = "ViT-Small"
    elif args.model == 'vim':
        if args.optim < 0 or args.optim >= len(VIM_OPTIMIZATION_CONFIGS):
            print(f"[ERROR] Invalid optim index: {args.optim}. Use 0-15.")
            sys.exit(1)
        optim_config = VIM_OPTIMIZATION_CONFIGS[args.optim]
        model = Create_Vim_Tiny(optim_config)
        model_name = "Vim Tiny"
    
    params = Count_Params(model)
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Model parameters: {params:,} ({params/1e6:.2f}M)")
    print()
    
    # 运行推理
    avg_ms = Run_Inference(model, input_tensor, args.runs, args.warmup)
    
    print()
    print("=" * 60)
    print("分析完成。使用VTune或perf查看详细性能数据。")
    print("=" * 60)


if __name__ == '__main__':
    main()
