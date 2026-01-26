#Presented by KeJi
#Date: 2026-01-26

"""
性能分析脚本 - 用于配合VTune/perf进行CPU性能分析

功能：
1. 创建~10M参数量的ResNet、ViT、Vim模型
2. 对于Vim支持16种优化配置选择
3. 运行指定次数的推理，供性能分析工具采样

使用方式：
=========

基本用法：
    python analysis.py --model resnet    # 测试ResNet
    python analysis.py --model vit       # 测试ViT
    python analysis.py --model vim       # 测试Vim (默认SIMD优化)
    
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
模型          IPC预期      L1 Miss预期     Backend Bound预期
ResNet        2.0-3.0      < 5%            < 30%
ViT           2.0-3.5      < 2%            < 30%
Vim (SSM)     0.5-0.8      10-20%          60-70%
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
TARGET_PARAMS = '10m'  # 目标~10M参数

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


def Create_Resnet_10M():
    """创建~10M参数的ResNet"""
    import torch.nn as nn
    
    class BasicBlock(nn.Module):
        expansion = 1
        
        def __init__(self, in_planes, planes, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                    nn.BatchNorm2d(planes)
                )
        
        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return self.relu(out)
    
    class CustomResNet(nn.Module):
        def __init__(self, block, num_blocks, width_mult=0.92, num_classes=1000):
            super().__init__()
            self.in_planes = int(64 * width_mult)
            
            self.conv1 = nn.Conv2d(3, self.in_planes, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, 2, 1)
            
            self.layer1 = self._make_layer(block, int(64*width_mult), num_blocks[0], 1)
            self.layer2 = self._make_layer(block, int(128*width_mult), num_blocks[1], 2)
            self.layer3 = self._make_layer(block, int(256*width_mult), num_blocks[2], 2)
            self.layer4 = self._make_layer(block, int(512*width_mult), num_blocks[3], 2)
            
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(int(512*width_mult), num_classes)
        
        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for s in strides:
                layers.append(block(self.in_planes, planes, s))
                self.in_planes = planes
            return nn.Sequential(*layers)
        
        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = CustomResNet(BasicBlock, [2, 2, 2, 2], width_mult=0.92)
    model.eval()
    return model


def Create_Vit_10M():
    """创建~10M参数的ViT"""
    try:
        import timm
        model = timm.models.vision_transformer.VisionTransformer(
            img_size=IMG_SIZE,
            patch_size=16,
            in_chans=3,
            num_classes=NUM_CLASSES,
            embed_dim=384,
            depth=6,
            num_heads=6,
            mlp_ratio=4.0,
            qkv_bias=True,
        )
        model.eval()
        return model
    except ImportError:
        print("[ERROR] timm not available. Install with: pip install timm")
        sys.exit(1)


def Create_Vim_10M(optim_config):
    """创建~10M参数的Vim"""
    try:
        from timm.models import create_model
        import models_mamba
        
        model = create_model(
            'vim_10m_patch16_224_bimambav2',
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
  python analysis.py --model resnet --runs 50      # 测试ResNet
  python analysis.py --model vit --runs 50         # 测试ViT
  python analysis.py --model vim --optim 12        # 测试Vim SIMD优化
  
配合VTune:
  vtune -collect uarch-exploration -- python analysis.py --model vim --runs 50
  
配合perf:
  perf stat -e cycles,instructions,cache-misses python analysis.py --model vim --runs 50
        """
    )
    
    parser.add_argument('--model', type=str, required=True, 
                        choices=['resnet', 'vit', 'vim'],
                        help='模型类型: resnet, vit, vim')
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
        model = Create_Resnet_10M()
    elif args.model == 'vit':
        model = Create_Vit_10M()
    elif args.model == 'vim':
        if args.optim < 0 or args.optim >= len(VIM_OPTIMIZATION_CONFIGS):
            print(f"[ERROR] Invalid optim index: {args.optim}. Use 0-15.")
            sys.exit(1)
        optim_config = VIM_OPTIMIZATION_CONFIGS[args.optim]
        model = Create_Vim_10M(optim_config)
    
    params = Count_Params(model)
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
