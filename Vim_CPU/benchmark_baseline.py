#Presented by KeJi
#Date: 2026-01-26

"""
ResNet和ViT基准测试脚本

功能：
1. 创建10M/15M/20M参数量的ResNet和ViT模型
2. 测试CPU推理延迟
3. 与Vision Mamba进行对比

使用方式：
    python benchmark_baseline.py
"""

import torch
import time
import platform
import sys
from datetime import datetime

# 强制使用CPU，禁用CUDA
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_default_device('cpu')

# 尝试导入timm
try:
    import timm
    from timm.models import create_model
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[WARN] timm not available, install with: pip install timm")

# ==================== 模型配置 ====================
# ResNet配置：通过调整宽度和深度控制参数量
RESNET_CONFIGS = {
    '10m': {'model': 'resnet34', 'expected_params': 21.8},  # resnet34 ~21.8M, 最接近
    '15m': {'model': 'resnet34', 'expected_params': 21.8},  # 没有精确15M的预设
    '20m': {'model': 'resnet34', 'expected_params': 21.8},  # resnet34 ~21.8M
}

# ViT配置：通过调整embed_dim、depth、num_heads控制参数量
VIT_CONFIGS = {
    '10m': {'embed_dim': 384, 'depth': 6, 'num_heads': 6},   # ~10M
    '15m': {'embed_dim': 384, 'depth': 9, 'num_heads': 6},   # ~15M
    '20m': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},  # ~20M
}

IMG_SIZE = 128
NUM_CLASSES = 1000


def Count_Params(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def Create_Resnet(target_params='10m'):
    """
    创建ResNet模型
    
    timm预设模型:
    - resnet18: 11.7M
    - resnet34: 21.8M
    - resnet50: 25.6M
    
    自定义参数量需要构建自定义ResNet
    """
    # 使用timm预设模型
    if target_params == '10m':
        model = create_model('resnet18', pretrained=False, num_classes=NUM_CLASSES)
    elif target_params == '15m':
        # 没有精确15M的预设，使用resnet18+宽度调整
        # 或者使用wide_resnet50_2的缩减版
        model = create_model('resnet18', pretrained=False, num_classes=NUM_CLASSES)
    elif target_params == '20m':
        model = create_model('resnet34', pretrained=False, num_classes=NUM_CLASSES)
    else:
        model = create_model('resnet18', pretrained=False, num_classes=NUM_CLASSES)
    
    model.eval()
    return model


def Create_Custom_Resnet(width_mult=1.0, depth_mult=1.0):
    """
    创建自定义宽度/深度的ResNet
    
    基于ResNet18结构：[2,2,2,2]
    通道数：[64,128,256,512] * width_mult
    """
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
        def __init__(self, block, num_blocks, width_mult=1.0, num_classes=1000):
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
    
    # 根据目标参数量调整width_mult
    # ResNet18 base: 11.7M
    # 10M: width_mult ≈ 0.92
    # 15M: width_mult ≈ 1.13
    # 20M: width_mult ≈ 1.30
    
    model = CustomResNet(BasicBlock, [2, 2, 2, 2], width_mult=width_mult)
    model.eval()
    return model


def Create_Vit(target_params='10m'):
    """
    创建ViT模型
    
    ViT参数量公式：
    Params ≈ embed_dim^2 * (12*depth + 4) + embed_dim * (patch_dim^2 * 3 + num_classes)
    
    调整embed_dim和depth来控制参数量
    """
    config = VIT_CONFIGS.get(target_params, VIT_CONFIGS['10m'])
    
    # 使用timm创建自定义ViT
    model = timm.models.vision_transformer.VisionTransformer(
        img_size=IMG_SIZE,
        patch_size=16,
        in_chans=3,
        num_classes=NUM_CLASSES,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=4.0,
        qkv_bias=True,
    )
    
    model.eval()
    return model


def Warmup_Inference(model, input_tensor, num_warmup=3):
    """模型预热"""
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)


def Benchmark_Inference(model, input_tensor, num_runs=10):
    """性能基准测试"""
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    return avg_time, min_time, max_time


def main():
    """主函数"""
    print("=" * 80)
    print("ResNet & ViT CPU推理性能基准测试")
    print("=" * 80)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"平台: {platform.machine()}")
    print(f"PyTorch: {torch.__version__} (CPU模式)")
    print(f"CUDA可用: {torch.cuda.is_available()} (已禁用)")
    print(f"CPU线程: {torch.get_num_threads()}")
    
    if not TIMM_AVAILABLE:
        print("[ERROR] timm库不可用，请安装: pip install timm")
        return
    
    print(f"timm: {timm.__version__}")
    print()
    
    # 测试配置
    input_tensor = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    NUM_WARMUP = 3
    NUM_RUNS = 10
    
    print(f"输入尺寸: {input_tensor.shape}")
    print(f"预热次数: {NUM_WARMUP}, 测试次数: {NUM_RUNS}")
    print()
    
    results = {}
    
    # ==================== ResNet测试 ====================
    print("=" * 80)
    print("ResNet测试")
    print("=" * 80)
    
    # 自定义宽度系数来精确控制参数量
    resnet_configs = [
        ('ResNet-10M', 0.92),   # ~10M params
        ('ResNet-15M', 1.13),   # ~15M params
        ('ResNet-20M', 1.30),   # ~20M params
    ]
    
    for name, width_mult in resnet_configs:
        try:
            model = Create_Custom_Resnet(width_mult=width_mult)
            params = Count_Params(model)
            
            Warmup_Inference(model, input_tensor, NUM_WARMUP)
            avg, min_t, max_t = Benchmark_Inference(model, input_tensor, NUM_RUNS)
            
            results[name] = {'params': params, 'avg': avg, 'min': min_t, 'max': max_t}
            print(f"{name:<15} | Params: {params/1e6:>6.2f}M | "
                  f"Avg: {avg:>7.2f}ms | Min: {min_t:>7.2f}ms | Max: {max_t:>7.2f}ms")
            
            del model
        except Exception as e:
            print(f"{name:<15} | ERROR: {e}")
    
    print()
    
    # ==================== ViT测试 ====================
    print("=" * 80)
    print("ViT测试")
    print("=" * 80)
    
    # ViT配置：调整embed_dim和depth
    vit_configs = [
        ('ViT-10M', {'embed_dim': 384, 'depth': 6, 'num_heads': 6}),    # ~10M
        ('ViT-15M', {'embed_dim': 384, 'depth': 9, 'num_heads': 6}),    # ~15M
        ('ViT-20M', {'embed_dim': 384, 'depth': 12, 'num_heads': 6}),   # ~20M
    ]
    
    for name, config in vit_configs:
        try:
            model = timm.models.vision_transformer.VisionTransformer(
                img_size=IMG_SIZE,
                patch_size=16,
                in_chans=3,
                num_classes=NUM_CLASSES,
                embed_dim=config['embed_dim'],
                depth=config['depth'],
                num_heads=config['num_heads'],
                mlp_ratio=4.0,
                qkv_bias=True,
            )
            model.eval()
            params = Count_Params(model)
            
            Warmup_Inference(model, input_tensor, NUM_WARMUP)
            avg, min_t, max_t = Benchmark_Inference(model, input_tensor, NUM_RUNS)
            
            results[name] = {'params': params, 'avg': avg, 'min': min_t, 'max': max_t}
            print(f"{name:<15} | Params: {params/1e6:>6.2f}M | "
                  f"Avg: {avg:>7.2f}ms | Min: {min_t:>7.2f}ms | Max: {max_t:>7.2f}ms")
            
            del model
        except Exception as e:
            print(f"{name:<15} | ERROR: {e}")
    
    # ==================== 汇总 ====================
    print()
    print("=" * 80)
    print("结果汇总")
    print("=" * 80)
    print(f"{'Model':<15} | {'Params':>10} | {'Latency(ms)':>12}")
    print("-" * 45)
    for name, data in results.items():
        print(f"{name:<15} | {data['params']/1e6:>9.2f}M | {data['avg']:>12.2f}")
    
    print()
    print("=" * 80)
    print("测试完成")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
