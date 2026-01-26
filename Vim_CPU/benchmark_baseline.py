#Presented by KeJi
#Date: 2026-01-26

"""
ResNet-50和ViT-Small基准测试脚本

功能：
1. 测试ResNet-50 (~25.6M) 和 ViT-Small (~22M) 模型
2. 比较128、160、224三种分辨率下的CPU推理延迟
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

# 尝试导入torchvision
try:
    from torchvision.models import resnet50
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("[WARN] torchvision not available, install with: pip install torchvision")

# ==================== 配置 ====================
NUM_CLASSES = 1000
# 测试分辨率
RESOLUTIONS = [128, 160, 224]


def Count_Params(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def Create_Resnet50():
    """创建ResNet-50模型 (~25.6M参数)"""
    model = resnet50(weights=None, num_classes=NUM_CLASSES)
    model.eval()
    return model


def Create_Vit_Small(img_size=224):
    """创建ViT-Small模型 (~22M参数)"""
    model = timm.create_model(
        'vit_small_patch16_224',
        pretrained=False,
        num_classes=NUM_CLASSES,
        img_size=img_size,
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
    print("ResNet-50 & ViT-Small CPU推理性能基准测试")
    print("=" * 80)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"平台: {platform.machine()}")
    print(f"PyTorch: {torch.__version__} (CPU模式)")
    print(f"CUDA可用: {torch.cuda.is_available()} (已禁用)")
    print(f"CPU线程: {torch.get_num_threads()}")
    
    if not TIMM_AVAILABLE:
        print("[ERROR] timm库不可用，请安装: pip install timm")
        return
    
    if not TORCHVISION_AVAILABLE:
        print("[ERROR] torchvision库不可用，请安装: pip install torchvision")
        return
    
    print(f"timm: {timm.__version__}")
    print()
    
    # 测试配置
    NUM_WARMUP = 3
    NUM_RUNS = 10
    
    print(f"测试分辨率: {RESOLUTIONS}")
    print(f"预热次数: {NUM_WARMUP}, 测试次数: {NUM_RUNS}")
    print()
    
    results = {}
    
    # ==================== ResNet-50测试 ====================
    print("=" * 80)
    print("ResNet-50测试 (~25.6M参数)")
    print("=" * 80)
    
    model_resnet = Create_Resnet50()
    params_resnet = Count_Params(model_resnet)
    print(f"模型参数量: {params_resnet/1e6:.2f}M")
    print()
    
    for resolution in RESOLUTIONS:
        input_tensor = torch.randn(1, 3, resolution, resolution)
        name = f"ResNet-50-{resolution}"
        
        try:
            Warmup_Inference(model_resnet, input_tensor, NUM_WARMUP)
            avg, min_t, max_t = Benchmark_Inference(model_resnet, input_tensor, NUM_RUNS)
            
            results[name] = {
                'params': params_resnet,
                'resolution': resolution,
                'avg': avg,
                'min': min_t,
                'max': max_t
            }
            print(f"Resolution {resolution:>3}x{resolution:<3} | "
                  f"Avg: {avg:>7.2f}ms | Min: {min_t:>7.2f}ms | Max: {max_t:>7.2f}ms")
        except Exception as e:
            print(f"Resolution {resolution:>3}x{resolution:<3} | ERROR: {e}")
    
    del model_resnet
    print()
    
    # ==================== ViT-Small测试 ====================
    print("=" * 80)
    print("ViT-Small测试 (~22M参数)")
    print("=" * 80)
    
    for resolution in RESOLUTIONS:
        input_tensor = torch.randn(1, 3, resolution, resolution)
        name = f"ViT-Small-{resolution}"
        
        try:
            # 为每个分辨率创建新模型（ViT需要指定img_size）
            model_vit = Create_Vit_Small(img_size=resolution)
            params_vit = Count_Params(model_vit)
            
            if resolution == RESOLUTIONS[0]:
                print(f"模型参数量: {params_vit/1e6:.2f}M")
                print()
            
            Warmup_Inference(model_vit, input_tensor, NUM_WARMUP)
            avg, min_t, max_t = Benchmark_Inference(model_vit, input_tensor, NUM_RUNS)
            
            results[name] = {
                'params': params_vit,
                'resolution': resolution,
                'avg': avg,
                'min': min_t,
                'max': max_t
            }
            print(f"Resolution {resolution:>3}x{resolution:<3} | "
                  f"Avg: {avg:>7.2f}ms | Min: {min_t:>7.2f}ms | Max: {max_t:>7.2f}ms")
            
            del model_vit
        except Exception as e:
            print(f"Resolution {resolution:>3}x{resolution:<3} | ERROR: {e}")
    
    # ==================== 汇总 ====================
    print()
    print("=" * 80)
    print("结果汇总")
    print("=" * 80)
    print(f"{'Model':<20} | {'Resolution':>12} | {'Params':>10} | {'Latency(ms)':>12}")
    print("-" * 65)
    for name, data in results.items():
        res_str = f"{data['resolution']}x{data['resolution']}"
        print(f"{name:<20} | {res_str:>12} | {data['params']/1e6:>9.2f}M | {data['avg']:>12.2f}")
    
    # 按模型分组输出延迟对比
    print()
    print("=" * 80)
    print("延迟对比 (按分辨率)")
    print("=" * 80)
    print(f"{'Resolution':<12} | {'ResNet-50(ms)':>15} | {'ViT-Small(ms)':>15} | {'Ratio(ViT/Res)':>15}")
    print("-" * 65)
    for resolution in RESOLUTIONS:
        resnet_key = f"ResNet-50-{resolution}"
        vit_key = f"ViT-Small-{resolution}"
        if resnet_key in results and vit_key in results:
            res_latency = results[resnet_key]['avg']
            vit_latency = results[vit_key]['avg']
            ratio = vit_latency / res_latency
            print(f"{resolution}x{resolution:<6} | {res_latency:>15.2f} | {vit_latency:>15.2f} | {ratio:>15.2f}")
    
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
