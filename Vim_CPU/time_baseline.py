#Presented by KeJi
#Date: 2026-01-06

"""
基准测试脚本：对比不同规模的CNN和ViT推理时间

模型选择：
- ResNet系列: ResNet-18(~11.7M), ResNet-34(~21.8M), ResNet-50(~25.6M)
- ViT系列: ViT-Tiny(~5.7M), ViT-Small(~22M), ViT-Base(~86M)
"""

import torch
import time
import platform


def Get_Model_Params(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def Warmup_Inference(model, input_tensor, num_warmup=10):
    """模型预热"""
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)


def Benchmark_Inference(model, input_tensor, num_runs=100):
    """性能基准测试"""
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return avg_time, min_time, max_time


def Test_Model(name, model_fn, input_tensor, device, results):
    """测试单个模型"""
    print(f"\n测试 {name}...")
    
    try:
        model = model_fn()
        model = model.to(device)
        model.eval()
        
        params = Get_Model_Params(model)
        print(f"  参数量: {params:,} ({params/1e6:.2f}M)")
        
        print(f"  预热中...")
        Warmup_Inference(model, input_tensor)
        
        print(f"  基准测试中...")
        avg, min_t, max_t = Benchmark_Inference(model, input_tensor)
        
        print(f"  平均时间: {avg:.2f} ms")
        
        results[name] = {'params': params, 'avg_time': avg, 'min_time': min_t, 'max_time': max_t}
        
        del model
        
    except Exception as e:
        print(f"  错误: {e}")


def main():
    print("=" * 70)
    print("基准测试：CNN vs ViT 不同规模推理时间对比 (CPU)")
    print("=" * 70)
    print(f"平台: {platform.machine()}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CPU线程数: {torch.get_num_threads()}")
    
    # 强制使用CPU
    device = torch.device('cpu')
    print(f"设备: {device}")
    
    # 输入张量
    input_tensor = torch.randn(1, 3, 224, 224, device=device)
    print(f"输入尺寸: {input_tensor.shape}")
    print(f"预热: 10次, 推理: 100次")
    
    results = {}
    
    # ===== ResNet系列 =====
    print("\n" + "=" * 70)
    print("ResNet系列 (torchvision)")
    print("=" * 70)
    
    from torchvision.models import resnet18, resnet34, resnet50
    
    Test_Model('ResNet-18', lambda: resnet18(weights=None), input_tensor, device, results)
    Test_Model('ResNet-34', lambda: resnet34(weights=None), input_tensor, device, results)
    Test_Model('ResNet-50', lambda: resnet50(weights=None), input_tensor, device, results)
    
    # ===== ViT系列 =====
    print("\n" + "=" * 70)
    print("ViT系列 (timm)")
    print("=" * 70)
    
    from timm import create_model
    
    Test_Model('ViT-Tiny', lambda: create_model('vit_tiny_patch16_224', pretrained=False), input_tensor, device, results)
    Test_Model('ViT-Small', lambda: create_model('vit_small_patch16_224', pretrained=False), input_tensor, device, results)
    Test_Model('ViT-Base', lambda: create_model('vit_base_patch16_224', pretrained=False), input_tensor, device, results)
    
    # ===== 结果汇总 =====
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    print(f"{'模型':<15} {'参数量':<15} {'平均时间':<15} {'最小时间':<15} {'最大时间':<15}")
    print("-" * 70)
    
    for name in ['ResNet-18', 'ResNet-34', 'ResNet-50', 'ViT-Tiny', 'ViT-Small', 'ViT-Base']:
        if name in results:
            data = results[name]
            print(f"{name:<15} {data['params']/1e6:>6.2f}M{'':<7} {data['avg_time']:>8.2f} ms{'':<5} {data['min_time']:>8.2f} ms{'':<5} {data['max_time']:>8.2f} ms")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
