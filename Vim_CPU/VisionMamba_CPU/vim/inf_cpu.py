#Presented by KeJi
#Date: 2026-01-07

"""
Vision Mamba CPU推理性能测试脚本

功能：
1. 测试9种模型：5种按参数量（5M/7M/10M/15M/20M） + 4种按FLOPs（2G/3G/4G/5G）
2. 每种模型测试16种优化配置（Python/C++/FullCPP/SIMD各4种）
3. 预热3轮，测试10轮

使用方式：
    python inf_cpu.py
"""

import os
import sys
import torch
import time
import logging
import platform
from datetime import datetime

os.environ['SELECTIVE_SCAN_FORCE_FALLBACK'] = 'TRUE'
os.environ['CAUSAL_CONV1D_FORCE_FALLBACK'] = 'TRUE'

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

mamba_dir = os.path.join(os.path.dirname(current_dir), 'mamba-1p1p1')
if os.path.exists(mamba_dir):
    sys.path.insert(0, mamba_dir)

modules_dir = os.path.join(mamba_dir, 'mamba_ssm', 'modules')
if os.path.exists(modules_dir):
    sys.path.insert(0, modules_dir)

from timm.models import create_model
import models_mamba

Img_size = 224

MODEL_CONFIGS = {
    # 按参数量分类
    'vim_5m': 'vim_5m_patch16_224_bimambav2',
    'vim_tiny': 'vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2',
    'vim_10m': 'vim_10m_patch16_224_bimambav2',
    'vim_15m': 'vim_15m_patch16_224_bimambav2',
    'vim_20m': 'vim_20m_patch16_224_bimambav2',
    # 按FLOPs分类
    'vim_2gflops': 'vim_2gflops_patch16_224_bimambav2',
    'vim_3gflops': 'vim_3gflops_patch16_224_bimambav2',
    'vim_4gflops': 'vim_4gflops_patch16_224_bimambav2',
    'vim_5gflops': 'vim_5gflops_patch16_224_bimambav2',
}

# 16种优化配置
OPTIMIZATION_CONFIGS = [
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


def Create_Vim_Model(model_name='vim_tiny', use_cpp_scan=False, use_fixlen_scan=False,
                     use_fused_bidirectional=False, use_full_cpp=False,
                     use_simd_scan=False, use_simd_fixlen_scan=False,
                     use_simd_fused_scan=False, use_simd_fused_fixlen_scan=False):
    """创建Vim模型"""
    timm_name = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['vim_tiny'])
    
    model = create_model(
        timm_name,
        pretrained=False,
        num_classes=1000,
        img_size = Img_size,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        use_cpp_scan=use_cpp_scan,
        use_fixlen_scan=use_fixlen_scan,
        use_fused_bidirectional=use_fused_bidirectional,
        use_full_cpp=use_full_cpp,
        use_simd_scan=use_simd_scan,
        use_simd_fixlen_scan=use_simd_fixlen_scan,
        use_simd_fused_scan=use_simd_fused_scan,
        use_simd_fused_fixlen_scan=use_simd_fused_fixlen_scan,
    )
    
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    
    return model, total_params


def Warmup_Inference(model, input_tensor, num_warmup=3):
    """模型预热"""
    with torch.no_grad():
        for i in range(num_warmup):
            _ = model(input_tensor)


def Benchmark_Inference(model, input_tensor, num_runs=10):
    """性能基准测试"""
    times = []
    with torch.no_grad():
        for i in range(num_runs):
            start = time.perf_counter()
            output = model(input_tensor)
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
    return sum(times) / len(times), min(times), max(times), output


def Setup_Logging():
    """设置日志记录"""
    log_file = 'test.log'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def Log_Print(msg, logger=None):
    if logger:
        logger.info(msg)
    else:
        print(msg)


def main():
    """主函数"""
    logger = Setup_Logging()
    start_time = datetime.now()


    
    Log_Print("=" * 100, logger)
    Log_Print("Vision Mamba CPU推理性能测试 - 9种模型(5种参数量+4种FLOPs) x 16种优化配置", logger)
    Log_Print("=" * 100, logger)
    Log_Print(f"测试时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", logger)
    Log_Print(f"平台: {platform.machine()}", logger)
    Log_Print(f"PyTorch: {torch.__version__}, Python: {sys.version.split()[0]}, CPU线程: {torch.get_num_threads()}", logger)
    
    # 检查C++扩展
    try:
        import selective_scan_cpp
        cpp_available = True
        Log_Print("C++扩展: ✓", logger)
    except ImportError:
        cpp_available = False
        Log_Print("C++扩展: ✗", logger)
    
    try:
        import vision_mamba_cpp
            # 显示OpenMP信息
        print(selective_scan_cpp.get_openmp_info())
        # 输出: [OpenMP] 已启用: 可用处理器=8, 最大线程数=8

        # 显示SIMD信息
        print(selective_scan_cpp.get_simd_info())
        # 输出: [SIMD] AVX/AVX2 (256位, 8x float)
        full_cpp_available = True
        Log_Print("全C++扩展: ✓", logger)
    except ImportError:
        full_cpp_available = False
        Log_Print("全C++扩展: ✗", logger)
    
    input_tensor = torch.randn(1, 3, Img_size, Img_size)
    NUM_WARMUP = 3
    NUM_RUNS = 10
    Log_Print(f"\n输入: {input_tensor.shape}, 预热: {NUM_WARMUP}次, 测试: {NUM_RUNS}次", logger)
    
    # 按参数量分类(5种) + 按FLOPs分类(4种) = 9种模型
    models_to_test = [
        # 按参数量
        'vim_5m', 'vim_tiny', 'vim_10m', 'vim_15m', 'vim_20m',
        # 按FLOPs
        # 'vim_2gflops', 'vim_3gflops', 'vim_4gflops', 'vim_5gflops',
    ]
    all_results = {}
    
    for model_name in models_to_test:
        Log_Print("\n" + "=" * 100, logger)
        Log_Print(f"模型: {model_name}", logger)
        Log_Print("=" * 100, logger)
        
        # 创建基础模型保存参数
        base_model, total_params = Create_Vim_Model(model_name)
        base_state_dict = base_model.state_dict()
        Log_Print(f"参数量: {total_params:,} ({total_params/1e6:.2f}M)", logger)
        del base_model
        
        model_results = {}
        
        for config in OPTIMIZATION_CONFIGS:
            cfg_name = config['name']
            
            # 检查依赖
            use_cpp = config.get('use_cpp_scan', False)
            use_full_cpp = config.get('use_full_cpp', False)
            use_simd = config.get('use_simd_scan', False) or config.get('use_simd_fixlen_scan', False) or \
                       config.get('use_simd_fused_scan', False) or config.get('use_simd_fused_fixlen_scan', False)
            
            if use_cpp and not cpp_available:
                Log_Print(f"  {cfg_name:<22} 跳过(C++不可用)", logger)
                continue
            if use_full_cpp and not full_cpp_available:
                Log_Print(f"  {cfg_name:<22} 跳过(全C++不可用)", logger)
                continue
            if use_simd and not cpp_available:
                Log_Print(f"  {cfg_name:<22} 跳过(SIMD需要C++)", logger)
                continue
            
            try:
                model, _ = Create_Vim_Model(
                    model_name,
                    use_cpp_scan=config.get('use_cpp_scan', False),
                    use_fixlen_scan=config.get('use_fixlen_scan', False),
                    use_fused_bidirectional=config.get('use_fused_bidirectional', False),
                    use_full_cpp=config.get('use_full_cpp', False),
                    use_simd_scan=config.get('use_simd_scan', False),
                    use_simd_fixlen_scan=config.get('use_simd_fixlen_scan', False),
                    use_simd_fused_scan=config.get('use_simd_fused_scan', False),
                    use_simd_fused_fixlen_scan=config.get('use_simd_fused_fixlen_scan', False),
                )
                
                model.load_state_dict(base_state_dict)
                Warmup_Inference(model, input_tensor, NUM_WARMUP)
                avg_time, min_time, max_time, output = Benchmark_Inference(model, input_tensor, NUM_RUNS)
                
                model_results[cfg_name] = {'avg': avg_time, 'min': min_time, 'max': max_time}
                Log_Print(f"  {cfg_name:<22} {avg_time:>8.2f}ms (min:{min_time:.2f}, max:{max_time:.2f})", logger)
                
                del model
            except Exception as e:
                Log_Print(f"  {cfg_name:<22} 失败: {e}", logger)
        
        all_results[model_name] = {'params': total_params, 'configs': model_results}
        
        # 该模型性能对比
        if model_results:
            Log_Print(f"\n  --- {model_name} 性能对比 ---", logger)
            ref_time = model_results.get('Python-Original', list(model_results.values())[0])['avg']
            for cfg_name, result in model_results.items():
                speedup = ref_time / result['avg']
                marker = "⭐" if speedup > 2.0 else ("✓" if speedup > 1.0 else "")
                Log_Print(f"  {cfg_name:<22} {result['avg']:>8.2f}ms  {speedup:>5.2f}x {marker}", logger)
    
    # ==================== 汇总表格 ====================
    Log_Print("\n" + "=" * 100, logger)
    Log_Print("汇总表格：各模型各配置平均时间(ms)", logger)
    Log_Print("=" * 100, logger)
    
    # 表头
    header = f"{'配置':<22}"
    for model_name in models_to_test:
        params = all_results.get(model_name, {}).get('params', 0)
        header += f" {model_name}({params/1e6:.1f}M):>14"
    Log_Print(header, logger)
    Log_Print("-" * 100, logger)
    
    # 数据行
    for config in OPTIMIZATION_CONFIGS:
        cfg_name = config['name']
        row = f"{cfg_name:<22}"
        for model_name in models_to_test:
            model_data = all_results.get(model_name, {}).get('configs', {})
            if cfg_name in model_data:
                row += f" {model_data[cfg_name]['avg']:>14.2f}"
            else:
                row += f" {'N/A':>14}"
        Log_Print(row, logger)
    
    # 总结
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    Log_Print(f"\n总耗时: {elapsed:.1f}秒", logger)
    Log_Print("=" * 100, logger)
    Log_Print("测试结果已保存到 test.log", logger)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
