#Presented by KeJi
#Date: 2026-01-07

"""
统计Vision Mamba模型的FLOPs
使用thop库进行计算，包含Mamba自定义算子注册
"""

import os
import sys
import torch
import torch.nn as nn

os.environ['SELECTIVE_SCAN_FORCE_FALLBACK'] = 'TRUE'
os.environ['CAUSAL_CONV1D_FORCE_FALLBACK'] = 'TRUE'

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

mamba_dir = os.path.join(os.path.dirname(current_dir), 'mamba-1p1p1')
if os.path.exists(mamba_dir):
    sys.path.insert(0, mamba_dir)

from thop import profile, clever_format
from thop.vision.basic_hooks import count_convNd, count_linear, count_normalization
from timm.models import create_model
import models_mamba


# ==================== Mamba自定义算子FLOPs计算 ====================

def Count_Mamba_Flops(m, x, y):
    """
    计算Mamba层的FLOPs
    
    Mamba层包含:
    1. in_proj: Linear(d_model, d_inner*2) - 输入投影
    2. conv1d: Conv1d(d_inner, d_inner, kernel=4) - 因果卷积
    3. x_proj: Linear(d_inner, dt_rank + d_state*2) - 状态投影
    4. dt_proj: Linear(dt_rank, d_inner) - dt投影
    5. Selective Scan: 递推计算 - 核心SSM操作
    6. out_proj: Linear(d_inner, d_model) - 输出投影
    
    Selective Scan FLOPs公式:
    - 状态更新: x[t] = A * x[t-1] + B * u[t]  -> 2*L*B*D*N (乘加)
    - 输出计算: y[t] = C @ x[t]  -> L*B*D*N (点积)
    总计: 3 * L * B * D * N
    """
    # 获取Mamba模块参数
    d_model = m.d_model
    d_inner = m.d_inner  # 通常 = d_model * expand = d_model * 2
    d_state = m.d_state  # N, 通常 = 16
    dt_rank = m.dt_rank  # 通常 = ceil(d_model / 16)
    d_conv = m.d_conv    # 通常 = 4
    
    # 输入形状: (B, L, D) 或 (B, D, L)
    input_shape = x[0].shape
    if len(input_shape) == 3:
        batch_size = input_shape[0]
        seq_len = input_shape[1]
    else:
        batch_size = 1
        seq_len = input_shape[0]
    
    flops = 0
    
    # 1. in_proj: (B, L, d_model) -> (B, L, d_inner*2)
    # FLOPs = B * L * d_model * d_inner * 2
    flops += batch_size * seq_len * d_model * d_inner * 2 * 2  # *2 for bias add
    
    # 2. conv1d: (B, d_inner, L) -> (B, d_inner, L)
    # FLOPs = B * d_inner * L * d_conv
    flops += batch_size * d_inner * seq_len * d_conv * 2
    
    # 3. x_proj: (B, L, d_inner) -> (B, L, dt_rank + d_state*2)
    # FLOPs = B * L * d_inner * (dt_rank + d_state * 2)
    flops += batch_size * seq_len * d_inner * (dt_rank + d_state * 2) * 2
    
    # 4. dt_proj: (B, L, dt_rank) -> (B, L, d_inner)
    # FLOPs = B * L * dt_rank * d_inner
    flops += batch_size * seq_len * dt_rank * d_inner * 2
    
    # 5. Selective Scan (核心SSM操作)
    # 状态更新: x = A*x + B*u -> 2*L*B*D*N (每步: N次乘法+N次加法)
    # 输出: y = C @ x -> L*B*D*N (每步: N次乘法+N-1次加法)
    # 双向: *2
    ssm_flops = 3 * seq_len * batch_size * d_inner * d_state
    if hasattr(m, 'bimamba_type') and m.bimamba_type in ['v1', 'v2']:
        ssm_flops *= 2  # 双向扫描
    flops += ssm_flops
    
    # 6. out_proj: (B, L, d_inner) -> (B, L, d_model)
    # FLOPs = B * L * d_inner * d_model
    flops += batch_size * seq_len * d_inner * d_model * 2
    
    return flops


def Register_Mamba_Hooks():
    """注册Mamba自定义算子到thop"""
    from thop.profile import register_hooks
    
    # 导入Mamba类
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
        register_hooks[Mamba] = Count_Mamba_Flops
        print("✓ 已注册Mamba层FLOPs计算")
    except ImportError:
        print("✗ 无法导入Mamba模块")


def main():
    print("=" * 60)
    print("Vision Mamba FLOPs统计 (包含Mamba自定义算子)")
    print("=" * 60)
    
    # 注册Mamba hooks
    Register_Mamba_Hooks()
    
    # 创建vim_tiny模型
    print("\n创建vim_tiny模型...")
    model = create_model(
        'vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2',
        pretrained=False,
        num_classes=1000,
    )
    model.eval()
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 创建输入
    input_tensor = torch.randn(1, 3, 224, 224)
    print(f"输入形状: {input_tensor.shape}")
    
    # 使用thop统计FLOPs
    print("\n使用thop统计FLOPs...")
    try:
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.2f")
        
        print(f"\n结果:")
        print(f"  FLOPs: {flops_str} ({flops/1e9:.2f}G)")
        print(f"  Params: {params_str} ({params/1e6:.2f}M)")
        
        # 计算每百万参数的GFLOPs
        gflops_per_mparam = (flops / 1e9) / (params / 1e6)
        print(f"  GFLOPs/M-Params: {gflops_per_mparam:.2f}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 分层FLOPs分析
    print("\n" + "=" * 60)
    print("理论FLOPs估算")
    print("=" * 60)
    
    # vim_tiny配置
    img_size = 224
    patch_size = 16
    in_chans = 3
    embed_dim = 192      # d_model
    depth = 24           # 层数
    d_state = 16         # N
    expand = 2           # d_inner倍数
    d_inner = embed_dim * expand  # 384
    dt_rank = (embed_dim + 15) // 16  # 12
    d_conv = 4
    num_patches = (img_size // patch_size) ** 2  # 196
    seq_len = num_patches + 1  # 197 (patches + cls_token)
    batch_size = 1
    num_classes = 1000
    
    # 1. PatchEmbed (Conv2d)
    # FLOPs = 2 * H_out * W_out * C_in * C_out * K_h * K_w
    patch_embed_flops = 2 * (img_size // patch_size) ** 2 * in_chans * embed_dim * patch_size * patch_size
    
    # 2. LayerNorm (每层2个: Block.norm + 最后norm_f)
    # FLOPs ≈ 2 * seq_len * embed_dim (mean + variance)
    layernorm_flops = 2 * seq_len * embed_dim * (depth + 1)
    
    # 3. 单层Mamba FLOPs
    single_mamba_flops = 0
    single_mamba_flops += batch_size * seq_len * embed_dim * d_inner * 2 * 2  # in_proj
    single_mamba_flops += batch_size * d_inner * seq_len * d_conv * 2         # conv1d
    single_mamba_flops += batch_size * seq_len * d_inner * (dt_rank + d_state * 2) * 2  # x_proj
    single_mamba_flops += batch_size * seq_len * dt_rank * d_inner * 2        # dt_proj
    single_mamba_flops += 3 * seq_len * batch_size * d_inner * d_state * 2    # SSM (双向)
    single_mamba_flops += batch_size * seq_len * d_inner * embed_dim * 2      # out_proj
    
    total_mamba_flops = single_mamba_flops * depth
    
    # 4. Classifier Head (Linear)
    # FLOPs = 2 * embed_dim * num_classes
    head_flops = 2 * embed_dim * num_classes
    
    # 总FLOPs
    total_flops = patch_embed_flops + layernorm_flops + total_mamba_flops + head_flops
    
    print(f"配置: img={img_size}, patch={patch_size}, embed_dim={embed_dim}, depth={depth}")
    print(f"       d_state={d_state}, d_inner={d_inner}, seq_len={seq_len}")
    print(f"\n分层FLOPs:")
    print(f"  PatchEmbed:     {patch_embed_flops/1e6:>8.2f}M")
    print(f"  LayerNorm:      {layernorm_flops/1e6:>8.2f}M")
    print(f"  Mamba({depth}层):   {total_mamba_flops/1e9:>8.2f}G ({single_mamba_flops/1e6:.2f}M/层)")
    print(f"  Classifier:     {head_flops/1e6:>8.2f}M")
    print(f"\n" + "-" * 40)
    print(f"  总FLOPs:        {total_flops/1e9:>8.2f}G")
    print("=" * 60)


if __name__ == '__main__':
    main()
