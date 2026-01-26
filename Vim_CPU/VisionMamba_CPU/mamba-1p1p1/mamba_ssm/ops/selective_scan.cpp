//Presented by KeJi
//Date: 2026-01-06

#include <torch/extension.h>
#include <vector>

// OpenMP头文件
#if defined(_OPENMP)
    #include <omp.h>
#endif

// SIMD头文件
#if defined(__AVX512F__)
    #include <immintrin.h>
    #define SIMD_WIDTH 16
    #define HAS_AVX512 1
#elif defined(__AVX2__) || defined(__AVX__)
    #include <immintrin.h>
    #define SIMD_WIDTH 8
    #define HAS_AVX 1
#elif defined(__SSE4_1__) || defined(__SSE2__)
    #include <emmintrin.h>
    #include <smmintrin.h>
    #define SIMD_WIDTH 4
    #define HAS_SSE 1
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    // MSVC默认支持SSE2
    #include <immintrin.h>
    #define SIMD_WIDTH 8
    #define HAS_AVX 1
#elif defined(__ARM_NEON) || defined(__aarch64__)
    // ARM NEON (树莓派, ARM64)
    #include <arm_neon.h>
    #define SIMD_WIDTH 4
    #define HAS_NEON 1
#else
    #define SIMD_WIDTH 1
    #define HAS_SCALAR_ONLY 1
#endif

/*
 * Vision Mamba Selective Scan C++ 完整实现
 *
 * 完全复刻selective_scan_interface.py:235的selective_scan_ref函数
 */

// 获取OpenMP信息的辅助函数
std::string Get_Openmp_Info() {
    std::string info;
#if defined(_OPENMP)
    int max_threads = omp_get_max_threads();
    int num_procs = omp_get_num_procs();
    info = "[OpenMP] =" + std::to_string(num_procs) +
           ", max thread=" + std::to_string(max_threads);
#else
    info = "[OpenMP] open";
#endif
    return info;
}

// 获取SIMD信息的辅助函数
std::string Get_Simd_Info() {
    std::string info = "[SIMD] ";
#if defined(HAS_AVX512)
    info += "AVX-512 (512bit, 16x float)";
#elif defined(HAS_AVX)
    info += "AVX/AVX2 (256bit, 8x float)";
#elif defined(HAS_SSE)
    info += "SSE (128bit, 4x float)";
#elif defined(HAS_NEON)
    info += "NEON (128bit, 4x float)";
#else
    info += "scalar (no SIMD)";
#endif
    return info;
}

// 辅助函数：检查tensor是否为None
inline bool is_none(const torch::Tensor& t) {
    return !t.defined() || t.numel() == 0;
}

// 完整复刻selective_scan_ref
torch::Tensor Selective_Scan_Ref_Cpu(
    const torch::Tensor& u,              // (B, D, L)
    const torch::Tensor& delta,          // (B, D, L)
    const torch::Tensor& A,              // (D, N)
    const torch::Tensor& B,              // (D, N) or (B, N, L)
    const torch::Tensor& C,              // (D, N) or (B, N, L)
    const torch::Tensor& D = torch::Tensor(),       // (D,) - 可选
    const torch::Tensor& z = torch::Tensor(),       // (B, D, L) - 可选
    const torch::Tensor& delta_bias = torch::Tensor(),  // (D,) - 可选
    bool delta_softplus = false,         // 是否对delta应用softplus
    bool return_last_state = false       // 是否返回last_state（简化：忽略）
) {
    // Line 250-252: 数据类型转换
    auto dtype_in = u.dtype();
    auto u_f = u.to(torch::kFloat32);
    auto delta_f = delta.to(torch::kFloat32);
    
    // Line 253-256: 处理delta_bias和delta_softplus
    if (!is_none(delta_bias)) {
        delta_f = delta_f + delta_bias.unsqueeze(-1).to(torch::kFloat32);
    }
    if (delta_softplus) {
        delta_f = torch::nn::functional::softplus(delta_f);
    }
    
    // Line 257: 获取维度
    const int64_t batch = u_f.size(0);
    const int64_t dim = A.size(0);
    const int64_t dstate = A.size(1);
    const int64_t seq_len = u_f.size(2);
    
    // Line 258-259: 判断B和C是否是variable
    bool is_variable_B = B.dim() >= 3;
    bool is_variable_C = C.dim() >= 3;
    
    // Line 265-267: 转换为float
    auto B_f = B.to(torch::kFloat32);
    auto C_f = C.to(torch::kFloat32);
    
    // Line 268: 初始化隐藏状态 x
    auto x = torch::zeros({batch, dim, dstate}, u_f.options());
    
    // Line 269: 初始化输出列表
    std::vector<torch::Tensor> ys;
    ys.reserve(seq_len);
    
    // Line 270: deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    // delta: (B,D,L), A: (D,N) -> (B,D,L,N)
    auto deltaA = torch::exp(
        delta_f.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2)
    );
    
    // Line 271-278: 计算deltaB_u
    torch::Tensor deltaB_u;
    if (!is_variable_B) {
        // Line 272: deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        // delta: (B,D,L), B: (D,N), u: (B,D,L) -> (B,D,L,N)
        deltaB_u = delta_f.unsqueeze(-1) * B_f.unsqueeze(0).unsqueeze(2) * u_f.unsqueeze(-1);
    } else {
        if (B_f.dim() == 3) {
            // Line 275: deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            // delta: (B,D,L), B: (B,N,L), u: (B,D,L) -> (B,D,L,N)
            // B: (B,N,L) -> (B,1,N,L) -> transpose to (B,1,L,N)
            deltaB_u = delta_f.unsqueeze(-1) * 
                      B_f.unsqueeze(1).transpose(2, 3) * 
                      u_f.unsqueeze(-1);
        } else {
            throw std::runtime_error("Grouped B (dim==4) not supported");
        }
    }
    
    // Line 282-295: 主循环
    for (int64_t i = 0; i < seq_len; i++) {
        // Line 283: x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        x = deltaA.index({torch::indexing::Slice(), torch::indexing::Slice(), i}) * x
            + deltaB_u.index({torch::indexing::Slice(), torch::indexing::Slice(), i});
        
        // Line 284-290: 计算y
        torch::Tensor y;
        if (!is_variable_C) {
            // Line 285: y = torch.einsum('bdn,dn->bd', x, C)
            // x: (B,D,N), C: (D,N) -> (B,D)
            y = torch::sum(x * C_f.unsqueeze(0), -1);
        } else {
            if (C_f.dim() == 3) {
                // Line 287-288: y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                // x: (B,D,N), C[:,:,i]: (B,N) -> (B,D)
                auto C_i = C_f.index({torch::indexing::Slice(), torch::indexing::Slice(), i});  // (B,N)
                y = torch::sum(x * C_i.unsqueeze(1), -1);  // (B,D)
            } else {
                throw std::runtime_error("Grouped C (dim==4) not supported");
            }
        }
        
        // Line 295: ys.append(y)
        ys.push_back(y);
    }
    
    // Line 296: y = torch.stack(ys, dim=2)  # (batch, dim, L)
    auto y = torch::stack(ys, 2);
    
    // Line 297: out = y if D is None else y + u * rearrange(D, "d -> d 1")
    torch::Tensor out;
    if (is_none(D)) {
        out = y;
    } else {
        // u: (B,D,L), D: (D,) -> (1,D,1) -> (B,D,L)
        out = y + u_f * D.to(torch::kFloat32).unsqueeze(0).unsqueeze(-1);
    }
    
    // Line 298-299: 应用z门控
    if (!is_none(z)) {
        out = out * torch::nn::functional::silu(z.to(torch::kFloat32));
    }
    
    // Line 300: 转换回原始数据类型
    out = out.to(dtype_in);
    
    return out;
}

// 优化的两阶段算法
torch::Tensor Selective_Scan_Ref_Fixlen_Cpu(
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    const torch::Tensor& D = torch::Tensor(),
    const torch::Tensor& z = torch::Tensor(),
    const torch::Tensor& delta_bias = torch::Tensor(),
    bool delta_softplus = false,
    bool return_last_state = false
) {
    // 前处理同原始版本
    auto dtype_in = u.dtype();
    auto u_f = u.to(torch::kFloat32);
    auto delta_f = delta.to(torch::kFloat32);
    
    if (!is_none(delta_bias)) {
        delta_f = delta_f + delta_bias.unsqueeze(-1).to(torch::kFloat32);
    }
    if (delta_softplus) {
        delta_f = torch::nn::functional::softplus(delta_f);
    }
    
    const int64_t batch = u_f.size(0);
    const int64_t dim = A.size(0);
    const int64_t dstate = A.size(1);
    const int64_t seq_len = u_f.size(2);
    
    bool is_variable_B = B.dim() >= 3;
    bool is_variable_C = C.dim() >= 3;
    
    auto B_f = B.to(torch::kFloat32);
    auto C_f = C.to(torch::kFloat32);
    
    // 计算deltaA
    auto deltaA = torch::exp(
        delta_f.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2)
    );
    
    // 计算deltaB_u
    torch::Tensor deltaB_u;
    if (!is_variable_B) {
        deltaB_u = delta_f.unsqueeze(-1) * B_f.unsqueeze(0).unsqueeze(2) * u_f.unsqueeze(-1);
    } else {
        if (B_f.dim() == 3) {
            deltaB_u = delta_f.unsqueeze(-1) * 
                      B_f.unsqueeze(1).transpose(2, 3) * 
                      u_f.unsqueeze(-1);
        } else {
            throw std::runtime_error("Grouped B not supported");
        }
    }
    
    // 阶段1：递推计算所有隐藏状态（使用select优化，避免clone）
    // 公式：deltaB_u[i] = deltaA[i] * deltaB_u[i-1] + deltaB_u[i]
    // 改写为：deltaB_u[i] += deltaA[i] * deltaB_u[i-1]
    for (int64_t i = 1; i < seq_len; i++) {
        auto deltaA_i = deltaA.select(2, i);       // (B, D, N) - 视图
        auto prev = deltaB_u.select(2, i - 1);     // (B, D, N) - 视图
        // 原地加法：curr += deltaA_i * prev
        deltaB_u.select(2, i).add_(deltaA_i * prev);
    }
    
    // 阶段2：批量计算输出
    torch::Tensor y;
    if (!is_variable_C) {
        y = torch::sum(deltaB_u * C_f.unsqueeze(0).unsqueeze(2), -1);
    } else {
        if (C_f.dim() == 3) {
            y = torch::sum(deltaB_u * C_f.unsqueeze(1).transpose(2, 3), -1);
        } else {
            throw std::runtime_error("Grouped C not supported");
        }
    }
    
    // 添加D和z
    torch::Tensor out;
    if (is_none(D)) {
        out = y;
    } else {
        out = y + u_f * D.to(torch::kFloat32).unsqueeze(0).unsqueeze(-1);
    }
    
    if (!is_none(z)) {
        out = out * torch::nn::functional::silu(z.to(torch::kFloat32));
    }
    
    out = out.to(dtype_in);
    
    return out;
}

// 融合双向Selective Scan（标准版本）
// 使用原始的逐步计算方式：每步计算隐藏状态x并输出y
torch::Tensor Selective_Fused_Scan_Cpu(
    const torch::Tensor& dt_fwd,            // (B, D, L)
    const torch::Tensor& dt_bwd,            // (B, D, L)
    const torch::Tensor& A_fwd,             // (D, N)
    const torch::Tensor& A_bwd,             // (D, N)
    const torch::Tensor& B_fwd,             // (B, N, L)
    const torch::Tensor& B_bwd,             // (B, N, L)
    const torch::Tensor& x_fwd_conv,        // (B, D, L)
    const torch::Tensor& x_bwd_conv_flip,   // (B, D, L)
    const torch::Tensor& C_fwd,             // (B, N, L)
    const torch::Tensor& C_bwd,             // (B, N, L)
    const torch::Tensor& D_fwd,             // (D,)
    const torch::Tensor& D_bwd,             // (D,)
    const torch::Tensor& z_fwd = torch::Tensor(),       // (B, D, L) - 可选
    const torch::Tensor& z_bwd_flip = torch::Tensor()   // (B, D, L) - 可选
) {
    auto dtype_in = dt_fwd.dtype();
    
    const int64_t batch = dt_fwd.size(0);
    const int64_t dim = dt_fwd.size(1);
    const int64_t seq_len = dt_fwd.size(2);
    const int64_t dstate = A_fwd.size(1);
    
    // 计算deltaA和deltaB_u（使用(B,D,L,2N)布局）
    auto deltaA_bi = torch::empty({batch, dim, seq_len, 2 * dstate}, dt_fwd.options().dtype(torch::kFloat32));
    auto deltaB_u_bi = torch::empty({batch, dim, seq_len, 2 * dstate}, dt_fwd.options().dtype(torch::kFloat32));
    
    // 正向部分 [:,:,:,:n]
    deltaA_bi.narrow(3, 0, dstate).copy_(
        torch::exp(dt_fwd.unsqueeze(-1) * A_fwd.unsqueeze(0).unsqueeze(2))
    );
    deltaB_u_bi.narrow(3, 0, dstate).copy_(
        dt_fwd.unsqueeze(-1) * B_fwd.unsqueeze(1).transpose(2, 3) * x_fwd_conv.unsqueeze(-1)
    );
    
    // 反向部分 [:,:,:,n:]
    deltaA_bi.narrow(3, dstate, dstate).copy_(
        torch::exp(dt_bwd.unsqueeze(-1) * A_bwd.unsqueeze(0).unsqueeze(2))
    );
    deltaB_u_bi.narrow(3, dstate, dstate).copy_(
        dt_bwd.unsqueeze(-1) * B_bwd.unsqueeze(1).transpose(2, 3) * x_bwd_conv_flip.unsqueeze(-1)
    );
    
    // C矩阵转置为(B,L,N)
    auto C_fwd_t = C_fwd.transpose(1, 2);  // (B, L, N)
    auto C_bwd_t = C_bwd.transpose(1, 2);  // (B, L, N)
    
    // 初始化隐藏状态
    auto x_bi = torch::zeros({batch, dim, 2 * dstate}, dt_fwd.options().dtype(torch::kFloat32));
    
    // 逐步计算
    std::vector<torch::Tensor> ys_fwd;
    std::vector<torch::Tensor> ys_bwd;
    ys_fwd.reserve(seq_len);
    ys_bwd.reserve(seq_len);
    
    for (int64_t i = 0; i < seq_len; i++) {
        // 更新隐藏状态: x = deltaA * x + deltaB_u
        x_bi = deltaA_bi.select(2, i) * x_bi + deltaB_u_bi.select(2, i);  // (B, D, 2N)
        
        // 分离正向和反向隐藏状态
        auto x_fwd = x_bi.narrow(2, 0, dstate);        // (B, D, N)
        auto x_bwd = x_bi.narrow(2, dstate, dstate);   // (B, D, N)
        
        // 计算输出: y = einsum('bdn,bn->bd')
        auto C_fwd_i = C_fwd_t.select(1, i);  // (B, N)
        auto C_bwd_i = C_bwd_t.select(1, i);  // (B, N)
        auto y_fwd = torch::sum(x_fwd * C_fwd_i.unsqueeze(1), -1);  // (B, D)
        auto y_bwd = torch::sum(x_bwd * C_bwd_i.unsqueeze(1), -1);  // (B, D)
        
        ys_fwd.push_back(y_fwd);
        ys_bwd.push_back(y_bwd);
    }
    
    // 堆叠输出
    auto y_fwd = torch::stack(ys_fwd, 2);  // (B, D, L)
    auto y_bwd = torch::stack(ys_bwd, 2);  // (B, D, L)
    
    // 添加D项
    y_fwd = y_fwd + x_fwd_conv * D_fwd.unsqueeze(0).unsqueeze(-1);
    y_bwd = y_bwd + x_bwd_conv_flip * D_bwd.unsqueeze(0).unsqueeze(-1);
    
    // 门控
    if (!is_none(z_fwd)) {
        y_fwd = y_fwd * torch::nn::functional::silu(z_fwd);
    }
    if (!is_none(z_bwd_flip)) {
        y_bwd = y_bwd * torch::nn::functional::silu(z_bwd_flip);
    }
    
    // 反转并合并输出
    y_bwd = y_bwd.flip({2});  // 反转回原序列顺序
    auto out = y_fwd + y_bwd;  // (B, D, L)
    
    return out.to(dtype_in);
}

// 融合双向Selective Scan（优化版本，使用select避免clone + 预分配消除cat + (B,D,L,2N)布局避免permute）
torch::Tensor Selective_Fused_Scan_Fixlen_Cpu(
    const torch::Tensor& dt_fwd,            // (B, D, L)
    const torch::Tensor& dt_bwd,            // (B, D, L)
    const torch::Tensor& A_fwd,             // (D, N)
    const torch::Tensor& A_bwd,             // (D, N)
    const torch::Tensor& B_fwd,             // (B, N, L)
    const torch::Tensor& B_bwd,             // (B, N, L)
    const torch::Tensor& x_fwd_conv,        // (B, D, L)
    const torch::Tensor& x_bwd_conv_flip,   // (B, D, L)
    const torch::Tensor& C_fwd,             // (B, N, L)
    const torch::Tensor& C_bwd,             // (B, N, L)
    const torch::Tensor& D_fwd,             // (D,)
    const torch::Tensor& D_bwd,             // (D,)
    const torch::Tensor& z_fwd = torch::Tensor(),       // (B, D, L) - 可选
    const torch::Tensor& z_bwd_flip = torch::Tensor()   // (B, D, L) - 可选
) {
    auto dtype_in = dt_fwd.dtype();
    
    const int64_t batch = dt_fwd.size(0);
    const int64_t dim = dt_fwd.size(1);
    const int64_t seq_len = dt_fwd.size(2);
    const int64_t dstate = A_fwd.size(1);
    
    // 使用(B,D,L,2N)布局，与广播操作兼容，避免permute
    auto deltaA_bi = torch::empty({batch, dim, seq_len, 2 * dstate}, dt_fwd.options().dtype(torch::kFloat32));
    auto deltaB_u_bi = torch::empty({batch, dim, seq_len, 2 * dstate}, dt_fwd.options().dtype(torch::kFloat32));
    
    // 计算并写入（无需permute）
    // 正向部分 [:,:,:,:n]
    deltaA_bi.narrow(3, 0, dstate).copy_(
        torch::exp(dt_fwd.unsqueeze(-1) * A_fwd.unsqueeze(0).unsqueeze(2))
    );
    deltaB_u_bi.narrow(3, 0, dstate).copy_(
        dt_fwd.unsqueeze(-1) * B_fwd.unsqueeze(1).transpose(2, 3) * x_fwd_conv.unsqueeze(-1)
    );
    
    // 反向部分 [:,:,:,n:]
    deltaA_bi.narrow(3, dstate, dstate).copy_(
        torch::exp(dt_bwd.unsqueeze(-1) * A_bwd.unsqueeze(0).unsqueeze(2))
    );
    deltaB_u_bi.narrow(3, dstate, dstate).copy_(
        dt_bwd.unsqueeze(-1) * B_bwd.unsqueeze(1).transpose(2, 3) * x_bwd_conv_flip.unsqueeze(-1)
    );
    
    // 阶段1：递推（原地修改，(B,D,L,2N)布局）
    for (int64_t i = 1; i < seq_len; i++) {
        auto deltaA_i = deltaA_bi.select(2, i);       // (B, D, 2N) - 视图
        auto prev = deltaB_u_bi.select(2, i - 1);     // (B, D, 2N) - 视图
        // 原地加法：curr += deltaA_i * prev
        deltaB_u_bi.select(2, i).add_(deltaA_i * prev);
    }
    
    // 阶段2：分离并计算输出（无需转置）
    auto deltaB_u_fwd_out = deltaB_u_bi.narrow(3, 0, dstate);       // (b, d, l, n)
    auto deltaB_u_bwd_out = deltaB_u_bi.narrow(3, dstate, dstate);  // (b, d, l, n)
    
    // 批量计算输出
    auto y_fwd = torch::sum(deltaB_u_fwd_out * C_fwd.unsqueeze(1).transpose(2, 3), -1);
    auto y_bwd = torch::sum(deltaB_u_bwd_out * C_bwd.unsqueeze(1).transpose(2, 3), -1);
    
    // 添加D项
    y_fwd = y_fwd + x_fwd_conv * D_fwd.unsqueeze(0).unsqueeze(-1);
    y_bwd = y_bwd + x_bwd_conv_flip * D_bwd.unsqueeze(0).unsqueeze(-1);
    
    // 门控
    if (!is_none(z_fwd)) {
        y_fwd = y_fwd * torch::nn::functional::silu(z_fwd);
    }
    if (!is_none(z_bwd_flip)) {
        y_bwd = y_bwd * torch::nn::functional::silu(z_bwd_flip);
    }
    
    // 反转并合并输出
    y_bwd = y_bwd.flip({2});
    auto out = y_fwd + y_bwd;
    
    return out.to(dtype_in);
}

// ========== SIMD优化实现 ==========

// SIMD辅助函数：水平求和
#if defined(HAS_AVX512)
inline float Horizontal_Sum_Avx512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}
#endif

#if defined(HAS_AVX) || defined(_MSC_VER)
inline float Horizontal_Sum_Avx(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
#endif

#if defined(HAS_SSE)
inline float Horizontal_Sum_Sse(__m128 v) {
    __m128 sum = _mm_hadd_ps(v, v);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
#endif

#if defined(HAS_NEON)
inline float Horizontal_Sum_Neon(float32x4_t v) {
#if defined(__aarch64__)
    // ARM64: 使用vaddvq_f32直接归约
    return vaddvq_f32(v);
#else
    // ARM32: 手动归约
    float32x2_t sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    sum = vpadd_f32(sum, sum);
    return vget_lane_f32(sum, 0);
#endif
}
#endif

// SIMD版本的Selective Scan（原版算法，N维度向量化）
torch::Tensor Selective_Scan_Simd_Cpu(
    const torch::Tensor& u,              // (B, D, L)
    const torch::Tensor& delta,          // (B, D, L)
    const torch::Tensor& A,              // (D, N)
    const torch::Tensor& B,              // (D, N) or (B, N, L)
    const torch::Tensor& C,              // (D, N) or (B, N, L)
    const torch::Tensor& D_param = torch::Tensor(),   // (D,) - 可选
    const torch::Tensor& z = torch::Tensor(),         // (B, D, L) - 可选
    const torch::Tensor& delta_bias = torch::Tensor(),  // (D,) - 可选
    bool delta_softplus = false,
    bool return_last_state = false
) {
    // 数据类型转换
    auto dtype_in = u.dtype();
    auto u_f = u.to(torch::kFloat32).contiguous();
    auto delta_f = delta.to(torch::kFloat32).contiguous();
    
    // 处理delta_bias和delta_softplus
    if (!is_none(delta_bias)) {
        delta_f = delta_f + delta_bias.unsqueeze(-1).to(torch::kFloat32);
    }
    if (delta_softplus) {
        delta_f = torch::nn::functional::softplus(delta_f);
    }
    
    // 获取维度
    const int64_t batch = u_f.size(0);
    const int64_t dim = A.size(0);
    const int64_t dstate = A.size(1);
    const int64_t seq_len = u_f.size(2);
    
    // 判断B和C是否是variable
    bool is_variable_B = B.dim() >= 3;
    bool is_variable_C = C.dim() >= 3;
    
    // 转换为float并确保contiguous
    auto B_f = B.to(torch::kFloat32).contiguous();
    auto C_f = C.to(torch::kFloat32).contiguous();
    
    // 计算deltaA: (B, D, L, N)
    auto deltaA = torch::exp(
        delta_f.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2)
    ).contiguous();
    
    // 计算deltaB_u: (B, D, L, N)
    torch::Tensor deltaB_u;
    if (!is_variable_B) {
        deltaB_u = (delta_f.unsqueeze(-1) * B_f.unsqueeze(0).unsqueeze(2) * u_f.unsqueeze(-1)).contiguous();
    } else {
        if (B_f.dim() == 3) {
            deltaB_u = (delta_f.unsqueeze(-1) * B_f.unsqueeze(1).transpose(2, 3) * u_f.unsqueeze(-1)).contiguous();
        } else {
            throw std::runtime_error("Grouped B (dim==4) not supported in SIMD version");
        }
    }
    
    // 准备C矩阵：根据is_variable_C决定布局
    torch::Tensor C_t;
    if (is_variable_C) {
        if (C_f.dim() == 3) {
            C_t = C_f.transpose(1, 2).contiguous();  // (B, N, L) -> (B, L, N)
        } else {
            throw std::runtime_error("Grouped C (dim==4) not supported in SIMD version");
        }
    }
    
    // 分配输出tensor
    auto y = torch::empty({batch, dim, seq_len}, u_f.options());
    
    // 获取数据指针
    float* deltaA_ptr = deltaA.data_ptr<float>();
    float* deltaB_u_ptr = deltaB_u.data_ptr<float>();
    float* C_ptr = is_variable_C ? C_t.data_ptr<float>() : C_f.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();
    
    // 分配隐藏状态x: (B, D, N)
    auto x = torch::zeros({batch, dim, dstate}, u_f.options());
    float* x_ptr = x.data_ptr<float>();
    
    // SIMD主循环（延迟归约优化 + OpenMP并行）
    #if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t d = 0; d < dim; d++) {
            const int64_t x_base = (b * dim + d) * dstate;
            
            for (int64_t i = 0; i < seq_len; i++) {
                const int64_t dA_base = ((b * dim + d) * seq_len + i) * dstate;
                const int64_t y_idx = (b * dim + d) * seq_len + i;
                
                // 状态更新：x = deltaA * x + deltaB_u
                float sum = 0.0f;
                int64_t n = 0;
                
#if defined(HAS_AVX512)
                // AVX-512延迟归约
                __m512 acc_512 = _mm512_setzero_ps();
                for (; n + 16 <= dstate; n += 16) {
                    __m512 x_v = _mm512_loadu_ps(&x_ptr[x_base + n]);
                    __m512 dA_v = _mm512_loadu_ps(&deltaA_ptr[dA_base + n]);
                    __m512 dBu_v = _mm512_loadu_ps(&deltaB_u_ptr[dA_base + n]);
                    
                    x_v = _mm512_fmadd_ps(dA_v, x_v, dBu_v);
                    _mm512_storeu_ps(&x_ptr[x_base + n], x_v);
                    
                    __m512 c_v;
                    if (is_variable_C) {
                        const int64_t c_idx = (b * seq_len + i) * dstate + n;
                        c_v = _mm512_loadu_ps(&C_ptr[c_idx]);
                    } else {
                        const int64_t c_idx = d * dstate + n;
                        c_v = _mm512_loadu_ps(&C_ptr[c_idx]);
                    }
                    acc_512 = _mm512_fmadd_ps(x_v, c_v, acc_512);
                }
                sum += Horizontal_Sum_Avx512(acc_512);
#endif

#if defined(HAS_AVX) || defined(_MSC_VER)
                // AVX延迟归约
                __m256 acc_256 = _mm256_setzero_ps();
                for (; n + 8 <= dstate; n += 8) {
                    __m256 x_v = _mm256_loadu_ps(&x_ptr[x_base + n]);
                    __m256 dA_v = _mm256_loadu_ps(&deltaA_ptr[dA_base + n]);
                    __m256 dBu_v = _mm256_loadu_ps(&deltaB_u_ptr[dA_base + n]);
                    
                    #if defined(__FMA__)
                    x_v = _mm256_fmadd_ps(dA_v, x_v, dBu_v);
                    #else
                    x_v = _mm256_add_ps(_mm256_mul_ps(dA_v, x_v), dBu_v);
                    #endif
                    _mm256_storeu_ps(&x_ptr[x_base + n], x_v);
                    
                    __m256 c_v;
                    if (is_variable_C) {
                        const int64_t c_idx = (b * seq_len + i) * dstate + n;
                        c_v = _mm256_loadu_ps(&C_ptr[c_idx]);
                    } else {
                        const int64_t c_idx = d * dstate + n;
                        c_v = _mm256_loadu_ps(&C_ptr[c_idx]);
                    }
                    #if defined(__FMA__)
                    acc_256 = _mm256_fmadd_ps(x_v, c_v, acc_256);
                    #else
                    acc_256 = _mm256_add_ps(acc_256, _mm256_mul_ps(x_v, c_v));
                    #endif
                }
                sum += Horizontal_Sum_Avx(acc_256);
#endif

#if defined(HAS_SSE)
                // SSE延迟归约
                __m128 acc_128 = _mm_setzero_ps();
                for (; n + 4 <= dstate; n += 4) {
                    __m128 x_v = _mm_loadu_ps(&x_ptr[x_base + n]);
                    __m128 dA_v = _mm_loadu_ps(&deltaA_ptr[dA_base + n]);
                    __m128 dBu_v = _mm_loadu_ps(&deltaB_u_ptr[dA_base + n]);
                    
                    x_v = _mm_add_ps(_mm_mul_ps(dA_v, x_v), dBu_v);
                    _mm_storeu_ps(&x_ptr[x_base + n], x_v);
                    
                    __m128 c_v;
                    if (is_variable_C) {
                        const int64_t c_idx = (b * seq_len + i) * dstate + n;
                        c_v = _mm_loadu_ps(&C_ptr[c_idx]);
                    } else {
                        const int64_t c_idx = d * dstate + n;
                        c_v = _mm_loadu_ps(&C_ptr[c_idx]);
                    }
                    acc_128 = _mm_add_ps(acc_128, _mm_mul_ps(x_v, c_v));
                }
                sum += Horizontal_Sum_Sse(acc_128);
#endif

#if defined(HAS_NEON)
                // NEON延迟归约
                float32x4_t acc_neon = vdupq_n_f32(0.0f);
                for (; n + 4 <= dstate; n += 4) {
                    float32x4_t x_v = vld1q_f32(&x_ptr[x_base + n]);
                    float32x4_t dA_v = vld1q_f32(&deltaA_ptr[dA_base + n]);
                    float32x4_t dBu_v = vld1q_f32(&deltaB_u_ptr[dA_base + n]);
                    
                    // x = dA * x + dBu (使用FMA如果可用)
                    x_v = vmlaq_f32(dBu_v, dA_v, x_v);
                    vst1q_f32(&x_ptr[x_base + n], x_v);
                    
                    float32x4_t c_v;
                    if (is_variable_C) {
                        const int64_t c_idx = (b * seq_len + i) * dstate + n;
                        c_v = vld1q_f32(&C_ptr[c_idx]);
                    } else {
                        const int64_t c_idx = d * dstate + n;
                        c_v = vld1q_f32(&C_ptr[c_idx]);
                    }
                    acc_neon = vmlaq_f32(acc_neon, x_v, c_v);
                }
                sum += Horizontal_Sum_Neon(acc_neon);
#endif
                
                // 标量回退处理剩余元素
                for (; n < dstate; n++) {
                    float x_val = deltaA_ptr[dA_base + n] * x_ptr[x_base + n] + deltaB_u_ptr[dA_base + n];
                    x_ptr[x_base + n] = x_val;
                    
                    float c_val;
                    if (is_variable_C) {
                        c_val = C_ptr[(b * seq_len + i) * dstate + n];
                    } else {
                        c_val = C_ptr[d * dstate + n];
                    }
                    sum += x_val * c_val;
                }
                
                y_ptr[y_idx] = sum;
            }
        }
    }
    
    // 后处理：添加D项
    torch::Tensor out;
    if (is_none(D_param)) {
        out = y;
    } else {
        out = y + u_f * D_param.to(torch::kFloat32).unsqueeze(0).unsqueeze(-1);
    }
    
    // 应用z门控
    if (!is_none(z)) {
        out = out * torch::nn::functional::silu(z.to(torch::kFloat32));
    }
    
    // 转换回原始数据类型
    out = out.to(dtype_in);
    
    return out;
}

// SIMD优化 + Fixlen两阶段算法（N维度向量化）
torch::Tensor Selective_Scan_Simd_Fixlen_Cpu(
    const torch::Tensor& u,              // (B, D, L)
    const torch::Tensor& delta,          // (B, D, L)
    const torch::Tensor& A,              // (D, N)
    const torch::Tensor& B,              // (D, N) or (B, N, L)
    const torch::Tensor& C,              // (D, N) or (B, N, L)
    const torch::Tensor& D_param = torch::Tensor(),   // (D,) - 可选
    const torch::Tensor& z = torch::Tensor(),         // (B, D, L) - 可选
    const torch::Tensor& delta_bias = torch::Tensor(),  // (D,) - 可选
    bool delta_softplus = false,
    bool return_last_state = false
) {
    // 数据类型转换和预处理
    auto dtype_in = u.dtype();
    auto u_f = u.to(torch::kFloat32).contiguous();
    auto delta_f = delta.to(torch::kFloat32).contiguous();
    
    if (!is_none(delta_bias)) {
        delta_f = delta_f + delta_bias.unsqueeze(-1).to(torch::kFloat32);
    }
    if (delta_softplus) {
        delta_f = torch::nn::functional::softplus(delta_f);
    }
    
    const int64_t batch = u_f.size(0);
    const int64_t dim = A.size(0);
    const int64_t dstate = A.size(1);
    const int64_t seq_len = u_f.size(2);
    
    bool is_variable_B = B.dim() >= 3;
    bool is_variable_C = C.dim() >= 3;
    
    auto B_f = B.to(torch::kFloat32).contiguous();
    auto C_f = C.to(torch::kFloat32).contiguous();
    
    // 计算deltaA: (B, D, L, N)
    auto deltaA = torch::exp(
        delta_f.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2)
    ).contiguous();
    
    // 计算deltaB_u: (B, D, L, N)
    torch::Tensor deltaB_u;
    if (!is_variable_B) {
        deltaB_u = (delta_f.unsqueeze(-1) * B_f.unsqueeze(0).unsqueeze(2) * u_f.unsqueeze(-1)).contiguous();
    } else {
        if (B_f.dim() == 3) {
            deltaB_u = (delta_f.unsqueeze(-1) * B_f.unsqueeze(1).transpose(2, 3) * u_f.unsqueeze(-1)).contiguous();
        } else {
            throw std::runtime_error("Grouped B not supported in SIMD version");
        }
    }
    
    // 获取数据指针
    float* deltaA_ptr = deltaA.data_ptr<float>();
    float* deltaB_u_ptr = deltaB_u.data_ptr<float>();
    
    // 阶段1：SIMD递推计算（原地修改deltaB_u）
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t d = 0; d < dim; d++) {
            for (int64_t i = 1; i < seq_len; i++) {
                const int64_t curr_base = ((b * dim + d) * seq_len + i) * dstate;
                const int64_t prev_base = ((b * dim + d) * seq_len + i - 1) * dstate;
                
                int64_t n = 0;
#if defined(HAS_AVX512)
                for (; n + 16 <= dstate; n += 16) {
                    __m512 prev_v = _mm512_loadu_ps(&deltaB_u_ptr[prev_base + n]);
                    __m512 dA_v = _mm512_loadu_ps(&deltaA_ptr[curr_base + n]);
                    __m512 curr_v = _mm512_loadu_ps(&deltaB_u_ptr[curr_base + n]);
                    curr_v = _mm512_fmadd_ps(dA_v, prev_v, curr_v);
                    _mm512_storeu_ps(&deltaB_u_ptr[curr_base + n], curr_v);
                }
#endif
#if defined(HAS_AVX) || defined(_MSC_VER)
                for (; n + 8 <= dstate; n += 8) {
                    __m256 prev_v = _mm256_loadu_ps(&deltaB_u_ptr[prev_base + n]);
                    __m256 dA_v = _mm256_loadu_ps(&deltaA_ptr[curr_base + n]);
                    __m256 curr_v = _mm256_loadu_ps(&deltaB_u_ptr[curr_base + n]);
                    #if defined(__FMA__)
                    curr_v = _mm256_fmadd_ps(dA_v, prev_v, curr_v);
                    #else
                    curr_v = _mm256_add_ps(_mm256_mul_ps(dA_v, prev_v), curr_v);
                    #endif
                    _mm256_storeu_ps(&deltaB_u_ptr[curr_base + n], curr_v);
                }
#endif
#if defined(HAS_SSE)
                for (; n + 4 <= dstate; n += 4) {
                    __m128 prev_v = _mm_loadu_ps(&deltaB_u_ptr[prev_base + n]);
                    __m128 dA_v = _mm_loadu_ps(&deltaA_ptr[curr_base + n]);
                    __m128 curr_v = _mm_loadu_ps(&deltaB_u_ptr[curr_base + n]);
                    curr_v = _mm_add_ps(_mm_mul_ps(dA_v, prev_v), curr_v);
                    _mm_storeu_ps(&deltaB_u_ptr[curr_base + n], curr_v);
                }
#endif
#if defined(HAS_NEON)
                for (; n + 4 <= dstate; n += 4) {
                    float32x4_t prev_v = vld1q_f32(&deltaB_u_ptr[prev_base + n]);
                    float32x4_t dA_v = vld1q_f32(&deltaA_ptr[curr_base + n]);
                    float32x4_t curr_v = vld1q_f32(&deltaB_u_ptr[curr_base + n]);
                    curr_v = vmlaq_f32(curr_v, dA_v, prev_v);
                    vst1q_f32(&deltaB_u_ptr[curr_base + n], curr_v);
                }
#endif
                // 标量回退
                for (; n < dstate; n++) {
                    deltaB_u_ptr[curr_base + n] += deltaA_ptr[curr_base + n] * deltaB_u_ptr[prev_base + n];
                }
            }
        }
    }
    
    // 阶段2：SIMD批量计算输出（替换原来的torch::sum）
    // === 原始实现（已注释，保留用于回退） ===
    // torch::Tensor y;
    // if (!is_variable_C) {
    //     y = torch::sum(deltaB_u * C_f.unsqueeze(0).unsqueeze(2), -1);
    // } else {
    //     if (C_f.dim() == 3) {
    //         y = torch::sum(deltaB_u * C_f.unsqueeze(1).transpose(2, 3), -1);
    //     } else {
    //         throw std::runtime_error("Grouped C not supported");
    //     }
    // }
    // === SIMD优化实现 ===
    
    // 准备C矩阵指针
    torch::Tensor C_t;
    if (is_variable_C) {
        if (C_f.dim() == 3) {
            C_t = C_f.transpose(1, 2).contiguous();  // (B, N, L) -> (B, L, N)
        } else {
            throw std::runtime_error("Grouped C not supported in SIMD version");
        }
    }
    float* C_ptr = is_variable_C ? C_t.data_ptr<float>() : C_f.data_ptr<float>();
    
    // 分配输出tensor
    auto y = torch::empty({batch, dim, seq_len}, deltaB_u.options());
    float* y_ptr = y.data_ptr<float>();
    
    // SIMD阶段2：计算输出 y = sum(deltaB_u * C, -1)
    #if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t d = 0; d < dim; d++) {
            for (int64_t l = 0; l < seq_len; l++) {
                const int64_t x_base = ((b * dim + d) * seq_len + l) * dstate;
                const int64_t y_idx = (b * dim + d) * seq_len + l;
                
                float sum = 0.0f;
                int64_t n = 0;
                
#if defined(HAS_AVX512)
                __m512 acc_512 = _mm512_setzero_ps();
                for (; n + 16 <= dstate; n += 16) {
                    __m512 x_v = _mm512_loadu_ps(&deltaB_u_ptr[x_base + n]);
                    __m512 c_v;
                    if (is_variable_C) {
                        const int64_t c_idx = (b * seq_len + l) * dstate + n;
                        c_v = _mm512_loadu_ps(&C_ptr[c_idx]);
                    } else {
                        const int64_t c_idx = d * dstate + n;
                        c_v = _mm512_loadu_ps(&C_ptr[c_idx]);
                    }
                    acc_512 = _mm512_fmadd_ps(x_v, c_v, acc_512);
                }
                sum += Horizontal_Sum_Avx512(acc_512);
#endif

#if defined(HAS_AVX) || defined(_MSC_VER)
                __m256 acc_256 = _mm256_setzero_ps();
                for (; n + 8 <= dstate; n += 8) {
                    __m256 x_v = _mm256_loadu_ps(&deltaB_u_ptr[x_base + n]);
                    __m256 c_v;
                    if (is_variable_C) {
                        const int64_t c_idx = (b * seq_len + l) * dstate + n;
                        c_v = _mm256_loadu_ps(&C_ptr[c_idx]);
                    } else {
                        const int64_t c_idx = d * dstate + n;
                        c_v = _mm256_loadu_ps(&C_ptr[c_idx]);
                    }
                    #if defined(__FMA__)
                    acc_256 = _mm256_fmadd_ps(x_v, c_v, acc_256);
                    #else
                    acc_256 = _mm256_add_ps(acc_256, _mm256_mul_ps(x_v, c_v));
                    #endif
                }
                sum += Horizontal_Sum_Avx(acc_256);
#endif

#if defined(HAS_SSE)
                __m128 acc_128 = _mm_setzero_ps();
                for (; n + 4 <= dstate; n += 4) {
                    __m128 x_v = _mm_loadu_ps(&deltaB_u_ptr[x_base + n]);
                    __m128 c_v;
                    if (is_variable_C) {
                        const int64_t c_idx = (b * seq_len + l) * dstate + n;
                        c_v = _mm_loadu_ps(&C_ptr[c_idx]);
                    } else {
                        const int64_t c_idx = d * dstate + n;
                        c_v = _mm_loadu_ps(&C_ptr[c_idx]);
                    }
                    acc_128 = _mm_add_ps(acc_128, _mm_mul_ps(x_v, c_v));
                }
                sum += Horizontal_Sum_Sse(acc_128);
#endif

#if defined(HAS_NEON)
                float32x4_t acc_neon = vdupq_n_f32(0.0f);
                for (; n + 4 <= dstate; n += 4) {
                    float32x4_t x_v = vld1q_f32(&deltaB_u_ptr[x_base + n]);
                    float32x4_t c_v;
                    if (is_variable_C) {
                        const int64_t c_idx = (b * seq_len + l) * dstate + n;
                        c_v = vld1q_f32(&C_ptr[c_idx]);
                    } else {
                        const int64_t c_idx = d * dstate + n;
                        c_v = vld1q_f32(&C_ptr[c_idx]);
                    }
                    acc_neon = vmlaq_f32(acc_neon, x_v, c_v);
                }
                sum += Horizontal_Sum_Neon(acc_neon);
#endif
                
                // 标量回退处理剩余元素
                for (; n < dstate; n++) {
                    float c_val;
                    if (is_variable_C) {
                        c_val = C_ptr[(b * seq_len + l) * dstate + n];
                    } else {
                        c_val = C_ptr[d * dstate + n];
                    }
                    sum += deltaB_u_ptr[x_base + n] * c_val;
                }
                
                y_ptr[y_idx] = sum;
            }
        }
    }
    
    // 后处理
    torch::Tensor out;
    if (is_none(D_param)) {
        out = y;
    } else {
        out = y + u_f * D_param.to(torch::kFloat32).unsqueeze(0).unsqueeze(-1);
    }
    
    if (!is_none(z)) {
        out = out * torch::nn::functional::silu(z.to(torch::kFloat32));
    }
    
    return out.to(dtype_in);
}

// SIMD优化 + Fused双向扫描（N维度向量化，逐步计算）
torch::Tensor Selective_Fused_Scan_Simd_Cpu(
    const torch::Tensor& dt_fwd,            // (B, D, L)
    const torch::Tensor& dt_bwd,            // (B, D, L)
    const torch::Tensor& A_fwd,             // (D, N)
    const torch::Tensor& A_bwd,             // (D, N)
    const torch::Tensor& B_fwd,             // (B, N, L)
    const torch::Tensor& B_bwd,             // (B, N, L)
    const torch::Tensor& x_fwd_conv,        // (B, D, L)
    const torch::Tensor& x_bwd_conv_flip,   // (B, D, L)
    const torch::Tensor& C_fwd,             // (B, N, L)
    const torch::Tensor& C_bwd,             // (B, N, L)
    const torch::Tensor& D_fwd,             // (D,)
    const torch::Tensor& D_bwd,             // (D,)
    const torch::Tensor& z_fwd = torch::Tensor(),
    const torch::Tensor& z_bwd_flip = torch::Tensor()
) {
    auto dtype_in = dt_fwd.dtype();
    
    const int64_t batch = dt_fwd.size(0);
    const int64_t dim = dt_fwd.size(1);
    const int64_t seq_len = dt_fwd.size(2);
    const int64_t dstate = A_fwd.size(1);
    
    // 计算deltaA和deltaB_u（使用(B,D,L,2N)布局）
    auto deltaA_bi = torch::empty({batch, dim, seq_len, 2 * dstate}, dt_fwd.options().dtype(torch::kFloat32)).contiguous();
    auto deltaB_u_bi = torch::empty({batch, dim, seq_len, 2 * dstate}, dt_fwd.options().dtype(torch::kFloat32)).contiguous();
    
    // 正向部分 [:,:,:,:n]
    deltaA_bi.narrow(3, 0, dstate).copy_(
        torch::exp(dt_fwd.unsqueeze(-1) * A_fwd.unsqueeze(0).unsqueeze(2))
    );
    deltaB_u_bi.narrow(3, 0, dstate).copy_(
        dt_fwd.unsqueeze(-1) * B_fwd.unsqueeze(1).transpose(2, 3) * x_fwd_conv.unsqueeze(-1)
    );
    
    // 反向部分 [:,:,:,n:]
    deltaA_bi.narrow(3, dstate, dstate).copy_(
        torch::exp(dt_bwd.unsqueeze(-1) * A_bwd.unsqueeze(0).unsqueeze(2))
    );
    deltaB_u_bi.narrow(3, dstate, dstate).copy_(
        dt_bwd.unsqueeze(-1) * B_bwd.unsqueeze(1).transpose(2, 3) * x_bwd_conv_flip.unsqueeze(-1)
    );
    
    // C矩阵转置为(B,L,N)
    auto C_fwd_t = C_fwd.transpose(1, 2).contiguous();  // (B, L, N)
    auto C_bwd_t = C_bwd.transpose(1, 2).contiguous();  // (B, L, N)
    
    // 分配输出和隐藏状态
    auto y_fwd = torch::empty({batch, dim, seq_len}, dt_fwd.options().dtype(torch::kFloat32));
    auto y_bwd = torch::empty({batch, dim, seq_len}, dt_fwd.options().dtype(torch::kFloat32));
    auto x_bi = torch::zeros({batch, dim, 2 * dstate}, dt_fwd.options().dtype(torch::kFloat32)).contiguous();
    
    // 获取数据指针
    float* deltaA_ptr = deltaA_bi.data_ptr<float>();
    float* deltaB_u_ptr = deltaB_u_bi.data_ptr<float>();
    float* x_ptr = x_bi.data_ptr<float>();
    float* C_fwd_ptr = C_fwd_t.data_ptr<float>();
    float* C_bwd_ptr = C_bwd_t.data_ptr<float>();
    float* y_fwd_ptr = y_fwd.data_ptr<float>();
    float* y_bwd_ptr = y_bwd.data_ptr<float>();
    
    const int64_t dstate2 = 2 * dstate;
    
    // ========== 方案：只融合状态更新 ==========
    // 状态更新：一次处理2N个连续元素（fwd+bwd一起更新）
    // 输出计算：分别计算fwd和bwd的输出
    // 好处：状态更新利用2N=32个元素，更好利用SIMD宽度；内存访问连续
    
    #if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t d = 0; d < dim; d++) {
            const int64_t x_base = (b * dim + d) * dstate2;  // 隐藏状态基址 (2N连续)
            
            for (int64_t i = 0; i < seq_len; i++) {
                const int64_t dA_base = ((b * dim + d) * seq_len + i) * dstate2;
                const int64_t y_idx = (b * dim + d) * seq_len + i;
                const int64_t c_base = (b * seq_len + i) * dstate;
                
                // === 阶段1：融合状态更新（一次处理2N个元素） ===
                int64_t n = 0;
#if defined(HAS_AVX512)
                for (; n + 16 <= dstate2; n += 16) {
                    __m512 x_v = _mm512_loadu_ps(&x_ptr[x_base + n]);
                    __m512 dA_v = _mm512_loadu_ps(&deltaA_ptr[dA_base + n]);
                    __m512 dBu_v = _mm512_loadu_ps(&deltaB_u_ptr[dA_base + n]);
                    x_v = _mm512_fmadd_ps(dA_v, x_v, dBu_v);
                    _mm512_storeu_ps(&x_ptr[x_base + n], x_v);
                }
#endif
#if defined(HAS_AVX) || defined(_MSC_VER)
                for (; n + 8 <= dstate2; n += 8) {
                    __m256 x_v = _mm256_loadu_ps(&x_ptr[x_base + n]);
                    __m256 dA_v = _mm256_loadu_ps(&deltaA_ptr[dA_base + n]);
                    __m256 dBu_v = _mm256_loadu_ps(&deltaB_u_ptr[dA_base + n]);
                    #if defined(__FMA__)
                    x_v = _mm256_fmadd_ps(dA_v, x_v, dBu_v);
                    #else
                    x_v = _mm256_add_ps(_mm256_mul_ps(dA_v, x_v), dBu_v);
                    #endif
                    _mm256_storeu_ps(&x_ptr[x_base + n], x_v);
                }
#endif
#if defined(HAS_SSE)
                for (; n + 4 <= dstate2; n += 4) {
                    __m128 x_v = _mm_loadu_ps(&x_ptr[x_base + n]);
                    __m128 dA_v = _mm_loadu_ps(&deltaA_ptr[dA_base + n]);
                    __m128 dBu_v = _mm_loadu_ps(&deltaB_u_ptr[dA_base + n]);
                    x_v = _mm_add_ps(_mm_mul_ps(dA_v, x_v), dBu_v);
                    _mm_storeu_ps(&x_ptr[x_base + n], x_v);
                }
#endif
#if defined(HAS_NEON)
                for (; n + 4 <= dstate2; n += 4) {
                    float32x4_t x_v = vld1q_f32(&x_ptr[x_base + n]);
                    float32x4_t dA_v = vld1q_f32(&deltaA_ptr[dA_base + n]);
                    float32x4_t dBu_v = vld1q_f32(&deltaB_u_ptr[dA_base + n]);
                    x_v = vmlaq_f32(dBu_v, dA_v, x_v);
                    vst1q_f32(&x_ptr[x_base + n], x_v);
                }
#endif
                // 标量处理剩余元素
                for (; n < dstate2; n++) {
                    x_ptr[x_base + n] = deltaA_ptr[dA_base + n] * x_ptr[x_base + n] + deltaB_u_ptr[dA_base + n];
                }
                
                // === 阶段2：分别计算输出 ===
                // 正向输出：y_fwd = sum(x_fwd * C_fwd)
                float sum_fwd = 0.0f;
                n = 0;
#if defined(HAS_AVX512)
                __m512 acc_fwd_512 = _mm512_setzero_ps();
                for (; n + 16 <= dstate; n += 16) {
                    __m512 x_v = _mm512_loadu_ps(&x_ptr[x_base + n]);  // x_fwd部分
                    __m512 c_v = _mm512_loadu_ps(&C_fwd_ptr[c_base + n]);
                    acc_fwd_512 = _mm512_fmadd_ps(x_v, c_v, acc_fwd_512);
                }
                sum_fwd = Horizontal_Sum_Avx512(acc_fwd_512);
#endif
#if defined(HAS_AVX) || defined(_MSC_VER)
                __m256 acc_fwd_256 = _mm256_setzero_ps();
                for (; n + 8 <= dstate; n += 8) {
                    __m256 x_v = _mm256_loadu_ps(&x_ptr[x_base + n]);
                    __m256 c_v = _mm256_loadu_ps(&C_fwd_ptr[c_base + n]);
                    #if defined(__FMA__)
                    acc_fwd_256 = _mm256_fmadd_ps(x_v, c_v, acc_fwd_256);
                    #else
                    acc_fwd_256 = _mm256_add_ps(acc_fwd_256, _mm256_mul_ps(x_v, c_v));
                    #endif
                }
                sum_fwd += Horizontal_Sum_Avx(acc_fwd_256);
#endif
#if defined(HAS_SSE)
                __m128 acc_fwd_128 = _mm_setzero_ps();
                for (; n + 4 <= dstate; n += 4) {
                    __m128 x_v = _mm_loadu_ps(&x_ptr[x_base + n]);
                    __m128 c_v = _mm_loadu_ps(&C_fwd_ptr[c_base + n]);
                    acc_fwd_128 = _mm_add_ps(acc_fwd_128, _mm_mul_ps(x_v, c_v));
                }
                sum_fwd += Horizontal_Sum_Sse(acc_fwd_128);
#endif
#if defined(HAS_NEON)
                float32x4_t acc_fwd_neon = vdupq_n_f32(0.0f);
                for (; n + 4 <= dstate; n += 4) {
                    float32x4_t x_v = vld1q_f32(&x_ptr[x_base + n]);
                    float32x4_t c_v = vld1q_f32(&C_fwd_ptr[c_base + n]);
                    acc_fwd_neon = vmlaq_f32(acc_fwd_neon, x_v, c_v);
                }
                sum_fwd += Horizontal_Sum_Neon(acc_fwd_neon);
#endif
                for (; n < dstate; n++) {
                    sum_fwd += x_ptr[x_base + n] * C_fwd_ptr[c_base + n];
                }
                
                // 反向输出：y_bwd = sum(x_bwd * C_bwd)
                float sum_bwd = 0.0f;
                const int64_t x_base_bwd = x_base + dstate;  // x_bwd在x_fwd之后
                n = 0;
#if defined(HAS_AVX512)
                __m512 acc_bwd_512 = _mm512_setzero_ps();
                for (; n + 16 <= dstate; n += 16) {
                    __m512 x_v = _mm512_loadu_ps(&x_ptr[x_base_bwd + n]);
                    __m512 c_v = _mm512_loadu_ps(&C_bwd_ptr[c_base + n]);
                    acc_bwd_512 = _mm512_fmadd_ps(x_v, c_v, acc_bwd_512);
                }
                sum_bwd = Horizontal_Sum_Avx512(acc_bwd_512);
#endif
#if defined(HAS_AVX) || defined(_MSC_VER)
                __m256 acc_bwd_256 = _mm256_setzero_ps();
                for (; n + 8 <= dstate; n += 8) {
                    __m256 x_v = _mm256_loadu_ps(&x_ptr[x_base_bwd + n]);
                    __m256 c_v = _mm256_loadu_ps(&C_bwd_ptr[c_base + n]);
                    #if defined(__FMA__)
                    acc_bwd_256 = _mm256_fmadd_ps(x_v, c_v, acc_bwd_256);
                    #else
                    acc_bwd_256 = _mm256_add_ps(acc_bwd_256, _mm256_mul_ps(x_v, c_v));
                    #endif
                }
                sum_bwd += Horizontal_Sum_Avx(acc_bwd_256);
#endif
#if defined(HAS_SSE)
                __m128 acc_bwd_128 = _mm_setzero_ps();
                for (; n + 4 <= dstate; n += 4) {
                    __m128 x_v = _mm_loadu_ps(&x_ptr[x_base_bwd + n]);
                    __m128 c_v = _mm_loadu_ps(&C_bwd_ptr[c_base + n]);
                    acc_bwd_128 = _mm_add_ps(acc_bwd_128, _mm_mul_ps(x_v, c_v));
                }
                sum_bwd += Horizontal_Sum_Sse(acc_bwd_128);
#endif
#if defined(HAS_NEON)
                float32x4_t acc_bwd_neon = vdupq_n_f32(0.0f);
                for (; n + 4 <= dstate; n += 4) {
                    float32x4_t x_v = vld1q_f32(&x_ptr[x_base_bwd + n]);
                    float32x4_t c_v = vld1q_f32(&C_bwd_ptr[c_base + n]);
                    acc_bwd_neon = vmlaq_f32(acc_bwd_neon, x_v, c_v);
                }
                sum_bwd += Horizontal_Sum_Neon(acc_bwd_neon);
#endif
                for (; n < dstate; n++) {
                    sum_bwd += x_ptr[x_base_bwd + n] * C_bwd_ptr[c_base + n];
                }
                
                y_fwd_ptr[y_idx] = sum_fwd;
                y_bwd_ptr[y_idx] = sum_bwd;
            }
        }
    }
    
    // 添加D项
    y_fwd = y_fwd + x_fwd_conv * D_fwd.unsqueeze(0).unsqueeze(-1);
    y_bwd = y_bwd + x_bwd_conv_flip * D_bwd.unsqueeze(0).unsqueeze(-1);
    
    // 门控
    if (!is_none(z_fwd)) {
        y_fwd = y_fwd * torch::nn::functional::silu(z_fwd);
    }
    if (!is_none(z_bwd_flip)) {
        y_bwd = y_bwd * torch::nn::functional::silu(z_bwd_flip);
    }
    
    // 反转并合并输出
    y_bwd = y_bwd.flip({2});
    auto out = y_fwd + y_bwd;
    
    return out.to(dtype_in);
}

// SIMD优化 + Fused双向 + Fixlen两阶段（最高优化级别）
torch::Tensor Selective_Fused_Scan_Simd_Fixlen_Cpu(
    const torch::Tensor& dt_fwd,            // (B, D, L)
    const torch::Tensor& dt_bwd,            // (B, D, L)
    const torch::Tensor& A_fwd,             // (D, N)
    const torch::Tensor& A_bwd,             // (D, N)
    const torch::Tensor& B_fwd,             // (B, N, L)
    const torch::Tensor& B_bwd,             // (B, N, L)
    const torch::Tensor& x_fwd_conv,        // (B, D, L)
    const torch::Tensor& x_bwd_conv_flip,   // (B, D, L)
    const torch::Tensor& C_fwd,             // (B, N, L)
    const torch::Tensor& C_bwd,             // (B, N, L)
    const torch::Tensor& D_fwd,             // (D,)
    const torch::Tensor& D_bwd,             // (D,)
    const torch::Tensor& z_fwd = torch::Tensor(),
    const torch::Tensor& z_bwd_flip = torch::Tensor()
) {
    auto dtype_in = dt_fwd.dtype();
    
    const int64_t batch = dt_fwd.size(0);
    const int64_t dim = dt_fwd.size(1);
    const int64_t seq_len = dt_fwd.size(2);
    const int64_t dstate = A_fwd.size(1);
    
    // 使用(B,D,L,2N)布局
    auto deltaA_bi = torch::empty({batch, dim, seq_len, 2 * dstate}, dt_fwd.options().dtype(torch::kFloat32)).contiguous();
    auto deltaB_u_bi = torch::empty({batch, dim, seq_len, 2 * dstate}, dt_fwd.options().dtype(torch::kFloat32)).contiguous();
    
    // 正向部分
    deltaA_bi.narrow(3, 0, dstate).copy_(
        torch::exp(dt_fwd.unsqueeze(-1) * A_fwd.unsqueeze(0).unsqueeze(2))
    );
    deltaB_u_bi.narrow(3, 0, dstate).copy_(
        dt_fwd.unsqueeze(-1) * B_fwd.unsqueeze(1).transpose(2, 3) * x_fwd_conv.unsqueeze(-1)
    );
    
    // 反向部分
    deltaA_bi.narrow(3, dstate, dstate).copy_(
        torch::exp(dt_bwd.unsqueeze(-1) * A_bwd.unsqueeze(0).unsqueeze(2))
    );
    deltaB_u_bi.narrow(3, dstate, dstate).copy_(
        dt_bwd.unsqueeze(-1) * B_bwd.unsqueeze(1).transpose(2, 3) * x_bwd_conv_flip.unsqueeze(-1)
    );
    
    // 获取数据指针
    float* deltaA_ptr = deltaA_bi.data_ptr<float>();
    float* deltaB_u_ptr = deltaB_u_bi.data_ptr<float>();
    const int64_t dstate2 = 2 * dstate;
    
    // 阶段1：SIMD递推计算（原地修改deltaB_u_bi）
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t d = 0; d < dim; d++) {
            for (int64_t i = 1; i < seq_len; i++) {
                const int64_t curr_base = ((b * dim + d) * seq_len + i) * dstate2;
                const int64_t prev_base = ((b * dim + d) * seq_len + i - 1) * dstate2;
                
                int64_t n = 0;
#if defined(HAS_AVX512)

                for (; n + 16 <= dstate2; n += 16) {
                    __m512 prev_v = _mm512_loadu_ps(&deltaB_u_ptr[prev_base + n]);
                    __m512 dA_v = _mm512_loadu_ps(&deltaA_ptr[curr_base + n]);
                    __m512 curr_v = _mm512_loadu_ps(&deltaB_u_ptr[curr_base + n]);
                    curr_v = _mm512_fmadd_ps(dA_v, prev_v, curr_v);
                    _mm512_storeu_ps(&deltaB_u_ptr[curr_base + n], curr_v);
                }
#endif
#if defined(HAS_AVX) || defined(_MSC_VER)
   
                for (; n + 8 <= dstate2; n += 8) {
                    __m256 prev_v = _mm256_loadu_ps(&deltaB_u_ptr[prev_base + n]);
                    __m256 dA_v = _mm256_loadu_ps(&deltaA_ptr[curr_base + n]);
                    __m256 curr_v = _mm256_loadu_ps(&deltaB_u_ptr[curr_base + n]);
                    #if defined(__FMA__)
                    curr_v = _mm256_fmadd_ps(dA_v, prev_v, curr_v);
                    #else
                    curr_v = _mm256_add_ps(_mm256_mul_ps(dA_v, prev_v), curr_v);
                    #endif
                    _mm256_storeu_ps(&deltaB_u_ptr[curr_base + n], curr_v);
                }
#endif
#if defined(HAS_SSE)

                for (; n + 4 <= dstate2; n += 4) {
                    __m128 prev_v = _mm_loadu_ps(&deltaB_u_ptr[prev_base + n]);
                    __m128 dA_v = _mm_loadu_ps(&deltaA_ptr[curr_base + n]);
                    __m128 curr_v = _mm_loadu_ps(&deltaB_u_ptr[curr_base + n]);
                    curr_v = _mm_add_ps(_mm_mul_ps(dA_v, prev_v), curr_v);
                    _mm_storeu_ps(&deltaB_u_ptr[curr_base + n], curr_v);
                }
#endif
#if defined(HAS_NEON)
                for (; n + 4 <= dstate2; n += 4) {
                    float32x4_t prev_v = vld1q_f32(&deltaB_u_ptr[prev_base + n]);
                    float32x4_t dA_v = vld1q_f32(&deltaA_ptr[curr_base + n]);
                    float32x4_t curr_v = vld1q_f32(&deltaB_u_ptr[curr_base + n]);
                    curr_v = vmlaq_f32(curr_v, dA_v, prev_v);
                    vst1q_f32(&deltaB_u_ptr[curr_base + n], curr_v);
                }
#endif
                for (; n < dstate2; n++) {
                    deltaB_u_ptr[curr_base + n] += deltaA_ptr[curr_base + n] * deltaB_u_ptr[prev_base + n];
                }
            }
        }
    }
    
    // 阶段2：SIMD计算输出（替换torch::sum）
    // C矩阵转置为(B,L,N)
    auto C_fwd_t = C_fwd.transpose(1, 2).contiguous();  // (B, L, N)
    auto C_bwd_t = C_bwd.transpose(1, 2).contiguous();  // (B, L, N)
    float* C_fwd_ptr = C_fwd_t.data_ptr<float>();
    float* C_bwd_ptr = C_bwd_t.data_ptr<float>();
    
    // 分配输出tensor
    auto y_fwd = torch::empty({batch, dim, seq_len}, dt_fwd.options().dtype(torch::kFloat32));
    auto y_bwd = torch::empty({batch, dim, seq_len}, dt_fwd.options().dtype(torch::kFloat32));
    float* y_fwd_ptr = y_fwd.data_ptr<float>();
    float* y_bwd_ptr = y_bwd.data_ptr<float>();
    
    // SIMD阶段2：计算输出 y_fwd = sum(deltaB_u_fwd * C_fwd), y_bwd = sum(deltaB_u_bwd * C_bwd)
    #if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t d = 0; d < dim; d++) {
            for (int64_t l = 0; l < seq_len; l++) {
                const int64_t x_base = ((b * dim + d) * seq_len + l) * dstate2;
                const int64_t y_idx = (b * dim + d) * seq_len + l;
                const int64_t c_base = (b * seq_len + l) * dstate;
                
                // 正向输出
                float sum_fwd = 0.0f;
                int64_t n = 0;
#if defined(HAS_AVX512)
                __m512 acc_fwd_512 = _mm512_setzero_ps();
                for (; n + 16 <= dstate; n += 16) {
                    __m512 x_v = _mm512_loadu_ps(&deltaB_u_ptr[x_base + n]);
                    __m512 c_v = _mm512_loadu_ps(&C_fwd_ptr[c_base + n]);
                    acc_fwd_512 = _mm512_fmadd_ps(x_v, c_v, acc_fwd_512);
                }
                sum_fwd = Horizontal_Sum_Avx512(acc_fwd_512);
#endif
#if defined(HAS_AVX) || defined(_MSC_VER)
                __m256 acc_fwd_256 = _mm256_setzero_ps();
                for (; n + 8 <= dstate; n += 8) {
                    __m256 x_v = _mm256_loadu_ps(&deltaB_u_ptr[x_base + n]);
                    __m256 c_v = _mm256_loadu_ps(&C_fwd_ptr[c_base + n]);
                    #if defined(__FMA__)
                    acc_fwd_256 = _mm256_fmadd_ps(x_v, c_v, acc_fwd_256);
                    #else
                    acc_fwd_256 = _mm256_add_ps(acc_fwd_256, _mm256_mul_ps(x_v, c_v));
                    #endif
                }
                sum_fwd += Horizontal_Sum_Avx(acc_fwd_256);
#endif
#if defined(HAS_SSE)
                __m128 acc_fwd_128 = _mm_setzero_ps();
                for (; n + 4 <= dstate; n += 4) {
                    __m128 x_v = _mm_loadu_ps(&deltaB_u_ptr[x_base + n]);
                    __m128 c_v = _mm_loadu_ps(&C_fwd_ptr[c_base + n]);
                    acc_fwd_128 = _mm_add_ps(acc_fwd_128, _mm_mul_ps(x_v, c_v));
                }
                sum_fwd += Horizontal_Sum_Sse(acc_fwd_128);
#endif
#if defined(HAS_NEON)
                float32x4_t acc_fwd_neon = vdupq_n_f32(0.0f);
                for (; n + 4 <= dstate; n += 4) {
                    float32x4_t x_v = vld1q_f32(&deltaB_u_ptr[x_base + n]);
                    float32x4_t c_v = vld1q_f32(&C_fwd_ptr[c_base + n]);
                    acc_fwd_neon = vmlaq_f32(acc_fwd_neon, x_v, c_v);
                }
                sum_fwd += Horizontal_Sum_Neon(acc_fwd_neon);
#endif
                for (; n < dstate; n++) {
                    sum_fwd += deltaB_u_ptr[x_base + n] * C_fwd_ptr[c_base + n];
                }
                
                // 反向输出
                float sum_bwd = 0.0f;
                const int64_t x_base_bwd = x_base + dstate;
                n = 0;
#if defined(HAS_AVX512)
                __m512 acc_bwd_512 = _mm512_setzero_ps();
                for (; n + 16 <= dstate; n += 16) {
                    __m512 x_v = _mm512_loadu_ps(&deltaB_u_ptr[x_base_bwd + n]);
                    __m512 c_v = _mm512_loadu_ps(&C_bwd_ptr[c_base + n]);
                    acc_bwd_512 = _mm512_fmadd_ps(x_v, c_v, acc_bwd_512);
                }
                sum_bwd = Horizontal_Sum_Avx512(acc_bwd_512);
#endif
#if defined(HAS_AVX) || defined(_MSC_VER)
                __m256 acc_bwd_256 = _mm256_setzero_ps();
                for (; n + 8 <= dstate; n += 8) {
                    __m256 x_v = _mm256_loadu_ps(&deltaB_u_ptr[x_base_bwd + n]);
                    __m256 c_v = _mm256_loadu_ps(&C_bwd_ptr[c_base + n]);
                    #if defined(__FMA__)
                    acc_bwd_256 = _mm256_fmadd_ps(x_v, c_v, acc_bwd_256);
                    #else
                    acc_bwd_256 = _mm256_add_ps(acc_bwd_256, _mm256_mul_ps(x_v, c_v));
                    #endif
                }
                sum_bwd += Horizontal_Sum_Avx(acc_bwd_256);
#endif
#if defined(HAS_SSE)
                __m128 acc_bwd_128 = _mm_setzero_ps();
                for (; n + 4 <= dstate; n += 4) {
                    __m128 x_v = _mm_loadu_ps(&deltaB_u_ptr[x_base_bwd + n]);
                    __m128 c_v = _mm_loadu_ps(&C_bwd_ptr[c_base + n]);
                    acc_bwd_128 = _mm_add_ps(acc_bwd_128, _mm_mul_ps(x_v, c_v));
                }
                sum_bwd += Horizontal_Sum_Sse(acc_bwd_128);
#endif
#if defined(HAS_NEON)
                float32x4_t acc_bwd_neon = vdupq_n_f32(0.0f);
                for (; n + 4 <= dstate; n += 4) {
                    float32x4_t x_v = vld1q_f32(&deltaB_u_ptr[x_base_bwd + n]);
                    float32x4_t c_v = vld1q_f32(&C_bwd_ptr[c_base + n]);
                    acc_bwd_neon = vmlaq_f32(acc_bwd_neon, x_v, c_v);
                }
                sum_bwd += Horizontal_Sum_Neon(acc_bwd_neon);
#endif
                for (; n < dstate; n++) {
                    sum_bwd += deltaB_u_ptr[x_base_bwd + n] * C_bwd_ptr[c_base + n];
                }
                
                y_fwd_ptr[y_idx] = sum_fwd;
                y_bwd_ptr[y_idx] = sum_bwd;
            }
        }
    }
    
    // 添加D项
    y_fwd = y_fwd + x_fwd_conv * D_fwd.unsqueeze(0).unsqueeze(-1);
    y_bwd = y_bwd + x_bwd_conv_flip * D_bwd.unsqueeze(0).unsqueeze(-1);
    
    // 门控
    if (!is_none(z_fwd)) {
        y_fwd = y_fwd * torch::nn::functional::silu(z_fwd);
    }
    if (!is_none(z_bwd_flip)) {
        y_bwd = y_bwd * torch::nn::functional::silu(z_bwd_flip);
    }
    
    // 反转并合并输出
    y_bwd = y_bwd.flip({2});
    auto out = y_fwd + y_bwd;
    
    return out.to(dtype_in);
}

// Python绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 信息查询函数
    m.def("get_openmp_info", &Get_Openmp_Info,
          "Get OpenMP configuration info (thread count, processor count)");
    m.def("get_simd_info", &Get_Simd_Info,
          "Get SIMD instruction set info (AVX512/AVX/SSE/NEON)");
    
    m.def("selective_scan", &Selective_Scan_Ref_Cpu,
          "Selective Scan CPU - complete replication of selective_scan_ref",
          py::arg("u"),
          py::arg("delta"),
          py::arg("A"),
          py::arg("B"),
          py::arg("C"),
          py::arg("D") = torch::Tensor(),
          py::arg("z") = torch::Tensor(),
          py::arg("delta_bias") = torch::Tensor(),
          py::arg("delta_softplus") = false,
          py::arg("return_last_state") = false);
    
    m.def("selective_scan_fixlen", &Selective_Scan_Ref_Fixlen_Cpu,
          "Selective Scan CPU - optimized two-stage algorithm",
          py::arg("u"),
          py::arg("delta"),
          py::arg("A"),
          py::arg("B"),
          py::arg("C"),
          py::arg("D") = torch::Tensor(),
          py::arg("z") = torch::Tensor(),
          py::arg("delta_bias") = torch::Tensor(),
          py::arg("delta_softplus") = false,
          py::arg("return_last_state") = false);
    
    m.def("selective_fused_scan", &Selective_Fused_Scan_Cpu,
          "Fused Bidirectional Selective Scan CPU - standard version",
          py::arg("dt_fwd"),
          py::arg("dt_bwd"),
          py::arg("A_fwd"),
          py::arg("A_bwd"),
          py::arg("B_fwd"),
          py::arg("B_bwd"),
          py::arg("x_fwd_conv"),
          py::arg("x_bwd_conv_flip"),
          py::arg("C_fwd"),
          py::arg("C_bwd"),
          py::arg("D_fwd"),
          py::arg("D_bwd"),
          py::arg("z_fwd") = torch::Tensor(),
          py::arg("z_bwd_flip") = torch::Tensor());
    
    m.def("selective_fused_scan_fixlen", &Selective_Fused_Scan_Fixlen_Cpu,
          "Fused Bidirectional Selective Scan CPU - optimized with select",
          py::arg("dt_fwd"),
          py::arg("dt_bwd"),
          py::arg("A_fwd"),
          py::arg("A_bwd"),
          py::arg("B_fwd"),
          py::arg("B_bwd"),
          py::arg("x_fwd_conv"),
          py::arg("x_bwd_conv_flip"),
          py::arg("C_fwd"),
          py::arg("C_bwd"),
          py::arg("D_fwd"),
          py::arg("D_bwd"),
          py::arg("z_fwd") = torch::Tensor(),
          py::arg("z_bwd_flip") = torch::Tensor());
    
    m.def("selective_scan_simd", &Selective_Scan_Simd_Cpu,
          "Selective Scan CPU - SIMD optimized (N dimension vectorized)",
          py::arg("u"),
          py::arg("delta"),
          py::arg("A"),
          py::arg("B"),
          py::arg("C"),
          py::arg("D") = torch::Tensor(),
          py::arg("z") = torch::Tensor(),
          py::arg("delta_bias") = torch::Tensor(),
          py::arg("delta_softplus") = false,
          py::arg("return_last_state") = false);
    
    m.def("selective_scan_simd_fixlen", &Selective_Scan_Simd_Fixlen_Cpu,
          "Selective Scan CPU - SIMD + Fixlen two-stage optimization",
          py::arg("u"),
          py::arg("delta"),
          py::arg("A"),
          py::arg("B"),
          py::arg("C"),
          py::arg("D") = torch::Tensor(),
          py::arg("z") = torch::Tensor(),
          py::arg("delta_bias") = torch::Tensor(),
          py::arg("delta_softplus") = false,
          py::arg("return_last_state") = false);
    
    m.def("selective_fused_scan_simd", &Selective_Fused_Scan_Simd_Cpu,
          "Fused Bidirectional Selective Scan CPU - SIMD optimized",
          py::arg("dt_fwd"),
          py::arg("dt_bwd"),
          py::arg("A_fwd"),
          py::arg("A_bwd"),
          py::arg("B_fwd"),
          py::arg("B_bwd"),
          py::arg("x_fwd_conv"),
          py::arg("x_bwd_conv_flip"),
          py::arg("C_fwd"),
          py::arg("C_bwd"),
          py::arg("D_fwd"),
          py::arg("D_bwd"),
          py::arg("z_fwd") = torch::Tensor(),
          py::arg("z_bwd_flip") = torch::Tensor());
    
    m.def("selective_fused_scan_simd_fixlen", &Selective_Fused_Scan_Simd_Fixlen_Cpu,
          "Fused Bidirectional Selective Scan CPU - SIMD + Fixlen optimization",
          py::arg("dt_fwd"),
          py::arg("dt_bwd"),
          py::arg("A_fwd"),
          py::arg("A_bwd"),
          py::arg("B_fwd"),
          py::arg("B_bwd"),
          py::arg("x_fwd_conv"),
          py::arg("x_bwd_conv_flip"),
          py::arg("C_fwd"),
          py::arg("C_bwd"),
          py::arg("D_fwd"),
          py::arg("D_bwd"),
          py::arg("z_fwd") = torch::Tensor(),
          py::arg("z_bwd_flip") = torch::Tensor());
}
