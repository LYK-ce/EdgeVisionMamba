# Presented by KeJi
# Date: 2026-01-06

# Selective Scan SIMD优化分析

## 1. 核心计算分析

### 1.1 递推公式
```
x[i] = deltaA[i] * x[i-1] + deltaB_u[i]    # 状态更新
y[i] = sum(x[i] * C[i], dim=-1)            # 输出计算
```

### 1.2 张量维度
| 变量 | 维度 | 说明 |
|------|------|------|
| deltaA | (B, D, L, N) | 状态转移系数 |
| deltaB_u | (B, D, L, N) | 输入项 |
| x | (B, D, N) | 隐藏状态 |
| C | (B, N, L) | 输出投影 |
| y | (B, D, L) | 输出 |

### 1.3 依赖关系
- **L维度**：存在循环依赖 `x[i]` 依赖 `x[i-1]`，不可并行
- **B维度**：batch间独立，可并行
- **D维度**：通道间独立，可并行
- **N维度**：状态维度独立，可并行

## 2. SIMD优化方案

### 2.1 目标维度
在**N维度**(dstate)上应用SIMD，原因：
1. 最内层循环，cache友好
2. 通常N=16，适合AVX2(256bit=8xfloat32)或AVX-512(512bit=16xfloat32)
3. 计算模式固定：乘加操作(FMA)

### 2.2 伪代码
```cpp
// 原始循环
for (int i = 0; i < L; i++) {
    for (int b = 0; b < B; b++) {
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < N; n++) {
                x[b][d][n] = deltaA[b][d][i][n] * x[b][d][n] + deltaB_u[b][d][i][n];
            }
        }
    }
}

// SIMD优化 (N维度向量化)
for (int i = 0; i < L; i++) {
    for (int b = 0; b < B; b++) {
        for (int d = 0; d < D; d++) {
            // N维度使用SIMD (假设N=16, AVX-512)
            __m512 x_vec = _mm512_load_ps(&x[b][d][0]);
            __m512 deltaA_vec = _mm512_load_ps(&deltaA[b][d][i][0]);
            __m512 deltaB_u_vec = _mm512_load_ps(&deltaB_u[b][d][i][0]);
            
            // x = deltaA * x + deltaB_u (FMA)
            x_vec = _mm512_fmadd_ps(deltaA_vec, x_vec, deltaB_u_vec);
            
            _mm512_store_ps(&x[b][d][0], x_vec);
        }
    }
}
```

### 2.3 输出计算SIMD化
```cpp
// y[i] = sum(x * C[i])
// N维度归约求和
__m512 x_vec = _mm512_load_ps(&x[b][d][0]);
__m512 c_vec = _mm512_load_ps(&C[b][0][i]);  // 需要预处理C布局为(B,L,N)
__m512 prod = _mm512_mul_ps(x_vec, c_vec);
float y_val = _mm512_reduce_add_ps(prod);    // 水平求和
```

## 3. 内存布局优化

### 3.1 当前布局问题
- deltaA/deltaB_u: (B, D, L, N) - N在最内层，SIMD友好
- C: (B, N, L) - N不在最内层，需转置

### 3.2 推荐布局
```
deltaA, deltaB_u: (B, D, L, N)  // 保持
C转置为: (B, L, N)              // 预处理
x: (B, D, N)                    // 保持，对齐到SIMD宽度
```

### 3.3 内存对齐
```cpp
// 确保N维度按SIMD宽度对齐
const int SIMD_WIDTH = 16;  // AVX-512
const int N_aligned = ((N + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;

// 分配对齐内存
float* x = (float*)aligned_alloc(64, B * D * N_aligned * sizeof(float));
```

## 4. 实现策略

### 4.1 纯C++实现 (使用Intrinsics)
```cpp
#include <immintrin.h>

void Selective_Scan_Simd(
    const float* deltaA,      // (B, D, L, N) contiguous
    const float* deltaB_u,    // (B, D, L, N) contiguous
    float* x,                 // (B, D, N) contiguous, 需要修改
    const float* C_t,         // (B, L, N) 转置后
    float* y,                 // (B, D, L) output
    int B, int D, int L, int N
) {
    const int N_vec = N / 16;  // AVX-512 向量数
    
    for (int i = 0; i < L; i++) {
        for (int b = 0; b < B; b++) {
            for (int d = 0; d < D; d++) {
                int x_offset = (b * D + d) * N;
                int dA_offset = ((b * D + d) * L + i) * N;
                
                // 状态更新 (N维度SIMD)
                for (int v = 0; v < N_vec; v++) {
                    __m512 x_v = _mm512_load_ps(&x[x_offset + v * 16]);
                    __m512 dA_v = _mm512_load_ps(&deltaA[dA_offset + v * 16]);
                    __m512 dBu_v = _mm512_load_ps(&deltaB_u[dA_offset + v * 16]);
                    x_v = _mm512_fmadd_ps(dA_v, x_v, dBu_v);
                    _mm512_store_ps(&x[x_offset + v * 16], x_v);
                }
                
                // 输出计算 (归约)
                __m512 sum = _mm512_setzero_ps();
                int C_offset = (b * L + i) * N;
                for (int v = 0; v < N_vec; v++) {
                    __m512 x_v = _mm512_load_ps(&x[x_offset + v * 16]);
                    __m512 c_v = _mm512_load_ps(&C_t[C_offset + v * 16]);
                    sum = _mm512_fmadd_ps(x_v, c_v, sum);
                }
                y[(b * D + d) * L + i] = _mm512_reduce_add_ps(sum);
            }
        }
    }
}
```

### 4.2 PyTorch C++扩展实现
```cpp
torch::Tensor Selective_Scan_Simd_Torch(
    const torch::Tensor& deltaA,      // (B, D, L, N)
    const torch::Tensor& deltaB_u,    // (B, D, L, N)
    const torch::Tensor& C,           // (B, N, L)
    const torch::Tensor& D_param,     // (D,)
    const torch::Tensor& u,           // (B, D, L)
    const torch::Tensor& z            // (B, D, L) optional
) {
    // 确保contiguous和float32
    auto dA = deltaA.contiguous();
    auto dBu = deltaB_u.contiguous();
    auto C_t = C.transpose(1, 2).contiguous();  // (B, L, N)
    
    const int64_t B = dA.size(0);
    const int64_t D = dA.size(1);
    const int64_t L = dA.size(2);
    const int64_t N = dA.size(3);
    
    // 分配对齐的x和y
    auto x = torch::zeros({B, D, N}, dA.options());
    auto y = torch::empty({B, D, L}, dA.options());
    
    // 获取数据指针
    float* dA_ptr = dA.data_ptr<float>();
    float* dBu_ptr = dBu.data_ptr<float>();
    float* C_ptr = C_t.data_ptr<float>();
    float* x_ptr = x.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();
    
    // 调用SIMD内核
    Selective_Scan_Simd(dA_ptr, dBu_ptr, x_ptr, C_ptr, y_ptr, B, D, L, N);
    
    // 后处理: y + u*D, z门控
    // ...
    
    return y;
}
```

## 5. 性能预期

### 5.1 理论加速
| 指令集 | 向量宽度 | N=16加速比 |
|--------|----------|------------|
| SSE | 4xf32 | ~4x |
| AVX2 | 8xf32 | ~8x |
| AVX-512 | 16xf32 | ~16x |

### 5.2 实际限制
1. **内存带宽**：L循环遍历大量数据，可能成为瓶颈
2. **归约开销**：`_mm512_reduce_add_ps`有延迟
3. **对齐要求**：N不是16倍数时需要mask操作

### 5.3 预期效果
- 当前Python实现：~100ms (推测)
- C++ Tensor操作：~30ms (已实现)
- C++ SIMD：~10-15ms (预期)
- 加速比：约2-3x相对于C++ Tensor

## 6. 实现建议

### 6.1 分步实现
1. **Phase 1**: 实现纯C++ SIMD内核，验证正确性
2. **Phase 2**: 集成到PyTorch扩展
3. **Phase 3**: 添加AVX2回退支持
4. **Phase 4**: 性能调优(prefetch, unroll等)

### 6.2 关键优化点
1. 使用FMA指令(`_mm512_fmadd_ps`)
2. 预取数据(`_mm_prefetch`)
3. 循环展开(针对D维度)
4. 避免分支(使用mask操作)

### 6.3 注意事项
1. 检测CPU支持的SIMD指令集
2. 提供fallback实现(纯标量)
3. 处理N不对齐的情况
4. 数值精度验证(FMA顺序可能影响结果)

## 7. 总结

SIMD优化策略是在**N维度(dstate)**上向量化计算，主要优化两个操作：
1. **状态更新**：`x = deltaA * x + deltaB_u`
2. **输出计算**：`y = sum(x * C)`

预期在现有C++实现基础上获得**2-3x**额外加速。
