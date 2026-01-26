# Presented by KeJi
# Date: 2026-01-25

# CPU Optimization Strategies for Vision Mamba

本文档介绍了为在CPU上高效运行Vision Mamba模型所采用的三种优化策略：**Loop Optimization**、**Bidirectional Scan Fusion**和**SIMD Optimization**。

---

## 1. Background: Selective Scan Algorithm

Vision Mamba的核心是Selective Scan算法，其递推公式为：

```
x[i] = deltaA[i] * x[i-1] + deltaB_u[i]    # 状态更新
y[i] = sum(x[i] * C[i], dim=-1)            # 输出计算
```

其中：
- `deltaA`: `(B, D, L, N)` - 状态转移系数
- `deltaB_u`: `(B, D, L, N)` - 输入项
- `x`: `(B, D, N)` - 隐藏状态
- `C`: `(B, N, L)` - 输出投影矩阵
- `y`: `(B, D, L)` - 输出

由于L维度存在循环依赖（`x[i]`依赖`x[i-1]`），无法直接并行化，这是优化的主要挑战。

---

## 2. Loop Optimization

### 2.1 Problem

原始实现在每个时间步同时计算状态更新和输出：

```python
for i in range(L):
    x = deltaA[:,:,i] * x + deltaB_u[:,:,i]  # 状态更新
    y = einsum('bdn,bn->bd', x, C[:,:,i])    # 输出计算
```

这导致每次迭代都需要执行复杂的矩阵运算（einsum），且无法充分利用向量化优化。

### 2.2 Solution

将计算分为两个阶段：

**Phase 1: State Propagation** - 仅进行状态递推，原地修改
```python
for i in range(1, L):
    deltaB_u[:,:,i] += deltaA[:,:,i] * deltaB_u[:,:,i-1]
```

**Phase 2: Batch Output Computation** - 一次性计算所有输出
```python
y = einsum('bdln,bnl->bdl', deltaB_u, C)
```

### 2.3 Benefits

1. **Phase 1**: 每步仅执行简单的乘加运算，计算量最小化
2. **Phase 2**: 完全向量化，充分利用BLAS库和SIMD指令
3. **减少循环开销**: 循环内操作更轻量

### 2.4 Speedup

- Python实现: 1.61x
- C++实现: 2.93x

---

## 3. Bidirectional Scan Fusion

### 3.1 Problem

Vision Mamba需要双向扫描（forward + backward），传统方法调用两次`selective_scan`：

```python
y_fwd = selective_scan(x, delta, A_fwd, B, C, ...)
y_bwd = selective_scan(x.flip(-1), delta.flip(-1), A_bwd, B.flip(-1), C.flip(-1), ...)
y = y_fwd + y_bwd.flip(-1)
```

这导致：
- 两次独立的循环遍历
- 重复的内存分配和数据准备
- 缓存利用率低

### 3.2 Solution

在N维度上拼接正向和反向参数，用单次扫描同时处理两个方向：

```python
# 融合正向和反向参数 (2N状态维度)
deltaA_bi = cat([deltaA_fwd, deltaA_bwd], dim=-1)      # (B,D,L,2N)
deltaB_u_bi = cat([deltaB_u_fwd, deltaB_u_bwd], dim=-1) # (B,D,L,2N)

# 单次递推循环同时处理两个方向
for i in range(1, L):
    deltaB_u_bi[:,:,i] += deltaA_bi[:,:,i] * deltaB_u_bi[:,:,i-1]

# 分别取出正向和反向输出
y_fwd = einsum('bdln,bnl->bdl', deltaB_u_bi[:,:,:,:N], C_fwd)
y_bwd = einsum('bdln,bnl->bdl', deltaB_u_bi[:,:,:,N:], C_bwd)
```

### 3.3 Memory Layout Optimization

采用`(B,D,L,2N)`布局：
- 与einsum输出直接匹配，无需permute
- 避免额外的内存拷贝开销

预分配缓冲区而非使用`torch.cat`：
```python
deltaA_bi = torch.empty(B, D, L, 2*N, ...)
deltaA_bi[:,:,:,:N] = exp(einsum('bdl,dn->bdln', dt_fwd, A_fwd))
deltaA_bi[:,:,:,N:] = exp(einsum('bdl,dn->bdln', dt_bwd, A_bwd))
```

### 3.4 Benefits

1. **减少循环遍历**: 从两次减少到一次
2. **更好的缓存利用**: 单一连续缓冲区
3. **减少Python解释器开销**: 更少的函数调用
4. **消除flip操作**: 反向数据预处理集成到循环中

### 3.5 Speedup

- Python实现: 2.45x
- C++实现: 3.60x

---

## 4. SIMD Optimization

### 4.1 Problem

即使使用C++实现，PyTorch Tensor操作仍有开销：
- 函数调用开销
- 内存分配检查
- 无法针对特定计算模式优化

### 4.2 Solution

直接使用SIMD intrinsics在N维度上向量化计算：

**目标维度选择**: N维度(dstate)，原因：
- 最内层维度，内存访问连续
- 典型值N=16，完美匹配AVX-512(16×float32)
- 计算模式固定：乘加操作(FMA)

**状态更新SIMD化**:
```cpp
// AVX-512: 一次处理16个float
for (int v = 0; v < N / 16; v++) {
    __m512 x_v = _mm512_load_ps(&x[x_offset + v * 16]);
    __m512 dA_v = _mm512_load_ps(&deltaA[dA_offset + v * 16]);
    __m512 dBu_v = _mm512_load_ps(&deltaB_u[dA_offset + v * 16]);
    
    // FMA: x = deltaA * x + deltaB_u
    x_v = _mm512_fmadd_ps(dA_v, x_v, dBu_v);
    
    _mm512_store_ps(&x[x_offset + v * 16], x_v);
}
```

**输出计算SIMD化**:
```cpp
// 水平归约: y = sum(x * C)
__m512 sum = _mm512_setzero_ps();
for (int v = 0; v < N / 16; v++) {
    __m512 x_v = _mm512_load_ps(&x[x_offset + v * 16]);
    __m512 c_v = _mm512_load_ps(&C_t[C_offset + v * 16]);
    sum = _mm512_fmadd_ps(x_v, c_v, sum);
}
y[y_offset] = _mm512_reduce_add_ps(sum);  // 水平求和
```

### 4.3 Cross-Platform Support

支持多种SIMD指令集：

| 指令集 | 向量宽度 | 平台 |
|--------|----------|------|
| AVX-512 | 16×f32 | Intel/AMD x86-64 |
| AVX2 | 8×f32 | Intel/AMD x86-64 |
| SSE | 4×f32 | x86/x86-64 |
| NEON | 4×f32 | ARM (树莓派/RK3588) |

自动检测CPU支持的指令集：
```cpp
#if defined(__AVX512F__)
    // AVX-512 path
#elif defined(__AVX__) || defined(__AVX2__)
    // AVX/AVX2 path
#elif defined(__SSE__) || defined(_M_X64)
    // SSE path
#elif defined(__ARM_NEON) || defined(__aarch64__)
    // ARM NEON path
#endif
```

### 4.4 Optimization Combinations

SIMD优化可与前两种策略组合：

| 配置 | 描述 |
|------|------|
| SIMD | 基础SIMD + 原始算法 |
| SIMD + Loop | SIMD + Loop Optimization |
| SIMD + Fusion | SIMD + Bidirectional Scan Fusion |
| SIMD + Loop + Fusion | 三种优化全部启用 (最佳) |

### 4.5 Benefits

1. **FMA指令**: 融合乘加，减少指令数量
2. **数据级并行**: 单条指令处理多个数据
3. **消除函数调用开销**: 直接操作原始内存
4. **优化内存访问**: 预取指令提升缓存命中

### 4.6 Speedup

相对于C++ Tensor实现额外获得约2-3x加速。

---


## 5. Performance Anomaly Analysis

### 5.1 观测到的现象

在实际测试中，SIMD与其他优化组合后，性能并不总是最优：

**RK3588 (ARM NEON)**:
| 配置 | vim_5m | vim_tiny | vim_10m |
|------|--------|----------|---------|
| SIMD | **549ms** | **731ms** | **792ms** |
| SIMD-Fixlen | 641ms | 852ms | 903ms |
| SIMD-Fused | 588ms | 748ms | 855ms |
| SIMD-Fused-Fixlen | 668ms | 806ms | 847ms |

**Laptop (x86 AVX2)**:
| 配置 | vim_5m | vim_tiny | vim_10m |
|------|--------|----------|---------|
| SIMD | 156ms | 199ms | **166ms** |
| SIMD-Fixlen | **125ms** | **152ms** | 173ms |
| SIMD-Fused | 130ms | 179ms | 206ms |
| SIMD-Fused-Fixlen | 125ms | 164ms | 203ms |

### 5.2 原因分析

#### 5.2.1 Loop Optimization的额外开销

Loop Optimization需要存储所有时间步的状态历史：

```python
# 原始SIMD: 只维护当前状态 x (B,D,N)
for i in range(L):
    x = deltaA * x + deltaB_u  # 原地更新

# SIMD + Loop: 需要存储state_history (B,D,L,N)
for i in range(L):
    state_history[:,:,i] = deltaA * state_history[:,:,i-1] + deltaB_u[:,:,i]
```

**内存开销对比**:
- 原始SIMD: `B × D × N × 4bytes`
- SIMD + Loop: `B × D × L × N × 4bytes` (L倍增长!)

对于vim_tiny (B=1, D=192, L=197, N=16):
- 原始: 12KB
- Loop: 2.4MB

**影响**: 当数据量超出L1/L2缓存时，内存访问变成瓶颈。

#### 5.2.2 Bidirectional Scan Fusion的状态膨胀

Fusion将状态维度从N扩展到2N：

```cpp
// 原始SIMD: N维度向量化
for (int n = 0; n < N; n += 16) {  // N=16时，1次迭代
    __m512 x_v = _mm512_load_ps(&x[n]);
    // ...
}

// SIMD + Fusion: 2N维度向量化
for (int n = 0; n < 2*N; n += 16) {  // N=16时，2次迭代
    __m512 x_v = _mm512_load_ps(&x_bi[n]);
    // ...
}
```

**影响**:
- 单次循环处理2N数据
- 缓存压力增加
- 当N=16恰好匹配AVX-512宽度时，Fusion反而需要2次加载

#### 5.2.3 平台差异 - SIMD不总是最优

**树莓派 (Cortex-A72, 4核)**:
| 配置 | vim_5m | vim_tiny | vim_10m |
|------|--------|----------|---------|
| **Python-Fused-Fixlen** | **2136ms** | **2931ms** | **2859ms** |
| FullCPP-Fixlen | 2786ms | 3534ms | 3323ms |
| SIMD | 3113ms | 4031ms | 3703ms |
| SIMD-Fused | 2887ms | 3704ms | 3016ms |

**关键发现**: 在树莓派上，SIMD (3113ms) 比 Python-Fused-Fixlen (2136ms) 慢 **46%**！

**原因分析**:
1. **PyTorch底层库深度优化**: Python版einsum调用的是OpenBLAS/BLIS等专业数学库
2. **专业BLAS库在低端ARM上更高效**: 这些库经过多年优化，包含：
   - 针对特定CPU微架构的手动调优
   - 复杂的循环展开和预取策略
   - 更好的指令调度
3. **手写SIMD的局限性**:
   - 我们的NEON实现相对简单
   - 缺少针对A72的专门优化
   - 内存布局转换的额外开销
4. **4核CPU算力瓶颈**: 树莓派算力有限，SIMD实现的开销更明显

**ARM高端 (RK3588, 8核 A76)**:
- 8核并行能力强
- A76核心性能远超A72
- 更大缓存和更高内存带宽
- **结论**: SIMD实现超过PyTorch库，纯SIMD最优

**x86 (Laptop AVX2)**:
- AVX2寄存器256-bit (8×float32)
- 内存带宽充足
- 大缓存
- **结论**: SIMD/SIMD-Fixlen最优

### 5.3 最佳实践建议

| 平台 | 小模型 (<10M) | 大模型 (>10M) |
|------|---------------|---------------|
| 树莓派 (低端ARM) | **Python-Fused-Fixlen** | **Python-Fused-Fixlen** |
| RK3588 (高端ARM) | SIMD | SIMD |
| x86 (AVX2) | SIMD-Fixlen | SIMD |

**选择原则**:
1. **低端ARM** (树莓派): 使用Python-Fused-Fixlen，依赖PyTorch底层BLAS优化
2. **高端ARM** (RK3588): 使用纯SIMD，手写NEON超过BLAS
3. **x86平台**: SIMD系列最优，小模型可用SIMD-Fixlen
4. **通用建议**: 实际部署前必须在目标平台benchmarking

---

## 6. Future Work: OpenBLAS Integration

### 6.1 为什么Python-Fused-Fixlen在树莓派上更快？

Python-Fused-Fixlen的关键优化是将输出计算batch化：

```python
# Loop Optimization Phase 2: 一次性输出计算
y = einsum('bdln,bnl->bdl', deltaB_u, C)
```

PyTorch的einsum底层会将此映射为**Batched GEMM**，并调用OpenBLAS/MKL的`cblas_sgemm_batch`。

### 6.2 能否在C++中直接调用OpenBLAS？

**可以！** Selective Scan的输出计算可以映射到BLAS：

```cpp
#include <cblas.h>

// einsum('bdln,bnl->bdl') 等价于多个矩阵乘法
// 对于每个 (b,d)，计算 y[l] = sum_n(deltaB_u[l,n] * C[n,l])

for (int b = 0; b < B; b++) {
    for (int d = 0; d < D; d++) {
        // deltaB_u[b,d]: (L, N) 矩阵
        // C[b]: (N, L) 矩阵
        // 输出 y[b,d]: (L,) 向量
        
        // 实际上是 (L,N) × (N,L) 的对角元素，等价于行向量点积
        for (int l = 0; l < L; l++) {
            y[b*D*L + d*L + l] = cblas_sdot(
                N,                                    // 向量长度
                &deltaB_u[b*D*L*N + d*L*N + l*N], 1, // deltaB_u[b,d,l,:]
                &C[b*N*L + l], L                      // C[b,:,l] (stride=L)
            );
        }
    }
}
```

### 6.3 潜在的优化方案

**方案1: BLAS + Loop混合**
```cpp
// Phase 1: 状态递推 (仍需循环，但可用SIMD)
for (int i = 1; i < L; i++) {
    // SIMD vectorized: deltaB_u[:,:,i] += deltaA[:,:,i] * deltaB_u[:,:,i-1]
}

// Phase 2: Batched GEMM输出
cblas_sgemm_batch(...);  // 或循环调用cblas_sdot
```

**方案2: 重构为标准GEMM**
```cpp
// 将 (B,D,L,N) 重排为 (B*D*L, N)
// 与 C (B, N, L) 进行批次矩阵乘法
// 需要额外的内存重排开销
```

### 6.4 为什么当前SIMD实现未调用OpenBLAS？

1. **循环依赖问题**: 状态递推 `x[i] = A*x[i-1] + B` 无法用BLAS表达
2. **数据布局不匹配**: BLAS期望标准矩阵布局，Selective Scan的4D张量需要转换
3. **额外开销**: 小规模计算时，BLAS函数调用开销可能超过优化收益

### 6.5 建议实现路径

对于低端ARM平台，推荐实现路径：

1. **Phase 1** (状态递推): 保持SIMD实现或简单循环
2. **Phase 2** (输出计算): 调用OpenBLAS的cblas_sdot或cblas_sgemm

这样可以结合两者优势：
- SIMD处理简单的逐元素运算
- BLAS处理复杂的矩阵运算

---

## 7. Conclusion

三种优化策略分别针对不同层面：

1. **Loop Optimization**: 算法层面，分离状态递推与输出计算
2. **Bidirectional Scan Fusion**: 架构层面，融合双向扫描减少冗余
3. **SIMD Optimization**: 硬件层面，利用向量指令并行计算

**重要发现**:
- 优化策略的组合效果**依赖于硬件平台和模型规模**
- 在低端ARM (树莓派)上，PyTorch底层的OpenBLAS比手写SIMD更优
- 在高端ARM (RK3588)和x86上，手写SIMD最优
- **未来优化方向**: C++直接调用OpenBLAS，可能在低端ARM上获得更好效果

实际部署时应根据目标平台和模型规模进行benchmarking，选择最优配置。
