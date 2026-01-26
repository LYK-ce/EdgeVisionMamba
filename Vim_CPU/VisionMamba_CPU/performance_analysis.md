#Presented by KeJi
#Date: 2026-01-26

# Vision Mamba CPU性能分析报告

## 一、核心结论

**尽管通过SIMD极致优化了SSM计算，Vim在CPU上仍然明显慢于同参数量的ViT。根本原因是SSM的计算特性与CPU架构本质不匹配。**

## 二、SSM计算特性分析

### 2.1 递推依赖本质

SSM的核心递推公式：
```
h[t] = Ā[t] ⊙ h[t-1] + B̄[t]·u[t]
y[t] = C[t]·h[t]
```

**关键问题**：`h[t]` 必须等待 `h[t-1]` 计算完成，形成严格的串行依赖链。

```
h[0] ──► h[1] ──► h[2] ──► ... ──► h[L-1]
         │         │                  │
         └────必须等待前一步完成───────┘
```

### 2.2 计算特征对比

| 特性 | SSM (Vim) | GEMM (ViT) |
|------|-----------|------------|
| **单次计算量** | N=16 floats (64 bytes) | 矩阵块 (数MB) |
| **循环次数** | L=196次 | 少量大块 |
| **依赖关系** | 严格串行 | 可并行 |
| **内存访问** | 跨步访问 (128 bytes/步) | 连续访问 |

## 三、性能瓶颈详解

### 3.1 瓶颈1：零碎小块计算

每个时间步仅处理 N=16 个浮点数：
- AVX-256：2次迭代即可处理完
- 循环开销、函数调用开销占比极高
- 计算密度极低

```
ViT: ████████████████████████████  (大块GEMM，充分利用SIMD)
SSM: ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪  (196个小块，大量开销)
```

### 3.2 瓶颈2：跨步访问模式

数据布局：`(B, D, L, N)` = `(1, 384, 196, 16)`

每步访问间隔：
```
步长 = dstate × sizeof(float) = 16 × 4 = 64 bytes (刚好1个cache line)
双向SSM步长 = 2N × 4 = 128 bytes (跨越2个cache line)
```

**后果**：每步可能触发cache line加载，无法有效重用。

### 3.3 瓶颈3：串行依赖不可消除

```cpp
// 无论如何优化，这个循环无法并行化
for (int64_t i = 0; i < seq_len; i++) {
    x = deltaA[i] * x + deltaB_u[i];  // h[t] 依赖 h[t-1]
}
```

SIMD只能加速**每步内部**的N=16维计算，无法消除**L=196步**的串行依赖。

## 四、为什么SIMD优化无法弥补差距

### 4.1 SIMD优化范围

| 维度 | 大小 | SIMD并行性 | 说明 |
|------|------|-----------|------|
| **L (序列)** | 196 | ❌ 串行 | 递推依赖，无法并行 |
| **N (状态)** | 16 | ✅ 2次AVX迭代 | SIMD仅优化这部分 |
| **D (通道)** | 384 | ✅ OpenMP | 外层并行 |
| **B (批次)** | 1 | ✅ | 推理通常batch=1 |

### 4.2 Amdahl定律限制

假设串行部分占比70%：
```
最大加速比 = 1 / (0.7 + 0.3/∞) = 1.43x
```

即使SIMD将可并行部分加速到无穷快，整体加速也受限于串行部分。

### 4.3 为什么不用Parallel Scan

GPU上的Parallel Scan可以将O(L)串行变为O(log L)并行，但在CPU上：

| 对比 | 串行递推 | Parallel Scan |
|------|---------|---------------|
| 计算量 | L次 | ~2L次（翻倍） |
| 并行度 | 1 | log₂(L)≈8 |
| CPU利用 | 已用SIMD+OpenMP | 核心数不足以利用 |

**结论**：CPU核心数有限（8-16），Parallel Scan的计算量翻倍换不来足够的并行收益，是负优化。

## 五、量化指标

### 5.1 关键性能指标对比

| 指标 | SSM (Vim) 预期 | GEMM (ViT) 预期 |
|------|---------------|----------------|
| **IPC** | 0.5-0.8 | 2.0-3.5 |
| **L1 Cache Miss Rate** | 10-20% | <2% |
| **Arithmetic Intensity** | ~0.5 FLOPs/byte | 50-100 FLOPs/byte |
| **Backend Bound** | 60-70% | <30% |

### 5.2 计算强度分析

```
SSM: 16 FMA × 2 = 32 FLOPs / 64 bytes = 0.5 FLOPs/byte
GEMM: N³ FLOPs / N² bytes ≈ N FLOPs/byte (N越大越高)
```

### 5.3 Roofline模型

```
FLOPS ▲
      │                    ╱ 计算瓶颈区
      │                  ╱
      │                ╱
      │              ╱ ← GEMM在这里（计算密集）
      │            ╱
      │          ╱
      │        ╱
      │ SSM→ •  内存瓶颈区（内存受限）
      │────────────────────────────────►
              Arithmetic Intensity (FLOPs/Byte)
              
SSM (0.5 FLOPs/B): 受限于内存带宽，计算单元空闲
GEMM (>50 FLOPs/B): 受限于计算能力，充分利用硬件
```

### 5.4 测量命令

#### Linux (perf)

```bash
# 基本性能计数器
perf stat -e cycles,instructions,cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses \
python benchmark.py

# 详细cache分析
perf stat -e l1d.replacement,l2_rqsts.miss,\
mem_load_retired.l1_hit,mem_load_retired.l1_miss \
python benchmark.py
```

#### Windows 测试方案

**方案1：Intel VTune Profiler (推荐)**

```powershell
# 安装后使用命令行
vtune -collect hotspots -knob sampling-mode=hw -- python benchmark.py

# 或收集微架构分析
vtune -collect uarch-exploration -- python benchmark.py

# 查看报告
vtune -report summary -result-dir r000hs
```

VTune可测量指标：
- CPI (Cycles Per Instruction) = 1/IPC
- Cache Misses (L1/L2/L3)
- Memory Bound / Core Bound
- Retiring / Bad Speculation / Frontend Bound / Backend Bound

**方案2：Windows Performance Analyzer (WPA)**

```powershell
# 使用Windows Performance Recorder录制
wpr -start CPU -start FileIO
python benchmark.py
wpr -stop output.etl

# 用WPA分析output.etl
wpa output.etl
```

**方案3：Visual Studio Profiler**

1. 打开Visual Studio → Debug → Performance Profiler
2. 选择"CPU Usage"或"Instrumentation"
3. 运行benchmark并查看火焰图

**方案4：AMD uProf (AMD CPU)**

```powershell
# 收集硬件性能数据
AMDuProfPcm.exe -m ipc,l1,l2,l3 -- python benchmark.py
```

**方案5：Python内置（简易方案）**

```python
import cProfile
import pstats

# 函数级profiling
cProfile.run('model(input)', 'output.prof')
stats = pstats.Stats('output.prof')
stats.sort_stats('cumulative').print_stats(20)
```

```python
# 使用py-spy（采样profiler）
# pip install py-spy
# py-spy record -o profile.svg -- python benchmark.py
```

#### 跨平台工具

**PyTorch Profiler (推荐)**

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    model(input)

# 打印结果
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

# 导出Chrome trace
prof.export_chrome_trace("trace.json")
```

## 六、结论

### 6.1 问题本质

SSM的递推依赖导致：
1. **零碎小块计算**：每步仅处理N=16元素，循环开销大
2. **跨步访问模式**：每步跳过128 bytes，Cache利用率低

### 6.2 量化体现

- **Cache Miss率**：高（跨步访问）
- **IPC**：低（流水线stall等待内存）
- **Roofline**：处于内存瓶颈区（计算强度太低）

### 6.3 最终结论

**SSM的计算特性与CPU架构本质不匹配。这不是实现问题，而是算法与硬件的固有矛盾。SIMD优化已达到极限，无法通过软件优化弥补这一差距。**

---

## 七、参考

- 代码实现：[`selective_scan.cpp`](mamba-1p1p1/mamba_ssm/ops/selective_scan.cpp)
- 数学推导：[`mamba_math_derivation.md`](../../mamba_math_derivation.md)
- 架构分析：[`vim_analysis.md`](../../vim_analysis.md)
