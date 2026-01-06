# Workbook

## Task 1
Start: 2026-01-06T12:17+08:00
End: 2026-01-06T12:19+08:00
Status: DONE
Target: selective_fused_scan_ref修正，移除fixlen优化

### Changes
- selective_scan_interface.py:320-409 改用逐步计算(x_bi状态+ys列表)
- selective_scan.cpp:236-330 同步改用逐步计算(vector<Tensor>)
- 区别于fixlen版本：不再使用两阶段递推，改为每步计算隐藏状态并输出

## Task 2
Start: 2026-01-06T12:25+08:00
End: 2026-01-06T12:26+08:00
Status: DONE
Target: SIMD优化分析文档

### Output
- VisionMamba_CPU/SIMD.md
- 核心策略：N维度(dstate)SIMD向量化
- 两个优化点：状态更新(FMA)、输出归约(reduce_add)
- 预期加速：2-3x(相对C++ Tensor实现)

## Task 3
Start: 2026-01-06T12:33+08:00
End: 2026-01-06T12:36+08:00
Status: DONE
Target: 实现SIMD版selective scan

### Changes
- selective_scan.cpp:437-673 添加Selective_Scan_Simd_Cpu函数
  - SIMD头文件检测(AVX512/AVX/SSE/Scalar)
  - Horizontal_Sum辅助函数(AVX512/AVX/SSE)
  - N维度SIMD向量化主循环
  - FMA指令优化状态更新
  - 标量回退处理剩余元素
- selective_scan.cpp:737-748 添加selective_scan_simd绑定
- setup.py:33 添加/arch:AVX2编译选项
- selective_scan_interface.py:120-142 添加selective_scan_simd_fn封装函数

## Task 4
Start: 2026-01-06T12:40+08:00
End: 2026-01-06T12:42+08:00
Status: DONE
Target: 添加SIMD选项到VIM模型并测试

### Changes
- mamba_simple.py:20 导入selective_scan_simd_fn
- mamba_simple.py:86 添加use_simd_scan参数
- mamba_simple.py:103 保存use_simd_scan属性
- mamba_simple.py:536-565 添加SIMD调用分支
- models_mamba.py:172 create_block添加use_simd_scan参数
- models_mamba.py:179 mixer_cls添加use_simd_scan
- models_mamba.py:297 VisionMamba添加use_simd_scan参数
- models_mamba.py:373 layers创建添加use_simd_scan
- inf_cpu.py:53 create_vim_tiny_model添加use_simd_scan参数
- inf_cpu.py:97 create_model添加use_simd_scan
- inf_cpu.py:348-356 添加SIMD测试配置
- inf_cpu.py:407 调用时获取use_simd_scan

## Task 5
Start: 2026-01-06T12:49+08:00
End: 2026-01-06T12:56+08:00
Status: DONE
Target: SIMD+fixlen/fused/fused_fixlen优化实现

### Changes
- selective_scan.cpp:676-850 添加Selective_Scan_Simd_Fixlen_Cpu
  - Phase1: SIMD递推计算state_history (L,B,D,N)
  - Phase2: einsum批量输出
- selective_scan.cpp:852-990 添加Selective_Fused_Scan_Simd_Cpu
  - 2N状态向量SIMD化(fwd+bwd融合)
  - 分离的C_fwd/C_bwd输出矩阵
- selective_scan.cpp:992-1180 添加Selective_Fused_Scan_Simd_Fixlen_Cpu
  - Phase1: 2N状态SIMD递推
  - Phase2: einsum批量输出
  - 最高优化级别
- selective_scan.cpp:1192-1247 添加pybind11绑定
- selective_scan_interface.py:144-208 添加3个Python封装函数
- mamba_simple.py:19-23 导入新函数
- mamba_simple.py:90-92 添加3个新use_xxx参数
- mamba_simple.py:111-113 保存参数为实例属性
- mamba_simple.py:548-590 _forward_reference添加simd_fixlen分支
- mamba_simple.py:663-704 _forward_fuse_reference添加simd_fused分支
- models_mamba.py:171-184 create_block添加3个新参数
- models_mamba.py:300-302 VisionMamba添加3个新参数
- models_mamba.py:379-381 layers创建传递新参数
- inf_cpu.py:50-68 create_vim_tiny_model添加3个新参数
- inf_cpu.py:96-113 create_model传递新参数
- inf_cpu.py:361-395 添加4个SIMD测试配置(含原SIMD)
- inf_cpu.py:424-439 调用时获取新参数

## Task 6 (额外)
Start: 2026-01-06T13:22+08:00
End: 2026-01-06T13:24+08:00
Status: DONE
Target: 添加ARM NEON支持(树莓派兼容性)

### Changes
- selective_scan.cpp:31-35 添加ARM NEON头文件检测(`__ARM_NEON`/`__aarch64__`)
- selective_scan.cpp:475-486 添加Horizontal_Sum_Neon(ARM32/ARM64两种实现)
- selective_scan.cpp:665-688 Selective_Scan_Simd_Cpu添加NEON循环(vld1q_f32/vmlaq_f32/vst1q_f32)
- selective_scan.cpp:824-832 Selective_Scan_Simd_Fixlen_Cpu添加NEON循环
- selective_scan.cpp:998-1009 Selective_Fused_Scan_Simd_Cpu正向NEON循环
- selective_scan.cpp:1066-1077 Selective_Fused_Scan_Simd_Cpu反向NEON循环
- selective_scan.cpp:1202-1209 Selective_Fused_Scan_Simd_Fixlen_Cpu添加NEON循环

### NEON特性
- 128-bit寄存器(4 floats)同SSE
- vmlaq_f32实现FMA: c = a*b + c
- ARM64使用vaddvq_f32直接归约
- ARM32使用vadd_f32/vpadd_f32手动归约

## Task 6
Start: 2026-01-06T13:46+08:00
End: 2026-01-06T13:46+08:00
Status: DONE
Target: 修改inf_cpu.py预热和推理次数

### Changes
- inf_cpu.py:477 warmup num_warmup=2 → num_warmup=10
- inf_cpu.py:480 benchmark num_runs=10 → num_runs=100

## Task 7
Start: 2026-01-06T13:47+08:00
End: 2026-01-06T13:47+08:00
Status: DONE
Target: 移除viztracer相关内容

### Changes
- inf_cpu.py:4-15 更新文档说明
- inf_cpu.py:47 移除from viztracer import VizTracer
- inf_cpu.py:169-194 移除profile_with_viztracer函数
- inf_cpu.py:484-486 移除VizTracer调用
- inf_cpu.py:534-548 移除性能分析文件打印和提示

## Task 9
Start: 2026-01-06T14:03+08:00
End: 2026-01-06T14:03+08:00
Status: DONE
Target: 编写time_baseline.py测试ResNet和ViT推理时间

### Output
- Tmp/time_baseline.py
- ResNet系列: ResNet-18(~11.7M), ResNet-34(~21.8M), ResNet-50(~25.6M)
- ViT系列: ViT-Tiny(~5.7M), ViT-Small(~22M), ViT-Base(~86M)
- 预热10次、推理100次
