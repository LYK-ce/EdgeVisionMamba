# Workbook

## T1: draw_column.py
- S: 2026-01-19T10:45 UTC+8
- E: 2026-01-19T10:47 UTC+8
- F: Result/draw_column.py
- D: 无依赖
- O:
  - 解析result_parsed.csv（9模型×16配置×4设备）
  - 支持参数：--model/-m, --configs/-c, --all/-a, --output/-o
  - 功能：标准柱状图、加速比图(--speedup)、多设备子图(--multi-device)
  - 学术风格：Times New Roman、300dpi、colorblind-friendly配色、hatch patterns
  - 示例：`python draw_column.py --model vim_5m --configs Python-Original,SIMD -o out.png`

## T2: 修改draw_column.py支持xlsx
- S: 2026-01-19T12:03 UTC+8
- E: 2026-01-19T12:12 UTC+8
- F: Result/draw_column.py
- D: T1
- O:
  - 添加Parse_Xlsx()解析result_final.xlsx，处理编码问题（通过": vim_"和"Intel"/"ARM"模式检测）
  - 更新配置名称：Python Original, Loop Optimization, Bidirectional Fusion, Loop + Fusion, SIMD, SIMD + Loop, SIMD + Fusion, SIMD + Loop + Fusion
  - 默认数据源改为result_final.xlsx
  - 自动识别xlsx/csv格式
  - 已测试：vim_5m.png生成成功

## T3: 放大图例字体
- S: 2026-01-19T12:19 UTC+8
- E: 2026-01-19T12:20 UTC+8
- F: Result/draw_column.py
- D: T2
- O:
  - font.size: 10→12, axes.titlesize: 12→14, axes.labelsize: 11→13
  - xtick/ytick.labelsize: 9→11, legend.fontsize: 8→11, figure.titlesize: 14→16
  - axes.linewidth: 0.8→1.0

## T4: training_curve.py
- S: 2026-01-22T15:43 UTC+8
- E: 2026-01-22T16:00 UTC+8
- F: Result/training_curve.py
- D: T3(draw.md学术规范)
- O:
  - 读取result.xlsx Summary sheet
  - 6模型：EdgeVim Max/Random1/Random2/Min, Vim tiny/Small
  - Random1/Random2→散点图(marker P/X)，其余→折线图
  - Min Model每20epoch一点，自动处理NaN
  - NATURE配色：#CC247C,#E95351,#F7A24F,#4EA660,#5C9AD4,#AA77E9
  - 支持参数：--input,-i/--output,-o/--title,-t/--zoomed,-z/--all,-a/--ylim
  - 功能：Draw_Training_Curve(完整)/Draw_Zoomed_Curve(epoch放大)
  - ylim参数支持y轴范围限制(如70,80)放大特定准确率区间
  - --all模式生成3个图：完整/_epoch_zoomed/_acc_zoomed(70-80)
  - annotate_final=True：在epoch 299处为Max Model和Vim Small添加标记点(s=60)
  - 输出PNG+PDF，300DPI

## T5: extract_random_config.py
- S: 2026-01-22T16:21 UTC+8
- E: 2026-01-22T16:22 UTC+8
- F: Result/Log/extract_random_config.py
- D: 无依赖
- O:
  - 读取random_search_all_results.json
  - 提取每个模型的Model_ID, Depth, Params, Params_M, FLOPs, FLOPs_G, Accuracy
  - 保留原始sheets，新增RandomSearch sheet到result.xlsx
  - 输出数据范围统计

## T6: 放大绘图字号适配双栏
- S: 2026-01-26T10:56 UTC+8
- E: 2026-01-26T10:57 UTC+8
- F: Result/draw_column.py, Result/training_curve.py
- D: T3
- O:
  - font.size: 12→14, axes.titlesize: 14→16, axes.labelsize: 13→15
  - xtick/ytick.labelsize: 11→13, legend.fontsize: 11→13
  - figure.titlesize: 16→18, axes.linewidth: 1.0→1.2

## T7: SIMD+Fixlen阶段2 SIMD优化
- S: 2026-01-26T12:57 UTC+8
- E: 2026-01-26T12:58 UTC+8
- F: Vim_CPU/VisionMamba_CPU/mamba-1p1p1/mamba_ssm/ops/selective_scan.cpp
- D: 无依赖
- O:
  - Selective_Scan_Simd_Fixlen_Cpu (L840-L852)
  - 原torch::sum已注释保留(回退用)
  - 新增SIMD阶段2：遍历(b,d,l)计算sum(x*C)
  - 支持AVX512/AVX/SSE/NEON + OpenMP并行
  - 解决问题：原阶段2用PyTorch API无SIMD加速

## T8: Selective_Fused_Scan_Simd_Cpu真正融合优化
- S: 2026-01-26T13:11 UTC+8
- E: 2026-01-26T13:12 UTC+8
- F: Vim_CPU/VisionMamba_CPU/mamba-1p1p1/mamba_ssm/ops/selective_scan.cpp
- D: T7
- O:
  - 修改Selective_Fused_Scan_Simd_Cpu (L1050-L1206)
  - 原实现：交替处理fwd(N)+bwd(N)，两个独立循环
  - 新实现：真正融合，只融合状态更新
    - 阶段1：一次SIMD处理2N个连续元素(fwd+bwd状态一起更新)
    - 阶段2：分别计算fwd和bwd输出(各自N个元素)
  - 好处：状态更新利用2N=32元素更好利用SIMD宽度，内存访问连续
  - 支持AVX512(16)/AVX(8)/SSE(4)/NEON(4) + OpenMP并行

## T9: ResNet/ViT基准测试脚本
- S: 2026-01-26T13:45 UTC+8
- E: 2026-01-26T13:45 UTC+8
- F: Vim_CPU/benchmark_baseline.py
- D: 无依赖
- O:
  - 创建ResNet和ViT的CPU延迟基准测试脚本
  - ResNet: 使用自定义BasicBlock，width_mult调整参数量(0.92/1.13/1.30→10M/15M/20M)
  - ViT: 调整embed_dim=384, depth=6/9/12→10M/15M/20M
  - 预热3次，测试10次，输出avg/min/max延迟
  - 用于与Vision Mamba进行性能对比

## T10: SIMD-Fused-Fixlen阶段2 SIMD优化
- S: 2026-01-26T13:48 UTC+8
- E: 2026-01-26T13:49 UTC+8
- F: Vim_CPU/VisionMamba_CPU/mamba-1p1p1/mamba_ssm/ops/selective_scan.cpp
- D: T8
- O:
  - 修改Selective_Fused_Scan_Simd_Fixlen_Cpu (L1372-L1492)
  - 原阶段2：使用torch::sum计算输出（无SIMD）
  - 新阶段2：完整SIMD实现
    - C矩阵转置为(B,L,N)
    - 遍历(b,d,l)计算sum_fwd和sum_bwd
    - 支持AVX512/AVX/SSE/NEON + OpenMP并行
  - 结合：阶段1 SIMD递推(2N元素) + 阶段2 SIMD输出(分开fwd/bwd)
