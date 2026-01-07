1. 完成以下任务
   - 浏览VisionMamba_CPU这个目录，确定这个目录里的内容是做什么的。
   - 当前是针对Vision mamba在cpu上进行了优化，但是selective_fused_scan_ref应该只使用fused优化方式，但是它错误的包含了fixlen优化方式，把python版本和cpp版本的实现都修正过来。
完成
2. 完成以下任务：
   - 当前在进行selective scan计算的时候，受限于循环依赖，无法快速的完成运算。因此，我们决定使用最后一种方案，就是SIMD，在维度方向上应该是可以并行的。写一个SIMD.md，分析一下如何使用SIMD来优化这个运算过程。
  完成

3. 按照SIMD.md的优化方案，在当前的selective scan.cpp当中实现一个selective scan simd方法，这种方法采用SIMD进行原版的selective scan，然后更新当前的setup.py，并且在测试用例当中增加此测试
   完成
4. 修改当前vim的参数，增加一个simd的选项，然后selective scan调用上面实现的方法。然后在inf cpu。py当中增加此测试用例
   完成

5. 当前的SIMD优化效果非常好，我们在此基础上结合之前提出的优化方案，进行更进一步的优化。完成下面的任务
   - 在selective scan.cpp当中实现selective scan simd fixlen方法，结合fixlen优化与simd方法
   - 在selective scan.cpp当中实现selective scan simd fused方法，结合fused优化与simd方法
   - 在selective scan.cpp当中实现selective scan simd fused fixlen方法，结合fused优化，fixlen优化与simd方法
   - 更新setup.py
   - 修改当前vim实现，增加以上三种方案，更新inf cpu.py
    完成

6. 现在修改当前的inf cpu.py，这一次预热10个epoch，然后推理100次求平均值
   完成

7. 移除viztracer相关内容，现在我们不需要分析性能了
   完成
8. 在tmp目录下编写一个time baseline.py，输出7M参数的卷积神经网络ResNet和7M参数的vision transformer vit的推理时间
   完成
9.  当前只测试了一个vim tiny，在models mamba.py文件里面，增加几个@register_model，使其参数量分别为5M，10M，15M，20M左右。主要通过调整模型的深度，宽度来调整总参数量。然后在inf cpu.py当中，增加这些模型的测试。并且，测试时，预热3轮，测试跑10轮即可。
    完成
10. 我们已经安装了thop库，先写个简单testflops.py看看vim tiny的FLOPs是多少
    完成
11. 现在继续增加模型，按照FLOPs在2G，3G，4G，5G设置模型，仍然通过调整模型深度，宽度来调整FLOPs，然后在inf cpu.py当中，增加这些模型的测试，同样的每一个模型也都要测试16种优化配置方案