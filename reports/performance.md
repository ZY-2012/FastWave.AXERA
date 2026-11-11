# FastWave.AXERA — 性能报告

| 项目 | 数值 |
|---|---|
| 平台 | AX650 / 三核 NPU（NPU3，BSP V3.10.2，NPU clk 800 MHz） |
| axmodel | 11.9 MB（单次 NFE 图，含 15 个 FFC 块的矩阵乘 FFT） |
| 量化 | U16 激活 / S8 权重 / U16 输出，MinMax 标定，Pulsar2 8.0，`compiler.check=2` |
| 单次 NFE（chunk 32768 = 0.68 s @48k） | **499.8 ms**（min 499.5 / max 500.1，n=10） |
| 8 步 / chunk | 3.99 s |
| 端到端 RTF（8 步，50% overlap） | **10.97**（C++）/ **10.96**（Python） |
| 端到端 RTF（4 步，50% overlap） | **5.51**（C++）/ **5.49**（Python） |
| 端到端 RTF（8 步，`--overlap 0`） | **6.29**（C++，无重叠拼接） |

## 说明

- chunk 固定 32768 点，约 0.68 s；每 chunk 调 8 次（或 4 次）NPU，扩散采样在 host 侧串联。
- 8 步与 4 步共用同一个 axmodel（sigma/系数由 host 提供），`--steps 4` 直接可用。
- 端到端耗时 ≈ 音频时长 × 2（50% overlap 的重叠系数）× NFE 次数 × 0.5 s。

## 单 NFE 0.5 s 的构成（实测探针）

| 组成 | 15 份合计 |
|---|---|
| Conv（12.8 GMAC，同尺寸 1x1 Conv 探针 ≈ 570 GMAC/s） | ~25 ms |
| STFT（15 份） | 15 ms |
| iSTFT（15 份） | 37 ms |
| sigmoid×tanh（128×32768） | 31 ms |
| GRN（64×32768） | 22 ms |
| 其余逐元素 / reshape / gather / overlap-add（15 个 FFC 块，2600+ 个 op） | ~370 ms |

单次 NFE 的真正瓶颈不是卷积，而是 15 个 FFC 块里 STFT/iSTFT 带来的大量
4.2 M 元素级逐元素算子（每 NFE 约 4 GB DDR 访存）。图是严格串行的，
AX650 的三核无法并行（实测只有 core0 在工作，多 session 并行也无加速）。

可选的优化方向（按收益排序）：

1. **换成 int8 激活**（S8/U8）再测一次：同尺寸探针 U8 比 U16 快 2.8×，访存减半，
   但 FFT 分支的动态范围较大，需要重新看精度；
2. 增大 chunk（如 65536）减少每秒的算子启动次数；
3. 把 FFT 的 32×32 MatMul 换成一次更大的 GEMM（合并 stage1/stage2 的批维）；
4. 用 `compiler.npu_perf` / `dump_simulator_cycle_trace` 做逐算子定位后再定点优化。
