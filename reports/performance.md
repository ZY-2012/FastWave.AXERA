# FastWave.AXERA — 性能报告

| 项目 | 数值 |
|---|---|
| 平台 | AX650 / 三核 NPU（NPU3，BSP V3.10.2，NPU clk 800 MHz） |
| axmodel | 11.9 MB（单次 NFE 图，含 15 个 FFC 块的矩阵乘 FFT） |
| 量化 | U16 激活 / S8 权重 / U16 输出，MinMax 标定，Pulsar2 8.0，`compiler.check=2` |
| 单次 NFE（chunk 32768 = 0.68 s @48k） | **499.8 ms**（min 499.5 / max 500.1，n=10） |
| 8 步 / chunk | 3.99 s |
| 端到端 RTF（8 步，50% overlap） | **10.97**（C++）/ **10.96**（Python）；板端 6 个示例 10.92–11.71 |
| 端到端 RTF（4 步，50% overlap） | **5.51**（C++）/ **5.49**（Python） |
| 端到端 RTF（8 步，`--overlap 0`） | **6.29**（C++，无重叠拼接） |

## 后处理（可选）

默认关闭。`--denoise` 打开谱减降噪：静音段噪声 0.00132 → 0.00053，语音段电平不变
（0.1238 → 0.1231），HF 约 -1.1 dB。

## 说明

- chunk 固定 32768 点，约 0.68 s；每 chunk 调 8 次（或 4 次）NPU，扩散采样在 host 侧串联。
- 8 步与 4 步共用同一个 axmodel（sigma/系数由 host 提供），`--steps 4` 直接可用。
- 端到端耗时 ≈ 音频时长 × 2（50% overlap 的重叠系数）× NFE 次数 × 0.5 s。

## 单 NFE 0.5 s 的构成（实测探针）

Pulsar2 的逐算子调度报告（`compiler.npu_perf=true`）给出了确切答案：
单次 NFE 共 **24514 个 job**，按调度周期分类：

| 类别 | 调度周期占比 | op 数 |
|---|---:|---:|
| 数据搬运（Slice/assign/Transpose/Gather/Concat/Pad/Reshape/Squeeze） | 74.3% | 15449 |
| 逐元素（Mul/Add/Sub/Neg/Sigmoid/Tanh/Reduce） | 18.0% | 6624 |
| 矩阵乘（FullyConnected） | 4.7% | 1455 |
| 卷积（Conv） | 2.6% | 779 |
| 其他 | 0.5% | 207 |

总周期 1529M，调度 job 数 24514


**算力只占 7%（MatMul 4.7% + Conv 2.6%），74% 是数据搬运**（Slice/assign/Transpose/
Gather/Concat/Pad/Reshape）。也就是说这个模型在 AX650 上不是"算不动"，而是
"搬不动"：15 个 FFC 块里的 STFT/iSTFT 需要在时间轴（T 连续）上做 FFT，而 NPU 的
矩阵乘友好布局是通道轴，每次 layout 变化都被编译器拆成大量 Slice+assign 拷贝
（6124 个 Slice + 5376 个 assign）。

图是严格串行的，AX650 三核无法并行（实测只有 core0 在工作，多 session 并行也无加速）。

可选的优化方向（按收益排序）：

1. **重写 FFT 的数据布局**（最大收益）：避免每次 FFT 的 3 次 transpose，把
   1024 点 FFT 拆成"沿 T 的 depthwise Conv（kernel=32, stride=32）+ 沿通道的 MatMul"，
   或者用更大的 chunk 摊薄布局转换开销；预计 1.5–2×；
2. **换成 int8 激活**（S8/U8）：访存直接减半，同尺寸探针 U8 比 U16 快 2.8×，
   但 FFT 分支动态范围大，需要重新验精度；
3. 增大 chunk（如 65536）减少每秒的布局转换次数（线性摊薄）；
4. 逐算子 profile：`compiler.npu_perf=true`，产物在
   `compile/work_perf/compiler/debug/subgraph_npu_0/b1/op_profile.csv`。
