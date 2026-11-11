# 量化精度分析（Pulsar2 EndToEnd，量化工件内）

配置：U16 激活 / S8 权重 / U16 输出，MinMax，64 个真实样本，`compiler.check=2`。

- 统计算子数：2245
- 输出 `y` cosine（量化 vs 浮点）：**0.99982**（MSE 0.00000）
- 逐算子 cosine：min 0.90369，mean 0.99962

| cosine 区间 | 算子数 |
|---|---|
| >=0.999 | 2196 |
| <0.999 | 35 |
| <0.99 | 12 |
| <0.95 | 2 |

最差 6 个算子（FFT 的 twiddle/矩阵乘，位于 STFT 分支内部，量级很小）：

| 算子 | 类型 | 输出 shape | dtype | cosine |
|---|---|---|---|---|
| /resid… | AxMul | /residu… | FP32 | 0.90369 |
| /resid… | AxMul | /residu… | FP32 | 0.94242 |
| /resid… | AxMul | /residu… | FP32 | 0.95310 |
| /resid… | AxMul | /residu… | FP32 | 0.95744 |
| /resid… | AxMul | /residu… | FP32 | 0.97668 |
| /resid… | AxMul | /residu… | FP32 | 0.98177 |

