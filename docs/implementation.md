# 实现说明：把带 STFT 的扩散模型搬到 AX650 NPU

## 1. 难点：模型内部有 STFT

FastWave / NU-Wave 2 的 FFC 块把特征分成 local / global 两半，global 分支是：

```
STFT(1024/256, hann, center, normalized)  →  BSFT(频带条件调制)  →  ReLU
   → 复数 1x1 卷积  →  iSTFT  →  与 local 分支相加
```

15 个残差块每块都要做一次 STFT + iSTFT，且中间夹着 ReLU（频域非线性），无法把频谱
变换移到 host 而不产生 15×8 次 NPU↔CPU 往返和上百 MB 的数据搬运。

AX NPU 不支持 FFT 算子，也不支持复数张量，所以必须把 STFT/iSTFT 拆成 NPU 友好的基本算子。

### 1.1 1024 点 FFT = 4 个 32×32 MatMul（Cooley-Tukey）

```
n = 32*n1 + n0 ,  k = 32*k1 + k0
A[n0,k0] = Σ_n1 x[32*n1+n0] · W32[n1,k0]        # 实数输入, 一次 MatMul
B[n0,k0] = A[n0,k0] · W1024[n0,k0]              # twiddle
X[k1*32+k0] = Σ_n0 B[n0,k0] · W32[n0,k1]        # 每个 k0 一次 MatMul（批量）
```

复数运算全部展开成实部/虚部两块实数张量（`(a+bi)(c+di) = (ac-bd) + (ad+bc)i`），
图上只有 `MatMul / Mul / Add / Sub / Transpose / Reshape`。

### 1.2 iSTFT

1. 单边谱 (513) → `Gather` 常数索引补齐 Hermitian 对称谱 (1024)，虚部乘 ±1；
2. `ifft(X) = real(conj(fft(conj(X))))`，只保留实部（FFT 同上）；
3. 乘 hann 窗；
4. overlap-add：跳长 256、窗长 1024 ⇒ 每个输出点恰好 4 帧叠加，拆成 4 个 256 点块移位相加
   （`Slice`+`Pad`+`Add`，不用 `ConvTranspose`，避免 1024×1024 的稠密矩阵）；
5. 除以窗平方包络（常数 buffer，倒数预先算好）；
6. 去掉两端 512 点（center=True）。

与 `torch.stft/istft(center=True, normalized=True, onesided=True, pad_mode='reflect')`
逐值对分：max abs err 1.2e-7，cosine 1.0000007。

### 1.3 计算量

单帧单通道 1024 点 FFT ≈ 135 k 次乘加（2 个 32×32 实数 MatMul + 复数 twiddle），
30 个 FFT（15 块 × 正逆变换）每 0.68 s chunk 约 200 MMAC。加上骨干网络 8 步约 100 GFLOP，
在 AX650 NPU3 上实测见 `reports/performance.md`。

## 2. 图外计算（host）

| 量 | 为什么放 host |
|---|---|
| `c_skip / c_in / c_out` | 仅依赖 sigma（8 个固定值），避免图上 log/sqrt/div |
| `noise_emb` (1×512) | 正弦编码 + 2 层 MLP，128→512→512，避免 Exp/Sin/Cos/Gemm |
| `band_onehot` (2×513) | 输入采样率决定，避免 OneHot |
| 低质量条件 `audio_low` | 重采样（polyphase FIR），numpy 实现 |
| 分块 / overlap-add | Hann 加权拼接 |

host 侧每步只有几十次标量运算 + 一次 512×512 MatMul（0.5 MFLOP），可忽略。

## 3. 量化

- 数据类型：**U16 激活 / S8 权重 / U16 输出**（layer_configs 用 `DEFAULT` 覆盖全图），`highest_mix_precision=false`
- 标定方法：MinMax
- 标定数据：真实 48 kHz 语音 → 8/12/16/24 kHz 退化，真实跑 8 步 EDM 采样，逐步保存
  `x / audio_low / band / c_* / noise_emb`（64 样本）。关键是第一步 `x ~ N(0, 80²)`，
  如果只用随机小信号标定，第一步就会被截断。
- `layer_configs` 用 `start_tensor_names/end_tensor_names = ["DEFAULT"]` 表示全图；
  `compiler.check` 默认 0（不仿真），精度直接上板验证。

## 4. 已知限制 / 后续

- chunk 固定 32768 点（图静态）。更小的 chunk 可降低延迟（需重新编译），更大的 chunk
  可减少拼接次数。
- 每块之间独立做扩散采样，靠 50% 重叠 + Hann 加权消除接缝；如需严格流式，可按
  `streaming_inference.py` 的思路做 pipeline 并行（每块 8 步流水）。
- 8 步与 4 步共用同一个 axmodel（sigma 由 host 提供），`--steps 4` 直接可用。
