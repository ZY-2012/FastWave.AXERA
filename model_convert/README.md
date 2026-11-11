# 模型转换（FastWave → AX650 AXMODEL）

整条链路在本机（x86_64 + Pulsar2 8.0 独立包）完成，不需要 Docker。

```
上游 FastWave 权重
   └─ export_onnx.py     → export/model.onnx     （单次 NFE 静态图, 4.98 MB）
      └─ make_calib.py   → calib/data/*.npy      （真实 48 kHz 语音, 8 个采样步）
         └─ prepare_compile.py → compile/pulsar2_config.json (U16)
            └─ compile.sh → compile/fastwave_nfe.axmodel (11.9 MB)
               └─ extract_assets.py → python/fastwave_sdk/assets
```

## 0. 依赖

```bash
pip install torch==2.2 onnx onnxruntime numpy scipy soundfile omegaconf
# Pulsar2 8.0 独立包（ax_pulsar2_8.0_lite_package.tar.gz）
export PULSAR2_HOME=/path/to/pulsar2/8.0
export PATH=$PULSAR2_HOME/bin:$PATH
```

上游权重：[FastWave checkpoint](https://drive.google.com/file/d/1oNCxrKjgiWsYGW6P49rsI84vFYR5G3m8/view)
（`checkpoint-epoch140.pth`，1.28 M 参数，MIT License）。

## 1. 导出 ONNX

```bash
python export_onnx.py --ckpt origin/checkpoint-epoch140.pth --out export/model.onnx
```

关键改动（详见 `fw_model.py` 文件头）：

- **STFT/iSTFT 换成矩阵乘 FFT**（`fw_spectral.py`）：模型 15 个 FFC 块内部都有
  `torch.stft → 复数 1x1 卷积 → torch.istft`，AX NPU 不支持 FFT/复数算子。这里用
  32×32 Cooley-Tukey 分解把 1024 点 FFT 变成 4 个 `MatMul`，iSTFT 用 Gather 构造
  Hermitian 谱 + 反向 FFT + 4 段移位 overlap-add + 窗包络归一化。与 `torch.stft/istft`
  误差 < 1e-7。
- **EDM 系数前置**：`c_skip/c_in/c_out` 只依赖 sigma，作为 (1,1) 输入。
- **扩散 embedding 前置**：正弦位置编码 + 两层 MLP（128→512→512）在 host 端算好，
  作为 `noise_emb`(1,512) 输入，图上没有 Exp/Sin/Cos/Gemm。
- **band one-hot 作为输入**：(1,2,513)，一个模型覆盖 6–48 kHz 输入。
- GRN 的 L2 norm 展开为 `ReduceSum + Sqrt`（避免 `ReduceL2`）。

图输入（全部 float32）：

| 名称 | shape | 说明 |
|---|---|---|
| `x` | 1×32768 | 扩散 latent（含噪信号） |
| `audio_low` | 1×32768 | 带限（输入采样率）条件信号 @48 kHz |
| `band_onehot` | 1×2×513 | 输入采样率对应频带 |
| `c_skip` / `c_in` / `c_out` | 1×1 | EDM preconditioning |
| `noise_emb` | 1×512 | 扩散 embedding |

chunk 必须是 `hop_length=256` 的整数倍（默认 32768 = 0.68 s @48 kHz）。

## 2. 校准集

```bash
python make_calib.py --wav-dir calib/wav48 --out-dir calib/data
```

对每段真实 48 kHz 语音（示例用 8/12/16/24 kHz 退化输入）真实跑一遍 8 步 EDM 采样，
把每一步真正喂给网络的张量都存下来。这样 `x` 的标定范围覆盖 `N(0,80)`（第一步）到
`N(0,1)`（最后一步），不会出现第一步被截断的问题。

## 3. 编译

```bash
python prepare_compile.py          # 打包 calib tar.gz + 生成 U16 配置
bash compile.sh                    # pulsar2 build
```

量化配置：

```json
"quant": {
  "calibration_method": "MinMax",
  "layer_configs": [{
    "start_tensor_names": ["DEFAULT"],
    "end_tensor_names": ["DEFAULT"],
    "data_type": "U16",
    "weight_data_type": "S8",
    "output_data_type": "U16"
  }],
  "highest_mix_precision": false
}
```

> `DEFAULT` 表示全图张量（U16 激活 / S8 权重 / U16 输出）。
> `compiler.check=0` 不仿真、编译最快，精度直接上板验证；量化精度有问题时再改
> `check=2`（统计校验）或 3（余弦校验，大图很慢）。

## 4. host 端资产

```bash
python extract_assets.py --ckpt origin/checkpoint-epoch140.pth --out ../python/fastwave_sdk/assets
```

导出扩散 embedding 的 4 个权重（`emb_p1_w/b`、`emb_p2_w/b`）与 `model_config.json`，
板端只跑 numpy，不需要 torch。

## 5. 上板验证

```bash
# 精度：held-out NFE 输入 vs 浮点 ONNX
python ../simulate/board_eval.py --model compile/fastwave_nfe.axmodel \
    --in-dir ../simulate/eval/in --out-dir ../simulate/eval_npu
python ../simulate/compare.py

# 端到端
BOARD=root@<ip> bash ../deploy_board.sh
```
