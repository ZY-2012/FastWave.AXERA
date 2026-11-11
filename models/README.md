# Models

本目录在 GitHub 仓库中不包含 axmodel 文件。

预编译 AX650 模型请从 HuggingFace 下载：

<https://huggingface.co/HY-2012/FastWave.AXERA/tree/main/models>

- `fastwave_nfe.axmodel` — 单次 NFE 图（11.9 MB）
- `model_config.json` — chunk / 采样参数

编译配置：Pulsar2 8.0，AX650 / NPU3，**U16 激活 / S8 权重 / U16 输出**，MinMax 标定，
`highest_mix_precision=false`。精度与性能见 `reports/accuracy.md`、`reports/performance.md`。
