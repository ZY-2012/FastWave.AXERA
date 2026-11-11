# 示例音频

## 输入

| 文件 | 采样率 | 来源 |
|---|---|---|
| `listen/8k/OSR_us_000_0010_input_8k.wav` | 8 kHz | Open Speech Repository (English) |
| `listen/8k/OSR_us_000_0013_input_8k.wav` | 8 kHz | Open Speech Repository (English) |
| `listen/8k/OSR_cn_000_0072_input_8k.wav` | 8 kHz | Open Speech Repository (Chinese) |
| `listen/8k/OSR_cn_000_0074_input_8k.wav` | 8 kHz | Open Speech Repository (Chinese) |
| `listen/16k/speech_2_input_16k.wav` | 16 kHz | 公开人声样例降采样 |
| `listen/16k/speech_3_input_16k.wav` | 16 kHz | 公开人声样例降采样 |

- Open Speech Repository: http://voiptroubleshooter.com/open_speech/index.html

## 输出

`listen/48k_from8k/`、`listen/48k_from16k/` 为 FastWave.AXERA 在 AX650 板端的实际输出
（8 步 EDM 采样，chunk 32768 / 50% overlap）。

`listen/spectrograms/` 为输入/输出频谱对比图。
