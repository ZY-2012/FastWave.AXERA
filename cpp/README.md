# FastWave.AXERA C++ runtime

AX650 C++17 runtime (AX_ENGINE). No Python / torch / onnxruntime dependency.
Host DSP (WAV I/O, polyphase resampling, band one-hot, chunking, overlap-add,
spectral-subtraction post filter) is implemented in `src/audio.hpp` /
`src/postfilter.hpp`; the 8-step EDM sampler calls the axmodel once per step.

## Build

```bash
bash cpp/download_toolchains.sh       # Arm GNU 9.2 + AX650 BSP (once)
TOOLCHAIN_ROOT=$PWD/cpp/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu \
BSP_MSP_DIR=$PWD/cpp/toolchains/ax650n_bsp_sdk/msp/out \
bash cpp/build_ax650.sh
```

The binary is written to `cpp/bin/fastwave_ax650` and shipped in `bin/`.

## Run on board

```bash
export LD_LIBRARY_PATH=/soc/lib:$LD_LIBRARY_PATH
./bin/fastwave_ax650 models/fastwave_nfe.axmodel input.wav output.wav \
    --assets models/assets [--steps 8] [--overlap 0.5] [--no-denoise]
```

Input: 6–48 kHz wav (PCM16/24/32 or float32). Output: 48 kHz PCM16.
