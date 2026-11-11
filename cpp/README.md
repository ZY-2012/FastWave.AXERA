# FastWave.AXERA C++ runtime

AX650 C++17 runtime (AX_ENGINE). No Python / torch / onnxruntime dependency.
Host DSP (WAV I/O, polyphase resampling, band one-hot, chunking, overlap-add,
spectral-subtraction post filter) is implemented in `src/`; the 8-step EDM
sampler calls the axmodel once per step.

## Build

```bash
bash cpp/download_toolchains.sh     # Arm GNU 9.2 aarch64 + AX650 BSP SDK (~1.1 GB, once)
bash cpp/build_ax650.sh             # -> bin/fastwave_ax650
```

`build_ax650.sh` calls `download_toolchains.sh` automatically if the toolchain or
the BSP SDK is missing. Already have them? Point the scripts at your copies:

```bash
TOOLCHAIN_ROOT=/path/to/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu \
BSP_MSP_DIR=/path/to/ax650n_bsp_sdk/msp/out \
bash cpp/build_ax650.sh
```

## Run on board

```bash
export LD_LIBRARY_PATH=/soc/lib:$LD_LIBRARY_PATH
./bin/fastwave_ax650 models/fastwave_nfe.axmodel input.wav output.wav \
    --assets models/assets [--steps 8] [--overlap 0.5] [--denoise]
```

Input: 6–48 kHz wav (PCM16/24/32 or float32). Output: 48 kHz PCM16.
