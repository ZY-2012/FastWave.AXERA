#!/usr/bin/env python3
"""Pack calibration npy files + write the Pulsar2 8.0 build config.

Quantization: data type U16 for every op type (weights U16 as well), which keeps
this diffusion + FFT graph stable. If accuracy is not enough, switch individual
op types (e.g. MatMul) to S16/U8 or add a second layer config.

usage:
  python prepare_compile.py --model export/model.onnx --calib calib/data \
      --out compile/pulsar2_config.json
"""
import argparse
import glob
import json
import os
import tarfile

NAMES = ["x", "audio_low", "band_onehot", "c_skip", "c_in", "c_out", "noise_emb"]


def pack(calib_dir, tar_dir):
    os.makedirs(tar_dir, exist_ok=True)
    info = {}
    for n in NAMES:
        files = sorted(glob.glob(os.path.join(calib_dir, n, "*.npy")))
        assert files, f"no calibration samples for input '{n}'"
        tar = os.path.join(tar_dir, f"{n}.tar.gz")
        with tarfile.open(tar, "w:gz") as tf:
            for f in files:
                tf.add(f, arcname=os.path.basename(f))
        info[n] = (tar, len(files))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="export/model.onnx")
    ap.add_argument("--calib", default="calib/data")
    ap.add_argument("--tar-dir", default="export/calib_data")
    ap.add_argument("--out", default="compile/pulsar2_config.json")
    ap.add_argument("--target-hardware", default="AX650")
    ap.add_argument("--npu-mode", default="NPU3", choices=["NPU1", "NPU2", "NPU3"])
    ap.add_argument("--data-type", default="U16", help="activation/output type")
    ap.add_argument("--weight-type", default="S8", help="weight type (S8 or FP32)")
    ap.add_argument("--chunk", type=int, default=32768)
    args = ap.parse_args()

    info = pack(args.calib, args.tar_dir)
    n_samples = info["x"][1]
    shapes = {"x": f"x:1x{args.chunk}", "audio_low": f"audio_low:1x{args.chunk}",
              "band_onehot": "band_onehot:1x2x513", "noise_emb": "noise_emb:1x512"}
    cfg = {
        "input": os.path.abspath(args.model),
        "output_dir": os.path.abspath(os.path.dirname(args.out)),
        "output_name": "fastwave_nfe.axmodel",
        "work_dir": os.path.abspath(os.path.join(os.path.dirname(args.out), "work")),
        "model_type": "ONNX",
        "target_hardware": args.target_hardware,
        "npu_mode": args.npu_mode,
        "input_shapes": ";".join(shapes.get(k, f"{k}:1x1") for k in NAMES),
        "onnx_opt": {"disable_onnx_optimization": False, "enable_onnxsim": False,
                     "model_check": True},
        "quant": {
            "calibration_method": "MinMax",
            "input_configs": [
                {"tensor_name": k, "calibration_dataset": os.path.abspath(info[k][0]),
                 "calibration_format": "Numpy", "calibration_size": n_samples,
                 "calibration_mean": [], "calibration_std": []}
                for k in NAMES
            ],
            # "DEFAULT" = 全部张量；U16 激活 / S8 权重 / U16 输出
            "layer_configs": [
                {"start_tensor_names": ["DEFAULT"], "end_tensor_names": ["DEFAULT"],
                 "data_type": args.data_type, "weight_data_type": args.weight_type,
                 "output_data_type": args.data_type}
            ],
            "highest_mix_precision": False,
            "precision_analysis": True,
            "precision_analysis_method": "EndToEnd",
        },
        "input_processors": [
            {"tensor_name": k, "tensor_layout": "NCHW", "src_dtype": "FP32",
             "src_layout": "NCHW"} for k in NAMES
        ],
        # check=0 不仿真（编译最快），精度直接上板验证；量化有问题再改 2（统计）/3（余弦）
        "compiler": {"check": 0},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"config -> {args.out}  ({n_samples} calibration samples, "
          f"{args.data_type} activations / {args.weight_type} weights)")


if __name__ == "__main__":
    main()
