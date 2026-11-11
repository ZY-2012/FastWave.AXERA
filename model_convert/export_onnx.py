#!/usr/bin/env python3
"""Export FastWave one-NFE graph to a static ONNX for Pulsar2.

usage:
  python export_onnx.py --ckpt origin/checkpoint-epoch140.pth --out export/model.onnx
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from fw_model import build_nfe, edm_coeffs, band_onehot

NAMES = ["x", "audio_low", "band_onehot", "c_skip", "c_in", "c_out", "noise_emb"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="origin/checkpoint-epoch140.pth")
    ap.add_argument("--out", default="export/model.onnx")
    ap.add_argument("--chunk", type=int, default=32768, help="chunk samples @48k (multiple of 256)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    net = build_nfe(args.ckpt, chunk=args.chunk)
    T = args.chunk
    band = band_onehot(16000 / 48000)
    cs, ci, co, cn = edm_coeffs(torch.tensor([4.0]))
    emb = net.model.diffusion_embedding(cn[:, None])
    args_in = (torch.randn(1, T) * 3.0, torch.randn(1, T) * 0.1, band,
               cs[:, None], ci[:, None], co[:, None], emb)

    with torch.no_grad():
        y = net(*args_in)
    print("torch output", tuple(y.shape), "std", float(y.std()))

    torch.onnx.export(net, args_in, args.out, opset_version=13, input_names=NAMES,
                      output_names=["y"], dynamo=False, do_constant_folding=True)
    print(f"onnx -> {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB)")

    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        feed = {n: a.detach().numpy() for n, a in zip(NAMES, args_in)}
        yo = torch.from_numpy(sess.run(None, feed)[0])
        cos = float(F.cosine_similarity(yo.flatten(), y.flatten(), dim=0))
        print(f"torch vs onnx: cosine {cos:.7f}  max abs err {float((yo-y).abs().max()):.2e}")
    except ImportError:
        pass

    meta = {
        "model": "FastWave",
        "task": "audio_super_resolution",
        "inputs": [{"name": n, "shape": list(a.shape), "dtype": "float32"}
                   for n, a in zip(NAMES, args_in)],
        "outputs": [{"name": "y", "shape": [1, T], "dtype": "float32"}],
        "chunk_samples": T,
        "sample_rate": 48000,
        "nfe": 8,
        "stft": {"n_fft": 1024, "hop": 256, "frames": T // 256 + 1},
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "model_meta.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
