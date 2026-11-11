#!/usr/bin/env python3
"""Extract host-side assets (diffusion embedding MLP) from the FastWave checkpoint.

The NPU graph receives `noise_emb` as an input, so the sinusoidal embedding +
two-layer MLP (128 -> 512 -> 512) runs on the host. Those weights are saved as
npy files next to the Python SDK.
"""
import argparse
import json
import os

import numpy as np

from fw_model import load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="origin/checkpoint-epoch140.pth")
ap.add_argument("--out", default="../python/fastwave_sdk/assets")
a = ap.parse_args()

hparams, sd = load_checkpoint(a.ckpt)
os.makedirs(a.out, exist_ok=True)
for key, name in (("projection1.weight", "emb_p1_w"), ("projection1.bias", "emb_p1_b"),
                  ("projection2.weight", "emb_p2_w"), ("projection2.bias", "emb_p2_b")):
    np.save(os.path.join(a.out, name + ".npy"),
            sd[f"model.diffusion_embedding.{key}"].detach().cpu().numpy().astype(np.float32))

cfg = {
    "name": "FastWave", "sample_rate": 48000, "chunk_samples": 32768,
    "n_fft": 1024, "hop_length": 256, "num_steps": 8,
    "sigma_min": 0.002, "sigma_max": 80.0, "rho": 7.0,
    "pos_emb_channels": int(hparams.dpm.pos_emb_channels),
    "pos_emb_scale": float(hparams.dpm.pos_emb_scale),
    "pos_emb_dim": int(hparams.arch.pos_emb_dim),
    "axmodel": "fastwave_nfe.axmodel",
}
with open(os.path.join(a.out, "model_config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
print("assets ->", a.out)
