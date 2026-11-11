#!/usr/bin/env python3
"""Build a real-data calibration set for the FastWave NFE graph.

For every chunk of real 48 kHz speech we run the actual EDM sampler (8 steps) and
store the tensors that reach the AXMODEL at every step. This makes the MinMax
calibration match the real inference distribution (x ranges from ~N(0, 80) at
step 0 down to ~N(0, 1) at the last step).

usage:
  python make_calib.py --wav-dir calib/wav48 --out-dir calib/data
"""
import argparse
import glob
import json
import os

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from fw_model import build_nfe, band_onehot, edm_sampler

NAMES = ["x", "audio_low", "band_onehot", "c_skip", "c_in", "c_out", "noise_emb"]
SR = 48000


def resample_exact(x, orig_sr, target_sr, target_len):
    out = resample_poly(x, target_sr, orig_sr).astype(np.float32)
    if len(out) > target_len:
        out = out[:target_len]
    elif len(out) < target_len:
        out = np.pad(out, (0, target_len - len(out)))
    return out


def low_quality(hr, in_sr):
    low = resample_exact(hr, SR, in_sr, int(round(len(hr) * in_sr / SR)))
    return resample_exact(low, in_sr, SR, len(hr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="origin/checkpoint-epoch140.pth")
    ap.add_argument("--wav-dir", default="calib/wav48")
    ap.add_argument("--out-dir", default="calib/data")
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--chunks-per-sr", type=int, default=2)
    ap.add_argument("--input-srs", default="8000,12000,16000,24000")
    args = ap.parse_args()

    net = build_nfe(args.ckpt, chunk=args.chunk)
    T = args.chunk
    files = sorted(glob.glob(os.path.join(args.wav_dir, "*.wav")))
    assert files, f"no wav in {args.wav_dir}"
    for n in NAMES:
        os.makedirs(os.path.join(args.out_dir, n), exist_ok=True)

    idx, meta = 0, []
    for in_sr in [int(s) for s in args.input_srs.split(",")]:
        band = band_onehot(in_sr / SR)
        for ci in range(args.chunks_per_sr):
            fn = files[(idx + ci) % len(files)]
            wav, sr = sf.read(fn, dtype="float32", always_2d=True)
            assert sr == SR, f"{fn} must be 48 kHz"
            wav = wav.mean(axis=1)
            wav = wav / (np.max(np.abs(wav)) + 1e-8)
            start = (ci * 7 + in_sr // 4000) * T % max(1, len(wav) - T - 1)
            hr = wav[start:start + T]
            if len(hr) < T:
                continue
            low = low_quality(hr, in_sr)
            with torch.no_grad():
                noise = torch.randn(1, T)
                _, rec = edm_sampler(net, noise, torch.from_numpy(low)[None], band,
                                     num_steps=8, trace=True)
            print(f"chunk {idx}: {os.path.basename(fn)} in_sr={in_sr}")
            for step, (x, low_t, band_t, cs, ci_, co, emb) in enumerate(rec):
                vals = {"x": x, "audio_low": low_t, "band_onehot": band_t,
                        "c_skip": cs[:, None], "c_in": ci_[:, None],
                        "c_out": co[:, None], "noise_emb": emb}
                for n, v in vals.items():
                    np.save(os.path.join(args.out_dir, n, f"{idx*8+step:04d}.npy"),
                            v.numpy().astype(np.float32))
            meta.append({"file": os.path.basename(fn), "in_sr": in_sr, "start": int(start)})
            idx += 1

    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"chunks": meta, "n_samples": idx * 8, "chunk_samples": T}, f, indent=2)
    print("total calibration samples:", idx * 8)


if __name__ == "__main__":
    main()
