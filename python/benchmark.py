#!/usr/bin/env python3
"""Board benchmark: single-NFE latency and end-to-end RTF."""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fastwave_sdk import FastWaveSR  # noqa: E402
from fastwave_sdk import audio as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "..", "models", "fastwave_nfe.axmodel"))
    ap.add_argument("--wav", default=os.path.join(HERE, "..", "samples", "listen", "8k",
                                                  "OSR_us_000_0010_input_8k.wav"))
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--overlap", type=float, default=0.5)
    a = ap.parse_args()

    m = FastWaveSR(a.model, num_steps=a.steps, overlap=a.overlap)
    wav, sr = A.read_wav(a.wav)
    wav = wav[: int(a.seconds * sr)]
    low = A.low_quality(wav / (np.abs(wav).max() + 1e-8), sr)
    band = A.band_onehot(sr).reshape(1, 2, -1)
    seg = np.ascontiguousarray(low[: m.chunk].reshape(1, -1))
    x = np.zeros((1, m.chunk), dtype=np.float32)

    for _ in range(5):
        m.nfe(x, seg, band, 0)
    ts = []
    for i in range(a.iters):
        t0 = time.perf_counter()
        m.nfe(x, seg, band, i % m.num_steps)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    print(f"NFE latency: mean {ts.mean():.2f} ms  min {ts.min():.2f}  max {ts.max():.2f}")

    t0 = time.perf_counter()
    out = m.enhance(wav, sr)
    dt = time.perf_counter() - t0
    dur = len(out) / 48000.0
    print(f"end-to-end {dur:.2f}s audio in {dt:.2f}s -> RTF {dt/dur:.4f}")


if __name__ == "__main__":
    main()
