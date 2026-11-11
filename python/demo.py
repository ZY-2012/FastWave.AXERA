#!/usr/bin/env python3
"""FastWave.AXERA demo: any sample rate -> 48 kHz super resolution on AX650 NPU."""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fastwave_sdk import FastWaveSR  # noqa: E402
from fastwave_sdk import audio as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="FastWave.AXERA 48 kHz audio super resolution")
    ap.add_argument("input", help="input wav (6 kHz - 48 kHz)")
    ap.add_argument("output", help="output wav (48 kHz)")
    ap.add_argument("--model", default=os.path.join(HERE, "..", "models", "fastwave_nfe.axmodel"))
    ap.add_argument("--chunk", type=int, default=None, help="chunk samples at 48 kHz (must match the axmodel, default from model_config.json)")
    ap.add_argument("--overlap", type=float, default=0.5, help="chunk overlap ratio")
    ap.add_argument("--steps", type=int, default=8, help="EDM sampling steps (4 or 8)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-denoise", action="store_true", help="disable the output noise post filter")
    args = ap.parse_args()

    wav, sr = A.read_wav(args.input)
    print(f"[input ] {args.input}  {len(wav)/sr:.2f}s @ {sr} Hz")
    model = FastWaveSR(args.model, num_steps=args.steps, chunk=args.chunk,
                       overlap=args.overlap, seed=args.seed)
    t0 = time.time()
    out = model.enhance(wav, sr, denoise=not args.no_denoise,
                        progress=lambda i, n: print(f"\r[chunk ] {i}/{n}", end="", flush=True))
    dt = time.time() - t0
    print()
    A.write_wav(args.output, out, 48000)
    dur = len(out) / 48000.0
    print(f"[output] {args.output}  {dur:.2f}s @ 48000 Hz")
    print(f"[perf  ] {dt:.2f}s  RTF {dt / dur:.3f}")


if __name__ == "__main__":
    main()
