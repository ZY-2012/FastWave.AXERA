"""FastWave EDM sampler on AX650 (pyaxengine) — numpy only.

The AXMODEL is one function evaluation (NFE):
  y = D(x, sigma) = c_skip * x + c_out * F(c_in * x, audio_low, band, noise_emb)

Everything that only depends on sigma (the 8 EDM steps) is precomputed on the
host: c_skip / c_in / c_out and the diffusion embedding.
"""
import json
import os

import numpy as np

from . import audio as A
from .postfilter import denoise as _denoise

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _silu(x):
    return x / (1.0 + np.exp(-x))


class FastWaveSR:
    def __init__(self, model_path, assets_dir=ASSETS, num_steps=8, provider="AxEngineExecutionProvider",
                 chunk=None, overlap=0.5, seed=1234, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        import axengine as axe

        self.sess = axe.InferenceSession(model_path, providers=[provider])
        self.input_names = [i.name for i in self.sess.get_inputs()]
        meta = json.load(open(os.path.join(assets_dir, "model_config.json")))
        self.meta = meta
        self.chunk = int(chunk or meta.get("chunk_samples", 32768))
        if self.chunk != int(meta.get("chunk_samples", 32768)):
            print(f"[warn] chunk={self.chunk} differs from the compiled axmodel "
                  f"({meta.get('chunk_samples')}); recompile the model for a new chunk size")
        self.overlap = float(overlap)
        self.hop = (max(256, int(round(self.chunk * overlap)) // 256 * 256)
                    if overlap > 0 else self.chunk)
        self.num_steps = int(num_steps)
        self.rng = np.random.default_rng(seed)

        self.sr_out = int(meta.get("sample_rate", A.SR_OUT))

        p1w = np.load(os.path.join(assets_dir, "emb_p1_w.npy"))
        p1b = np.load(os.path.join(assets_dir, "emb_p1_b.npy"))
        p2w = np.load(os.path.join(assets_dir, "emb_p2_w.npy"))
        p2b = np.load(os.path.join(assets_dir, "emb_p2_b.npy"))
        half = int(p1w.shape[1]) // 2
        scale = float(meta.get("pos_emb_scale", 50000.0))
        freqs = np.exp(np.arange(half, dtype=np.float64) * (-np.log(10000.0) / (half - 1)))

        idx = np.arange(self.num_steps, dtype=np.float64)
        t = (sigma_max ** (1 / rho) + idx / (self.num_steps - 1)
             * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        self.t_steps = np.concatenate([t, [0.0]]).astype(np.float32)

        self.steps = []
        for s in self.t_steps[:-1].astype(np.float64):
            c_skip = 0.5 ** 2 / (s ** 2 + 0.5 ** 2)
            c_in = 1.0 / np.sqrt(s ** 2 + 0.5 ** 2)
            c_out = s * 0.5 / np.sqrt(s ** 2 + 0.5 ** 2)
            c_noise = np.log(s) / 4.0
            ang = scale * c_noise * freqs
            emb = np.concatenate([np.sin(ang), np.cos(ang)]).astype(np.float32)
            emb = _silu(emb @ p1w.T + p1b)
            emb = _silu(emb @ p2w.T + p2b)
            self.steps.append({
                "c_skip": np.array([[c_skip]], dtype=np.float32),
                "c_in": np.array([[c_in]], dtype=np.float32),
                "c_out": np.array([[c_out]], dtype=np.float32),
                "noise_emb": emb.reshape(1, -1).astype(np.float32),
            })

    # ------------------------------------------------------------------ graph
    def nfe(self, x, low, band, step):
        """One denoising step; x/low: (1, chunk) float32."""
        s = self.steps[step]
        feed = {
            "x": np.ascontiguousarray(x, dtype=np.float32),
            "audio_low": low,
            "band_onehot": band,
            "c_skip": s["c_skip"],
            "c_in": s["c_in"],
            "c_out": s["c_out"],
            "noise_emb": s["noise_emb"],
        }
        return self.sess.run(None, {k: feed[k] for k in self.input_names})[0]

    def denoise_chunk(self, low_chunk, band):
        x = self.rng.standard_normal((1, self.chunk)).astype(np.float32) * self.t_steps[0]
        for i in range(self.num_steps):
            t_cur = self.t_steps[i]
            t_next = self.t_steps[i + 1]
            denoised = self.nfe(x, low_chunk, band, i)
            x = x + (t_next - t_cur) * (x - denoised) / t_cur
        return np.clip(x[0], -1.0, 1.0 - 1e-7)

    # --------------------------------------------------------------- pipeline
    def enhance(self, wav, sr_in, progress=None, normalize=True, denoise=True, denoise_alpha=1.0):
        """wav: 1-D float32 at sr_in. Returns 48 kHz float32."""
        wav = np.asarray(wav, dtype=np.float32)
        peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
        if normalize and peak > 1e-8:
            wav = wav / peak
        else:
            peak = 1.0
        low = A.low_quality(wav, int(sr_in), self.sr_out)
        band = A.band_onehot(int(sr_in), self.sr_out)
        band3 = band.reshape(1, 2, -1)
        n = len(low)
        if n == 0:
            return low
        if self.overlap <= 0:
            # fastest: pad to whole chunks, no overlap / no windowing
            n_pad = int(np.ceil(n / self.chunk)) * self.chunk
            if n_pad > n:
                low = np.pad(low, (0, n_pad - n))
            starts = list(range(0, n_pad, self.chunk))
            outs = []
            for i, s in enumerate(starts):
                seg = low[s: s + self.chunk]
                outs.append(self.denoise_chunk(
                    np.ascontiguousarray(seg.reshape(1, -1)), band3))
                if progress:
                    progress(i + 1, len(starts))
            y = np.concatenate(outs)[:n].astype(np.float32)
            if denoise:
                y = _denoise(y, self.sr_out, alpha=denoise_alpha)
            return y * peak
        starts = A.chunk_bounds(n, self.chunk, self.hop)
        outs = []
        for i, s in enumerate(starts):
            seg = low[s: s + self.chunk]
            if len(seg) < self.chunk:
                seg = np.pad(seg, (0, self.chunk - len(seg)))
            outs.append(self.denoise_chunk(np.ascontiguousarray(seg.reshape(1, -1)), band3))
            if progress:
                progress(i + 1, len(starts))
        if len(starts) == 1:
            y = outs[0][:n].astype(np.float32)
        else:
            y = A.overlap_add(outs, starts, n, self.chunk)
        if denoise:
            y = _denoise(y, self.sr_out, alpha=denoise_alpha)
        return y * peak
