"""FastWave one-NFE (function evaluation) graph for AX NPU + host-side EDM sampler.

Model surgery relative to the upstream implementation (MIT, see baseline_model.py):
  1. STFT / iSTFT inside EfficientFourierUnit are replaced by a matmul-based
     Cooley-Tukey FFT (fw_spectral.py) so the graph has no FFT / complex ops
     and can run on the AX NPU.
  2. band one-hot and the EDM preconditioning coefficients are graph inputs
     (no OneHot / log / sqrt on device).
  3. the sinusoidal diffusion embedding is precomputed on the host and fed as
     `noise_emb` (removes Exp / Sin / Cos / Gemm from the graph).
  4. GRN uses an explicit L2 norm (ReduceSum + Sqrt instead of ReduceL2).
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import baseline_model as bm  # noqa: E402
from fw_spectral import STFT, FREQ  # noqa: E402


class FourierUnitNPU(nn.Module):
    """Same parameter names as EfficientFourierUnit, STFT via matmul FFT."""

    def __init__(self, orig, n_frames):
        super().__init__()
        self.conv_layer = orig.conv_layer
        self.bsft = orig.bsft
        self.register_buffer("hann_window", orig.hann_window)
        self.stft = STFT(n_frames)

    def forward(self, x, band):
        batch = x.shape[0]
        n_ch = x.shape[1]
        t_len = x.shape[-1]
        x = x.reshape(-1, t_len)
        re, im = self.stft.stft(x)                      # (M, 513, F)
        ffted = torch.stack((re, im), dim=1)            # (M, 2, 513, F)
        ffted = ffted.reshape(batch, -1, FREQ, self.stft.n_frames)
        ffted = bm.relu(self.bsft(ffted, band))
        ffted = self.conv_layer(ffted)
        ffted = ffted.reshape(-1, 2, FREQ, self.stft.n_frames)
        out = self.stft.istft(ffted[:, 0], ffted[:, 1])  # (M, T)
        return out.reshape(batch, n_ch, t_len)


def _grn_forward(self, x):
    gx = torch.sqrt((x * x).sum(dim=-1, keepdim=True))
    nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
    return self.gamma * (x * nx) + self.beta + x


def build_nuwave2(hparams, n_frames):
    bm.GlobalResponseNorm.forward = _grn_forward
    model = bm.NuWave2(hparams)
    for layer in model.residual_layers:
        fu = layer.ffc1.convg2g.fu
        layer.ffc1.convg2g.fu = FourierUnitNPU(fu, n_frames)
    return model


class FastWaveNFE(nn.Module):
    """(x, audio_low, band_onehot, c_skip, c_in, c_out, noise_emb) -> denoised

    x, audio_low              : (1, T) float32
    band_onehot               : (1, 2, 513) float32
    c_skip / c_in / c_out     : (1, 1) float32
    noise_emb                 : (1, 512) float32
    """

    def __init__(self, hparams, n_frames, return_fx=False):
        super().__init__()
        self.model = build_nuwave2(hparams, n_frames)
        self.sqrt_len_res = math.sqrt(len(self.model.residual_layers))
        self.return_fx = return_fx

    def load_fastwave_state(self, state_dict):
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        extra = [k for k in missing if ".stft." not in k]
        assert not extra, f"unexpected missing keys: {extra[:5]}"
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"
        return missing

    def forward(self, x, audio_low, band_onehot, c_skip, c_in, c_out, noise_emb):
        h = torch.stack((c_in * x, audio_low), dim=1)
        h = self.model.input_projection(h)
        h = F.silu(h)
        skip = 0.0
        for layer in self.model.residual_layers:
            h, skip_connection = layer(h, band_onehot, noise_emb)
            skip = skip + skip_connection
        h = skip / self.sqrt_len_res
        h = self.model.skip_projection(h)
        h = F.silu(h)
        f_x = self.model.output_projection(h).squeeze(1)
        if self.return_fx:
            return f_x
        return c_skip * x + c_out * f_x


# ---------------------------------------------------------------- host helpers
def load_checkpoint(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = OmegaConf.create(ck["config"]["model"]["hparams"])
    return hparams, ck["state_dict"]


def build_nfe(ckpt_path, chunk=32768, hop=256):
    assert chunk % hop == 0, "chunk must be a multiple of hop_length"
    hparams, state = load_checkpoint(ckpt_path)
    net = FastWaveNFE(hparams, n_frames=chunk // hop + 1)
    net.load_fastwave_state(state)
    net.eval()
    return net


def edm_coeffs(sigma, sigma_data=0.5):
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_in = 1.0 / torch.sqrt(sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data / torch.sqrt(sigma ** 2 + sigma_data ** 2)
    c_noise = sigma.log() / 4.0
    return c_skip, c_in, c_out, c_noise


def sigma_steps(num_steps=8, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    idx = torch.arange(num_steps, dtype=torch.float64)
    t = (sigma_max ** (1 / rho) + idx / (num_steps - 1)
         * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([t, torch.zeros(1, dtype=torch.float64)]).float()


def band_onehot(hi_ratio, freq=FREQ):
    """hi_ratio = input_sr / 48000"""
    band = torch.zeros(1, freq, dtype=torch.long)
    band[:, : int(hi_ratio * freq)] = 1
    return F.one_hot(band).transpose(1, -1).float()


@torch.no_grad()
def edm_sampler(net, x_init, audio_low, band, num_steps=8, sigma_min=0.002,
                sigma_max=80.0, rho=7.0, trace=False):
    ts = sigma_steps(num_steps, sigma_min, sigma_max, rho)
    x_next = x_init.to(torch.float32) * ts[0]
    records = []
    for i in range(num_steps):
        t_cur, t_next = ts[i], ts[i + 1]
        sigma = t_cur.expand(x_next.shape[0])
        c_skip, c_in, c_out, c_noise = edm_coeffs(sigma)
        emb = net.model.diffusion_embedding(c_noise[:, None])
        if trace:
            records.append((x_next.detach().clone(), audio_low.detach().clone(),
                            band.detach().clone(), c_skip.clone(), c_in.clone(),
                            c_out.clone(), emb.detach().clone()))
        denoised = net(x_next, audio_low, band, c_skip[:, None], c_in[:, None],
                       c_out[:, None], emb).to(torch.float32)
        d = (x_next - denoised) / t_cur
        x_next = x_next + (t_next - t_cur) * d
    return x_next, records
