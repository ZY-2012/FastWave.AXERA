"""NPU-friendly STFT/ISTFT for FastWave.

Matches torch.stft / torch.istft exactly for
  n_fft = win_length = 1024, hop_length = 256, hann window,
  center=True, pad_mode='reflect', normalized=True, onesided=True

FFT 1024 = 32 x 32 Cooley-Tukey, all complex math on real tensors so the
exported ONNX only uses MatMul / Reshape / Transpose / Mul / Add / Sub /
Gather / Concat / Slice -- no complex dtype, no FFT operator.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_FFT = 1024
HOP = 256
FREQ = N_FFT // 2 + 1  # 513


def _cos_sin(n, denom):
    i = torch.arange(n, dtype=torch.float64)
    ang = -2.0 * math.pi * i[:, None] * i[None, :] / denom
    return torch.cos(ang).float(), torch.sin(ang).float()


class FFT1024(nn.Module):
    """Radix-32 Cooley-Tukey FFT on the last dim (size 1024)."""

    def __init__(self, dtype=torch.float32):
        super().__init__()
        c, s = _cos_sin(32, 32)
        self.register_buffer("c32", c.to(dtype))
        self.register_buffer("s32", s.to(dtype))
        tw_r, tw_i = _cos_sin(32, 1024)
        self.register_buffer("tw_r", tw_r.to(dtype))
        self.register_buffer("tw_i", tw_i.to(dtype))

    def _cmatmul(self, x_re, x_im, m_re, m_im):
        """(..., m, 32) @ (m_re + i m_im)[32, 32] -> (..., m, 32)"""
        return (x_re @ m_re - x_im @ m_im, x_re @ m_im + x_im @ m_re)

    def rfft(self, x):
        """x: (..., 1024) real -> (re, im) each (..., 513)."""
        shp = x.shape[:-1]
        xr = x.reshape(-1, 32, 32).transpose(-1, -2)          # [n0, n1]
        a_re = xr @ self.c32                                  # [n0, k0]
        a_im = xr @ self.s32
        b_re = a_re * self.tw_r - a_im * self.tw_i
        b_im = a_re * self.tw_i + a_im * self.tw_r
        c_re, c_im = self._cmatmul(b_re.transpose(-1, -2), b_im.transpose(-1, -2),
                                   self.c32, self.s32)        # [k0, k1]
        out_re = c_re.transpose(-1, -2).reshape(*shp, 1024)
        out_im = c_im.transpose(-1, -2).reshape(*shp, 1024)
        return out_re[..., :FREQ], out_im[..., :FREQ]

    def fft_full(self, x_re, x_im):
        """(..., 1024) complex -> (..., 1024) complex"""
        shp = x_re.shape[:-1]
        a_re = x_re.reshape(-1, 32, 32).transpose(-1, -2)     # [n0, n1]
        a_im = x_im.reshape(-1, 32, 32).transpose(-1, -2)
        b_re, b_im = self._cmatmul(a_re, a_im, self.c32, self.s32)
        t_re = b_re * self.tw_r - b_im * self.tw_i
        t_im = b_re * self.tw_i + b_im * self.tw_r
        c_re, c_im = self._cmatmul(t_re.transpose(-1, -2), t_im.transpose(-1, -2),
                                   self.c32, self.s32)
        return (c_re.transpose(-1, -2).reshape(*shp, 1024),
                c_im.transpose(-1, -2).reshape(*shp, 1024))


class STFT(nn.Module):
    """torch.stft-compatible STFT/ISTFT (static frame count)."""

    def __init__(self, n_frames, dtype=torch.float32):
        super().__init__()
        self.n_frames = n_frames
        self.fft = FFT1024(dtype)
        w = torch.hann_window(N_FFT, dtype=dtype)
        self.register_buffer("win", w)
        idx = torch.arange(N_FFT, dtype=torch.long)
        # one-sided -> hermitian full spectrum
        rev = torch.cat([torch.arange(0, FREQ), torch.arange(FREQ - 2, 0, -1)])
        sign = torch.cat([torch.ones(FREQ), -torch.ones(FREQ - 2)])
        self.register_buffer("herm_idx", rev)
        self.register_buffer("herm_sign", sign)
        # reflect-pad index tensors (pad 512 each side)
        self.register_buffer("pad_l", torch.arange(N_FFT // 2, 0, -1))
        # right indices depend on signal length -> computed once for the fixed size
        out_len = (n_frames - 1) * HOP + N_FFT
        self.register_buffer("out_len", torch.tensor(out_len))
        # precomputed reciprocal window envelope for the fixed frame count
        env = self._env(out_len, torch.device("cpu"), dtype)
        self.register_buffer("env_recip", (1.0 / env).reshape(1, -1))

    # ---------- reflect padding ----------
    def _reflect_pad(self, x, T):
        left = x[:, self.pad_l]                   # (M, 512) = x[512..1]
        right = x[:, T - 1 - self.pad_l.flip(0)]  # x[T-2 .. T-513]
        return torch.cat([left, x, right], dim=-1)

    # ---------- STFT ----------
    def stft(self, x):
        """x: (M, T) -> (re, im) each (M, 513, n_frames)"""
        T = x.shape[-1]
        xp = self._reflect_pad(x, T)              # (M, T+1024)
        # framing with 4 shifted strided slices (unfold is not ONNX friendly)
        n = self.n_frames
        parts = [xp[:, r * HOP: r * HOP + n * HOP].reshape(-1, n, HOP) for r in range(4)]
        frames = torch.cat(parts, dim=-1) * self.win     # (M, n_frames, 1024)
        re, im = self.fft.rfft(frames)
        scale = 1.0 / math.sqrt(N_FFT)
        return re.transpose(1, 2) * scale, im.transpose(1, 2) * scale

    # ---------- ISTFT ----------
    def istft(self, re, im):
        """re, im: (M, 513, n_frames) -> (M, T)"""
        fre = re.transpose(1, 2)[:, :, self.herm_idx]
        fim = im.transpose(1, 2)[:, :, self.herm_idx]
        fim = fim * self.herm_sign
        # ifft: conj(fft(conj(X))) ; only real part needed
        z_re, _ = self.fft.fft_full(fre, -fim)
        y = z_re * (1.0 / math.sqrt(N_FFT))
        y = y * self.win                          # (M, n_frames, 1024)
        # overlap-add with 4 shifted block sums
        yb = y.reshape(y.shape[0], self.n_frames, 4, HOP)
        acc = None
        for r in range(4):
            part = yb[:, :, r, :]                 # (M, F, 256)
            pad = F.pad(part, (0, 0, r, 3 - r))
            acc = pad if acc is None else acc + pad
        out = acc.reshape(acc.shape[0], -1)        # (M, out_len)
        out = out * self.env_recip.to(out.dtype)
        s = N_FFT // 2
        return out[:, s:s + (self.n_frames - 1) * HOP]

    def _env(self, length, device, dtype):
        n = int(round((length - N_FFT) / HOP)) + 1
        w2 = self.win.to(dtype) ** 2
        frames = torch.zeros(n, N_FFT, device=device, dtype=dtype)
        frames[:] = w2
        yb = frames.reshape(n, 4, HOP)
        acc = None
        for r in range(4):
            part = yb[:, r, :]
            pad = F.pad(part, (0, 0, r, 3 - r))
            acc = pad if acc is None else acc + pad
        acc = acc.reshape(-1)
        return acc.clamp_min(1e-11)
