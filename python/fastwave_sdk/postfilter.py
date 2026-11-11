"""Light spectral-subtraction post filter for the FastWave output hiss (numpy).

The diffusion model leaves a broadband noise floor (mostly above the input band).
We estimate the noise magnitude per frequency from the quietest frames of the
output itself and apply a smoothed Wiener-like gain.
"""
import numpy as np

from .audio import hann


def denoise(x, sr=48000, n_fft=1024, hop=256, alpha=1.2, floor=0.10, quiet_percentile=20,
            smooth=3):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < n_fft:
        return x
    win = hann(n_fft)
    m = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(m)[:, None]
    fr = x[idx] * win
    X = np.fft.rfft(fr, axis=1)
    mag = np.abs(X)

    noise = np.percentile(mag, quiet_percentile, axis=0)
    gain = 1.0 - alpha * noise[None, :] / np.maximum(mag, 1e-8)
    gain = np.clip(gain, floor, 1.0)
    if smooth > 1 and m >= smooth:
        k = np.ones(smooth) / smooth
        gain = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, gain)

    Y = X * gain
    frames = np.fft.irfft(Y, n=n_fft, axis=1) * win

    out_len = (m - 1) * hop + n_fft
    acc = np.zeros(out_len, dtype=np.float64)
    wsum = np.zeros(out_len, dtype=np.float64)
    for i in range(m):
        acc[i * hop: i * hop + n_fft] += frames[i]
        wsum[i * hop: i * hop + n_fft] += win ** 2
    acc /= np.maximum(wsum, 1e-8)
    return acc[: len(x)].astype(np.float32)
