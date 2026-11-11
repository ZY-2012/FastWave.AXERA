"""Numpy-only audio helpers for the FastWave AX650 SDK.

- WAV read/write (PCM16 / PCM24 / float32) without soundfile
- polyphase resampling (scipy.signal.resample_poly compatible)
- band one-hot, hann window, overlap-add

Only numpy is required on the board.
"""
import math

import numpy as np

SR_OUT = 48000
N_FFT = 1024
HOP = 256
FREQ = N_FFT // 2 + 1
CHUNK = 32768


# --------------------------------------------------------------------------- WAV
def read_wav(path):
    """Return (float32 mono in [-1, 1], sample_rate)."""
    import wave

    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        sw = w.getsampwidth()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(np.float32) / (1 << 23)
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / (1 << 31)
    else:
        raise ValueError(f"unsupported sample width: {sw}")
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), sr


def write_wav(path, x, sr):
    import wave

    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


# --------------------------------------------------------------------- resampling
def _firwin(cutoff, numtaps, beta=5.0):
    """Lowpass FIR (same design as scipy.signal.firwin(..., window=('kaiser', beta)))."""
    n = np.arange(numtaps)
    alpha = (numtaps - 1) / 2.0
    m = n - alpha
    h = cutoff * np.sinc(cutoff * m)
    h *= np.kaiser(numtaps, beta)
    h /= h.sum()
    return h


def resample_poly(x, up, down):
    """Polyphase rational resampling, scipy.signal.resample_poly compatible."""
    x = np.asarray(x, dtype=np.float64)
    g = math.gcd(int(up), int(down))
    up, down = int(up) // g, int(down) // g
    if up == down:
        return x.astype(np.float32)
    max_rate = max(up, down)
    half = 10 * max_rate
    h = (_firwin(1.0 / max_rate, 2 * half + 1) * up).astype(np.float64)
    hl = len(h)
    n_out = int(math.ceil(len(x) * up / down))
    # polyphase evaluation: out[k] = sum_j h[j] * x[(half + k*down - j)/up]
    base = half + np.arange(n_out, dtype=np.int64) * down
    m0 = (base - hl + 1 + up - 1) // up
    m0 = np.maximum(m0, 0)
    acc = np.zeros(n_out, dtype=np.float64)
    for t in range(hl // up + 2):
        m = m0 + t
        j = base - m * up
        valid = (j >= 0) & (j < hl) & (m < len(x))
        acc[valid] += h[j[valid]] * x[m[valid]]
    return acc.astype(np.float32)


def resample_to(x, sr_in, sr_out):
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32).copy()
    return resample_poly(x, int(sr_out), int(sr_in))


def low_quality(x, sr_in, out_sr=SR_OUT):
    """Upsample the band-limited input to out_sr (same role as the training pipeline)."""
    n_out = int(round(len(x) * out_sr / sr_in))
    y = resample_to(x, sr_in, out_sr)
    if len(y) > n_out:
        y = y[:n_out]
    elif len(y) < n_out:
        y = np.pad(y, (0, n_out - len(y)))
    return y


# ------------------------------------------------------------------------ misc
def band_onehot(sr_in, sr_out=SR_OUT, freq=FREQ):
    """(2, freq) float32 one-hot of usable frequency bins."""
    hi = float(sr_in) / float(sr_out)
    band = np.zeros(freq, dtype=np.int64)
    band[: int(hi * freq)] = 1
    # matches F.one_hot(band).transpose(1, -1): ch0 = high band, ch1 = low band
    out = np.zeros((2, freq), dtype=np.float32)
    out[0, band == 0] = 1.0
    out[1, band == 1] = 1.0
    return out


def hann(n):
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


def chunk_bounds(n, chunk, hop):
    """Start positions of overlapping chunks covering n samples."""
    if n <= chunk:
        return [0]
    starts = list(range(0, n - chunk + 1, hop))
    if starts[-1] + chunk < n:
        starts.append(n - chunk)
    return starts


def overlap_add(chunks, starts, n, chunk):
    """Hann-window overlap-add."""
    w = hann(chunk)
    acc = np.zeros(n + chunk, dtype=np.float64)
    wsum = np.zeros(n + chunk, dtype=np.float64)
    for c, s in zip(chunks, starts):
        acc[s: s + chunk] += c * w
        wsum[s: s + chunk] += w
    wsum = np.maximum(wsum, 1e-8)
    return (acc[:n] / wsum[:n]).astype(np.float32)
