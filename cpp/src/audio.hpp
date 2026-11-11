#pragma once
// Numpy-free audio helpers: WAV I/O, polyphase resampling (scipy resample_poly
// compatible), band one-hot, Hann window / overlap-add.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace fastwave {

constexpr int kSampleRate = 48000;
constexpr int kNFft = 1024;
constexpr int kHop = 256;
constexpr int kFreq = kNFft / 2 + 1;
constexpr int kChunk = 32768;

// ------------------------------------------------------------------ WAV I/O
static inline bool read_wav(const std::string& path, std::vector<float>& out, int& sr) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open wav %s\n", path.c_str());
        return false;
    }
    char riff[4], wave[4];
    uint32_t size = 0;
    if (std::fread(riff, 1, 4, f) != 4 || std::fread(&size, 4, 1, f) != 1 ||
        std::fread(wave, 1, 4, f) != 4 || std::memcmp(riff, "RIFF", 4) || std::memcmp(wave, "WAVE", 4)) {
        std::fprintf(stderr, "not a RIFF/WAVE file: %s\n", path.c_str());
        std::fclose(f);
        return false;
    }
    uint16_t fmt = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    std::vector<uint8_t> data;
    while (true) {
        char id[4];
        uint32_t sz = 0;
        if (std::fread(id, 1, 4, f) != 4) break;
        if (std::fread(&sz, 4, 1, f) != 1) break;
        if (!std::memcmp(id, "fmt ", 4)) {
            std::vector<uint8_t> buf(sz);
            if (std::fread(buf.data(), 1, sz, f) != sz) break;
            std::memcpy(&fmt, buf.data(), 2);
            std::memcpy(&channels, buf.data() + 2, 2);
            std::memcpy(&rate, buf.data() + 4, 4);
            std::memcpy(&bits, buf.data() + 14, 2);
            if (sz & 1) std::fseek(f, 1, SEEK_CUR);
        } else if (!std::memcmp(id, "data", 4)) {
            data.resize(sz);
            if (std::fread(data.data(), 1, sz, f) != sz) data.resize(0);
            if (sz & 1) std::fseek(f, 1, SEEK_CUR);
        } else {
            std::fseek(f, (long)((sz + 1) & ~1u), SEEK_CUR);
        }
    }
    std::fclose(f);
    if (data.empty() || channels == 0) {
        std::fprintf(stderr, "empty wav: %s\n", path.c_str());
        return false;
    }
    const size_t frames = data.size() / (channels * (bits / 8));
    out.assign(frames, 0.0f);
    for (size_t i = 0; i < frames; ++i) {
        float acc = 0.0f;
        for (int c = 0; c < channels; ++c) {
            size_t off = (i * channels + c) * (bits / 8);
            float v = 0.0f;
            if (bits == 16) {
                int16_t s;
                std::memcpy(&s, data.data() + off, 2);
                v = s / 32768.0f;
            } else if (bits == 24) {
                int32_t s = (int32_t)data[off] | ((int32_t)data[off + 1] << 8) |
                            ((int32_t)data[off + 2] << 16);
                if (s & 0x800000) s -= 0x1000000;
                v = s / 8388608.0f;
            } else if (bits == 32) {
                if (fmt == 3) {
                    std::memcpy(&v, data.data() + off, 4);
                } else {
                    int32_t s;
                    std::memcpy(&s, data.data() + off, 4);
                    v = s / 2147483648.0f;
                }
            } else {
                std::fprintf(stderr, "unsupported wav bit depth %d\n", bits);
                return false;
            }
            acc += v;
        }
        out[i] = acc / channels;
    }
    sr = static_cast<int>(rate);
    return true;
}

static inline bool write_wav(const std::string& path, const std::vector<float>& x, int sr) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    uint32_t data_bytes = static_cast<uint32_t>(x.size() * 2);
    uint32_t riff_size = 36 + data_bytes;
    uint16_t channels = 1, bits = 16, block_align = 2, fmt = 1;
    uint32_t byte_rate = static_cast<uint32_t>(sr) * 2;
    std::fwrite("RIFF", 1, 4, f);
    std::fwrite(&riff_size, 4, 1, f);
    std::fwrite("WAVEfmt ", 1, 8, f);
    uint32_t fmtsz = 16;
    std::fwrite(&fmtsz, 4, 1, f);
    std::fwrite(&fmt, 2, 1, f);
    std::fwrite(&channels, 2, 1, f);
    std::fwrite(&sr, 4, 1, f);
    std::fwrite(&byte_rate, 4, 1, f);
    std::fwrite(&block_align, 2, 1, f);
    std::fwrite(&bits, 2, 1, f);
    std::fwrite("data", 1, 4, f);
    std::fwrite(&data_bytes, 4, 1, f);
    std::vector<int16_t> pcm(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float v = std::max(-1.0f, std::min(1.0f, x[i]));
        pcm[i] = static_cast<int16_t>(std::lround(v * 32767.0f));
    }
    std::fwrite(pcm.data(), 2, pcm.size(), f);
    std::fclose(f);
    return true;
}

// -------------------------------------------------------------- resampling
static inline double bessel_i0(double x) {
    double s = 1.0, t = 1.0;
    for (int k = 1; k < 50; ++k) {
        t *= (x * 0.5 / k) * (x * 0.5 / k);
        s += t;
        if (t < 1e-12 * s) break;
    }
    return s;
}

static inline std::vector<float> firwin_lowpass(double cutoff, int numtaps, double beta = 5.0) {
    std::vector<float> h(numtaps);
    const double alpha = (numtaps - 1) / 2.0;
    const double denom = bessel_i0(beta);
    double sum = 0.0;
    for (int i = 0; i < numtaps; ++i) {
        double m = i - alpha;
        double sinc = (std::fabs(m) < 1e-12) ? cutoff : std::sin(M_PI * cutoff * m) / (M_PI * m);
        double r = 2.0 * i / (numtaps - 1) - 1.0;
        double w = bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / denom;
        h[i] = static_cast<float>(cutoff * sinc * w);
        sum += h[i];
    }
    for (auto& v : h) v = static_cast<float>(v / sum);
    return h;
}

static inline int gcd_int(int a, int b) { return b == 0 ? a : gcd_int(b, a % b); }

static inline std::vector<float> resample_poly(const std::vector<float>& x, int up, int down) {
    if (up == down || x.empty()) return x;
    int g = gcd_int(up, down);
    up /= g;
    down /= g;
    const int max_rate = std::max(up, down);
    const int half = 10 * max_rate;
    std::vector<float> h = firwin_lowpass(1.0 / max_rate, 2 * half + 1);
    for (auto& v : h) v = static_cast<float>(v * up);

    // polyphase evaluation: out[k] = sum_j h[j] * x[(half + k*down - j)/up]
    // (only terms where the upsampled index is an integer multiple of `up`)
    const size_t n_out = static_cast<size_t>(std::ceil(double(x.size()) * up / down));
    std::vector<float> out(n_out, 0.0f);
    const long hl = static_cast<long>(h.size());
    for (size_t k = 0; k < n_out; ++k) {
        const long base = static_cast<long>(half) + static_cast<long>(k) * down;  // j = base - m*up
        // valid m: 0 <= base - m*up <= hl-1  ->  (base-hl+1)/up <= m <= base/up
        long m_lo = (base - hl + 1 + up - 1) / up;
        if (m_lo < 0) m_lo = 0;
        long m_hi = base / up;
        if (m_hi >= static_cast<long>(x.size())) m_hi = static_cast<long>(x.size()) - 1;
        double acc = 0.0;
        for (long m = m_lo; m <= m_hi; ++m) {
            const long j = base - m * up;
            if (j >= 0 && j < hl) acc += double(h[j]) * x[m];
        }
        out[k] = static_cast<float>(acc);
    }
    return out;
}

static inline std::vector<float> low_quality(const std::vector<float>& x, int sr_in, int sr_out) {
    const size_t n_out = static_cast<size_t>(std::llround(double(x.size()) * sr_out / sr_in));
    std::vector<float> y = resample_poly(x, sr_out, sr_in);
    y.resize(n_out, 0.0f);
    return y;
}

// ---------------------------------------------------------------- helpers
static inline void band_onehot(int sr_in, std::vector<float>& out) {
    out.assign(2 * kFreq, 0.0f);
    const double hi = double(sr_in) / double(kSampleRate);
    const int n = int(hi * kFreq);
    for (int f = 0; f < kFreq; ++f) {
        if (f < n) out[kFreq + f] = 1.0f;  // ch1 = low band (matches F.one_hot(band).transpose)
        else out[f] = 1.0f;                // ch0 = high band
    }
}

static inline std::vector<float> hann_window(int n) {
    std::vector<float> w(n);
    for (int i = 0; i < n; ++i) w[i] = 0.5f - 0.5f * std::cos(2.0 * M_PI * i / n);
    return w;
}

static inline std::vector<size_t> chunk_bounds(size_t n, size_t chunk, size_t hop) {
    std::vector<size_t> starts;
    if (n <= chunk) {
        starts.push_back(0);
        return starts;
    }
    for (size_t s = 0; s + chunk <= n; s += hop) starts.push_back(s);
    if (starts.back() + chunk < n) starts.push_back(n - chunk);
    return starts;
}

static inline std::vector<float> overlap_add(const std::vector<std::vector<float>>& chunks,
                                             const std::vector<size_t>& starts, size_t n,
                                             size_t chunk) {
    std::vector<float> w = hann_window(int(chunk));
    std::vector<double> acc(n + chunk, 0.0), wsum(n + chunk, 0.0);
    for (size_t i = 0; i < chunks.size(); ++i) {
        const size_t s = starts[i];
        for (size_t j = 0; j < chunk; ++j) {
            acc[s + j] += double(chunks[i][j]) * w[j];
            wsum[s + j] += w[j];
        }
    }
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = static_cast<float>(acc[i] / std::max(1e-8, wsum[i]));
    return out;
}

}  // namespace fastwave
