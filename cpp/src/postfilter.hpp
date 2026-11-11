#pragma once
// Spectral-subtraction post filter (same behaviour as python/fastwave_sdk/postfilter.py).
#include <algorithm>
#include <cmath>
#include <vector>

#include "audio.hpp"

namespace fastwave {

// ---------------------------------------------------------- radix-2 FFT 1024
struct FFT {
    int n;
    std::vector<int> rev;
    std::vector<float> cos_t, sin_t;
    explicit FFT(int n_) : n(n_) {
        rev.resize(n);
        int lg = 0;
        while ((1 << lg) < n) ++lg;
        for (int i = 0; i < n; ++i) {
            int r = 0;
            for (int b = 0; b < lg; ++b)
                if (i & (1 << b)) r |= 1 << (lg - 1 - b);
            rev[i] = r;
        }
        cos_t.resize(n / 2);
        sin_t.resize(n / 2);
        for (int i = 0; i < n / 2; ++i) {
            cos_t[i] = std::cos(2.0 * M_PI * i / n);
            sin_t[i] = std::sin(2.0 * M_PI * i / n);
        }
    }
    void transform(std::vector<float>& re, std::vector<float>& im) const {
        for (int i = 0; i < n; ++i) {
            int j = rev[i];
            if (j > i) {
                std::swap(re[i], re[j]);
                std::swap(im[i], im[j]);
            }
        }
        for (int len = 2; len <= n; len <<= 1) {
            const int half = len >> 1;
            const int step = n / len;
            for (int i = 0; i < n; i += len) {
                for (int k = 0; k < half; ++k) {
                    const float wr = cos_t[k * step];
                    const float wi = -sin_t[k * step];
                    const int a = i + k, b = a + half;
                    const float tr = re[b] * wr - im[b] * wi;
                    const float ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
        }
    }
};

inline std::vector<float> postfilter(const std::vector<float>& x, int sr = kSampleRate,
                                     float alpha = 1.0f, float floor_gain = 0.15f,
                                     int quiet_percentile = 20) {
    const int n = kNFft, hop = kHop;
    if (static_cast<int>(x.size()) < n) return x;
    const int m = 1 + (static_cast<int>(x.size()) - n) / hop;
    const std::vector<float> win = hann_window(n);
    static const FFT fft(n);

    std::vector<float> re(n), im(n);
    std::vector<std::vector<float>> mag(m, std::vector<float>(n / 2 + 1));
    std::vector<std::vector<float>> ph_re(m, std::vector<float>(n / 2 + 1));
    std::vector<std::vector<float>> ph_im(m, std::vector<float>(n / 2 + 1));
    for (int f = 0; f < m; ++f) {
        for (int i = 0; i < n; ++i) {
            re[i] = x[f * hop + i] * win[i];
            im[i] = 0.0f;
        }
        fft.transform(re, im);
        for (int k = 0; k <= n / 2; ++k) {
            mag[f][k] = std::sqrt(re[k] * re[k] + im[k] * im[k]);
            ph_re[f][k] = re[k];
            ph_im[f][k] = im[k];
        }
    }
    // per-frequency noise estimate: <percentile> of magnitudes
    std::vector<float> noise(n / 2 + 1);
    std::vector<float> col(m);
    for (int k = 0; k <= n / 2; ++k) {
        for (int f = 0; f < m; ++f) col[f] = mag[f][k];
        const int idx = std::max(0, std::min(m - 1, (int)std::lround(quiet_percentile / 100.0 * (m - 1))));
        std::nth_element(col.begin(), col.begin() + idx, col.end());
        noise[k] = col[idx];
    }
    // gains + 3-frame smoothing
    std::vector<std::vector<float>> gain(m, std::vector<float>(n / 2 + 1, 1.0f));
    for (int f = 0; f < m; ++f)
        for (int k = 0; k <= n / 2; ++k) {
            const float g = 1.0f - alpha * noise[k] / std::max(1e-8f, mag[f][k]);
            gain[f][k] = std::min(1.0f, std::max(floor_gain, g));
        }
    std::vector<std::vector<float>> gs = gain;
    for (int f = 0; f < m; ++f) {
        for (int k = 0; k <= n / 2; ++k) {
            float acc = 0.0f, cnt = 0.0f;
            for (int d = -1; d <= 1; ++d) {
                const int ff = f + d;
                if (ff < 0 || ff >= m) continue;
                acc += gain[ff][k];
                cnt += 1.0f;
            }
            gs[f][k] = acc / cnt;
        }
    }

    std::vector<double> out(static_cast<size_t>(m - 1) * hop + n, 0.0);
    std::vector<double> wsum(static_cast<size_t>(m - 1) * hop + n, 0.0);
    std::vector<float> fre(n), fim(n);
    for (int f = 0; f < m; ++f) {
        for (int k = 0; k <= n / 2; ++k) {
            fre[k] = ph_re[f][k] * gs[f][k];
            fim[k] = ph_im[f][k] * gs[f][k];
        }
        for (int k = n / 2 + 1; k < n; ++k) {
            fre[k] = fre[n - k];
            fim[k] = -fim[n - k];
        }
        // inverse: conj(fft(conj(X))) / n ; only the real part is needed
        for (int i = 0; i < n; ++i) fim[i] = -fim[i];
        fft.transform(fre, fim);
        for (int i = 0; i < n; ++i) {
            out[f * hop + i] += double(fre[i] / n) * win[i];
            wsum[f * hop + i] += double(win[i]) * win[i];
        }
    }
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = static_cast<float>(out[i] / std::max(1e-8, wsum[i]));
    return y;
}

}  // namespace fastwave
