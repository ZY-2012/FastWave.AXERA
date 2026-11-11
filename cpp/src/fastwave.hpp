#pragma once
// FastWave EDM sampler on AX650 (AX_ENGINE): chunked diffusion + overlap-add.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "audio.hpp"
#include "ax_runner.hpp"
#include "postfilter.hpp"

namespace fastwave {

// ------------------------------------------------------------- npy loading
static inline bool load_npy_f32(const std::string& path, std::vector<float>& data,
                                std::vector<int>* shape = nullptr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return false;
    }
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        std::fprintf(stderr, "not a npy file: %s\n", path.c_str());
        return false;
    }
    unsigned char ver[2];
    f.read(reinterpret_cast<char*>(ver), 2);
    size_t hlen = 0;
    if (ver[0] == 1) {
        uint16_t l;
        f.read(reinterpret_cast<char*>(&l), 2);
        hlen = l;
    } else {
        uint32_t l;
        f.read(reinterpret_cast<char*>(&l), 4);
        hlen = l;
    }
    std::string header(hlen, '\0');
    f.read(&header[0], hlen);
    if (header.find("<f4") == std::string::npos) {
        std::fprintf(stderr, "only float32 npy supported: %s\n", path.c_str());
        return false;
    }
    if (shape) {
        shape->clear();
        size_t p = header.find("'shape': (");
        if (p != std::string::npos) {
            p += 10;
            size_t end = header.find(')', p);
            std::string dims = header.substr(p, end - p);
            int v = 0;
            bool has = false;
            for (char c : dims) {
                if (c >= '0' && c <= '9') {
                    v = v * 10 + (c - '0');
                    has = true;
                } else if (has) {
                    shape->push_back(v);
                    v = 0;
                    has = false;
                }
            }
            if (has) shape->push_back(v);
        }
    }
    f.seekg(0, std::ios::end);
    const size_t total = static_cast<size_t>(f.tellg());
    const size_t data_bytes = total - 10 - hlen;
    data.resize(data_bytes / sizeof(float));
    f.seekg(10 + hlen, std::ios::beg);
    f.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    return true;
}

struct StepParams {
    float c_skip, c_in, c_out;
    std::vector<float> emb;  // (512)
};

class FastWave {
public:
    bool load(const std::string& model_path, const std::string& assets_dir, int chunk = kChunk,
              int steps = 8) {
        chunk_ = chunk;
        steps_ = steps;
        if (!model_.load(model_path)) return false;
        in_x_ = model_.input_index("x");
        in_low_ = model_.input_index("audio_low");
        in_band_ = model_.input_index("band_onehot");
        in_cs_ = model_.input_index("c_skip");
        in_ci_ = model_.input_index("c_in");
        in_co_ = model_.input_index("c_out");
        in_emb_ = model_.input_index("noise_emb");
        out_y_ = model_.output_index("y");
        if (in_x_ < 0 || in_low_ < 0 || in_band_ < 0 || out_y_ < 0) {
            std::fprintf(stderr, "axmodel is missing FastWave inputs/outputs\n");
            return false;
        }

        std::vector<float> w1, b1, w2, b2;
        std::vector<int> s1, s2;
        const std::string dir = assets_dir;
        if (!load_npy_f32(dir + "/emb_p1_w.npy", w1, &s1) ||
            !load_npy_f32(dir + "/emb_p1_b.npy", b1) ||
            !load_npy_f32(dir + "/emb_p2_w.npy", w2, &s2) ||
            !load_npy_f32(dir + "/emb_p2_b.npy", b2)) {
            std::fprintf(stderr, "cannot load diffusion embedding weights from %s\n", dir.c_str());
            return false;
        }
        if (s1.size() != 2 || s2.size() != 2) {
            std::fprintf(stderr, "unexpected embedding weight shapes\n");
            return false;
        }
        emb_in_ = s1[1];
        emb_hid_ = s1[0];
        if (s2[0] != emb_hid_ || s2[1] != emb_hid_) {
            std::fprintf(stderr, "unexpected projection2 shape\n");
            return false;
        }

        // sigma schedule (identical to the Python reference)
        std::vector<float> ts(steps + 1);
        const double sigma_min = 0.002, sigma_max = 80.0, rho = 7.0;
        for (int i = 0; i < steps; ++i) {
            double t = std::pow(std::pow(sigma_max, 1.0 / rho) +
                                    (double)i / (steps - 1) *
                                        (std::pow(sigma_min, 1.0 / rho) - std::pow(sigma_max, 1.0 / rho)),
                                rho);
            ts[i] = static_cast<float>(t);
        }
        ts[steps] = 0.0f;
        t_steps_ = ts;

        std::vector<float> freqs(emb_in_ / 2);
        for (size_t j = 0; j < freqs.size(); ++j)
            freqs[j] = static_cast<float>(std::exp(double(j) * (-std::log(10000.0) / (freqs.size() - 1))));

        params_.resize(steps);
        for (int i = 0; i < steps; ++i) {
            const double s = ts[i];
            StepParams& p = params_[i];
            p.c_skip = static_cast<float>(0.25 / (s * s + 0.25));
            p.c_in = static_cast<float>(1.0 / std::sqrt(s * s + 0.25));
            p.c_out = static_cast<float>(0.5 * s / std::sqrt(s * s + 0.25));
            const double c_noise = std::log(s) / 4.0;
            const size_t half = freqs.size();
            std::vector<float> emb(half * 2);
            for (size_t j = 0; j < half; ++j) {
                const double ang = 50000.0 * c_noise * freqs[j];
                emb[j] = static_cast<float>(std::sin(ang));
                emb[half + j] = static_cast<float>(std::cos(ang));
            }
            std::vector<float> h1(emb_hid_);
            for (int r = 0; r < emb_hid_; ++r) {
                double acc = b1[r];
                const float* row = w1.data() + (size_t)r * emb_in_;
                for (int c = 0; c < emb_in_; ++c) acc += row[c] * emb[c];
                h1[r] = static_cast<float>(acc / (1.0 + std::exp(-acc)));
            }
            p.emb.resize(emb_hid_);
            for (int r = 0; r < emb_hid_; ++r) {
                double acc = b2[r];
                const float* row = w2.data() + (size_t)r * emb_hid_;
                for (int c = 0; c < emb_hid_; ++c) acc += row[c] * h1[c];
                p.emb[r] = static_cast<float>(acc / (1.0 + std::exp(-acc)));
            }
        }
        rng_ = 1234u;
        return true;
    }

    // xorshift RNG so results are reproducible run to run
    float randn() {
        rng_ ^= rng_ << 13;
        rng_ ^= rng_ >> 17;
        rng_ ^= rng_ << 5;
        const float u1 = std::max(1e-7f, (rng_ & 0xFFFFFF) / 16777216.0f);
        rng_ ^= rng_ << 13;
        rng_ ^= rng_ >> 17;
        rng_ ^= rng_ << 5;
        const float u2 = (rng_ & 0xFFFFFF) / 16777216.0f;
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * (float)M_PI * u2);
    }

    bool denoise_chunk(const float* low, const std::vector<float>& band, std::vector<float>& out) {
        const int T = chunk_;
        out.assign(T, 0.0f);
        std::vector<float> x(static_cast<size_t>(T));
        for (int i = 0; i < T; ++i) x[i] = randn() * t_steps_[0];
        std::vector<float> y(static_cast<size_t>(T));
        for (int s = 0; s < steps_; ++s) {
            const StepParams& p = params_[s];
            model_.set_input(in_x_, x.data(), x.size());
            model_.set_input(in_low_, low, static_cast<size_t>(T));
            if (in_band_ >= 0) model_.set_input(in_band_, band.data(), band.size());
            model_.set_input(in_cs_, &p.c_skip, 1);
            model_.set_input(in_ci_, &p.c_in, 1);
            model_.set_input(in_co_, &p.c_out, 1);
            model_.set_input(in_emb_, p.emb.data(), p.emb.size());
            if (!model_.run()) return false;
            if (!model_.get_output(out_y_, y.data(), y.size())) return false;
            const float t_cur = t_steps_[s];
            const float t_next = t_steps_[s + 1];
            const float k = (t_next - t_cur) / t_cur;
            for (int i = 0; i < T; ++i) x[i] = x[i] + k * (x[i] - y[i]);
        }
        for (int i = 0; i < T; ++i) out[i] = std::min(1.0f - 1e-7f, std::max(-1.0f, x[i]));
        return true;
    }

    bool enhance(const std::vector<float>& wav, int sr_in, double overlap, std::vector<float>& out,
                 bool denoise = true) {
        float peak = 0.0f;
        for (float v : wav) peak = std::max(peak, std::fabs(v));
        if (peak < 1e-8f) peak = 1.0f;
        std::vector<float> norm(wav.size());
        for (size_t i = 0; i < wav.size(); ++i) norm[i] = wav[i] / peak;

        std::vector<float> low = low_quality(norm, sr_in, kSampleRate);
        std::vector<float> band;
        band_onehot(sr_in, band);
        const size_t n = low.size();
        if (n == 0) return false;

        if (overlap <= 0.0) {
            const size_t n_pad = ((n + chunk_ - 1) / chunk_) * chunk_;
            low.resize(n_pad, 0.0f);
            std::vector<std::vector<float>> outs;
            for (size_t s = 0; s < n_pad; s += chunk_) {
                std::vector<float> y;
                if (!denoise_chunk(low.data() + s, band, y)) return false;
                outs.push_back(std::move(y));
                std::fprintf(stderr, "\r[chunk] %zu/%zu", outs.size(), n_pad / chunk_);
            }
            std::fprintf(stderr, "\n");
            out.resize(n);
            size_t idx = 0;
            for (size_t c = 0; c < outs.size() && idx < n; ++c)
                for (size_t i = 0; i < outs[c].size() && idx < n; ++i) out[idx++] = outs[c][i];
        } else {
            size_t hop = static_cast<size_t>(std::llround(chunk_ * overlap));
            hop = std::max<size_t>(256, (hop / 256) * 256);
            hop = std::min<size_t>(hop, chunk_);
            std::vector<size_t> starts = chunk_bounds(n, chunk_, hop);
            std::vector<std::vector<float>> outs;
            outs.reserve(starts.size());
            for (size_t i = 0; i < starts.size(); ++i) {
                const size_t s = starts[i];
                std::vector<float> seg(low.begin() + s,
                                       low.begin() + std::min(low.size(), s + chunk_));
                seg.resize(chunk_, 0.0f);
                std::vector<float> y;
                if (!denoise_chunk(seg.data(), band, y)) return false;
                outs.push_back(std::move(y));
                std::fprintf(stderr, "\r[chunk] %zu/%zu", i + 1, starts.size());
            }
            std::fprintf(stderr, "\n");
            out = overlap_add(outs, starts, n, chunk_);
        }
        if (denoise) out = postfilter(out, kSampleRate);
        for (auto& v : out) v *= peak;
        return true;
    }

    int chunk() const { return chunk_; }

private:
    AxModel model_;
    int in_x_ = -1, in_low_ = -1, in_band_ = -1, in_cs_ = -1, in_ci_ = -1, in_co_ = -1,
        in_emb_ = -1, out_y_ = -1;
    int chunk_ = kChunk, steps_ = 8;
    int emb_in_ = 128, emb_hid_ = 512;
    std::vector<float> t_steps_;
    std::vector<StepParams> params_;
    uint32_t rng_ = 1234u;
};

}  // namespace fastwave
