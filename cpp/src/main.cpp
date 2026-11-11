// FastWave.AXERA — AX650 C++ demo: any sample rate -> 48 kHz super resolution.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "fastwave.hpp"

static void usage(const char* argv0) {
    std::printf(
        "usage: %s <model.axmodel> <input.wav> <output.wav> [options]\n"
        "  --assets <dir>   diffusion-embedding npy files (default: models/assets)\n"
        "  --steps <n>      EDM sampling steps (default 8, 4 = faster)\n"
        "  --overlap <r>    chunk overlap ratio (default 0.5, 0 = fastest)\n"
        "  --denoise        enable the spectral-subtraction noise post filter\n",
        argv0);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        usage(argv[0]);
        return 1;
    }
    const std::string model = argv[1];
    const std::string in_path = argv[2];
    const std::string out_path = argv[3];
    std::string assets = "models/assets";
    int steps = 8;
    double overlap = 0.5;
    bool denoise = false;
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--assets") && i + 1 < argc) assets = argv[++i];
        else if (!std::strcmp(argv[i], "--steps") && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--overlap") && i + 1 < argc) overlap = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--denoise")) denoise = true;
        else {
            usage(argv[0]);
            return 1;
        }
    }

    std::vector<float> wav;
    int sr = 0;
    if (!fastwave::read_wav(in_path, wav, sr)) return 2;
    std::printf("[input ] %s  %.2fs @ %d Hz\n", in_path.c_str(), double(wav.size()) / sr, sr);

    fastwave::FastWave fw;
    if (!fw.load(model, assets, fastwave::kChunk, steps)) {
        std::fprintf(stderr, "failed to load model %s\n", model.c_str());
        return 3;
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> out;
    if (!fw.enhance(wav, sr, overlap, out, denoise)) {
        std::fprintf(stderr, "inference failed\n");
        return 4;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(t1 - t0).count();

    if (!fastwave::write_wav(out_path, out, fastwave::kSampleRate)) {
        std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
        return 5;
    }
    const double dur = double(out.size()) / fastwave::kSampleRate;
    std::printf("[output] %s  %.2fs @ 48000 Hz\n", out_path.c_str(), dur);
    std::printf("[perf  ] %.2fs  RTF %.3f\n", dt, dt / dur);
    return 0;
}
