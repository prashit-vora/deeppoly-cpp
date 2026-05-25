#pragma once
// Binary serialization for per-image attack + verify results.
// attack.cpp writes the file; verify.cpp loads and enriches it; show.cpp reads it.

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

struct ImageResult {
    int   idx        = 0;
    int   true_label = 0;
    int   pred_clean = 0;
    int   pred_adv   = 0;
    bool  attacked   = false;   // pred_adv != pred_clean
    bool  robust     = false;   // DeepPoly certified
    float margin     = 0.0f;    // DeepPoly margin (>0 iff robust)
    std::vector<float> orig_pixels;   // 784 floats in [0,1]
    std::vector<float> adv_pixels;    // 784 floats in [0,1]
};

inline void save_results(const std::vector<ImageResult>& v,
                          const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    uint32_t n = (uint32_t)v.size();
    f.write(reinterpret_cast<const char*>(&n), 4);
    for (const auto& r : v) {
        int32_t ints[4] = {r.idx, r.true_label, r.pred_clean, r.pred_adv};
        f.write(reinterpret_cast<const char*>(ints), 16);
        uint8_t flags[2] = {(uint8_t)r.attacked, (uint8_t)r.robust};
        f.write(reinterpret_cast<const char*>(flags), 2);
        f.write(reinterpret_cast<const char*>(&r.margin), 4);
        f.write(reinterpret_cast<const char*>(r.orig_pixels.data()), 784 * 4);
        f.write(reinterpret_cast<const char*>(r.adv_pixels.data()),  784 * 4);
    }
}

inline std::vector<ImageResult> load_results(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    uint32_t n;
    f.read(reinterpret_cast<char*>(&n), 4);
    std::vector<ImageResult> v(n);
    for (auto& r : v) {
        int32_t ints[4];
        f.read(reinterpret_cast<char*>(ints), 16);
        r.idx = ints[0]; r.true_label = ints[1];
        r.pred_clean = ints[2]; r.pred_adv = ints[3];
        uint8_t flags[2];
        f.read(reinterpret_cast<char*>(flags), 2);
        r.attacked = (bool)flags[0];
        r.robust   = (bool)flags[1];
        f.read(reinterpret_cast<char*>(&r.margin), 4);
        r.orig_pixels.resize(784);
        r.adv_pixels.resize(784);
        f.read(reinterpret_cast<char*>(r.orig_pixels.data()), 784 * 4);
        f.read(reinterpret_cast<char*>(r.adv_pixels.data()),  784 * 4);
    }
    return v;
}
