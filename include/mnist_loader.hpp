#pragma once
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mnist {

struct Sample {
    std::vector<float> pixels;  // 784 values in [0.0, 1.0]
    int label;
};

static uint32_t read_u32(std::ifstream& f) {
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) |
           (uint32_t(b[2]) << 8)  |  uint32_t(b[3]);
}

inline std::vector<Sample> load(const std::string& img_path,
                                 const std::string& lbl_path) {
    std::ifstream img_f(img_path, std::ios::binary);
    std::ifstream lbl_f(lbl_path, std::ios::binary);

    if (!img_f) throw std::runtime_error("Cannot open: " + img_path);
    if (!lbl_f) throw std::runtime_error("Cannot open: " + lbl_path);

    if (read_u32(img_f) != 2051)
        throw std::runtime_error("Bad image magic in: " + img_path);
    uint32_t n    = read_u32(img_f);
    uint32_t rows = read_u32(img_f);
    uint32_t cols = read_u32(img_f);

    if (read_u32(lbl_f) != 2049)
        throw std::runtime_error("Bad label magic in: " + lbl_path);
    if (read_u32(lbl_f) != n)
        throw std::runtime_error("Image/label count mismatch");

    size_t np = (size_t)rows * cols;
    std::vector<uint8_t> buf(np);
    std::vector<Sample> samples(n);

    for (size_t i = 0; i < n; ++i) {
        img_f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)np);
        samples[i].pixels.resize(np);
        for (size_t j = 0; j < np; ++j)
            samples[i].pixels[j] = buf[j] / 255.0f;

        uint8_t lbl;
        lbl_f.read(reinterpret_cast<char*>(&lbl), 1);
        samples[i].label = (int)lbl;
    }
    return samples;
}

} // namespace mnist
