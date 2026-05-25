#pragma once
// Fast Gradient Sign Method (FGSM) adversarial attack.
//
// Key insight: nn::NeuralNetwork::backprop() accumulates the gradient w.r.t.
// the input layer into g.as[0] as a byproduct of backpropagation.
// We use that directly — no manual backprop needed.

#include <algorithm>
#include <vector>
#include "nn.h"

namespace fgsm {

// Forward pass + argmax prediction.
inline int predict(nn::NeuralNetwork& net, const std::vector<float>& pixels) {
    auto& inp = net.get_input();
    for (size_t j = 0; j < pixels.size(); ++j)
        inp(0, j) = pixels[j];
    net.forward(nn::Activation::Relu, nn::Activation::Sigmoid);
    const auto& out = net.get_output();
    int best = 0;
    for (int i = 1; i < 10; ++i)
        if (out(0, (size_t)i) > out(0, (size_t)best)) best = i;
    return best;
}

// Returns adversarial pixels: x + ε·sign(∂L/∂x), clamped to [0,1].
// Uses the gradient stored in g.as[0] after backprop on a single sample.
inline std::vector<float> attack(nn::NeuralNetwork& net,
                                  const std::vector<float>& pixels,
                                  int true_label,
                                  float epsilon) {
    const size_t n_in = pixels.size();

    // Build 1-row training matrix: [pixels | one-hot target]
    nn::Matrix row(1, n_in + 10, 0.0f);
    for (size_t j = 0; j < n_in; ++j)
        row(0, j) = pixels[j];
    row(0, n_in + (size_t)true_label) = 1.0f;

    // Backprop — g.as[0] accumulates ∂L/∂input as a side effect
    auto g = net.backprop(row, nn::Activation::Relu, nn::Activation::Sigmoid);

    // FGSM perturbation
    std::vector<float> adv(n_in);
    for (size_t i = 0; i < n_in; ++i) {
        float grad = g.as[0](0, i);
        float sign = (grad > 0.0f) ? 1.0f : (grad < 0.0f ? -1.0f : 0.0f);
        adv[i] = std::clamp(pixels[i] + epsilon * sign, 0.0f, 1.0f);
    }
    return adv;
}

} // namespace fgsm
