#pragma once
// Simplified DeepPoly abstract domain for certifying neural network robustness.
//
// Each neuron i maintains symbolic bounds over the n_inp input pixels:
//   lb(x_i) >= lb_bias[i] + sum_j lb_coef[i,j] * input[j]
//   ub(x_i) <= ub_bias[i] + sum_j ub_coef[i,j] * input[j]
// plus concrete interval bounds [l[i], u[i]].
//
// Supports: identity input, affine, ReLU (3-case), sigmoid (monotone, concrete only).
// Reference: Singh et al., "An Abstract Domain for Certifying Neural Networks", POPL 2019.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include "nn.h"

namespace deeppoly {

struct VerifyResult {
    bool  robust;   // true iff certified: lb(out_k) > ub(out_j) for all j != k
    float margin;   // min_{j!=k}(lb(out_k) - ub(out_j)); positive iff robust
};

// ── Abstract layer representation ─────────────────────────────────────────────
struct LayerAbs {
    size_t n_neurons;
    size_t n_inp;
    std::vector<float> lb_coef;   // [n_neurons × n_inp], row-major
    std::vector<float> ub_coef;
    std::vector<float> lb_bias;   // [n_neurons]
    std::vector<float> ub_bias;
    std::vector<float> l, u;      // concrete bounds

    LayerAbs(size_t neurons, size_t inp)
        : n_neurons(neurons), n_inp(inp),
          lb_coef(neurons * inp, 0.0f), ub_coef(neurons * inp, 0.0f),
          lb_bias(neurons, 0.0f),       ub_bias(neurons, 0.0f),
          l(neurons, 0.0f),             u(neurons, 0.0f) {}

    float& lbc(size_t i, size_t j) { return lb_coef[i * n_inp + j]; }
    float& ubc(size_t i, size_t j) { return ub_coef[i * n_inp + j]; }
    const float& lbc(size_t i, size_t j) const { return lb_coef[i * n_inp + j]; }
    const float& ubc(size_t i, size_t j) const { return ub_coef[i * n_inp + j]; }
};

// ── Helpers ────────────────────────────────────────────────────────────────────

// Evaluate concrete bounds from symbolic expressions + input box.
static void eval_concrete(LayerAbs& la,
                           const std::vector<float>& inp_l,
                           const std::vector<float>& inp_u) {
    for (size_t i = 0; i < la.n_neurons; ++i) {
        float lv = la.lb_bias[i];
        float uv = la.ub_bias[i];
        for (size_t j = 0; j < la.n_inp; ++j) {
            float lc = la.lbc(i, j);
            float uc = la.ubc(i, j);
            lv += (lc >= 0.0f) ? lc * inp_l[j] : lc * inp_u[j];
            uv += (uc >= 0.0f) ? uc * inp_u[j] : uc * inp_l[j];
        }
        la.l[i] = lv;
        la.u[i] = uv;
    }
}

// ── Transformers ───────────────────────────────────────────────────────────────

// First affine layer is a special case: the input neurons have identity symbolic
// bounds (lb_coef = ub_coef = e_i), so the output coefficients are just the weight
// columns. This avoids an O(n_inp^2 * n_out) loop.
static LayerAbs first_affine(const nn::Matrix& W,
                               const nn::Matrix& b,
                               const std::vector<float>& inp_l,
                               const std::vector<float>& inp_u) {
    size_t out   = W.cols;
    size_t n_inp = W.rows;
    LayerAbs curr(out, n_inp);
    for (size_t i = 0; i < out; ++i) {
        curr.lb_bias[i] = b(0, i);
        curr.ub_bias[i] = b(0, i);
        for (size_t j = 0; j < n_inp; ++j) {
            float w = W(j, i);
            curr.lbc(i, j) = w;   // lb == ub for affine (exact)
            curr.ubc(i, j) = w;
        }
    }
    eval_concrete(curr, inp_l, inp_u);
    return curr;
}

// General affine: curr[i] = bias[i] + sum_k W(k,i) * prev[k]
// Lower/upper bound coefficients are obtained by choosing prev[k]'s lb or ub
// depending on the sign of the weight.
static LayerAbs affine(const LayerAbs& prev,
                        const nn::Matrix& W,
                        const nn::Matrix& b,
                        const std::vector<float>& inp_l,
                        const std::vector<float>& inp_u) {
    size_t out   = W.cols;
    size_t in    = W.rows;
    LayerAbs curr(out, prev.n_inp);

    for (size_t i = 0; i < out; ++i) {
        curr.lb_bias[i] = b(0, i);
        curr.ub_bias[i] = b(0, i);
        for (size_t k = 0; k < in; ++k) {
            float w = W(k, i);
            if (w >= 0.0f) {
                for (size_t j = 0; j < prev.n_inp; ++j) {
                    curr.lbc(i, j) += w * prev.lbc(k, j);
                    curr.ubc(i, j) += w * prev.ubc(k, j);
                }
                curr.lb_bias[i] += w * prev.lb_bias[k];
                curr.ub_bias[i] += w * prev.ub_bias[k];
            } else {
                for (size_t j = 0; j < prev.n_inp; ++j) {
                    curr.lbc(i, j) += w * prev.ubc(k, j);
                    curr.ubc(i, j) += w * prev.lbc(k, j);
                }
                curr.lb_bias[i] += w * prev.ub_bias[k];
                curr.ub_bias[i] += w * prev.lb_bias[k];
            }
        }
    }
    eval_concrete(curr, inp_l, inp_u);
    return curr;
}

// ReLU: three cases from the paper (Section 4.1).
//   l >= 0 : exact identity
//   u <= 0 : exact zero
//   l < 0 < u : convex approximation
//     upper = slope * x + intercept  (where slope = u/(u-l))
//     lower = 0         if u <= -l  (area of triangle (b) is smaller)
//           = x         otherwise   (area of triangle (c) is smaller)
static LayerAbs relu(const LayerAbs& prev,
                      const std::vector<float>& inp_l,
                      const std::vector<float>& inp_u) {
    LayerAbs curr(prev.n_neurons, prev.n_inp);

    for (size_t i = 0; i < prev.n_neurons; ++i) {
        float li = prev.l[i];
        float ui = prev.u[i];

        if (ui <= 0.0f) {
            // x = 0: all zeros already; l=u=0
        } else if (li >= 0.0f) {
            // x = identity (exact)
            for (size_t j = 0; j < prev.n_inp; ++j) {
                curr.lbc(i, j) = prev.lbc(i, j);
                curr.ubc(i, j) = prev.ubc(i, j);
            }
            curr.lb_bias[i] = prev.lb_bias[i];
            curr.ub_bias[i] = prev.ub_bias[i];
            curr.l[i] = li;
            curr.u[i] = ui;
        } else {
            // Mixed: l < 0 < u
            float slope     = ui / (ui - li);
            float intercept = -ui * li / (ui - li);

            // Upper bound: slope * x + intercept (always this form)
            for (size_t j = 0; j < prev.n_inp; ++j)
                curr.ubc(i, j) = slope * prev.ubc(i, j);
            curr.ub_bias[i] = slope * prev.ub_bias[i] + intercept;
            curr.u[i] = ui;  // = slope*ui + intercept

            // Lower bound: minimum-area choice
            if (ui <= -li) {
                // lower = 0 (all zeros, l=0 already)
                curr.l[i] = 0.0f;
            } else {
                // lower = x (identity lower bound, conservative l=li)
                for (size_t j = 0; j < prev.n_inp; ++j)
                    curr.lbc(i, j) = prev.lbc(i, j);
                curr.lb_bias[i] = prev.lb_bias[i];
                curr.l[i] = li;
            }
        }
    }
    return curr;
}

// ── Main verification entry point ──────────────────────────────────────────────
inline VerifyResult verify(const nn::NeuralNetwork& net,
                            const std::vector<float>& pixels,
                            int true_label,
                            float epsilon) {
    const size_t n_inp    = pixels.size();
    const size_t n_layers = net.ws.size();

    // Input L∞ box: clamp to [0, 1]
    std::vector<float> inp_l(n_inp), inp_u(n_inp);
    for (size_t i = 0; i < n_inp; ++i) {
        inp_l[i] = std::max(0.0f, pixels[i] - epsilon);
        inp_u[i] = std::min(1.0f, pixels[i] + epsilon);
    }

    // First layer: use identity shortcut (O(n_inp * n_out) instead of O(n_inp^2 * n_out))
    LayerAbs curr = first_affine(net.ws[0], net.bs[0], inp_l, inp_u);
    if (n_layers > 1)
        curr = relu(curr, inp_l, inp_u);
    else {
        // Single-layer network with sigmoid output
        for (size_t i = 0; i < curr.n_neurons; ++i) {
            curr.l[i] = 1.0f / (1.0f + std::exp(-curr.l[i]));
            curr.u[i] = 1.0f / (1.0f + std::exp(-curr.u[i]));
        }
    }

    // Remaining layers
    for (size_t layer = 1; layer < n_layers; ++layer) {
        bool is_output = (layer == n_layers - 1);
        curr = affine(curr, net.ws[layer], net.bs[layer], inp_l, inp_u);
        if (!is_output) {
            curr = relu(curr, inp_l, inp_u);
        } else {
            // Sigmoid is monotone: lb(sig(z)) = sig(lb(z))
            for (size_t i = 0; i < curr.n_neurons; ++i) {
                curr.l[i] = 1.0f / (1.0f + std::exp(-curr.l[i]));
                curr.u[i] = 1.0f / (1.0f + std::exp(-curr.u[i]));
            }
        }
    }

    // Robustness: lb(out[k]) > ub(out[j]) for all j != k
    float margin = std::numeric_limits<float>::max();
    for (size_t j = 0; j < curr.n_neurons; ++j) {
        if ((int)j == true_label) continue;
        float diff = curr.l[(size_t)true_label] - curr.u[j];
        margin = std::min(margin, diff);
    }
    return {margin > 0.0f, margin};
}

} // namespace deeppoly
