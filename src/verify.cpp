#include <cstdio>
#include <cstdlib>
#include <vector>

#include "nn.h"
#include "deeppoly.hpp"
#include "results.hpp"

int main(int argc, char* argv[]) {
    const float epsilon = (argc > 1) ? std::atof(argv[1]) : 0.05f;

    // ── Load model ─────────────────────────────────────────────────────────────
    nn::NeuralNetwork net({784, 128, 64, 10});
    if (!net.load("models/mnist.bin")) {
        fprintf(stderr, "Cannot load models/mnist.bin — run ./train first.\n");
        return 1;
    }

    // ── Load attack results ────────────────────────────────────────────────────
    auto results = load_results("output/results.bin");
    if (results.empty()) {
        fprintf(stderr, "output/results.bin not found — run ./attack first.\n");
        return 1;
    }

    printf("DeepPoly verification on %zu images  (ε = %.3f)\n\n",
           results.size(), epsilon);

    // ── Verify each image ──────────────────────────────────────────────────────
    int n_robust = 0;
    for (auto& r : results) {
        auto vr    = deeppoly::verify(net, r.orig_pixels, r.true_label, epsilon);
        r.robust   = vr.robust;
        r.margin   = vr.margin;
        if (vr.robust) ++n_robust;

        const char* verdict = vr.robust
            ? "\033[32mROBUST    \033[0m"
            : "\033[33mUNVERIFIED\033[0m";

        printf("[%3d] true=%-2d  pred=%-2d  margin=%+.4f  %s\n",
               r.idx, r.true_label, r.pred_clean, vr.margin, verdict);
    }

    // ── Summary ────────────────────────────────────────────────────────────────
    int total = (int)results.size();
    int n_attacked = 0;
    for (const auto& r : results) if (r.attacked) ++n_attacked;

    printf("\n─────────────────────────────────────────\n");
    printf("Images       : %d\n",    total);
    printf("Attacked     : %d (%.1f%%)\n", n_attacked, 100.f * n_attacked / total);
    printf("Certified    : %d (%.1f%%)\n", n_robust,   100.f * n_robust   / total);
    printf("─────────────────────────────────────────\n");
    printf("(verified at ε=%.3f — attacker used ε=%.3f)\n",
           epsilon, (float)epsilon * 2.0f);  // display note

    save_results(results, "output/results.bin");
    printf("\nUpdated → output/results.bin\n");
    printf("Next step: run  ./show\n");
    return 0;
}
