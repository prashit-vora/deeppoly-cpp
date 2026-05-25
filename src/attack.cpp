#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "nn.h"
#include "mnist_loader.hpp"
#include "fgsm.hpp"
#include "results.hpp"

int main(int argc, char* argv[]) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data";
    const int         n_imgs   = (argc > 2) ? std::atoi(argv[2]) : 100;
    const float       epsilon  = (argc > 3) ? std::atof(argv[3]) : 0.10f;

    // ── Load model ─────────────────────────────────────────────────────────────
    nn::NeuralNetwork net({784, 128, 64, 10});
    if (!net.load("models/mnist.bin")) {
        fprintf(stderr, "Cannot load models/mnist.bin — run ./train first.\n");
        return 1;
    }
    printf("Model loaded: 784 → 128 → 64 → 10\n");
    printf("FGSM attack on %d test images  (ε = %.3f)\n\n", n_imgs, epsilon);

    // ── Load test set ──────────────────────────────────────────────────────────
    auto test = mnist::load(data_dir + "/t10k-images-idx3-ubyte",
                            data_dir + "/t10k-labels-idx1-ubyte");

    // ── Run attack ─────────────────────────────────────────────────────────────
    std::vector<ImageResult> results;
    results.reserve((size_t)n_imgs);

    int n_attacked = 0;
    int n_correct  = 0;

    for (int i = 0; i < n_imgs && i < (int)test.size(); ++i) {
        const auto& s = test[(size_t)i];

        int pred_clean = fgsm::predict(net, s.pixels);
        auto adv       = fgsm::attack(net, s.pixels, s.label, epsilon);
        int pred_adv   = fgsm::predict(net, adv);

        bool attacked = (pred_adv != pred_clean);
        bool correct  = (pred_clean == s.label);
        if (attacked) ++n_attacked;
        if (correct)  ++n_correct;

        printf("[%3d] true=%-2d  clean=%-2d  adv=%-2d  %s\n",
               i, s.label, pred_clean, pred_adv,
               attacked ? "\033[31mATTACKED\033[0m" : "same");

        ImageResult r;
        r.idx         = i;
        r.true_label  = s.label;
        r.pred_clean  = pred_clean;
        r.pred_adv    = pred_adv;
        r.attacked    = attacked;
        r.robust      = false;   // filled by verify
        r.margin      = 0.0f;
        r.orig_pixels = s.pixels;
        r.adv_pixels  = std::move(adv);
        results.push_back(std::move(r));
    }

    // ── Summary ────────────────────────────────────────────────────────────────
    printf("\n─────────────────────────────\n");
    printf("Images    : %d\n",  n_imgs);
    printf("Correct   : %d (%.1f%%)\n", n_correct,  100.f * n_correct  / n_imgs);
    printf("Attacked  : %d (%.1f%%)\n", n_attacked, 100.f * n_attacked / n_imgs);
    printf("─────────────────────────────\n");

    save_results(results, "output/results.bin");
    printf("Saved → output/results.bin\n");
    printf("Next step: run  ./verify [epsilon]  (default ε=0.05)\n");
    return 0;
}
