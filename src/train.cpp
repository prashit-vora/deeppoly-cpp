#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "nn.h"
#include "mnist_loader.hpp"

int main(int argc, char* argv[]) {
    const std::string data_dir  = (argc > 1) ? argv[1] : "data";
    const int         epochs    = 30;
    const float       lr        = 0.5f;
    const int         batch_sz  = 32;

    // ── Load data ──────────────────────────────────────────────────────────────
    printf("Loading MNIST from '%s'...\n", data_dir.c_str());
    auto train_s = mnist::load(data_dir + "/train-images-idx3-ubyte",
                               data_dir + "/train-labels-idx1-ubyte");
    auto test_s  = mnist::load(data_dir + "/t10k-images-idx3-ubyte",
                               data_dir + "/t10k-labels-idx1-ubyte");
    printf("Train: %zu   Test: %zu\n\n", train_s.size(), test_s.size());

    // Build full training matrix: each row = [784 pixels | 10 one-hot targets]
    printf("Building training matrix (%zu × 794)...\n", train_s.size());
    nn::Matrix train_mat((size_t)train_s.size(), 794, 0.0f);
    for (size_t i = 0; i < train_s.size(); ++i) {
        for (int j = 0; j < 784; ++j)
            train_mat(i, (size_t)j) = train_s[i].pixels[j];
        train_mat(i, 784 + (size_t)train_s[i].label) = 1.0f;
    }
    printf("Done.\n\n");

    // ── Network: 784 → 128 → 64 → 10 ─────────────────────────────────────────
    nn::NeuralNetwork net({784, 128, 64, 10});
    net.randomize(-0.1f, 0.1f);
    printf("Architecture: 784 → 128 → 64 → 10  (ReLU hidden, Sigmoid output)\n");
    printf("Epochs: %d   LR: %.4f   Batch: %d\n\n", epochs, lr, batch_sz);

    // ── Training loop ──────────────────────────────────────────────────────────
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto t0 = std::chrono::steady_clock::now();

        nn::Batch batch;
        do {
            batch.process((size_t)batch_sz, net, train_mat, lr,
                          nn::Activation::Relu, nn::Activation::Sigmoid);
        } while (!batch.finished);

        // Evaluate test accuracy
        int correct = 0;
        for (const auto& s : test_s) {
            auto& inp = net.get_input();
            for (int j = 0; j < 784; ++j)
                inp(0, (size_t)j) = s.pixels[j];
            net.forward(nn::Activation::Relu, nn::Activation::Sigmoid);
            const auto& out = net.get_output();
            int best = 0;
            for (int j = 1; j < 10; ++j)
                if (out(0, (size_t)j) > out(0, (size_t)best)) best = j;
            if (best == s.label) ++correct;
        }
        float acc = 100.0f * (float)correct / (float)test_s.size();

        auto t1   = std::chrono::steady_clock::now();
        float sec = std::chrono::duration<float>(t1 - t0).count();
        printf("Epoch %2d/%d | Loss: %.5f | Test Acc: %.2f%% | %.1fs\n",
               epoch + 1, epochs, batch.cost, acc, sec);
    }

    // ── Save ───────────────────────────────────────────────────────────────────
    if (net.save("models/mnist.bin"))
        printf("\nSaved → models/mnist.bin\n");
    else
        fprintf(stderr, "Failed to save model.\n");

    return 0;
}
