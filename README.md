# deeppoly-cpp

Adversarial robustness certification for neural networks, in C++.

Trains a fully connected network on MNIST, attacks it with FGSM, then formally
certifies per-image robustness using the DeepPoly abstract domain from
Singh et al., POPL 2019.

```
┌─────────────────────────────────────────────────────────────────┐
│  100 images   94 correct   73 attacked   31 certified robust    │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────┤
│ [7] → 3  │ [2] → 2  │ [9] → 4  │ [0] → 0  │ [5] → 5  │  ...    │
│ ATTACKED │  ROBUST  │ ATTACKED │ UNVERIF. │  ROBUST  │         │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## how it works

The verifier propagates a symbolic abstract element through the network
instead of running it on every possible perturbed input (exponentially many).

Each neuron tracks two linear bounds over the input pixels plus concrete
interval bounds. Three transformers do the work:

**Affine** — exact symbolic propagation through linear layers via backsubstitution
to input variables.

**ReLU** — three cases: zero (exact), identity (exact), mixed (convex hull
approximation with minimum area lower bound).

**Sigmoid** — monotone, so concrete bounds are sigmoid applied to the pre-activation
bounds directly.

If the symbolic lower bound on the true class output exceeds the upper bound of
every other class across the entire L∞ ball around the input, the network is
certified robust. No counterexample exists.

---

## stack

```
C++20        neural network + verifier
tiny-cpp-nn  header-only training library
Raylib 5.5   GPU-accelerated visualization window
CMake        build system, fetches dependencies automatically
```

---

## build

```bash
git clone https://github.com/jeetrex17/deeppoly-cpp
cd deeppoly-cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.logicalcpu)
```

---

## run

```bash
cd build/data && bash ../../data/download.sh && cd ..

./train  ../data
./attack ../data 100 0.10
./verify 0.05
./show
```

---

## architecture

```
include/mnist_loader.hpp    IDX binary parser
include/fgsm.hpp            one-step gradient sign attack
include/deeppoly.hpp        abstract domain with affine + ReLU transformers
include/results.hpp         binary serialization
src/train.cpp               784 → 128 → 64 → 10, saves models/mnist.bin
src/attack.cpp              FGSM on N images, saves output/results.bin
src/verify.cpp              DeepPoly certification, updates results
src/show.cpp                Raylib window, scrollable card grid
```

---

## reference

Gagandeep Singh, Timon Gehr, Markus Püschel, Martin Vechev.
*An Abstract Domain for Certifying Neural Networks.*
POPL 2019. https://doi.org/10.1145/3290354
