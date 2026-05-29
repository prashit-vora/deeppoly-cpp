# Research Idea: ZK Proof of Certified Accuracy via Abstract Interpretation

> A novel system that proves a neural network achieves ≥ X% certified robustness
> on a private test set — without revealing model weights or test data.

---

## The Problem

When a company deploys a safety-critical ML model (medical diagnosis, fraud detection,
autonomous driving), two questions arise that cannot both be answered today:

1. **Regulators ask:** "Prove your model is robust to adversarial inputs."
2. **Companies respond:** "Only if you let us hide our weights and test data."

There is no system that satisfies both simultaneously. Either the company reveals
proprietary model weights, or the regulator cannot verify the claim.

---

## Our Goal

Build a cryptographic system where a model owner can produce a single proof that says:

> *"My model achieves ≥ X% certified accuracy at perturbation radius ε on a private
> test set. You can verify this proof in under 1 second without ever seeing my model
> weights, test images, or labels."*

**Certified accuracy** is stronger than plain accuracy — it means for X% of test
images, no attacker within an ε-ball around the image can flip the prediction.
This is a formal, mathematical guarantee, not an empirical claim.

---

## Key Insight: LP Certificate Verification

The central bottleneck in all prior ZK-for-ML work is encoding ReLU activations
inside a ZK circuit. Exact ReLU requires bit decomposition — roughly 32 range
checks per neuron. A network with 1000 neurons needs ~32,000 constraints per image.

We sidestep this entirely using **LP certificate verification**:

```
Standard approach (expensive):
  ZK circuit must RECOMPUTE: run every ReLU inside the proof
  → 32,000 constraints per image

Our approach (cheap):
  1. Run DeepPoly OUTSIDE the circuit (offline, no speed limit)
  2. Save bounds [l_i, u_i] for every neuron
  3. ZK circuit only CHECKS: are these bounds consistent with the weights?
  → ~70 constraints per image
```

Checking consistency is just verifying linear inequalities — purely linear arithmetic,
no non-linear operations anywhere in the circuit.

### Why This Is Sound

Model weights are **committed** (cryptographically locked) at proof start.
The consistency check uses those committed weights. Fake bounds cannot pass
the check without breaking the commitment scheme — computationally infeasible.

### The Three Cases for ReLU (All Linear)

```
Case 1 — always active  (l ≥ 0):
    check: l_out = l_in,  u_out = u_in

Case 2 — always inactive  (u ≤ 0):
    check: l_out = 0,  u_out = 0

Case 3 — mixed  (l < 0 < u):
    check: u_out ≤ u/(u-l) · u_in      ← DeepPoly linear upper relaxation
    check: l_out ≥ 0                    ← DeepPoly linear lower bound
```

No bit decomposition. No sign checks. All cases reduce to multiply-and-compare.

---

## Upgraded System: Elliptic Curves Instead of Circuits

Arithmetic circuits are the standard ZK encoding but not the most efficient one for
our specific computation. Since LP certificate verification is almost entirely
**linear arithmetic**, two elliptic curve protocols eliminate circuits for the
heaviest parts entirely.

### KZG Polynomial Commitments — Lock the Weights

Instead of committing to each weight individually via a hash, represent all weights
as a polynomial and commit using an elliptic curve pairing:

```
Weights w[0], w[1], ..., w[n]  →  polynomial p(x)
Commitment: C = p(τ)·G          (τ is a secret trusted setup point)

Proving weight at position z has value y:
  → one elliptic curve pairing check
  → O(1) time regardless of how many weights exist
```

This means verifying any individual weight used in a bound check costs one pairing
operation — constant time, no circuit constraints.

### Bulletproof Inner Product Arguments — Linear Layers for Free

Each linear layer is matrix multiplication = a series of dot products.
Bulletproofs prove an inner product ⟨a, b⟩ = c in **O(log n) communication**
using only elliptic curve operations — no arithmetic circuit needed:

```
Standard circuit:    n constraints per dot product
Bulletproof:         O(log n) elliptic curve operations per dot product
```

Since LP certificate verification is all linear arithmetic, this eliminates
arithmetic circuits for the main computation entirely.

### What Still Needs a Circuit

Only the final certification check requires a range check:
```
lower_bound[true_class] > upper_bound[wrong_class]
```
One range check per image — tiny compared to the full computation.

### Upgraded Architecture

```
Model weights → KZG polynomial commitment (elliptic curve pairing)
                      ↓
Per-image: LP certificate verification
           → Bulletproof inner product arguments for linear layers
           → one range check for certification condition
                      ↓
Nova folding to aggregate N images (also elliptic curve internally)
                      ↓
Final proof: constant size, < 1 second to verify
```

Every component uses elliptic curves. No large arithmetic circuits anywhere.

### What the Upgrade Gains

| Component | Circuit approach | Elliptic curve approach |
|---|---|---|
| Weight commitment | Hash / Merkle tree | KZG polynomial commitment |
| Linear layer proof | n constraints | O(log n) inner product argument |
| ReLU (LP cert check) | Linear constraints | Same — still linear arithmetic |
| Final certification check | Range check | Range check (unavoidable) |
| Proof size | Linear in circuit | Logarithmic (Bulletproofs) |

---

## Full System Architecture

```
                    OFFLINE (no ZK constraints)
                    ┌──────────────────────────────┐
                    │  Run DeepPoly on each image  │
                    │  Save bounds [l_i, u_i]       │
                    │  for every neuron             │
                    └────────────┬─────────────────┘
                                 │
                    ONLINE (inside ZK proof)
                                 │
                    ┌────────────▼─────────────────┐
                    │  Per-image ZK circuit         │
                    │  (LP certificate verification)│
                    │                               │
                    │  1. Check bounds consistent   │
                    │     with committed weights    │
                    │  2. Check certification:      │
                    │     l[true] > u[wrong_class]  │
                    └────────────┬─────────────────┘
                                 │  × N images
                    ┌────────────▼─────────────────┐
                    │  Nova Folding                 │
                    │                               │
                    │  Aggregates N per-image       │
                    │  proofs into ONE proof        │
                    │  Proof size: constant in N    │
                    └────────────┬─────────────────┘
                                 │
                    ┌────────────▼─────────────────┐
                    │  Final Proof                  │
                    │                               │
                    │  "≥ X% of test images are    │
                    │   certified robust at ε"      │
                    │                               │
                    │  Verified in < 1 second       │
                    │  Proof size: < 10 KB          │
                    └──────────────────────────────┘
```

---

## Three Nested Contributions

```
┌─────────────────────────────────────────────────────┐
│   ZK proof of certified accuracy over test set      │  ← GOAL (the claim)
│                                                     │
│   ┌─────────────────────────────────────────────┐   │
│   │   DeepPoly + ZK + Nova folding combined     │   │  ← SYSTEM (the design)
│   │                                             │   │
│   │   ┌─────────────────────────────────────┐   │   │
│   │   │   DeepPoly abstract domain in ZK    │   │   │  ← TRICK (the novelty)
│   │   │   (LP certificate verification)     │   │   │
│   │   └─────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## What Makes This Novel

No prior paper has combined these three components:

| Component | Prior work exists? | Our use |
|---|---|---|
| DeepPoly certification | Yes (Singh et al., POPL 2019) | Offline — produces certificate |
| ZK proof of single inference | Yes (EZKL, zkLLM) | Not used — too expensive |
| ZK proof of fairness certificate | Yes (FairProof, ICML 2024) | Related — different certificate type |
| ZK proof of training | Yes (Kaizen, CCS 2024) | Related — different goal |
| Nova folding for ML | Partial | Applied to certification aggregation |
| **LP certificate verification for DeepPoly** | **No** | **Core contribution** |
| **ZK proof of certified accuracy over test set** | **No** | **Final result** |

### Key Differentiation from FairProof (Closest Paper)

FairProof (ICML 2024, UC San Diego + Stanford) is the most structurally similar work.
They use ZK to prove **individual fairness** of a model without revealing weights.

| | FairProof | Our Work |
|---|---|---|
| Certificate type | Local Lipschitz / sensitivity bound | DeepPoly polyhedral bounds |
| What is certified | Fairness (similar inputs → similar outputs) | Robustness (ε-ball → same class) |
| Scope | Single input | Full test set |
| Aggregation | None | Nova folding over N images |
| Output | One fairness certificate | Certified accuracy percentage |

---

## Concrete Constraint Count Comparison

For a small network (3 layers, 100 neurons per layer):

| Approach | Constraints per image | For 10,000 images |
|---|---|---|
| Exact ReLU (EZKL style) | ~10,000 | 100,000,000 |
| DeepPoly relaxation in circuit | ~500 | 5,000,000 |
| **LP certificate verification (ours)** | **~50** | **500,000** |

200× fewer constraints than exact ReLU. The gap grows with network depth.

---

## Implementation Plan

### Tools
- **Python** — run DeepPoly, save per-neuron bounds (already have this)
- **Gnark** (Go) — ZK circuit implementation (FairProof used this, reference code exists)
- **Nova-Scotia** — Nova folding wrapper for Gnark circuits
- **Bulletproofs** — inner product arguments for linear layers (optional upgrade)

### Phased Roadmap

Do NOT try to build everything at once. Each phase is independently publishable.

**Phase 1 — Read and understand (Month 1-2)**
Read FairProof end to end. Understand their Gnark circuit implementation.
This is your template. Also read the Nova paper and run their example code.
Output: deep understanding of the ZK engineering required.

**Phase 2 — First working result (Month 3-4)**
Modify DeepPoly to export per-neuron bounds `[l_i, u_i]` for every test image.
Implement LP certificate verification in Gnark for a single MNIST image.
No folding, no KZG, no Bulletproofs — just the core circuit.
Benchmark: constraint count vs exact ReLU.
Output: workshop paper. The constraint reduction result alone is publishable.

**Phase 3 — First conference paper (Month 5-6)**
Add Nova folding over N images. Each step: verify one certificate + increment counter.
Add Merkle tree commitment for test set and labels.
Run end-to-end on MNIST, benchmark all metrics.
Output: conference paper (ICLR workshop / SaTML / IEEE S&P workshop).

**Phase 4 — Top venue paper (Month 7-12)**
Replace Merkle commitments with KZG polynomial commitments.
Replace linear layer circuit constraints with Bulletproof inner product arguments.
Test on larger networks (CIFAR-10).
Output: full paper targeting CCS, S&P, or USENIX Security.

### Target Benchmarks to Report
- Constraint count per image vs exact ReLU (aim: 100-200× reduction)
- Proving time for 1,000 / 10,000 images
- Proof size (should be constant regardless of test set size — Nova property)
- Verification time (target: < 1 second)
- Certified accuracy on MNIST at ε = 0.1, 0.3

### Critical Requirement: Find a Collaborator

This project requires expertise in two hard fields simultaneously:
- **Certification side** (DeepPoly, CROWN, abstract interpretation) — you have this
- **ZK engineering side** (Gnark, Nova, KZG, Bulletproofs) — you need this

Options:
- Contact FairProof authors (UC San Diego) — they solved the Gnark engineering already
- Contact ETH Zurich SRI — built DeepPoly, expressed interest in ZK combinations
- Find a CS PhD student specializing in applied cryptography at your institution

Do not underestimate this requirement. The ZK implementation is where most ML
researchers attempting this kind of work get stuck.

---

## Expected Results

| Metric | Expected |
| --- | --- |
| Proof size | < 10 KB (constant in N — Nova property) |
| Verification time | < 1 second |
| Proving time (MNIST, 10K images) | Hours (offline, one-time cost) |
| Certified accuracy at ε=0.1 | ~60-70% (matches DeepPoly baseline) |
| Constraints vs exact ReLU | ~100-200× fewer |

---

## Paper Pitch

> *"We present the first zero-knowledge system for certifying neural network accuracy
> on a private test set. Our key insight is LP certificate verification: instead of
> re-running DeepPoly inside a ZK circuit, we verify pre-computed DeepPoly bounds
> using only linear arithmetic, reducing constraints by 100-200× versus exact-ReLU
> approaches. Combined with Nova folding, our system produces a constant-size proof
> that a network achieves ≥ X% certified robustness at radius ε — without revealing
> model weights, test images, or labels."*

**Target venues:** CCS, S&P (IEEE), USENIX Security, ICLR (ML track)

---

## Key Papers to Read

### Foundation
- **DeepPoly** — Singh et al., POPL 2019.
  Abstract interpretation for NN certification. Our offline component.
  [PDF](https://ggndpsngh.github.io/files/DeepPoly.pdf)

- **α,β-CROWN** — Wang et al., NeurIPS 2021.
  State-of-the-art certified training. Understand bound propagation.
  [arXiv:2103.06624](https://arxiv.org/abs/2103.06624)

### Most Important Related Work
- **FairProof** — Yadav et al., ICML 2024.
  ZK proof of fairness certificate. Closest structural analog to our work.
  [arXiv:2402.12572](https://arxiv.org/abs/2402.12572)

- **Kaizen** — Abbaszadeh et al., CCS 2024.
  ZK proof of training. Proves correct training without revealing model/data.
  [ePrint:2024/162](https://eprint.iacr.org/2024/162)

### ZK Infrastructure
- **Nova** — Kothapalli et al., CRYPTO 2022.
  Folding scheme for incremental verifiable computation. Our aggregation layer.
  [arXiv:2107.04315](https://arxiv.org/abs/2107.04315)

- **TeleSparse** — PoPETS 2025.
  ZK-friendly sparsification for NN inference. Compare against for circuit size.
  [arXiv:2504.19274](https://arxiv.org/abs/2504.19274)

- **EZKL** — Open-source toolkit for ZK inference proofs.
  [github.com/zkonduit/ezkl](https://github.com/zkonduit/ezkl)

### Survey
- **ZK Proof Based Verifiable ML Survey** — Peng et al., Feb 2025.
  Full map of the ZKML field 2017-2025.
  [arXiv:2502.18535](https://arxiv.org/abs/2502.18535)

### ZK Proof Systems
- **Gnark** — Go ZK framework. FairProof used this — reference implementation.
  [github.com/Consensys/gnark](https://github.com/consensys/gnark)

- **Bulletproofs** — Bünz et al., S&P 2018.
  Inner product arguments for linear arithmetic. Eliminates circuits for linear layers.
  [arXiv:1907.06381](https://arxiv.org/abs/1907.06381)

- **Nova** — Kothapalli et al., CRYPTO 2022.
  Folding scheme for IVC. Our aggregation layer over N images.
  [arXiv:2107.04315](https://arxiv.org/abs/2107.04315)

- **ZK AI Inference High Precision** — CCS 2025.
  Latest inference proof efficiency results to compare against.
  [ACM DL](https://dl.acm.org/doi/10.1145/3719027.3765056)

---

## Groups to Contact

| Group | Why |
|---|---|
| ETH Zurich SRI (Gagandeep Singh, Martin Vechev) | Built DeepPoly — ideal collaborators for the certification side |
| UC San Diego ML Group (FairProof authors) | Built the closest prior system — understand their circuit design |
| Consensys (Gnark team) | ZK engineering support |
