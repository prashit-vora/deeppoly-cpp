# Neural Network Certification — Research Landscape & Open Problems

> Personal research map for publishing in formal ML verification and cryptographic certification.
> Last updated: May 2026. All numbers are from peer-reviewed venues unless marked (blog).

---

## Table of Contents

1. [What Is Certification? (Quick Orientation)](#1-what-is-certification)
2. [Classical Certification — Abstract Interpretation](#2-classical-certification--abstract-interpretation)
3. [Certified Training — Getting More Images Certified](#3-certified-training--getting-more-images-certified)
4. [Scaling to Large Networks](#4-scaling-to-large-networks)
5. [Randomized Smoothing — The Scale Winner](#5-randomized-smoothing--the-scale-winner)
6. [Zero-Knowledge Proofs for ML (ZKML)](#6-zero-knowledge-proofs-for-ml-zkml)
7. [Other Cryptographic Approaches](#7-other-cryptographic-approaches)
8. [The Giant Open Gap — ZKP + Robustness Certification](#8-the-giant-open-gap--zkp--robustness-certification)
9. [Other Open Problems Worth Publishing On](#9-other-open-problems-worth-publishing-on)
10. [Key Benchmarks & Numbers to Beat](#10-key-benchmarks--numbers-to-beat)
11. [Where to Publish](#11-where-to-publish)
12. [Key Groups to Follow](#12-key-groups-to-follow)
13. [Speech & Audio Certification](#13-speech--audio-certification)
14. [Game Theory & Certification](#14-game-theory--certification)
15. [Research Direction Rankings](#15-research-direction-rankings)

---

## 1. What Is Certification?

Two fundamentally different questions in this field — don't confuse them:

| Question | Approach | What it proves |
|---|---|---|
| Is this model robust to perturbations? | Abstract interpretation (DeepPoly, IBP, CROWN) | For ALL x' in ε-ball around x, model predicts same class |
| Was this computation done correctly? | ZKP (zkLLM, EZKL, zkCNN) | This specific output came from this specific model on this input |

**DeepPoly (POPL 2019, Singh et al.)** is the foundation of the first approach. It propagates symbolic intervals through the network. The proof condition is:

```
lower_bound[true_label] > upper_bound[j]  for every wrong class j
```

If true → formally certified. No attacker within the ε-ball can flip the prediction.

---

## 2. Classical Certification — Abstract Interpretation

### 2.1 Verifiers (Test-Time Tools)

| Tool | Key idea | Network scale | Venue |
|---|---|---|---|
| **DeepPoly** | Symbolic polyhedral bounds per neuron | Small MLPs/CNNs | POPL 2019 |
| **CROWN / α-CROWN** | Optimized per-neuron slopes (learned α) | Medium CNNs | NeurIPS 2018, 2021 |
| **β-CROWN** | Adds neuron split constraints for completeness | Large CNNs | NeurIPS 2021 |
| **α,β-CROWN** | Combines α + β + BaB on GPU | WideResNet scale | VNN-COMP winner 2021–2025 |
| **MN-BaB** | Multi-neuron (PRIMA) constraints + BaB | WideResNet scale | ICLR 2022 (ETH SRI) |
| **GCP-CROWN** | Cutting planes via MIP solver | Large networks | NeurIPS 2022 |
| **BICCOS** | Cuts from BaB tree itself (no MIP) | Larger than GCP-CROWN | NeurIPS 2024 |
| **GenBaB** | Extends BaB to non-ReLU (Sigmoid, GeLU, ViT) | Small transformers | TACAS 2025 |

**α,β-CROWN** has won every VNN-COMP from 2021–2025. It is the tool to use and the tool to beat. GitHub: `Verified-Intelligence/alpha-beta-CROWN`.

**Key finding (JMLR 2024, "Critically Assessing..."):** Across 79 networks, NO single verifier dominates all instances. A portfolio of tools beats any single one — this is underexplored and publishable.

### 2.2 Why Verification Gets Hard

- Every "mixed" ReLU (interval straddles zero) introduces approximation error
- Error compounds layer by layer
- Larger networks = more mixed ReLUs = looser bounds = harder to certify
- Complete verification (BaB) is EXPTIME in the worst case

### 2.3 The Soundness Gap (New, Scary)

**"No Soundness in the Real World" (arXiv 2506.01054, 2025):** All tested verifiers can be fooled by adversarial networks that exploit floating-point ordering and precision. A verifier that is theoretically sound may not be practically sound on deployed hardware. This is a live attack surface and almost completely unaddressed.

---

## 3. Certified Training — Getting More Images Certified

The goal: train models so that the verifier can actually certify them. Simply training with cross-entropy produces models with loose symbolic bounds.

### 3.1 Methods

| Method | Core Idea | CIFAR-10 certified @ ε=8/255 | Venue |
|---|---|---|---|
| **IBP** | Propagate interval bounds through training loss | ~35% | ICML 2019 |
| **CROWN-IBP** | Warmup with CROWN, switch to IBP | ~33–36% | ICLR 2020 |
| **SABR** | Train on small sub-box instead of full IBP box (λ controls size) | ~35.3% | ICLR 2023 |
| **TAPS** | IBP for early layers + PGD for later layers | ~35% (22% on TinyImageNet @ ε=1/255) | NeurIPS 2023 |
| **STAPS** | SABR + TAPS hybrid | ~35% | NeurIPS 2023 |
| **MTL-IBP** | Convex combo of adversarial loss + IBP loss | 34.61% | ICLR 2024 |
| **MTL-IBP + diffusion data** | Above + EDM-generated training images | ~37–38% | NeurIPS 2024 |

The robust loss used in this project's notebook:
```
robust_loss = softplus(logsumexp(upper_wrong) - lower_true)
```
This is differentiable and penalizes certification failure directly. It's closest to the MTL-IBP family.

### 3.2 The Uncomfortable Truth

**CTBENCH (arXiv 2406.04848, ETH SRI, 2024):** Unified benchmark of all major methods. Finding: MTL-IBP improves IBP by >10 pp at ε=2/255 but by only **0.13 pp** at ε=8/255. The certified accuracy wall at large perturbations **has not meaningfully moved since 2020**. Progress at small ε is real; progress at large ε is near-zero.

The best empirical robustness (adversarial training, WideResNet) is ~65% at ε=8/255. Best certified is ~37%. That **~28 pp gap** is the central unsolved problem of certified training.

### 3.3 Current Best Trick: Diffusion-Generated Data

**NeurIPS 2024:** Using EDM (diffusion model) to generate extra training data gives **+3.95 pp certified on CIFAR-10 (L2)** and **+1.39 pp (L∞)**. The generalization gap (train-test accuracy gap) predicts how much you'll benefit. This is currently the most reliable practical improvement.

---

## 4. Scaling to Large Networks

### 4.1 What's Feasible Today (Complete Verification)

| Scale | Feasible? |
|---|---|
| Small MLP (784→32→16→10) | Yes, fully |
| CNN (WideResNet, CIFAR-10) | Yes, α,β-CROWN |
| ResNet-50, ImageNet | No — not with any non-trivial ε |
| ViT (small) | Barely — GenBaB (TACAS 2025) |
| ViT (production scale) | No |
| LLMs | No — EXPSPACE |

### 4.2 Transformers

Attention's softmax is a non-polynomial nonlinearity — breaks standard layer-by-layer propagation. **GenBaB** (2025) handles small ViTs using a pre-optimized lookup table and branching heuristic. No complete verifier handles production ViTs.

**CertViT (arXiv 2302.10287, 2023):** Certified robustness for ViTs via proximal-projection, small models only.

### 4.3 Lipschitz-Based Scaling (Different Approach)

Instead of post-hoc verification, design the architecture to have a bounded Lipschitz constant (≤1) by construction.

**LipNeXt (arXiv 2601.18513, ICLR 2026):** 1-Lipschitz architecture (manifold optimization + Spatial Shift Module). Scales to **2 billion parameters on ImageNet**, +8% certified robust accuracy over prior Lipschitz models. First Lipschitz certification at billion-parameter scale.

### 4.4 Geometric Robustness (Rotation, Scaling)

Not expressible as L∞ perturbations — fundamentally harder because the perturbation set is non-convex.

**arXiv 2408.13140 (2024):** Piecewise linear approximation for geometric transforms. Resolves 32% more verification cases than prior convex relaxation. Still open for large networks.

---

## 5. Randomized Smoothing — The Scale Winner

**Core idea (Cohen et al., ICML 2019):** Add Gaussian noise to the input, take majority vote over many noisy predictions. The resulting "smoothed classifier" has a provable L2 robustness certificate (no DP bound).

**Why it scales:** No symbolic propagation needed. Just repeated inference. Works on any black-box model.

**Limitation:** Probabilistic (not deterministic), primarily L2 (L∞ extensions are weak), and requires many forward passes at inference time.

### 5.1 State of the Art

| Method | Setting | Certified Acc | Venue |
|---|---|---|---|
| Cohen et al. baseline | CIFAR-10, L2 r=0.25 | ~61% | ICML 2019 |
| Cohen et al. baseline | CIFAR-10, L2 r=0.50 | ~43% | ICML 2019 |
| DiffSmooth | CIFAR-10, L2 r=0.50 | **59.2%** (+17 pp) | USENIX Sec 2023 |
| DiffSmooth | ImageNet, L2 r=1.5 | **53.0%** (from 36%) | USENIX Sec 2023 |
| Multi-scale DDS | CIFAR-10, L2 r=2.0 | 8.8% (from 6.4%) | NeurIPS 2023 |
| State-of-art (2024) | ImageNet, L2 r=0.25 | ~64.2% | Various 2024 |

**Key advance:** Using a diffusion model as the denoiser before classification. **DiffSmooth (arXiv 2308.14333)** is the key paper.

---

## 6. Zero-Knowledge Proofs for ML (ZKML)

### 6.1 What ZKPs Prove (vs What DeepPoly Proves)

These are **orthogonal** — they prove fundamentally different things:

| | DeepPoly | ZKP (ZKML) |
|---|---|---|
| **Proves** | Network is robust to perturbations (all x' in ε-ball → same class) | This specific output was correctly computed from this model on this input |
| **Secret kept** | Nothing — proof is transparent | Model weights can remain hidden |
| **Adversary modeled** | Input adversary making worst-case perturbations | Cheating prover claiming a false computation |
| **Network scale** | Hundreds of millions of params (α,β-CROWN) | 13B params inference (zkLLM) |
| **Floating point** | Exact | Quantized (fixed-point) or exact via ZIP (2025) |
| **Proof artifact** | None — local computation | 200 KB – 1.63 MB SNARK |
| **Verification cost** | Seconds to minutes (LP/MILP) | 12 ms – 130 ms |

No published work combines both. **This is the central open gap.**

### 6.2 Key ZKML Systems

| System | Task | Max Network | Proof Time | Venue |
|---|---|---|---|---|
| **ZEN** | Inference + accuracy | LeNet (MNIST) | 147–4710 s | ePrint 2021/087 |
| **zkCNN** | CNN inference | LeNet-5 (CIFAR-10) | 88.3 s | ACM CCS 2021 |
| **Mystique** | Inference | ResNet-101 (42.5M) | 28 min (private weights) | USENIX Sec 2021 |
| **ZKML** (EuroSys) | Inference | GPT-2 (1.5B) | 3651 s | EuroSys 2024 |
| **Kaizen** | Training | VGG-11 (10M) | 15 min / iteration | ACM CCS 2024 |
| **zkLLM** | Inference | LLaMA-2 13B | < 15 min, <200 KB proof | ACM CCS 2024 |
| **zkDL** | Training (backprop) | 200M params | <1 s / batch (10M, 8-layer) | IEEE TIFS 2024 |
| **Artemis** | Inference + commitment | GPT-2 / VGG | 1.2x overhead (vs 11.5x baseline) | arXiv 2409.12055 |
| **ZKAUDIT** | Arbitrary audit functions | MobileNet v2 | Not reported | arXiv 2404.04500 |
| **zkPyTorch** | Inference | VGG-16 | 6.3 s / image | ePrint 2025/535 |
| **ZIP** | High-precision inference | Full pipelines | Not reported | ACM CCS 2025 |
| **DeepProve-1** | Inference (production) | GPT-2 / Gemma3 | 54–158x faster than EZKL | Lagrange, 2025 (blog) |
| **ZKTorch** | Inference | LLaMA-2-7B, BERT, ResNet-50 | 6.2x speedup over prior | arXiv 2507.07031 |

### 6.3 Three Tasks in ZKML

1. **Verifiable inference (ZKPoI):** Prove a specific output came from a specific model on a specific input
2. **Verifiable training (ZKPoT):** Prove weights resulted from SGD on a committed dataset (Kaizen, zkDL)
3. **Verifiable testing (ZKPoTest):** Prove declared accuracy is correct on a committed test set (ZEN, Artemis)

**Robustness certification as a verified property doesn't appear in any category.** That's the gap.

### 6.4 The Arithmetization Problem

ML uses IEEE-754 floating-point. ZK circuits work over finite fields. The mismatch:

- ReLU requires bit decomposition: ~hundreds of constraints per neuron
- Softmax requires exponential → approximated or expanded
- BatchNorm requires division → handled via Lagrange tricks

**ZIP (ACM CCS 2025)** is the first system to provide exact IEEE-754 double-precision semantics in a ZK proof. Before ZIP, all ZKML systems quantize to fixed-point (introducing error and breaking the exact correspondence to the trained model).

---

## 7. Other Cryptographic Approaches

| Approach | Representative Systems | Proves | Privacy | Latency Overhead | Trust Assumption |
|---|---|---|---|---|---|
| **ZKP / ZK-SNARK** | zkCNN, zkLLM, EZKL | Correct execution of committed model | Weights/input optional | 100x–100,000x | Computational hardness |
| **MPC (2-party)** | DELPHI, CrypTen, Gazelle | Joint inference without sharing data | Input privacy (semi-honest) | 22x–1000x | Honest majority |
| **FHE** | HETAL | Inference on encrypted input | Input never decrypted | Very high (1000x+) | Computational hardness |
| **TEE (SGX)** | Slalom, ORCA | Hardware-attested execution | Code + data in enclave | 3–20x (good) | Hardware manufacturer |
| **Hybrid MPC+ZKP** | Mystique in Rosetta | Correct + private MPC | Both input and model private | Higher than either | Computational hardness |

**Slalom (ICLR 2019)** is a useful baseline: uses Intel SGX to verify inference. 20x and 6x throughput improvement over VGG16 and MobileNet respectively. Much faster than ZKP but relies on trusting Intel.

---

## 8. The Giant Open Gap — ZKP + Robustness Certification

**No published paper combines a ZKP with a DeepPoly-style robustness certificate.**

This is the most novel direction available in this field right now.

### 8.1 What Would "ZK Certified Robustness" Mean?

A model owner (who wants to keep their weights private) proves to a third party:

> "My model is ε-robust on this input — certified by abstract interpretation — and I ran the certifier correctly, without revealing my weights."

This combines:
- **Privacy:** Third party never sees the model weights
- **Soundness:** The certification bound is a valid over-approximation
- **Non-interactivity:** One proof, verified cheaply (12–130 ms)

### 8.2 Concrete Research Directions (Ordered by Novelty + Feasibility)

---

**Direction A — ZK Proof of Certified Robustness** ⭐ Highest novelty

**Goal:** Given committed weights W, prove in ZK that DeepPoly certifies the model as ε-robust for a given input, without revealing W.

**Why it's feasible:**
- DeepPoly's linear interval propagation (the `dp_linear` pass) is entirely linear algebra — maps perfectly to finite-field arithmetic
- The ReLU case splits require bit decomposition (~hundreds of constraints per neuron) — hard but standard in ZK circuits
- The key challenge is the mixed-ReLU case: the convex hull approximation involves division (`upper / (upper - lower)`) which requires field inversion

**Approach:**
1. Arithmetize the DeepPoly forward pass in halo2 or EZKL
2. The "positive" / "negative" / "mixed" ReLU case split becomes a conditional constraint — field-compatible
3. The final certification check (`lower[y] > upper[j]`) becomes a range proof
4. Wrap in a SNARK → compact, publicly verifiable certificate

**Key challenge:** DeepPoly operates in floating-point. Either: (a) quantize to fixed-point + bound the error introduced (the ZK certificate must still be a sound over-approximation), or (b) use ZIP's IEEE-754 ZK techniques (CCS 2025).

**Publication target:** IEEE S&P, ACM CCS, or CAV

---

**Direction B — ZK Proof of Certified Accuracy** ⭐⭐ Medium novelty, lower difficulty

**Goal:** Extend ZEN's "verifiable testing accuracy" to the robustness setting. Prove that a model achieves a declared certified accuracy (fraction of test inputs with valid robustness certificates) on a committed test set.

**Why interesting:** An AI company could publicly prove their model is X% certifiably robust without revealing the model or test data. Huge practical demand (regulatory AI audits, safety claims).

**Approach:**
- Use IBP as the inner certifier (simpler to arithmetize than DeepPoly — just interval arithmetic, no symbolic coefficients)
- Prove the IBP pass over each test image → accumulate certified count
- Wrap in recursive SNARK for efficiency over the full test set

**Key challenge:** Proving over 10,000 test images requires recursive proof composition (Nova, Halo2 recursion, or similar).

**Publication target:** IEEE S&P, USENIX Security, ACM CCS

---

**Direction C — ZK Attestation of Verifier Soundness** ⭐ Addresses the soundness gap

**Goal:** The "No Soundness in the Real World" paper (2025) shows deployed verifiers can be fooled by floating-point precision tricks. A ZK proof that the verifier was run correctly (with certified arithmetic) would close this gap.

**Approach:** Compile α,β-CROWN's LP oracle calls into a ZK-verifiable format. The model owner runs the verifier + generates a proof. Third party verifies the proof, not the verifier software.

**Related:** arXiv 2506.09455 ("Abstraction-Based Proof Production in Formal Verification of Neural Networks", Jun 2025) — produces certificates from formal verifiers, but without the ZK privacy layer.

---

**Direction D — Robustness Audit via ZKAUDIT** ⭐ Lowest barrier to entry

**ZKAUDIT (arXiv 2404.04500, 2024)** supports arbitrary audit functions over hidden weights via halo2 + AIR. Cast IBP certification as an audit function:
- The audit function runs the IBP forward pass on committed weights
- Returns whether the image is certified
- ZKAUDIT proves this computation without revealing weights

This may be the easiest entry point — no new ZK infrastructure needed, just needs an arithmetic circuit for the IBP pass.

---

**Direction E — MPC-Based Robustness Audit** (Collaborative)

Instead of ZKP, use two-party MPC between model owner and certified auditor. Both jointly run abstract interpretation — neither learns the other's private information.

MPC is more tractable than ZKP for abstract interpretation because the operations (interval arithmetic, linear algebra) map naturally to secret-shared field elements. Trade-off: interactive vs. non-interactive.

---

### 8.3 Why This Matters Beyond Academia

Real-world need:
- AI companies want to claim robustness without revealing their model weights (IP protection)
- Regulators want to verify robustness claims without trusting the company's word
- Medical/safety-critical AI needs third-party verified robustness proofs
- Current situation: company runs certifier themselves and publishes numbers — no way to verify without the model

A "ZK Certified Robustness" proof breaks this trust bottleneck. It's analogous to how zk-SNARKs in blockchain allow transaction validity to be verified without seeing the transaction contents.

---

## 9. Other Open Problems Worth Publishing On

### 9.1 The ε=8/255 Wall on CIFAR-10

Best certified L∞ accuracy: ~37% (MTL-IBP + diffusion data, NeurIPS 2024)
Best empirical adversarial robustness: ~65% (adversarial training)
**Gap: ~28 pp. Hasn't closed since 2020.**

Approach worth trying: Combine randomized smoothing (good for large radii) with IBP training (good for small radii) in a hybrid training objective. Neither alone closes the gap; a hybrid has not been thoroughly explored.

### 9.2 No Certification for Geometric Perturbations at Scale

Rotations, translations, scaling — not L∞ balls. The perturbation set is non-convex so standard DeepPoly doesn't apply. arXiv 2408.13140 (2024) solves 32% more cases via piecewise linear approximation, but only for small networks.

Open: geometric robustness certification for CNNs used in computer vision (object detection, segmentation).

### 9.3 Certified Training for Transformers

TAPS achieves 22% certified accuracy on TinyImageNet. Nothing comparable exists for transformer architectures. Attention's softmax breaks standard IBP propagation.

A certified training method for ViTs that achieves meaningful certified accuracy would be a significant contribution.

### 9.4 Algorithm Portfolios for Verification

JMLR 2024 ("Critically Assessing..."): Shapley value analysis shows combining verifiers in a portfolio beats any single tool. No practical portfolio system has been built and evaluated end-to-end. Straightforward systems paper with clear benchmark.

### 9.5 Formally Verified Certifier

arXiv 2505.06958 (May 2025): "A Formally Verified Robustness Certifier for Neural Networks" — implements a certification function in Dafny (a verification-aware language) and formally verifies its soundness. Extension to more expressive domains (DeepPoly vs just intervals) would be publishable.

---

## 10. Key Benchmarks & Numbers to Beat

### Certified Training — CIFAR-10 L∞

| ε | Current best certified acc | System | Target to beat |
|---|---|---|---|
| 2/255 | ~60%+ | MTL-IBP variants | >65% |
| 8/255 | ~37–38% | MTL-IBP + diffusion data | >40% |

### Randomized Smoothing — ImageNet L2

| Radius | Current best | System | Target |
|---|---|---|---|
| 0.25 | ~64.2% | 2024 ViT-based | >67% |
| 1.0 | ~40-45% | DiffSmooth variants | >50% |
| 1.5 | 53.0% | DiffSmooth | >58% |

### VNN-COMP Standard Benchmark

Any new verifier should beat α,β-CROWN on at least one benchmark family. Focus on: transformer benchmarks (GenBaB is weakest here), or the MaxPool tightening benchmarks (CVPR 2025 showed 78.6% improvement possible).

---

## 11. Where to Publish

| Venue | Best for | Deadline cycle |
|---|---|---|
| **IEEE S&P (Oakland)** | Security + privacy angle (ZKP + certification) | ~October each year |
| **ACM CCS** | Cryptographic ML, ZKP systems | ~May each year |
| **USENIX Security** | Systems-oriented ZKP/ML work | ~Feb + Oct each year |
| **ICLR** | Certified training, new abstract domains | ~October each year |
| **NeurIPS** | Certified training, randomized smoothing | ~May each year |
| **ICML** | Certified training, theoretical bounds | ~Feb each year |
| **CAV** | Formal verification, new abstract domains | ~Jan each year |
| **POPL** | Abstract interpretation theory | ~July each year |

**For the ZKP + certification gap:** IEEE S&P or ACM CCS are the natural homes. If the angle is more on the ML side, ICLR is possible.

---

## 12. Key Groups to Follow

| Group | Focus | Where |
|---|---|---|
| **ETH SRI Lab** (Singh, Gehr, Vechev, Mueller) | DeepPoly, PRIMA, MN-BaB, CTBench | eth-sri.github.io |
| **UIUC** (Huan Zhang) | α,β-CROWN, β-CROWN, GCP-CROWN | Verified-Intelligence/alpha-beta-CROWN |
| **Waterloo** (Hongyang Zhang) | zkLLM, zkDL, certified training | |
| **Berkeley** (Stoica group) | ZKML (EuroSys 2024) | |
| **CMU / MIT** | Randomized smoothing, IBP training | |
| **Lagrange Labs** | DeepProve (production ZKML) | lagrange.dev |
| **Polyhedra Network** | zkPyTorch, EZKL competitor | |
| **EZKL** | Open-source ZK inference | ezkl.xyz |

---

## Key Papers to Read First (Priority Order)

1. **DeepPoly** — Singh et al., POPL 2019. The foundation.
2. **α,β-CROWN** — Wang et al., NeurIPS 2021. The state-of-art verifier.
3. **CTBENCH** — arXiv 2406.04848. Honest assessment of certified training progress.
4. **"Critically Assessing..."** — JMLR 2024. Most honest picture of where the field stands.
5. **SABR** — Müller et al., ICLR 2023. Best simple certified training baseline.
6. **MTL-IBP** — De Palma et al., ICLR 2024. Current SOTA certified training.
7. **DiffSmooth** — Zhang et al., USENIX Sec 2023. Best smoothing result.
8. **zkLLM** — Sun et al., ACM CCS 2024. Scale frontier for ZKP.
9. **ZKML (EuroSys)** — Chen et al., EuroSys 2024. Best system overview.
10. **ZIP** — Riasi et al., ACM CCS 2025. Solves the floating-point arithmetization problem.
11. **"No Soundness in the Real World"** — arXiv 2506.01054, 2025. The soundness gap paper — important for motivation.
12. **ZKAUDIT** — arXiv 2404.04500. Closest existing work to Direction D above.
13. **GenBaB** — Shi et al., TACAS 2025. Transformers in ZK verification.
14. **LipNeXt** — Hu et al., arXiv 2601.18513. Billion-parameter certified models.

---

*The highest-novelty, highest-impact direction: **ZK Proof of Certified Robustness** (Direction A above). No paper has done it. The tools (halo2, EZKL, ZIP) exist. The theory (DeepPoly) is mature. The demand (private AI auditing) is real. This could be a top-4 venue paper.*

---

## 13. Speech & Audio Certification

### 13.1 Why Audio Is Harder Than Images for Certification

| Challenge | Image Domain | Audio Domain |
|---|---|---|
| **Perturbation norm** | L∞ pixel — simple, standard | Psychoacoustic masking threshold — non-convex, signal-dependent, non-Lp |
| **Preprocessing** | Normalization (differentiable) | STFT + mel filterbank + log compression (non-differentiable, non-monotone) |
| **Output type** | Scalar class label | Variable-length sequence → WER metric |
| **Model architecture** | CNNs (well-studied) | RNNs, LSTMs, Transformers (Whisper, wav2vec2) — largely uncertified |
| **Temporal structure** | None (IID pixels) | Sequential dependencies across 10-100ms frames |
| **Perceptibility metric** | SSIM, LPIPS | Auditory masking threshold — no certification framework exists for it |

### 13.2 Attack Landscape

| Attack | Paper | Constraint | Success Rate | Over-the-Air? |
|---|---|---|---|---|
| Carlini-Wagner Audio | DLS Workshop 2018 | L∞ waveform | **100%** targeted on DeepSpeech | No (digital) |
| Imperceptible + Robust | Qin, Carlini et al., ICML 2019 | Psychoacoustic masking | 100% digital; **>60% OTA** | Yes |
| Imperio | Schönherr et al., 2019 | Psychoacoustic + room impulse | ~50% without environment adaptation | Yes |
| Inaudible SV Attack | Du et al., 2020 | Below auditory threshold | **98.5%** cross-gender impersonation | No |
| PGD on Whisper | Olivier & Raj, Interspeech 2023 | L2, SNR 35–45 dB | Dramatic WER increase | No |
| Universal prefix mute | Various 2022–2024 | 0.64s audio prefix | **>97% mute rate** | Partially |

### 13.3 Certification Methods for Speech

| Method | Venue | Architecture | Preprocessing certified? | Metric | Key Numbers |
|---|---|---|---|---|---|
| **POPQORN** | ICML 2019 | LSTM, GRU, RNN | No | Certified min distortion | Baseline — Cert-RNN is 1.86x tighter |
| **Cert-RNN** | CCS 2021 | LSTM, RNN | No | Certified bound | 1.86x tighter than POPQORN on MNIST-seq |
| **PROVER** | CAV 2021 (ETH SRI) | 2-layer LSTM (50 hidden) | **Yes** (Square, Log, mel filterbank) | Certified classification acc | ~86% @ -90 dB, ~70% @ -80 dB (FSDD) |
| **Sequential Smoothing** | EMNLP 2021 | DeepSpeech2, Espresso Transformer | Via speech enhancement | **WER** (probabilistic) | Robust to inaudible attacks; breakable at high distortion |
| **Center Smoothing** | NeurIPS 2021 | Any (framework) | N/A | Structured output metric (e.g., edit distance) | Not applied to any large ASR system |
| **SV Certification** | AAAI 2025 | ECAPA-TDNN, Pyannote, CAM++ | No (embedding level) | Certified few-shot SV acc | σ=0.01, L2 only; gap vs. empirical acknowledged as loose |

**PROVER** (ETH SRI, CAV 2021) is the closest existing work to "DeepPoly for speech." It extends DeepPoly's abstract transformers to handle the Square and Log operations in the mel filterbank. The gap between polyhedral bounds (~86%) and interval-only bounds (~61%) at -90 dB on FSDD demonstrates the same tightness advantage DeepPoly has over IBP in the image domain. Code: `github.com/eth-sri/prover`.

### 13.4 Open Gaps — All Publishable

**Gap 1 — WER-level deterministic certification.** PROVER certifies classification (digit/keyword recognition), not full ASR. No paper produces a certified WER bound using abstract interpretation. Center Smoothing (NeurIPS 2021) handles structured outputs in principle but has not been applied to any large ASR system. Bridging this is the central open problem.

**Gap 2 — Transformer ASR certification.** Whisper and wav2vec2 are entirely uncertified. Whisper is demonstrably vulnerable at 35–45 dB SNR; a universal 0.64-second prefix achieves >97% mute rates. CertViT exists for vision transformers but has not been adapted for audio transformers. The cross-attention in encoder-decoder models (Whisper) adds further complexity.

**Gap 3 — Certified training for speech.** IBP, CROWN-IBP, SABR, and TAPS have all been developed exclusively for image classifiers. Applying certified training to a speech command classifier (FSDD, Google Speech Commands) using L2 waveform bounds is the most directly actionable gap — directly analogous to MNIST certified training, never done for audio.

**Gap 4 — Psychoacoustic norm certification.** All existing audio certification uses L2 or L∞ norms. The perceptually correct threat model is the psychoacoustic masking threshold, which is non-convex, time-frequency-dependent, and signal-dependent. No certification framework handles it. Anisotropic randomized smoothing (Eiras et al., arXiv 2207.05327) calibrated to per-frequency-bin masking thresholds is a promising direction.

**Gap 5 — Spectrogram-space vs. waveform-space connection.** When an adversary perturbs in waveform space, the mel-spectrogram perturbation is non-linear (STFT magnitude, log compression). PROVER handles this for small LSTMs but not for modern end-to-end ASR. Certifying robustness to perturbations in spectrogram space and bounding their waveform equivalents requires new abstraction techniques.

### 13.5 Papers to Read

1. **PROVER** — Ryou et al., CAV 2021. arXiv:2005.13300. The DeepPoly extension for speech.
2. **POPQORN** — Ko et al., ICML 2019. arXiv:1905.07387. First RNN/LSTM certifier.
3. **Sequential Smoothing** — Olivier & Raj, EMNLP 2021. arXiv:2112.03000. Only WER-level certification. Code: `github.com/RaphaelOlivier/smoothingASR`.
4. **SV Certification** — Korzh et al., AAAI 2025. arXiv:2404.18791. Most recent audio certified result.
5. **Imperceptible ASR attack** — Qin, Carlini et al., ICML 2019. The psychoacoustic threat model.
6. **Center Smoothing** — Kumar & Goldstein, NeurIPS 2021. arXiv:2102.09701. Framework for structured output certification.

---

## 14. Game Theory & Certification

### 14.1 The Minimax Foundation

Every certified training method is implicitly solving a minimax problem:

```
min_θ  max_{δ ∈ S}  L(θ, x + δ)
```

Madry et al. (ICLR 2018) formalized this as a Stackelberg game: the defender (leader) commits to weights θ, then the attacker (follower) optimizes δ. PGD solves the inner maximization. This is the implicit game-theoretic framing every certified training paper inherits — but almost none of them analyze it as a game.

### 14.2 Nash Equilibria in Adversarial ML

| Paper | Venue | Finding |
|---|---|---|
| **Meunier et al.** | ICML 2021 | Mixed Nash equilibria **provably exist** in the zero-sum attacker-classifier game (no duality gap). Both players must randomize at optimum. |
| **Balcan et al.** | AISTATS 2023 | A **unique pure Nash equilibrium** exists and is provably robust — but standard PGD adversarial training **may not converge to it**. Can cycle. |
| **Chen et al.** | Acta Math. Sci. 2022 | Stackelberg formulation (classifier as leader, adversary as follower) gives a **Stackelberg equilibrium** and optimal strategies analytically. |
| **Farnia & Ozdaglar** | ICML 2020 | GANs do **not always have Nash equilibria** — the generator-moves-first sequential structure requires proximal equilibrium. |

**Key implication:** The minimax training everyone uses is not guaranteed to converge to the game's solution. The procedure is heuristic even when the theory says a Nash equilibrium exists.

### 14.3 Game-Theoretic Certified Training

| Paper | Venue | Idea | Connection to Certification |
|---|---|---|---|
| **SALT** | EMNLP 2021 | Reformulates adversarial regularization as Stackelberg game; computes "Stackelberg gradient" via unrolled optimization | Not yet applied to CROWN/DeepPoly training |
| **MACER** | ICLR 2020 | Directly maximizes certified radius (randomized smoothing) as training objective — collapses the attacker's role | Closest to game-theoretically principled certified training |
| **Wasserstein DRO** | ICLR 2018 | Minimax over worst-case distribution in Wasserstein ball; certifies population-level robustness | Different certificate type than IBP/DeepPoly; not systematically compared |
| **Bai et al.** | NeurIPS 2023 | Extends WDRO to ReLU + GELU networks using piecewise-affine structure | Lipschitz certificates; not IBP/CROWN compatible |

**No paper frames DeepPoly or CROWN-IBP training as a Stackelberg or regret-minimization problem.** The minimax structure is present in every certified training objective but the game-theoretic consequences — convergence guarantees, regret bounds, optimal mixing — are unstudied.

### 14.4 Verification as a Two-Player Game

**DeepGame** (Wu, Kwiatkowska et al., TCS 2020) is the **only paper that explicitly frames neural network verification as a two-player game**:
- **Attacker:** MCTS to search for adversarial examples → computes upper bound on robustness
- **Verifier:** A*/Alpha-Beta pruning to certify safety → computes lower bound on robustness
- Produces both a maximum safe radius and feature robustness certificate with provable guarantees

No subsequent work has unified this game-based verification framing with certified training objectives.

### 14.5 Multi-Agent Robustness

**BARDec-POMDP** (Li et al., ICLR 2024): Byzantine adversaries in cooperative MARL are modeled as nature-dictated types in a **Bayesian game**. Solution concept: ex post robust Bayesian Markov perfect equilibrium (provably exists). A two-timescale actor-critic algorithm converges almost surely. Validated on StarCraft II against non-oblivious adversaries.

**AlphaZero is vulnerable** (Lan et al., NeurIPS 2022): KataGo (50 MCTS simulations) plays a losing move in 58% of self-play games when two meaningless stones are added. 90% of such adversarial examples also fool amateur human Go players. **No formal certification has been attempted on any game-playing agent architecture.**

### 14.6 Shapley Values in Verification

**König et al. (JMLR 2024)** is the **only published paper** using cooperative game theory to analyze verification tools. Across 79 networks and multiple verifiers, Shapley values quantified each tool's marginal contribution to portfolio coverage. Key finding: no single verifier dominates — a portfolio wins. Code: `github.com/ADA-research/nn-verification-assessment`.

**No other paper applies Shapley values, Banzhaf power indices, or any cooperative game measure to verifier analysis.** Extensions are open.

### 14.7 The Certification Paradox (Cullen et al., ICML 2024)

**"Et Tu Certifications"** (arXiv:2302.04379): Publicly disclosed robustness certificates become attack resources. A **certification-aware attacker** exploiting the certificate finds adversarial examples **74% more often** than an unconstrained attacker, with **>10% smaller perturbation norms**.

Strategic implication: releasing a certificate changes the game state — the attacker gains information the defender would prefer to withhold. This is an **information-revelation / mechanism design problem** that has not been formally analyzed.

### 14.8 Open Problems — All Novel

**Open Problem 1 — Stackelberg certified training via differentiable bounds.**
CROWN produces differentiable lower bounds. The SALT paper (EMNLP 2021) differentiates through an attacker's PGD update via unrolling. Combining these: a Stackelberg certified training algorithm where the "attacker" uses CROWN bounds rather than PGD. Requires: CROWN-IBP codebase + SALT's unrolling. Potentially closes the gap between adversarial training (~65% empirical) and certified training (~37% certified) by giving the training algorithm a tighter attack oracle.

**Open Problem 2 — Shapley attribution of bound tightness per layer.**
The König et al. approach attributes verification success to tools. An analogous cooperative game: each "player" is a network layer, the "value" of a coalition is the bound tightness achieved by jointly optimizing those layers' relaxations. Shapley values then identify which layers are the certification bottleneck — directly actionable for architecture and training decisions.

**Open Problem 3 — Mixed-strategy ensemble of certified classifiers.**
Meunier et al. proved optimal defense requires randomizing over classifiers. Train K IBP/CROWN-certified classifiers and compute the Nash-optimal mixing distribution. Test whether the ensemble certified bound is tighter than any individual classifier. Connects game-theoretic optimality to practical certification improvement.

**Open Problem 4 — Information design for certification disclosure.**
The Cullen et al. paradox raises the question: what information about a certificate should a defender reveal to a regulator without arming attackers? Optimal disclosure policies (full / partial / probabilistic certification release) are a mechanism design problem — completely unstudied in the ML security literature.

**Open Problem 5 — Certifying neural mechanism design components.**
Auction algorithms and fair division mechanisms now use neural networks (RegretNet, deep learning for combinatorial auctions). Certifying that these neural mechanisms satisfy incentive-compatibility or individual rationality under input perturbations is formally unposed but practically urgent — a manipulated auction is a concrete financial attack.

**Open Problem 6 — Regret-minimization over verifier calls.**
Frame certified training as an online learning problem: at each round the algorithm picks weights, an adversary picks a perturbation distribution, and loss is the certified worst-case loss. A no-regret algorithm would guarantee that time-averaged certified loss converges to the minimax value. This connects to robust optimization via online learning (arXiv:2101.11443) but that paper does not handle certified bounds.

### 14.9 Key Papers to Read

1. **Madry et al.** — ICLR 2018. The PGD saddle-point formulation. Foundational.
2. **Meunier et al.** — ICML 2021. arXiv:2102.06905. Mixed Nash existence proof.
3. **Balcan et al.** — AISTATS 2023. arXiv:2210.12606. Non-convergence of adversarial training.
4. **DeepGame** — Wu, Kwiatkowska et al., TCS 2020. arXiv:1807.03571. Verification as two-player game.
5. **SALT** — Zheng et al., EMNLP 2021. arXiv:2104.04886. Stackelberg gradient for adversarial regularization.
6. **BARDec-POMDP** — Li et al., ICLR 2024. arXiv:2305.12872. Byzantine MARL as Bayesian game.
7. **Cullen et al.** — ICML 2024. arXiv:2302.04379. The certification paradox.
8. **König et al.** — JMLR 2024. The Shapley verifier portfolio analysis.
9. **MACER** — Zhai et al., ICLR 2020. arXiv:2001.02378. Certified radius maximization as training objective.
10. **Wasserstein DRO** — Sinha et al., ICLR 2018. Certifiable distributional robustness.

---

## 15. Research Direction Rankings

Two independent rankings across all 20 identifiable open directions in this file. Use these to pick where to invest research time.

---

### Table 1 — Ranked by Easiness (1 = easiest)

| Rank | Direction | Difficulty | Why | Best Venue |
|------|-----------|-----------|-----|-----------|
| 1 | Shapley attribution of bound tightness per layer | ⬜ Easy | König et al. codebase exists; plug-in cooperative game on top of CROWN | NeurIPS, ICML |
| 2 | Mixed-strategy ensemble of certified classifiers | ⬜ Easy | Train K IBP models, compute Nash mixing — no new theory needed | ICLR, NeurIPS |
| 3 | Certified training for speech (LSTM + mel) | ⬜ Easy | PROVER + Cert-RNN codebases exist; extend to wav2vec2 | ICLR, Interspeech |
| 4 | Psychoacoustic certification norm | 🟡 Medium | dB(A) / masking threshold as Lp substitute — needs new math but scope is bounded | ICASSP, ICLR |
| 5 | Certified WER bound (speech) | 🟡 Medium | Compose phoneme-level certificate with WER formula; interval arithmetic over strings | EMNLP, ACL |
| 6 | Speaker verification certification | 🟡 Medium | Korzh 2025 did one model; extend to ECAPA-TDNN, x-vector | Interspeech, ICASSP |
| 7 | Stackelberg certified training via CROWN | 🟡 Medium | CROWN-IBP + SALT unrolling; engineering-heavy but clear roadmap | ICML, NeurIPS |
| 8 | Regret-minimization over verifier calls | 🟡 Medium | Online learning framing is well-studied; bridging to certified bounds is the gap | COLT, NeurIPS |
| 9 | Information design for certification disclosure | 🟡 Medium | Mechanism design is standard; applying it to Cullen paradox is novel | EC, NeurIPS |
| 10 | Certifying Whisper / wav2vec2 | 🟠 Medium-Hard | Large Transformer; IBP diverges — need CROWN or randomized smoothing adaptation | ICLR, ICASSP |
| 11 | Certifying game-playing agents (AlphaZero) | 🟠 Medium-Hard | MCTS state space explosion; DeepGame framework partially applies | IJCAI, AAAI |
| 12 | ZK Proof of Certified Accuracy | 🟠 Medium-Hard | Prove accuracy ≥ threshold without revealing test set; ZKP engineering complex | S&P, CCS |
| 13 | Scalable abstract interpretation (Transformers) | 🟠 Medium-Hard | Attention head bounds — active research; α,β-CROWN partially handles it | ICLR, NeurIPS |
| 14 | Tight randomized smoothing (L∞, Lp) | 🟠 Medium-Hard | Derandomization open problem; distribution design is theoretically hard | NeurIPS, ICML |
| 15 | Certified training for large models (ResNet+) | 🟠 Medium-Hard | IBP diverges; needs SABR/expressive relaxations + compute | ICLR, NeurIPS |
| 16 | ZKP + robustness combination (bound proof) | 🔴 Hard | Requires zkSNARK circuit for CROWN forward pass — large circuit depth | S&P, CCS, USENIX |
| 17 | Privacy-preserving certified inference | 🔴 Hard | MPC + bound propagation simultaneously — latency explosion | CCS, CRYPTO |
| 18 | ZK Proof of Certified Robustness | 🔴 Very Hard | Prove ∀x' in ε-ball f(x')=c inside a ZK circuit — no known efficient encoding | S&P, CCS |
| 19 | Certifying neural auction mechanisms | 🔴 Very Hard | Incentive-compatibility under perturbation — new formalism needed | EC, NeurIPS |
| 20 | Certifying neural mechanism design (general) | 🔴 Very Hard | No prior work; requires unifying formal verification + mechanism design theory | FOCS, STOC, EC |

---

### Table 2 — Ranked by Importance / Reward (1 = most impactful)

| Rank | Direction | Impact | Why It Matters |
|------|-----------|--------|---------------|
| 1 | ZK Proof of Certified Robustness | ⭐⭐⭐⭐⭐ | Would close the single largest gap in trustworthy AI — prove robustness without revealing weights; field-defining paper |
| 2 | ZK Proof of Certified Accuracy | ⭐⭐⭐⭐⭐ | Enables auditable AI regulation without IP disclosure; immediate policy relevance |
| 3 | Certifying Whisper / wav2vec2 | ⭐⭐⭐⭐⭐ | 911 dispatch, medical STT, legal transcription — high-stakes deployment with zero certified coverage today |
| 4 | Stackelberg certified training via CROWN | ⭐⭐⭐⭐ | Could close the empirical vs certified accuracy gap (65% → 37%) — the central bottleneck of the entire field |
| 5 | Certified training for large models (ResNet+) | ⭐⭐⭐⭐ | Makes certification deployment-practical; without it the field stalls at toy-scale |
| 6 | Tight randomized smoothing (L∞, Lp) | ⭐⭐⭐⭐ | L∞ smoothing would enable scaling *and* stronger guarantees — the best of both worlds |
| 7 | Scalable abstract interpretation (Transformers) | ⭐⭐⭐⭐ | LLMs and vision Transformers have zero deterministic certification; enormous deployment surface |
| 8 | Psychoacoustic certification norm | ⭐⭐⭐⭐ | Existing Lp norms are meaningless for audio; a correct threat model unlocks the whole speech sub-field |
| 9 | Information design for certification disclosure | ⭐⭐⭐⭐ | Cullen paradox shows naive disclosure harms defenders; mechanism design answer would affect regulation |
| 10 | Certified WER bound (speech) | ⭐⭐⭐ | First deterministic WER certificate would be landmark paper for NLP + speech communities |
| 11 | Certified training for speech (LSTM + mel) | ⭐⭐⭐ | Clear path exists; first to do it properly gets the ICLR/Interspeech paper |
| 12 | Regret-minimization over verifier calls | ⭐⭐⭐ | Connects online learning theory to certification; opens algorithmic robustness as new sub-field |
| 13 | Privacy-preserving certified inference | ⭐⭐⭐ | MPC + certification combo needed for federated + medical deployments |
| 14 | Certifying game-playing agents (AlphaZero) | ⭐⭐⭐ | Adversarial Go exploits documented; certification of game agents is entirely open |
| 15 | Speaker verification certification | ⭐⭐⭐ | Voice authentication attacks are real and deployed; formal certificates would have direct security impact |
| 16 | ZKP + robustness combination (bound proof) | ⭐⭐⭐ | First step toward the ZK robustness dream; publishable intermediate result |
| 17 | Shapley attribution of bound tightness per layer | ⭐⭐ | Useful diagnostic; identifies certification bottleneck layers; incremental over König et al. |
| 18 | Mixed-strategy ensemble of certified classifiers | ⭐⭐ | Game-theoretically optimal but marginal empirical gains expected |
| 19 | Certifying neural auction mechanisms | ⭐⭐ | Niche but novel; financial stakes are real once neural auctions deploy at scale |
| 20 | Certifying neural mechanism design (general) | ⭐⭐ | Theoretically interesting; very long time to real-world impact |

---

### Sweet Spot Summary — High Reward, Reasonable Effort

| Direction | Effort | Reward | Verdict |
|-----------|--------|--------|---------|
| Certified training for speech | Easy | ⭐⭐⭐ | **Start here** — fastest path to a real paper |
| Stackelberg certified training | Medium | ⭐⭐⭐⭐ | **Best medium-term target** — high impact, clear implementation path |
| ZK Proof of Certified Accuracy | Medium-Hard | ⭐⭐⭐⭐⭐ | **Best collaborative target** — needs ZKP engineering partner |
| ZK Proof of Certified Robustness | Very Hard | ⭐⭐⭐⭐⭐ | **The dream paper** — plan 2–3 years, publish intermediate results first |
