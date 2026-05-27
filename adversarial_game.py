#!/usr/bin/env python3
"""
Adversarial Training with DeepPoly Certification
=================================================
Trains a CNN on MNIST using PGD adversarial training (Madry et al. 2018)
and certifies robustness via DeepPoly bound propagation (Singh et al. 2019).

Architecture:
    Conv(1→16, 4×4, stride=2) → ReLU → Conv(16→32, 4×4, stride=2) → ReLU
    → Flatten → Dense(1568→64) → ReLU → Dense(64→10) → logits

Min-max adversarial game (two-player zero-sum):

    Defender (θ) minimises the worst-case loss against an Attacker (δ):

        θ* = argmin_θ  max_δ  E[ L(f_θ(x + δ), y) ]   (‖δ‖_∞ ≤ ε)

    • Attacker (inner maximisation): approximated by PGD-7 during
      training, evaluated by PGD-40 at test time.
    • Defender (outer minimisation): trains θ on adversarially-
      perturbed batches (Madry et al. 2018).

    Two notions of robustness:
        Empirical:  does PGD-40 find an adversarial example?
        Certified:  DeepPoly proves that *no* perturbation inside ε can
                    change the prediction (sound over-approximation).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import struct, gzip, urllib.request, os, time, sys
from pathlib import Path

plt.rcParams.update({'font.family': 'serif', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': 0.3, 'figure.dpi': 120})

DEVICE = torch.device('cpu')
print(f"Device: {DEVICE}")

# ────────────────────────────────────────────────────────────
# 1. DATA
# ────────────────────────────────────────────────────────────

BASE  = "https://ossci-datasets.s3.amazonaws.com/mnist"
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz",  "t10k-labels-idx1-ubyte.gz"]

Path("mnist").mkdir(exist_ok=True)
for fname in FILES:
    out = Path("mnist") / fname[:-3]
    if not out.exists():
        print(f"Downloading {fname}...")
        urllib.request.urlretrieve(f"{BASE}/{fname}", str(out) + ".gz")
        with gzip.open(str(out) + ".gz", "rb") as fi, open(out, "wb") as fo:
            fo.write(fi.read())
        os.remove(str(out) + ".gz")

def read_images(path):
    with open(path, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), np.uint8).reshape(n, r, c).astype(np.float32) / 255.0

def read_labels(path):
    with open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), np.uint8).astype(np.int64)

X_train_full = read_images("mnist/train-images-idx3-ubyte")
y_train_full = read_labels("mnist/train-labels-idx1-ubyte")
X_test_imgs  = read_images("mnist/t10k-images-idx3-ubyte")
y_test       = read_labels("mnist/t10k-labels-idx1-ubyte")
print(f"Data: train {X_train_full.shape}, test {X_test_imgs.shape}")

# ────────────────────────────────────────────────────────────
# 2. CNN
# ────────────────────────────────────────────────────────────

class CNN(nn.Module):
    """Small CNN for MNIST. Outputs raw logits (CrossEntropyLoss applies softmax)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.fc1   = nn.Linear(32 * 7 * 7, 64)
        self.fc2   = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, x):
        return self.forward(x).argmax(1)

    def get_weights(self):
        sd = self.state_dict()
        return {
            'conv1.k': sd['conv1.weight'].cpu().numpy(),
            'conv1.b': sd['conv1.bias'].cpu().numpy(),
            'conv2.k': sd['conv2.weight'].cpu().numpy(),
            'conv2.b': sd['conv2.bias'].cpu().numpy(),
            'fc1.W':   sd['fc1.weight'].cpu().numpy().T,
            'fc1.b':   sd['fc1.bias'].cpu().numpy(),
            'fc2.W':   sd['fc2.weight'].cpu().numpy().T,
            'fc2.b':   sd['fc2.bias'].cpu().numpy(),
        }

# ────────────────────────────────────────────────────────────
# 3. PGD ADVERSARIAL TRAINING
# ────────────────────────────────────────────────────────────

def pgd_batch(net, x, y, eps=0.10, steps=7):
    """PGD-7 on a batch for training. Random start, gradients w.r.t. inputs only."""
    step_size = eps / 4.0
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = net(x_adv)
        loss = nn.CrossEntropyLoss()(logits, y)
        grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]
        with torch.no_grad():
            x_adv = x_adv + step_size * grad.sign()
            x_adv = torch.clamp(x_adv, x - eps, x + eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()

def train(net, X_tr_np, y_tr_np, X_te_np, y_te_np,
          epochs=15, lr=0.001, batch=128, adv_eps=0.1, adv_steps=7,
          epoch_callback=None):
    """PGD adversarial training (Madry et al. 2018)."""

    X_tr = torch.tensor(X_tr_np, device=DEVICE)
    y_tr = torch.tensor(y_tr_np, device=DEVICE)
    X_te = torch.tensor(X_te_np, device=DEVICE)
    y_te = torch.tensor(y_te_np, device=DEVICE)

    opt = optim.Adam(net.parameters(), lr=lr)
    N = len(X_tr)

    for ep in range(epochs):
        t0 = time.time()
        perm = torch.randperm(N, device=DEVICE)
        losses = []
        for s in range(0, N, batch):
            idx = perm[s:s+batch]
            xb, yb = X_tr[idx], y_tr[idx]
            xb_adv = pgd_batch(net, xb, yb, eps=adv_eps, steps=adv_steps)

            opt.zero_grad()
            out = net(xb_adv)
            loss = nn.CrossEntropyLoss()(out, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        with torch.no_grad():
            acc = (net.predict(X_te) == y_te).float().mean().item() * 100
        print(f"  Epoch {ep+1:2d}/{epochs}  loss={np.mean(losses):.5f}  "
              f"acc={acc:.2f}%  ({time.time()-t0:.1f}s)")
        if epoch_callback:
            epoch_callback(ep)

# ────────────────────────────────────────────────────────────
# 4. EVALUATION ATTACK
# ────────────────────────────────────────────────────────────

def pgd(net, x_flat, y_true, eps, steps=40):
    """Multi-step PGD attack for evaluation."""
    x0 = torch.tensor(x_flat.reshape(1, 1, 28, 28), device=DEVICE)
    y = torch.tensor([y_true], device=DEVICE)
    adv = x0.clone().requires_grad_(True)
    step_size = eps / 10.0
    for _ in range(steps):
        out = net(adv)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        adv = adv + step_size * adv.grad.sign()
        adv = torch.clamp(adv, x0 - eps, x0 + eps)
        adv = torch.clamp(adv, 0.0, 1.0)
        adv = adv.detach().requires_grad_(True)
    return adv.detach().cpu().numpy().ravel()

# ────────────────────────────────────────────────────────────
# 5. ATTACKER  (min-max game: inner maximisation)
# ────────────────────────────────────────────────────────────
# Two-player zero-sum game:
#   • Attacker (max-player): chooses δ inside ε-ball to maximise loss
#   • Defender (min-player): chooses θ to minimise worst-case loss
# The trainer below is the Defender; the Attacker class here is used
# at evaluation time to approximate max_δ L(f_θ(x+δ), y).

class Attacker:
    """Approximates the inner maximisation δ* = argmax_δ L(f_θ(x+δ), y)."""
    def __init__(self, eps=0.10, steps=40):
        self.eps = eps
        self.steps = steps
    def attack(self, net, x_flat, y_true):
        return pgd(net, x_flat, y_true, self.eps, self.steps)

# ────────────────────────────────────────────────────────────
# 6. DEEPPOLY VERIFIER
# ────────────────────────────────────────────────────────────

def eval_concrete(lbc, ubc, lbb, ubb, il, iu):
    l = lbb + np.where(lbc >= 0, lbc * il, lbc * iu).sum(1)
    u = ubb + np.where(ubc >= 0, ubc * iu, ubc * il).sum(1)
    return l, u

def dp_conv2d(prev, kernel, bias, il, iu, c_in, h_in, w_in, stride=1, pad=0):
    lbc, ubc, lbb, ubb = prev[:4]
    c_out, _, k_h, k_w = kernel.shape
    h_out = (h_in + 2*pad - k_h) // stride + 1
    w_out = (w_in + 2*pad - k_w) // stride + 1
    n_inp = lbc.shape[1]

    L = torch.from_numpy(lbc.reshape(c_in, h_in, w_in, n_inp).transpose(3, 0, 1, 2))
    U = torch.from_numpy(ubc.reshape(c_in, h_in, w_in, n_inp).transpose(3, 0, 1, 2))
    Kp = torch.from_numpy(np.maximum(kernel, 0))
    Kn = torch.from_numpy(np.minimum(kernel, 0))

    Lbb = torch.from_numpy(lbb.reshape(1, c_in, h_in, w_in))
    Ubb = torch.from_numpy(ubb.reshape(1, c_in, h_in, w_in))

    nlbc_t = (torch.nn.functional.conv2d(L, Kp, padding=pad, stride=stride) +
              torch.nn.functional.conv2d(U, Kn, padding=pad, stride=stride))
    nubc_t = (torch.nn.functional.conv2d(U, Kp, padding=pad, stride=stride) +
              torch.nn.functional.conv2d(L, Kn, padding=pad, stride=stride))
    nlbb_t = (torch.nn.functional.conv2d(Lbb, Kp, padding=pad, stride=stride) +
              torch.nn.functional.conv2d(Ubb, Kn, padding=pad, stride=stride))
    nubb_t = (torch.nn.functional.conv2d(Ubb, Kp, padding=pad, stride=stride) +
              torch.nn.functional.conv2d(Lbb, Kn, padding=pad, stride=stride))

    nlbc = nlbc_t.permute(1, 2, 3, 0).reshape(c_out * h_out * w_out, -1).numpy()
    nubc = nubc_t.permute(1, 2, 3, 0).reshape(c_out * h_out * w_out, -1).numpy()
    nlbb = (nlbb_t[0].numpy() + bias.reshape(-1, 1, 1)).ravel()
    nubb = (nubb_t[0].numpy() + bias.reshape(-1, 1, 1)).ravel()

    l, u = eval_concrete(nlbc, nubc, nlbb, nubb, il, iu)
    return nlbc, nubc, nlbb, nubb, l, u

def dp_affine(prev, W, b, il, iu):
    lbc, ubc, lbb, ubb = prev[:4]
    Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
    nlbc = Wp.T @ lbc + Wn.T @ ubc
    nubc = Wp.T @ ubc + Wn.T @ lbc
    nlbb = Wp.T @ lbb + Wn.T @ ubb + b
    nubb = Wp.T @ ubb + Wn.T @ lbb + b
    l, u = eval_concrete(nlbc, nubc, nlbb, nubb, il, iu)
    return nlbc, nubc, nlbb, nubb, l, u

def dp_relu(prev, il, iu):
    lbc, ubc, lbb, ubb, pl, pu = prev
    n, m = lbc.shape
    nlbc = np.zeros_like(lbc);  nubc = np.zeros_like(ubc)
    nlbb = np.zeros(n, np.float32); nubb = np.zeros(n, np.float32)
    nl   = np.zeros(n, np.float32); nu   = np.zeros(n, np.float32)
    for i in range(n):
        li, ui = pl[i], pu[i]
        if ui <= 0:
            pass
        elif li >= 0:
            nlbc[i] = lbc[i]; nubc[i] = ubc[i]
            nlbb[i] = lbb[i]; nubb[i] = ubb[i]
            nl[i] = li; nu[i] = ui
        else:
            slope = ui / (ui - li)
            nubc[i] = slope * ubc[i]
            nubb[i] = slope * ubb[i] + (-ui*li/(ui-li))
            nu[i] = ui
            if ui > -li:
                nlbc[i] = lbc[i]; nlbb[i] = lbb[i]
            nl[i] = 0.0
    return nlbc, nubc, nlbb, nubb, nl, nu

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def deeppoly_verify(net, x_flat, true_label, eps):
    """Returns (is_robust, margin). margin > 0 → no attack inside ε-ball can change prediction."""
    il = np.clip(x_flat - eps, 0.0, 1.0).astype(np.float32)
    iu = np.clip(x_flat + eps, 0.0, 1.0).astype(np.float32)
    w = net.get_weights()

    prev = (np.eye(784, dtype=np.float32), np.eye(784, dtype=np.float32),
            np.zeros(784, np.float32), np.zeros(784, np.float32),
            il.copy(), iu.copy())

    curr = dp_conv2d(prev, w['conv1.k'], w['conv1.b'], il, iu, 1, 28, 28, 2, 1)
    curr = dp_relu(curr, il, iu)
    curr = dp_conv2d(curr, w['conv2.k'], w['conv2.b'], il, iu, 16, 14, 14, 2, 1)
    curr = dp_relu(curr, il, iu)
    curr = dp_affine(curr, w['fc1.W'], w['fc1.b'], il, iu)
    curr = dp_relu(curr, il, iu)
    curr = dp_affine(curr, w['fc2.W'], w['fc2.b'], il, iu)

    l_out, u_out = sigmoid(curr[4]), sigmoid(curr[5])
    margin = min(l_out[true_label] - u_out[j] for j in range(10) if j != true_label)
    return margin > 0, float(margin)

# ────────────────────────────────────────────────────────────
# 7. DEFENDER  (min-max game: outer minimisation)
# ────────────────────────────────────────────────────────────
# The Defender trains θ to solve the outer minimisation:
#     θ* = argmin_θ  max_δ  E[ L(f_θ(x + δ), y) ]
# The inner maximisation (δ) is handled by PGD-7 per batch during
# training, and evaluated with a stronger PGD-40 at test time.

class AdversarialTraining:
    """Single-pass PGD adversarial training with DeepPoly certification."""
    def __init__(self, n_train=5000, n_test=200, eps=0.10):
        self.n_test = n_test
        self.eps = eps

        rng = np.random.default_rng(42)
        n_per = n_train // 10
        idx = np.concatenate([rng.choice(np.where(y_train_full == c)[0], n_per, replace=False)
                              for c in range(10)])
        self.X_train = X_train_full[idx].reshape(-1, 1, 28, 28)
        self.y_train = y_train_full[idx]

        self.X_test = X_test_imgs[:n_test].reshape(-1, 1, 28, 28)
        self.y_test = y_test[:n_test]

        self.attacker = Attacker(eps=eps)
        self.epoch_history = []
        self.result = None

        print(f"Train: {len(self.X_train)}, Test: {self.n_test}")

    def _evaluate(self, net):
        t0 = time.time()
        t_dp = 0.0
        details = []
        for i in range(self.n_test):
            x_img = self.X_test[i]
            x_flat = x_img.ravel()
            y = int(self.y_test[i])
            x_t = torch.tensor(x_img[None], device=DEVICE)
            pc = int(net.predict(x_t).item())

            if pc == y:
                xadv_flat = self.attacker.attack(net, x_flat, y)
                xadv_t = torch.tensor(xadv_flat.reshape(1, 1, 28, 28), device=DEVICE)
                pa = int(net.predict(xadv_t).item())
                attacked = (pa != y)
            else:
                attacked = False

            t1 = time.time()
            robust, margin = deeppoly_verify(net, x_flat, y, self.eps)
            t_dp += time.time() - t1
            details.append(dict(idx=i, true=y, pc=pc, attacked=attacked, robust=robust))

        n_correct = sum(d['pc'] == d['true'] for d in details)
        n_attacked = sum(d['attacked'] for d in details)
        n_robust = sum(d['robust'] for d in details)
        t = time.time() - t0

        return {
            'accuracy': 100.0 * n_correct / self.n_test,
            'attack_success': 100.0 * n_attacked / max(n_correct, 1),
            'robustness': 100.0 * n_robust / self.n_test,
            'n_correct': n_correct, 'n_attacked': n_attacked, 'n_robust': n_robust,
            'details': details, 'time': t, 'time_dp': t_dp,
        }

    def _fast_evaluate(self, net):
        n_attacked = 0
        n_correct = 0
        for i in range(min(self.n_test, 50)):
            x_img = self.X_test[i]
            x_flat = x_img.ravel()
            y = int(self.y_test[i])
            x_t = torch.tensor(x_img[None], device=DEVICE)
            pc = int(net.predict(x_t).item())
            if pc == y:
                n_correct += 1
                xadv_flat = self.attacker.attack(net, x_flat, y)
                xadv_t = torch.tensor(xadv_flat.reshape(1, 1, 28, 28), device=DEVICE)
                pa = int(net.predict(xadv_t).item())
                if pa != y:
                    n_attacked += 1
        return n_correct, n_attacked

    def run(self):
        print(f"CNN: Conv(1→16,4,s2,p1)→ReLU→Conv(16→32,4,s2,p1)→ReLU"
              f"→Flatten→Dense(1568→64)→ReLU→Dense(64→10)")
        print(f"Attack: PGD-40 (eval), PGD-7 (train), ε={self.eps:.2f}")
        print(f"Verify: DeepPoly (ε={self.eps:.2f})")
        print(f"Train: {len(self.X_train)}  |  Test: {self.n_test}")

        net = CNN().to(DEVICE)

        def cb(ep):
            nc, na = self._fast_evaluate(net)
            self.epoch_history.append((ep, nc, na))

        print(f"\n--- TRAINING ---")
        train(net, self.X_train, self.y_train, self.X_test, self.y_test,
              adv_eps=self.eps, epoch_callback=cb)

        print(f"\n--- EVALUATION (PGD-40 + DeepPoly) ---")
        self.result = self._evaluate(net)
        r = self.result

        print(f"\n  Accuracy         : {r['accuracy']:.1f}%  ({r['n_correct']}/{self.n_test})")
        print(f"  Attack success   : {r['attack_success']:.1f}%  ({r['n_attacked']}/{r['n_correct']})")
        print(f"  Certified robust : {r['robustness']:.1f}%  ({r['n_robust']})")
        print(f"  PGD-40 time      : {r['time'] - r['time_dp']:.1f}s")
        print(f"  DeepPoly time    : {r['time_dp']:.1f}s")

        self._plot()

    def _plot(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        fig.suptitle("PGD Adversarial Training — CNN on MNIST", fontsize=13, y=1.02)

        ax = axes[0]
        if self.epoch_history:
            eps_epoch = [e[0] for e in self.epoch_history]
            atk_pct = [100.0 * e[2] / max(e[1], 1) for e in self.epoch_history]
            ax.plot(eps_epoch, atk_pct, '-', color='crimson', lw=2, label='Attack success')
            ax.axhline(y=self.result['robustness'], color='forestgreen', ls='--', lw=1.5,
                       label='Certified robust')
            ax.annotate(f'{self.result["robustness"]:.0f}%',
                        (eps_epoch[-1], self.result['robustness']),
                        textcoords='offset points', xytext=(8, 8), fontsize=9,
                        color='forestgreen', fontweight='bold')
        ax.set(xlabel='Epoch', ylabel='%', title='Attack Success over Training')
        ax.legend(fontsize=9)
        ax.set_ylim(0, 100)

        ax = axes[1]
        r = self.result
        nW = self.n_test - r['n_correct']
        nU = r['n_correct'] - r['n_robust']
        labels = ['Correct', 'Wrong', 'Certified', 'Uncertified']
        vals   = [r['n_correct'], nW, r['n_robust'], max(nU, 0)]
        colors = ['#1565c0', '#546e7a', '#2e7d32', '#e65100']
        bars = ax.bar(labels, vals, color=colors)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(val), ha='center', fontsize=10)
        ax.set(ylabel='Images', title=f'Accuracy vs Certification (attack ε={self.eps:.2f})')
        ax.set_ylim(0, self.n_test * 1.15)

        plt.tight_layout()
        plt.savefig("training_results.png", dpi=150, bbox_inches='tight')
        print("  [Saved → training_results.png]")
        plt.close('all')

if __name__ == '__main__':
    n_tr = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    n_te = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    eps  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
    AdversarialTraining(n_train=n_tr, n_test=n_te, eps=eps).run()
