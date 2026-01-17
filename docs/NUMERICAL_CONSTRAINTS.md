# ODIN: Numerical & Stability Constraints
## Non-Negotiable Requirements for a Stable Dynamical System

---

> **ODIN is not "just another neural net".**
> **It is a dynamical system with learning layered on top.**

These constraints are **existence conditions**, not optimizations.

---

## 1️⃣ Eigenvalue Stability

### Requirement
```
Re(λᵢ(A_t)) < 0  ∀i
```

### Why It Exists
Core equation: dh/dt = A_t h + B_t u

Solution: h(t) = exp(A_t · t) h(0)

If ANY eigenvalue has positive real part → state grows exponentially.

### What Breaks
- State norm explodes after 5-10 steps
- LayerNorm temporarily hides it
- Uncertainty calibration collapses
- Gradients vanish or explode

### Enforcement (ODIN)
```
D_t = -softplus(D_base + ΔD) - ε
A_t = diag(D_t) + U V^T
```

With spectral norm constraint on UV^T:
```
||U_t V^T|| < |min(D_t)|
```

---

## 2️⃣ Matrix Exponential Tractability

### Requirement
Only parameterize A_t such that exp(A_t Δ) is:
- Cheap to compute: O(d·r), not O(d³)
- Well-conditioned numerically

### Why It Exists
Dense expm with d=2048: **impossible** (8.6B ops per layer per step)

Even if computed:
- Numerical error grows with matrix norm
- Backprop through expm is unstable
- Floating-point errors accumulate

### Enforcement (ODIN)
Use diagonal + low-rank:
```
A_t = diag(D) + U V^T
Φ_t ≈ I + A_t Δ + (A_t Δ)²/2
```

All terms are O(d) or O(d·r²), not O(d³).

---

## 3️⃣ Contraction Margin

### Requirement
```
|Φ_t| ≤ 1 - δ,  δ > 0
```

### Why It Exists
With recurrence: h_{t+1} = Φ h_t + ε

Even tiny bias accumulates:
```
E[|h_t|] ≤ Σ |Φ|^k |ε_{t-k}|
```

If |Φ| ≈ 0.99: convergence very slow, accumulated bias large.

### What Breaks
State drifts from ground truth over many steps.

### Enforcement (ODIN)
- λ_margin = -0.01 (strict negative diagonal)
- softplus ensures gap from zero
- Residual scaling bounded

---

## 4️⃣ Gradient Flow (Critical Damping)

### Requirement
```
|Φ_t| ≈ 1  AND  Re(λ) < 0
```

This is a narrow band.

### Why It Exists
BPTT gradient contains:
```
∂h_T/∂h_0 = Π Φ_t
```

- |Φ| < 1 → vanishing gradients
- |Φ| > 1 → exploding gradients

### What Breaks
- Vanishing: early layers don't learn
- Exploding: NaN training

### Enforcement (ODIN)
- Diagonal dominance
- Residual identity paths
- LayerNorm
- Bounded low-rank corrections
- Test: d_eff ∈ [0.5, 2.0]

---

## 5️⃣ Bounded Noise Injection

### Requirement
```
Q_t = σ(h_t, u_t)² I
σ ≤ σ_max
```

Noise must be:
1. State-dependent
2. Bounded
3. Decay when confident

### Why It Exists
Noise in recurrent systems accumulates.
Variance can diverge.

### What Breaks
- ODIN becomes permanently uncertain
- Or falsely confident (noise masked)
- Uncertainty saturates

### Enforcement (ODIN)
```python
noise_std = sigmoid(noise_head(h)) * max_noise_std
w_t = randn() * noise_std
```

---

## 6️⃣ Residual Connections (Mandatory)

### Requirement
```
h_{t+1} = h_t + f(h_t)   # NOT just f(h_t)
```

### Why It Exists
Residuals:
- Preserve identity across layers
- Stabilize gradients
- Allow slow adaptation
- Enable skip connections for identity

In ODIN: residuals are **semantic continuity**.

### What Breaks
- Information lost between layers
- Catastrophic forgetting within episode
- Semantic jumps in reasoning

### Enforcement (ODIN)
```
h^(ℓ+1) = SSM(h^(ℓ), u) + h^(ℓ)  # Identity residual
h^(ℓ+1) = LayerNorm(h^(ℓ+1))
```

---

## 7️⃣ FP Precision Discipline

### Requirement
- Train: FP32 (or BF16)
- Inference: FP16 acceptable
- Critical scalars: FP32 if needed
- Never: INT8 dynamics, aggressive quantization

### Why It Exists
SSMs are sensitive to rounding.
Errors accumulate over time.

### What Breaks
Silent corruption. Hard to debug.

### Enforcement (ODIN)
```python
# Keep diagonal dynamics in FP32
self.D_base = nn.Parameter(torch.zeros(d, dtype=torch.float32))

# Mixed precision for bulk compute
with autocast():
    h_next = phi @ h + gamma @ u
```

---

## 8️⃣ Structure > Parameter Count

### Requirement
Expressivity from **structure**, not raw parameters.

85M well-structured > 1B poorly constrained.

### Why It Exists
- Low-rank enforces smoothness
- Gating enables selection
- Factorization enables specialization

### Enforcement (ODIN)
- Low-rank dynamics (r=32)
- Selection gating z_t
- Hierarchical state factorization

---

## 9️⃣ Summary: Constraint Checklist

| Constraint | Math | Test |
|------------|------|------|
| Eigenvalue stability | Re(λ) < 0 | Monte Carlo 10K samples |
| Tractable exp | O(d·r) | Latency < 10ms |
| Contraction margin | |Φ| < 1-δ | Check Φ norm << 1 |
| Gradient flow | d_eff ∈ [0.5, 2.0] | Gradient analysis |
| Bounded noise | σ ≤ σ_max | Variance monitoring |
| Residual paths | Identity + delta | Architecture audit |
| FP precision | FP32 critical | Numerical tests |
| Structured A_t | D + UV^T | Parameter count |

---

## Final Principle

> If you respect these constraints, ODIN will feel **alive but sane**.
> 
> If you ignore them, ODIN will feel **broken or erratic**.

These are not tuning tricks — they are **physics**.
