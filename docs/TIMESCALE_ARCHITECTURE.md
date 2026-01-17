# ODIN: Timescale Stratification Architecture
## Multi-Temporal Dynamics for Long-Horizon Intelligence

---

## The Core Insight

> **Intelligence scales with the number of independent time constants a system can coordinate.**

Not with:
- Parameter count alone
- Rank alone
- Raw dimension alone

ODIN's "window" is controlled by **how long information persists and interacts**, not token count.

---

## 1. State Decomposition by Timescale

Instead of one homogeneous state, decompose into temporal blocks:

```
h_t = [h_fast | h_mid | h_slow | h_ultra]
```

Each block has its own dynamics:

```
h^(k)_{t+1} = Φ^(k)_t h^(k)_t + Γ^(k)_t u_t
```

Where:
```
Φ^(k) ≈ exp(-Δ / τ_k)
τ_fast << τ_mid << τ_slow << τ_ultra
```

---

## 2. Timescale Configuration (8192-dim state)

| Block | Dim | τ (steps) | What It Captures |
|-------|-----|-----------|------------------|
| h_fast | 1024 | 1-3 | Emotion, urgency, interruptions |
| h_mid | 2048 | 10-30 | Project context, ongoing plans |
| h_slow | 2048 | 100-300 | Goals, preferences |
| h_ultra | 2048 | 1000+ | "Who you are to me" |

**Total:** 8192-dim state with multi-timescale coverage.

---

## 3. Rank Per Block

| Block | Rank | Rationale |
|-------|------|-----------|
| fast | 32-64 | Quick reactions, low complexity |
| mid | 64-128 | Project reasoning |
| slow | 128-256 | Goal integration |
| ultra | 256-512 | Deep patterns (tightly gated) |

**Instantaneous rank stays controlled, but total expressive capacity explodes over time.**

---

## 4. Cross-Timescale Coupling (Controlled)

Allow limited interaction from fast → slow (not reverse):

```
h^(k)_{t+1} += Σ_{j<k} W_{j→k} · h^(j)_t
```

Constraints:
- Only slower ← faster (no feedback loops)
- Low-rank couplings
- Gated by z_t

This gives **combinatorial expressivity over time**, not instant chaos.

---

## 5. Mathematical Formulation

### Per-Block Dynamics
```
D^(k)_t = -softplus(D^(k)_base / τ_k + ΔD^(k)) - ε
U^(k)_t = (U^(k)_base + ΔU^(k)) ⊙ z_t
A^(k)_t = diag(D^(k)_t) + U^(k)_t (V^(k))^T
```

### Cross-Timescale Coupling
```
C_{j→k} = W_{j→k} ⊙ gate_{j→k}(z_t)
h^(k)_coupled = h^(k) + Σ_{j<k} C_{j→k} h^(j)
```

### Final State
```
h_t = concat([h^(1), h^(2), h^(3), h^(4)])
```

---

## 6. What This Enables

| Capability | Single-Timescale | Multi-Timescale |
|------------|------------------|-----------------|
| React to interruption | ✅ | ✅ |
| Remember 5 steps ago | ✅ | ✅ |
| Remember 100 steps ago | ⚠️ Decayed | ✅ In h_slow |
| Remember 1000 steps ago | ❌ Gone | ✅ In h_ultra |
| Combine old + new context | ⚠️ Limited | ✅ Cross-coupling |
| "Know who you are" | ❌ | ✅ |

---

## 7. Parameter Allocation for 1B ODIN

| Component | Params | Notes |
|-----------|--------|-------|
| D^(k)_base (4 blocks) | 32K | Diagonal dynamics |
| U^(k), V^(k) (4 blocks) | 16M | Low-rank factors |
| Selection networks | 200M | Per-block selection |
| Cross-timescale gates | 100M | Controlled coupling |
| MLP_D, MLP_U (shared) | 400M | Delta networks |
| Output heads | 100M | Cognitive, action, memory |
| **TOTAL** | **~800M-1B** | |

---

## 8. Stability Across Timescales

### Each block maintains its own stability
```
Re(λ^(k)) < 0  ∀k
```

### Cross-coupling is bounded
```
||W_{j→k}|| < stability_margin
```

### Ultra-long block is heavily gated
```
gate_ultra = sigmoid(...) with bias toward 0
```

Only strong signals update ultra-long memory.

---

## 9. The JARVIS Effect

This architecture gives:

| JARVIS Trait | How Timescales Enable It |
|--------------|--------------------------|
| Instant reactions | h_fast responds in 1-3 steps |
| Project awareness | h_mid tracks ongoing work |
| Knows your preferences | h_slow learns over 100+ interactions |
| Feels like same entity | h_ultra persists identity |
| Smooth pivots | Cross-coupling allows gradual shifts |

---

## 10. Summary

```
Intelligence = timescale diversity × controlled coupling
```

Not:
```
Intelligence = more parameters × bigger rank
```

This is how ODIN achieves "large window = better intelligence" without instability.
