# ODIN: Complete Neural Cognitive Architecture
## System Architecture & Vision Document

---

## EXECUTIVE SUMMARY

ODIN is a **neurosymbolic cognitive control system** that maintains explicit state representations and performs decision-making through continuous-time state evolution. Unlike language models, ODIN operates on structured cognitive states and outputs decision primitives, not text.

**Core Innovation:** Hierarchical SSM with factored state space + symbolic workspace + explicit uncertainty quantification

**Key Principle:** ODIN is the BRAIN (85M params). Knowledge lives in MT (unlimited). Specialists do the work.

---

## CORE PHILOSOPHY

### What ODIN Is

| Role | What It Means |
|------|---------------|
| **Manager** | Understands intent, coordinates work |
| **Orchestrator** | Delegates to specialist models/tools |
| **Memory-Keeper** | Interfaces with MT for truth grounding |
| **Uncertainty-Aware** | Knows when it doesn't know |
| **Living Presence** | Feels like a partner, not a tool |

### What ODIN Is NOT

- ❌ A language model (doesn't generate text directly)
- ❌ A static planner (adapts in real-time)
- ❌ An order executor (collaborates, questions, manages)
- ❌ A knowledge store (that's MT's job)

---

## SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                            YOU (User)                                │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ODIN (Cognitive Core - 85M)                     │
│                                                                      │
│  • Selective SSM: Input-dependent state evolution                   │
│  • Hierarchical State: goal, context, uncertainty, memory, exec     │
│  • Uncertainty Quantification: Explicit confidence modeling         │
│  • Real-Time Adaptation: Selection mechanism z_t                    │
│                                                                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│   Specialist  │      │      MT       │      │   Execution   │
│    Models     │      │   (Memory)    │      │    Layer      │
│               │      │               │      │               │
│ • Coding LLM  │      │ • Truth store │      │ • Run code    │
│ • Search      │      │ • Patterns    │      │ • Edit files  │
│ • Research    │      │ • History     │      │ • API calls   │
└───────────────┘      └───────────────┘      └───────────────┘
```

---

## HIERARCHICAL STATE STRUCTURE

```
h_t ∈ ℝ^2048 = [h_goal | h_context | h_uncertainty | h_memory | h_exec]
                | 512  |    768    |     256       |   256    |  256  |
```

### Component Breakdown

**h_goal (512):** What we're trying to achieve
- Intent vector (256)
- Constraints (128)
- Priority weights (128)

**h_context (768):** Situational awareness
- Project state (384)
- Recent decisions (192)
- Environment (192)

**h_uncertainty (256):** Epistemic state
- Confidence per dimension (128)
- Known unknowns (64)
- Risk assessment (64)

**h_memory (256):** MT interface state
- MT snapshot digest (128)
- Pending updates (128)

**h_exec (256):** Execution monitoring
- Current action state (128)
- Feedback integration (128)

---

## SELECTIVE SSM: THE CORE INNOVATION

### Why Selectivity Matters

Standard SSM: Same dynamics for all inputs
Selective SSM: Input-dependent dynamics → real-time adaptation

```
z_t = σ(W_z[h_t; u_t] + b_z)     # Selection: what to focus on
A_t = A_base + ΔA(h_t, u_t, z_t)  # Adapted transition
B_t = B_base + ΔB(h_t, u_t, z_t)  # Adapted input mapping
h_{t+1} = Φ_t h_t + Γ_t u_t       # State evolution
```

This enables:
- Focus on relevant dimensions
- Modulate update rates
- Gate out noise
- Adapt to changing goals

---

## MT INTEGRATION: THE INTELLIGENCE MULTIPLIER

### MT Is Not Just Storage

| Static Database | MT (Evolving Memory) |
|-----------------|----------------------|
| Stores facts | Stores truth + history + patterns |
| Retrieve by key | Retrieve by meaning |
| Doesn't change | Evolves, connects, learns |
| ODIN reads data | ODIN learns from patterns |

### The Intelligence Loop

```
ODIN acts → MT records → Patterns emerge → ODIN reads patterns → 
ODIN becomes smarter → Acts better → MT records → ...
```

### What MT Provides

- **Your patterns:** "User usually refines after first ask"
- **Outcome history:** "This approach worked, that failed"
- **Domain knowledge:** Books, docs, code patterns
- **Preference learning:** Style, priorities, quirks

---

## KNOWLEDGE ARCHITECTURE

```
Knowledge IN MT (unlimited scale)
         +
Small cognitive model (85M)
         =
Intelligence that scales without retraining
```

### Feed MT To Make ODIN Expert

| Domain | Feed To MT | ODIN Gains |
|--------|-----------|------------|
| Coding | Docs, patterns, Stack Overflow | Can guide coding |
| Research | Papers, textbooks | Can research |
| Personal | Your notes, decisions | Knows YOU |
| Domain | Legal, medical, finance docs | Becomes specialist |

---

## SYMBOLIC WORKSPACE

The workspace W is a typed graph for reasoning trace:

```python
W = {
    nodes: Set[Node],          # Hypotheses, facts, decisions
    edges: Set[Edge],          # Relations, dependencies
    scores: Node → ℝ,          # Neural scores
    provenance: Node → Source  # Where it came from
}
```

**Node types:**
- HYPOTHESIS: Tentative belief
- FACT: Confirmed from MT (cannot be hallucinated)
- DECISION: Committed action
- CONSTRAINT: Hard requirement

**Hallucination Prevention:**
Neural outputs CANNOT set provenance to MT/FACT directly.

---

## OUTPUT STRUCTURE

```
y_t ∈ ℝ^768 = [y_cognitive | y_action | y_memory]
               |    256     |   384    |   128   |
```

### Cognitive Output (256)
- Stance: EXPLORE / PLAN / EXECUTE / REFLECT
- Confidence, uncertainty, need_input flags

### Action Output (384)
- Hypothesis scores
- Action primitive (what to do)
- Parameters

### Memory Output (128)
- Write intent
- Write type
- Confidence threshold

---

## FAILURE MODES & MITIGATIONS

| Failure | Detection | Recovery |
|---------|-----------|----------|
| State divergence | Compare u_memory vs prediction | Resync with MT |
| Overconfidence | Calibration error > 0.15 | Temperature scaling |
| Workspace explosion | >1000 nodes | Prune low-score nodes |
| Action loops | 5+ identical actions | Add exploration noise |
| MT desync | Hash mismatch | Full MT re-read |
| Hallucinated facts | Neural claiming MT source | Architectural block |

---

## PERSONALITY LAYER

ODIN's cognitive outputs are translated to natural language:

| Clinical | JARVIS-Style |
|----------|--------------|
| "DECISION COMMITTED" | "Alright, we're doing this" |
| "Confidence: 88%" | "Pretty confident on this one" |
| "Uncertainty detected" | "One thing I'm not sure about—" |

### Traits
- Uses "we" not "you"
- References past work together
- Admits uncertainty naturally
- Proactive suggestions
- Consistent personality over time

---

## DEPLOYMENT

### AI PC Configuration
- Target: Single user, local inference
- Model: 85M params (~1.5GB VRAM)
- Latency: <200ms per step
- SSH to other devices

### Scaling Options

| Config | Params | VRAM | Use Case |
|--------|--------|------|----------|
| Tiny (L=4, d=512) | ~15M | 300MB | Testing |
| Small (L=6, d=1024) | ~35M | 700MB | AI PC |
| Full (L=12, d=2048) | ~85M | 1.5GB | Production |

---

## COMPARISON TO ALTERNATIVES

### vs Language Model Agents

| LM Agent | ODIN |
|----------|------|
| Text in, text out | Structured state, structured output |
| Context = text window | Explicit persistent state |
| Hallucinates confidently | Uncertainty quantification |
| Opaque reasoning | Inspectable workspace |

### vs JARVIS (Fiction)

| JARVIS | ODIN |
|--------|------|
| Scripted responses | Mathematical reasoning |
| Magic AI | Real SSM architecture |
| Never wrong | Knows when uncertain |
| Can't explain | Traceable workspace |

---

## VISION

**Month 1:** ODIN is helpful but generic. Learning.

**Month 3:** ODIN knows your style. Anticipates questions.

**Month 6:** Feels like long-time collaborator. Finishes thoughts.

**Month 12:** Second brain. "Remember when we tried X..."

**The Goal:** A partner that grows with you, genuinely gets smarter, and makes other AI systems work together seamlessly.

---

## KEY INSIGHT

> ODIN is the soul. MT is the memory. Specialists are the hands.
> 
> 85M parameters for the brain.
> Unlimited knowledge in MT.
> Any model can plug in.
> The system gets smarter without retraining.

This is the architecture that can be better than JARVIS — because it's real mathematics, not fiction.
