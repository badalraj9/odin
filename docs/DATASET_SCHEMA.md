# ODIN Training Dataset Schema
## Scenario-Based Cognitive Training Data

---

## Dataset Structure

```
data/
├── scenarios/
│   ├── planning/       # Stage 1: Simple decisions
│   ├── revision/       # Stage 2: Multi-step with pivots
│   └── adversarial/    # Stage 3: Ambiguity, errors
└── scripts/
    ├── schema.py       # Data classes
    ├── generator.py    # Scenario generation
    └── curriculum.py   # Difficulty scaling
```

---

## Scenario Format

Each scenario is a JSON file:

```json
{
  "id": "planning_0001",
  "type": "planning",
  "difficulty": 0.3,
  "steps": [
    {
      "t": 0,
      "input": {
        "intent": "Build a REST API for user management",
        "mt_hash": "a1b2c3...",
        "mt_delta": [],
        "feedback": null,
        "system": {"urgency": 0.5, "timestamp": 1704825600}
      },
      "expected_output": {
        "stance": "PLAN",
        "confidence": 0.7,
        "action": "DECOMPOSE_TASK",
        "uncertainty": ["tech_stack", "scope"]
      },
      "reward": 1.0
    },
    {
      "t": 1,
      "input": {...},
      "expected_output": {...},
      "reward": 1.0
    }
  ],
  "metadata": {
    "domain": "software",
    "complexity": "medium",
    "required_pivots": 0
  }
}
```

---

## Scenario Types

### 1. Planning (Stage 1)
- Single goal
- Clear path
- Reward: correct action selection

### 2. Revision (Stage 2)
- Goal changes mid-scenario
- Requires state update
- Reward: successful pivot + completion

### 3. Adversarial (Stage 3)
- Ambiguous intent
- Error conditions
- Missing information
- Reward: appropriate uncertainty + recovery

---

## Action Space (10 primitives)

| ID | Action | When To Use |
|----|--------|-------------|
| 0 | EXPLORE | Low confidence, need more info |
| 1 | PLAN | Ready to structure approach |
| 2 | DECOMPOSE_TASK | Break complex into steps |
| 3 | QUERY_USER | Need clarification |
| 4 | DELEGATE | Pass to specialist model |
| 5 | EXECUTE | High confidence, do it |
| 6 | COMMIT | Finalize decision |
| 7 | WRITE_MT | Persist to memory |
| 8 | BACKTRACK | Realize mistake, undo |
| 9 | WAIT | Need external result |

---

## Reward Structure

| Outcome | Reward |
|---------|--------|
| Correct action, high confidence | +1.0 |
| Correct action, low confidence | +0.7 |
| Wrong action, low confidence (honest) | -0.3 |
| Wrong action, high confidence (bad) | -1.0 |
| Successful pivot after revision | +0.5 |
| Appropriate uncertainty flag | +0.3 |
| Unnecessary query (wasted step) | -0.2 |
| Caught error before execution | +0.8 |

---

## Curriculum Scaling

```python
difficulty(episode_k) = min(1.0, k / 30000)

scenario_distribution(difficulty):
    if difficulty < 0.3:
        return (0.8, 0.1, 0.1)  # (planning, revision, adversarial)
    elif difficulty < 0.7:
        return (0.5, 0.3, 0.2)
    else:
        return (0.4, 0.3, 0.3)
```

---

## Input Dimensions

| Field | Dimension | Source |
|-------|-----------|--------|
| intent | 384 | DistilBERT encoding |
| mt_hash | 128 | SHA-256 truncated |
| mt_delta | 128 | Change embedding |
| feedback | 192 | Execution result |
| system | 192 | Metadata |
| **Total u_t** | **1024** | |

---

## Target Output Dimensions

| Field | Dimension |
|-------|-----------|
| stance (4 modes) | 4 (softmax) |
| confidence | 1 (scalar) |
| action (10 types) | 10 (softmax) |
| uncertainty_flags | 64 |
| **Total y_t** | **768** |
