# ODIN Model

**ODIN: Complete Neural Cognitive Architecture**

A Selective State Space Model for cognitive reasoning with explicit uncertainty quantification.

## Documentation

- [Mathematical Specification](docs/MATHEMATICAL_SPECIFICATION.md) - All equations and hyperparameters
- [Architecture Vision](docs/ARCHITECTURE_VISION.md) - System design and philosophy
- [Development Plan](docs/DEVELOPMENT_PLAN.md) - 8-week implementation roadmap

## Quick Start

```python
from odin.core.model import ODINModel
from odin.core.config import ODINConfig

config = ODINConfig()
model = ODINModel(config)

# Single reasoning step
h_t = torch.randn(1, 2048) * 0.01
u_t = torch.randn(1, 1024)
h_next, y_t, diagnostics = model.step(h_t, u_t)

# y_t contains:
# - y_cognitive (256): stance, confidence, uncertainty
# - y_action (384): hypothesis scores, action primitive
# - y_memory (128): write intent, type, threshold
```

## Key Features

- **Selective SSM**: Input-dependent dynamics for real-time adaptation
- **Hierarchical State**: Goal, context, uncertainty, memory, execution
- **Explicit Uncertainty**: Knows when it doesn't know
- **MT Integration**: Grounds reasoning in truth via Memory Thread
- **85M Parameters**: Small model, knowledge in MT

## Architecture

```
ODIN (85M) ─── Cognitive Core (Selective SSM)
    │
    ├── MT ─── Evolving Memory (Knowledge)
    ├── Workspace ─── Reasoning Trace (Graph)
    └── Specialists ─── Execution (LLMs, Tools)
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU)

## Project Status

🚧 **Under Development** - Phase 0

## License

Proprietary - All rights reserved
