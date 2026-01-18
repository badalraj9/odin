"""
ODIN Configuration
------------------
All hyperparameters for the model.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import math


@dataclass
class ODINConfig:
    """
    Configuration for ODIN Selective SSM model.
    
    Supports multiple scales:
    - Tiny (~1M): For testing architecture
    - Small (~10M): For validating learning
    - Medium (~85M): Production baseline
    - Large (~1B): Full scale
    """
    
    # ===== Model Scale Presets =====
    # Override these for different scales
    
    # State dimensions (total = sum of timescales)
    state_dim: int = 256  # Tiny default
    timescales: List[int] = field(default_factory=lambda: [64, 64, 64, 64])
    
    # Timescale decay rates (tau values in steps)
    timescale_taus: List[float] = field(default_factory=lambda: [2.0, 10.0, 50.0, 200.0])
    
    # Low-rank dynamics
    rank: int = 8  # Tiny default
    
    # Layers
    num_layers: int = 4  # Tiny default
    
    # MLP hidden dimensions
    hidden_dim: int = 64  # Tiny default
    
    # ===== Input/Output Dimensions (Fixed) =====
    input_dim: int = 1024   # From encoders: 384+256+192+192
    output_dim: int = 768   # cognitive (256) + action (384) + memory (128)
    
    # Action space
    num_actions: int = 10
    num_stances: int = 4    # EXPLORE, PLAN, EXECUTE, REFLECT
    
    # ===== Stability Parameters =====
    stability_margin: float = 0.01   # λ_margin for eigenvalue constraint
    max_noise_std: float = 0.1       # Bounded noise injection
    
    # ===== Training Parameters =====
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_epochs: int = 30
    batch_size: int = 64
    gradient_clip: float = 1.0
    
    # Discretization timestep
    dt: float = 0.01
    
    # FP16 training
    use_fp16: bool = True
    
    # ===== Checkpoint =====
    checkpoint_every: int = 1  # Save every epoch
    
    def __post_init__(self):
        """Validate and compute derived values."""
        # Ensure timescales sum to state_dim
        assert sum(self.timescales) == self.state_dim, \
            f"Timescales {self.timescales} must sum to state_dim {self.state_dim}"
        
        # Ensure we have tau for each timescale
        assert len(self.timescale_taus) == len(self.timescales), \
            "Must have one tau per timescale block"
    
    @property
    def num_timescales(self) -> int:
        return len(self.timescales)
    
    def param_count_estimate(self) -> int:
        """Estimate total parameter count."""
        d = self.state_dim
        r = self.rank
        h = self.hidden_dim
        L = self.num_layers
        u = self.input_dim
        y = self.output_dim
        
        # Per layer
        per_layer = (
            d +                    # D_base (diagonal)
            2 * d * r +           # U_base, V_base (low-rank)
            2 * (h * (d + u + d) + h) +  # MLP_D, MLP_U first layer
            d + d * r +           # MLP output layers
            (d + u + d) * d + d   # Selection network
        )
        
        # Heads
        heads = (
            h * d + h + 256 +     # Cognitive head
            h * d + h + 384 +     # Action head
            h * d + h + 128       # Memory head
        )
        
        total = per_layer * L + heads
        return int(total)
    
    @classmethod
    def tiny(cls) -> 'ODINConfig':
        """~1.7M params, trains in 10-15 mins on Colab T4."""
        return cls(
            state_dim=256,
            timescales=[64, 64, 64, 64],
            timescale_taus=[2.0, 10.0, 50.0, 200.0],
            rank=8,
            num_layers=4,
            hidden_dim=64,
            batch_size=64,
            max_epochs=20,
        )
    
    @classmethod
    def mini(cls) -> 'ODINConfig':
        """~5M params, trains in 25-35 mins on Colab T4."""
        return cls(
            state_dim=384,
            timescales=[96, 96, 96, 96],
            timescale_taus=[2.0, 12.0, 75.0, 350.0],
            rank=12,
            num_layers=5,
            hidden_dim=96,
            batch_size=48,
            max_epochs=25,
        )
    
    @classmethod
    def small(cls) -> 'ODINConfig':
        """~5M params, trains in 1-2 hours on Colab T4."""
        return cls(
            state_dim=512,
            timescales=[128, 128, 128, 128],
            timescale_taus=[2.0, 15.0, 100.0, 500.0],
            rank=16,
            num_layers=6,
            hidden_dim=128,
            batch_size=32,
            max_epochs=30,
        )
    
    @classmethod
    def small_10m(cls) -> 'ODINConfig':
        """~9M params - matches the checkpoint from 2026-01-17 training."""
        return cls(
            state_dim=640,
            timescales=[160, 160, 160, 160],
            timescale_taus=[2.0, 15.0, 100.0, 500.0],
            rank=24,
            num_layers=8,
            hidden_dim=192,
            batch_size=32,
            max_epochs=35,
            learning_rate=5e-5,
        )
    
    @classmethod
    def medium(cls) -> 'ODINConfig':
        """~85M params, trains in 4-6 hours on Colab V100."""
        return cls(
            state_dim=2048,
            timescales=[512, 512, 512, 512],
            timescale_taus=[3.0, 30.0, 300.0, 1000.0],
            rank=32,
            num_layers=12,
            hidden_dim=256,
            batch_size=32,
            max_epochs=30,
        )
    
    @classmethod
    def production(cls) -> 'ODINConfig':
        """~30M params - the SWEET SPOT. 2 hours on Colab T4, runs on RTX 3050."""
        return cls(
            state_dim=768,                              # Larger state
            timescales=[192, 192, 192, 192],
            timescale_taus=[2.0, 15.0, 100.0, 500.0],
            rank=32,                                    # Higher rank
            num_layers=12,                              # More layers
            hidden_dim=256,                             # Larger hidden
            batch_size=24,                              # Smaller batch for memory
            max_epochs=35,
            learning_rate=3e-5,                         # Lower LR for larger model
        )
    
    @classmethod
    def large(cls) -> 'ODINConfig':
        """~1B params, needs A100 or better."""
        return cls(
            state_dim=8192,
            timescales=[1024, 2048, 2048, 2048],
            timescale_taus=[3.0, 30.0, 300.0, 3000.0],
            rank=128,
            num_layers=24,
            hidden_dim=512,
            batch_size=16,
            max_epochs=30,
        )
