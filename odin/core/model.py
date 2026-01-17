"""
ODIN Model
----------
Complete ODIN Selective SSM model with multi-timescale architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math

from .config import ODINConfig
from .ssm import TimescaleSSMBlock, SelectiveSSMLayer
from .heads import ODINOutputHeads


class CrossTimescaleCoupling(nn.Module):
    """
    Controlled coupling from faster timescales to slower ones.
    Only allows slower ← faster (no feedback loops).
    """
    
    def __init__(self, from_dim: int, to_dim: int, hidden_dim: int):
        super().__init__()
        
        self.gate = nn.Sequential(
            nn.Linear(from_dim + to_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        self.transform = nn.Sequential(
            nn.Linear(from_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, to_dim),
        )
        
        # Initialize to small values
        nn.init.zeros_(self.transform[-1].weight)
        nn.init.zeros_(self.transform[-1].bias)
    
    def forward(self, h_from: torch.Tensor, h_to: torch.Tensor) -> torch.Tensor:
        """
        Compute gated coupling contribution.
        
        Args:
            h_from: Source (faster) timescale state
            h_to: Target (slower) timescale state
            
        Returns:
            Contribution to add to h_to
        """
        gate = self.gate(torch.cat([h_from, h_to], dim=-1))
        contribution = self.transform(h_from)
        return gate * contribution


class ODINModel(nn.Module):
    """
    Complete ODIN model with multi-timescale SSM architecture.
    
    Architecture:
    - Input projection
    - 4 timescale blocks (fast/mid/slow/ultra)
    - Cross-timescale coupling
    - Output heads (cognitive/action/memory)
    """
    
    def __init__(self, config: ODINConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.state_dim)
        
        # Timescale SSM blocks
        layers_per_block = max(1, config.num_layers // config.num_timescales)
        
        self.timescale_blocks = nn.ModuleList()
        for i, (block_dim, tau) in enumerate(zip(config.timescales, config.timescale_taus)):
            block = TimescaleSSMBlock(
                state_dim=block_dim,
                input_dim=config.input_dim,
                rank=config.rank,
                hidden_dim=config.hidden_dim,
                tau=tau,
                num_layers=layers_per_block,
                dt=config.dt,
                stability_margin=config.stability_margin,
            )
            self.timescale_blocks.append(block)
        
        # Cross-timescale coupling (slower ← faster)
        self.couplings = nn.ModuleList()
        for i in range(1, config.num_timescales):
            coupling = CrossTimescaleCoupling(
                from_dim=config.timescales[i-1],
                to_dim=config.timescales[i],
                hidden_dim=config.hidden_dim,
            )
            self.couplings.append(coupling)
        
        # Output heads
        self.heads = ODINOutputHeads(
            state_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            num_stances=config.num_stances,
            num_actions=config.num_actions,
        )
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _split_state(self, h: torch.Tensor) -> List[torch.Tensor]:
        """Split full state into timescale blocks."""
        blocks = []
        idx = 0
        for dim in self.config.timescales:
            blocks.append(h[:, idx:idx+dim])
            idx += dim
        return blocks
    
    def _merge_state(self, blocks: List[torch.Tensor]) -> torch.Tensor:
        """Merge timescale blocks into full state."""
        return torch.cat(blocks, dim=-1)
    
    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize state to small random values."""
        return torch.randn(batch_size, self.config.state_dim, device=device) * 0.01
    
    def forward(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Single ODIN step.
        
        Args:
            h: Current state (batch, state_dim)
            u: Input (batch, input_dim)
            
        Returns:
            h_next: Next state (batch, state_dim)
            y: Output (batch, 768)
            info: Diagnostics
        """
        # Split state into timescale blocks
        h_blocks = self._split_state(h)
        
        # Process each timescale block
        h_blocks_next = []
        all_info = {}
        
        for i, (block, h_block) in enumerate(zip(self.timescale_blocks, h_blocks)):
            h_new, info = block(h_block, u)
            h_blocks_next.append(h_new)
            all_info[f'timescale_{i}'] = info
        
        # Apply cross-timescale coupling (slower ← faster)
        for i, coupling in enumerate(self.couplings):
            contribution = coupling(h_blocks_next[i], h_blocks_next[i+1])
            h_blocks_next[i+1] = h_blocks_next[i+1] + contribution
        
        # Merge back to full state
        h_next = self._merge_state(h_blocks_next)
        
        # Compute outputs
        y, outputs = self.heads(h_next)
        
        all_info['outputs'] = {
            'stance_probs': F.softmax(outputs['cognitive']['stance_logits'], dim=-1).mean(0).tolist(),
            'confidence': outputs['cognitive']['confidence'].mean().item(),
        }
        all_info['h_norm'] = h_next.norm(dim=-1).mean().item()
        
        return h_next, y, all_info
    
    def run_episode(
        self,
        inputs: List[torch.Tensor],
        h_init: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
        """
        Run a full episode (sequence of steps).
        
        Args:
            inputs: List of input tensors, one per timestep
            h_init: Optional initial state
            
        Returns:
            states: List of states
            outputs: List of outputs
            infos: List of diagnostic dicts
        """
        device = inputs[0].device
        batch_size = inputs[0].shape[0]
        
        if h_init is None:
            h = self.init_state(batch_size, device)
        else:
            h = h_init
        
        states = [h]
        outputs = []
        infos = []
        
        for u in inputs:
            h, y, info = self.forward(h, u)
            states.append(h)
            outputs.append(y)
            infos.append(info)
        
        return states, outputs, infos
    
    @torch.no_grad()
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self) -> ODINConfig:
        """Get model config."""
        return self.config


def create_model(config_name: str = 'tiny', device: str = 'cpu') -> ODINModel:
    """
    Create ODIN model with preset configuration.
    
    Args:
        config_name: 'tiny', 'small', 'medium', or 'large'
        device: 'cpu' or 'cuda'
    """
    config_map = {
        'tiny': ODINConfig.tiny,
        'small': ODINConfig.small,
        'medium': ODINConfig.medium,
        'large': ODINConfig.large,
    }
    
    config = config_map[config_name]()
    model = ODINModel(config)
    model = model.to(device)
    
    print(f"Created ODIN-{config_name} with {model.count_parameters():,} parameters")
    
    return model
