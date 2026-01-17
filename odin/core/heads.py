"""
ODIN Output Heads
-----------------
Cognitive, Action, and Memory heads for decision output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class CognitiveHead(nn.Module):
    """
    Outputs cognitive state: stance, confidence, uncertainty.
    
    Output (256-dim):
    - stance: 4-dim (softmax over EXPLORE/PLAN/EXECUTE/REFLECT)  
    - confidence: 1-dim (sigmoid)
    - uncertainty_flags: 64-dim
    - need_input: 64-dim
    - reserved: 123-dim
    """
    
    def __init__(self, state_dim: int, hidden_dim: int, num_stances: int = 4):
        super().__init__()
        self.num_stances = num_stances
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 256),
        )
    
    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: State (batch, state_dim)
        Returns:
            Dict with stance_logits, confidence, uncertainty, need_input
        """
        out = self.net(h)  # (batch, 256)
        
        return {
            'stance_logits': out[:, :4],  # (batch, 4) - apply softmax externally
            'confidence': torch.sigmoid(out[:, 4:5]),  # (batch, 1)
            'uncertainty': torch.sigmoid(out[:, 5:69]),  # (batch, 64)
            'need_input': torch.sigmoid(out[:, 69:133]),  # (batch, 64)
            'raw': out,  # (batch, 256)
        }


class ActionHead(nn.Module):
    """
    Outputs action decision: hypothesis scores, action primitive, parameters.
    
    Output (384-dim):
    - hypothesis_scores: 192-dim
    - action_logits: 10-dim (10 action primitives)
    - parameters: 182-dim
    """
    
    def __init__(self, state_dim: int, hidden_dim: int, num_actions: int = 10):
        super().__init__()
        self.num_actions = num_actions
        
        # Use goal component for action decisions
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 384),
        )
    
    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: State (batch, state_dim)
        Returns:
            Dict with hypothesis_scores, action_logits, parameters
        """
        out = self.net(h)  # (batch, 384)
        
        return {
            'hypothesis_scores': out[:, :192],  # (batch, 192)
            'action_logits': out[:, 192:202],  # (batch, 10)
            'parameters': out[:, 202:],  # (batch, 182)
            'raw': out,  # (batch, 384)
        }


class MemoryHead(nn.Module):
    """
    Outputs memory operations for MT interaction.
    
    Output (128-dim):
    - write_intent: 64-dim
    - write_type: 32-dim
    - confidence_threshold: 32-dim
    """
    
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 128),
        )
    
    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: State (batch, state_dim)
        Returns:
            Dict with write_intent, write_type, confidence_threshold
        """
        out = self.net(h)  # (batch, 128)
        
        return {
            'write_intent': torch.sigmoid(out[:, :64]),  # (batch, 64)
            'write_type': out[:, 64:96],  # (batch, 32)
            'confidence_threshold': torch.sigmoid(out[:, 96:]),  # (batch, 32)
            'raw': out,  # (batch, 128)
        }


class ODINOutputHeads(nn.Module):
    """
    Combined output heads producing full y_t output.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_stances: int = 4,
        num_actions: int = 10,
    ):
        super().__init__()
        
        self.cognitive = CognitiveHead(state_dim, hidden_dim, num_stances)
        self.action = ActionHead(state_dim, hidden_dim, num_actions)
        self.memory = MemoryHead(state_dim, hidden_dim)
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            h: State (batch, state_dim)
        Returns:
            y: Full output (batch, 768)
            outputs: Dict with parsed outputs from each head
        """
        cog_out = self.cognitive(h)
        act_out = self.action(h)
        mem_out = self.memory(h)
        
        # Concatenate raw outputs
        y = torch.cat([cog_out['raw'], act_out['raw'], mem_out['raw']], dim=-1)
        
        outputs = {
            'cognitive': cog_out,
            'action': act_out,
            'memory': mem_out,
        }
        
        return y, outputs
