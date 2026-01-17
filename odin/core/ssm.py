"""
ODIN Selective SSM Layer
------------------------
Low-rank diagonal + UV^T dynamics with selection mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SelectionNetwork(nn.Module):
    """
    Computes selection vector z_t = σ(W_z[h_t; u_t] + b_z)
    Controls which dimensions of state are updated.
    """
    
    def __init__(self, state_dim: int, input_dim: int, hidden_dim: int):
        super().__init__()
        concat_dim = state_dim + input_dim
        
        self.net = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: State (batch, state_dim)
            u: Input (batch, input_dim)
        Returns:
            z: Selection vector (batch, state_dim) in [0, 1]
        """
        x = torch.cat([h, u], dim=-1)
        return self.net(x)


class DeltaNetwork(nn.Module):
    """
    Computes delta adjustments for dynamics.
    Used for ΔD (diagonal) and ΔU (low-rank factor).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Initialize output layer to zero for stability
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelectiveSSMLayer(nn.Module):
    """
    Single Selective SSM layer with low-rank dynamics.
    
    Dynamics: A_t = diag(D_t) + U_t @ V^T
    State update: h_{t+1} = Φ_t @ h_t + Γ_t @ u_t
    
    Where Φ_t ≈ exp(A_t * dt) computed efficiently via low-rank structure.
    """
    
    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        rank: int,
        hidden_dim: int,
        tau: float,
        dt: float = 0.01,
        stability_margin: float = 0.01,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.rank = rank
        self.tau = tau
        self.dt = dt
        self.stability_margin = stability_margin
        
        # Base dynamics (learnable)
        # Initialize D_base based on timescale tau
        init_d = -1.0 / tau  # Decay rate
        self.D_base = nn.Parameter(torch.full((state_dim,), init_d))
        
        # Low-rank factors
        self.U_base = nn.Parameter(torch.randn(state_dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(state_dim, rank) * 0.01)
        
        # Input projection
        self.B = nn.Linear(input_dim, state_dim, bias=False)
        nn.init.xavier_uniform_(self.B.weight, gain=0.1)
        
        # Selection network
        self.selection = SelectionNetwork(state_dim, input_dim, hidden_dim)
        
        # Delta networks for input-dependent dynamics
        concat_dim = state_dim + input_dim + state_dim  # h, u, z
        self.delta_D = DeltaNetwork(concat_dim, hidden_dim, state_dim)
        self.delta_U = DeltaNetwork(concat_dim, hidden_dim, state_dim * rank)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(state_dim)
    
    def _enforce_stability(self, D: torch.Tensor) -> torch.Tensor:
        """Ensure all diagonal entries are negative (Re(λ) < 0)."""
        return -F.softplus(D) - self.stability_margin
    
    def _compute_phi_h(
        self,
        h: torch.Tensor,
        D: torch.Tensor,
        U: torch.Tensor,
    ) -> torch.Tensor:
        """
        Efficiently compute Φ @ h where Φ = exp((D + UV^T) * dt).
        Uses diagonal + low-rank approximation.
        
        For small dt, uses truncated Taylor series:
        exp(A*dt) ≈ I + A*dt + (A*dt)²/2
        """
        # Diagonal exponential
        exp_D = torch.exp(D * self.dt)  # (batch, state_dim)
        
        # Diagonal contribution
        h_diag = exp_D * h  # (batch, state_dim)
        
        # Low-rank contribution (first order approximation)
        # (UV^T)h * dt * exp(D*dt)
        Vh = torch.einsum('dr,bd->br', self.V, h)  # (batch, rank)
        UVh = torch.einsum('bdr,br->bd', U.unsqueeze(0) if U.dim() == 2 else U, Vh)  # (batch, state_dim)
        h_lowrank = exp_D * UVh * self.dt
        
        return h_diag + h_lowrank
    
    def forward(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Single SSM step.
        
        Args:
            h: Current state (batch, state_dim)
            u: Input (batch, input_dim)
            
        Returns:
            h_next: Next state (batch, state_dim)
            info: Diagnostics dict
        """
        batch_size = h.shape[0]
        
        # 1. Compute selection vector
        z = self.selection(h, u)  # (batch, state_dim)
        
        # 2. Compute input-dependent dynamics
        concat = torch.cat([h, u, z], dim=-1)  # (batch, state_dim + input_dim + state_dim)
        
        delta_d = self.delta_D(concat)  # (batch, state_dim)
        delta_u = self.delta_U(concat).view(batch_size, self.state_dim, self.rank)  # (batch, state_dim, rank)
        
        # 3. Apply dynamics with stability constraint
        D_t = self._enforce_stability(self.D_base.unsqueeze(0) + delta_d)  # (batch, state_dim)
        U_t = (self.U_base.unsqueeze(0) + delta_u) * z.unsqueeze(-1)  # (batch, state_dim, rank)
        
        # 4. Compute state transition
        h_evolved = self._compute_phi_h(h, D_t, U_t)
        
        # 5. Add input contribution
        Bu = self.B(u) * z  # Gated input
        h_next = h_evolved + Bu * self.dt
        
        # 6. Residual connection + layer norm
        h_next = self.layer_norm(h_next + h)
        
        # Diagnostics
        info = {
            'D_min': D_t.min().item(),
            'D_max': D_t.max().item(),
            'z_mean': z.mean().item(),
            'h_norm': h_next.norm(dim=-1).mean().item(),
        }
        
        return h_next, info


class TimescaleSSMBlock(nn.Module):
    """
    SSM block for a single timescale (fast/mid/slow/ultra).
    Contains multiple SSM layers with shared timescale.
    """
    
    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        rank: int,
        hidden_dim: int,
        tau: float,
        num_layers: int = 1,
        dt: float = 0.01,
        stability_margin: float = 0.01,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.tau = tau
        
        self.layers = nn.ModuleList([
            SelectiveSSMLayer(
                state_dim=state_dim,
                input_dim=input_dim if i == 0 else state_dim,
                rank=rank,
                hidden_dim=hidden_dim,
                tau=tau,
                dt=dt,
                stability_margin=stability_margin,
            )
            for i in range(num_layers)
        ])
    
    def forward(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward through all layers in this timescale block.
        """
        all_info = {}
        
        for i, layer in enumerate(self.layers):
            input_to_layer = u if i == 0 else h
            h, info = layer(h, input_to_layer)
            all_info[f'layer_{i}'] = info
        
        return h, all_info
