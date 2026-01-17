"""
ODIN Training Script
--------------------
Complete training loop with checkpointing and auto-resume.
Ready for Google Colab.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import os
import json
import time
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from odin.core import ODINModel, ODINConfig
from odin.dataset import create_dataloader


class ODINTrainer:
    """
    Trainer for ODIN model with:
    - Automatic checkpointing
    - Resume from checkpoint
    - Mixed precision training
    - Gradient clipping
    - Learning rate scheduling
    - Logging
    """
    
    def __init__(
        self,
        model: ODINModel,
        config: ODINConfig,
        train_loader,
        checkpoint_dir: str = 'checkpoints',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        total_steps = config.max_epochs * len(train_loader)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=config.learning_rate / 100,
        )
        
        # Mixed precision
        self.scaler = GradScaler(enabled=config.use_fp16 and device == 'cuda')
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'stance_acc': [], 'action_acc': []}
    
    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses."""
        
        # Get predictions
        stance_logits = outputs['cognitive']['stance_logits']  # (batch, 4)
        action_logits = outputs['action']['action_logits']  # (batch, 10)
        confidence = outputs['cognitive']['confidence'].squeeze(-1)  # (batch,)
        
        # Get targets (take first valid timestep for now - simplified)
        stance_targets = batch['stance_targets'][:, 0]  # (batch,)
        action_targets = batch['action_targets'][:, 0]  # (batch,)
        confidence_targets = batch['confidence_targets'][:, 0]  # (batch,)
        
        # Losses
        stance_loss = F.cross_entropy(stance_logits, stance_targets)
        action_loss = F.cross_entropy(action_logits, action_targets)
        confidence_loss = F.mse_loss(confidence, confidence_targets)
        
        # Total loss
        total_loss = stance_loss + action_loss + 0.5 * confidence_loss
        
        # Accuracy
        stance_acc = (stance_logits.argmax(-1) == stance_targets).float().mean()
        action_acc = (action_logits.argmax(-1) == action_targets).float().mean()
        
        return {
            'loss': total_loss,
            'stance_loss': stance_loss,
            'action_loss': action_loss,
            'confidence_loss': confidence_loss,
            'stance_acc': stance_acc,
            'action_acc': action_acc,
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_stance_acc = 0
        total_action_acc = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch in pbar:
            # Move to device
            inputs = batch['inputs'].to(self.device)  # (batch, max_steps, 1024)
            mask = batch['mask'].to(self.device)
            
            # Move targets
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Take first timestep input (simplified training)
            u = inputs[:, 0, :]  # (batch, 1024)
            
            # Initialize state
            h = self.model.init_state(u.shape[0], self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_fp16 and self.device == 'cuda'):
                h_next, y, info = self.model(h, u)
                
                # Parse outputs for loss
                _, outputs = self.model.heads(h_next)
                
                losses = self._compute_loss(outputs, batch, mask)
            
            # Backward
            self.scaler.scale(losses['loss']).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            # Track
            total_loss += losses['loss'].item()
            total_stance_acc += losses['stance_acc'].item()
            total_action_acc += losses['action_acc'].item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['loss'].item():.4f}",
                'stance': f"{losses['stance_acc'].item():.2%}",
                'action': f"{losses['action_acc'].item():.2%}",
            })
        
        metrics = {
            'train_loss': total_loss / num_batches,
            'stance_acc': total_stance_acc / num_batches,
            'action_acc': total_action_acc / num_batches,
        }
        
        return metrics
    
    def save_checkpoint(self, filename: str = 'latest.pt'):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
            'config': {
                'state_dim': self.config.state_dim,
                'timescales': self.config.timescales,
                'rank': self.config.rank,
                'num_layers': self.config.num_layers,
                'hidden_dim': self.config.hidden_dim,
            },
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, filename: str = 'latest.pt') -> bool:
        """Load checkpoint if exists. Returns True if loaded."""
        path = self.checkpoint_dir / filename
        
        if not path.exists():
            print(f"No checkpoint found at {path}")
            return False
        
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint['history']
        
        print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
        return True
    
    def train(self, resume: bool = True):
        """
        Full training loop.
        
        Args:
            resume: If True, attempt to resume from checkpoint
        """
        # Try to resume
        if resume:
            self.load_checkpoint()
        
        print(f"\n{'='*60}")
        print(f"Training ODIN Model")
        print(f"{'='*60}")
        print(f"Parameters: {self.model.count_parameters():,}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.current_epoch} -> {self.config.max_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            
            metrics = self.train_epoch()
            
            # Track history
            self.history['train_loss'].append(metrics['train_loss'])
            self.history['stance_acc'].append(metrics['stance_acc'])
            self.history['action_acc'].append(metrics['action_acc'])
            
            # Log
            elapsed = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{self.config.max_epochs}")
            print(f"  Loss: {metrics['train_loss']:.4f}")
            print(f"  Stance Acc: {metrics['stance_acc']:.2%}")
            print(f"  Action Acc: {metrics['action_acc']:.2%}")
            print(f"  Time: {elapsed/60:.1f} min")
            
            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_every == 0:
                self.save_checkpoint('latest.pt')
                
                if metrics['train_loss'] < self.best_loss:
                    self.best_loss = metrics['train_loss']
                    self.save_checkpoint('best.pt')
        
        # Final save
        self.save_checkpoint('final.pt')
        print(f"\nTraining complete! Total time: {(time.time()-start_time)/60:.1f} min")
        
        return self.history
    
    def export_model(self, path: str = 'odin_final.pt'):
        """Export trained model weights only."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'state_dim': self.config.state_dim,
                'timescales': self.config.timescales,
                'rank': self.config.rank,
                'num_layers': self.config.num_layers,
                'hidden_dim': self.config.hidden_dim,
            },
        }, path)
        print(f"Model exported to {path}")


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ODIN model')
    parser.add_argument('--config', choices=['tiny', 'small', 'medium', 'large'], 
                       default='tiny', help='Model size')
    parser.add_argument('--data', type=str, default='data/scenarios',
                       help='Path to scenario data')
    parser.add_argument('--checkpoint', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override max epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh, do not resume')
    
    args = parser.parse_args()
    
    # Create config
    config_map = {
        'tiny': ODINConfig.tiny,
        'small': ODINConfig.small,
        'medium': ODINConfig.medium,
        'large': ODINConfig.large,
    }
    config = config_map[args.config]()
    
    # Override if specified
    if args.epochs:
        config.max_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ODINModel(config).to(device)
    
    print(f"Created ODIN-{args.config} with {model.count_parameters():,} parameters")
    
    # Create dataloader
    train_loader = create_dataloader(
        scenario_dir=args.data,
        batch_size=config.batch_size,
        shuffle=True,
        max_steps=10,
    )
    
    # Create trainer
    trainer = ODINTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        checkpoint_dir=args.checkpoint,
        device=device,
    )
    
    # Train
    history = trainer.train(resume=not args.no_resume)
    
    # Export
    trainer.export_model(f'odin_{args.config}.pt')
    
    print("\nDone!")


if __name__ == '__main__':
    main()
