"""
ODIN Dataset Loader
-------------------
Loads scenario data for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import sys

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'data' / 'scripts'))

try:
    from encoders import FullInputEncoder
except ImportError:
    # Fallback for when run from different directory
    from data.scripts.encoders import FullInputEncoder


class ODINDataset(Dataset):
    """
    Dataset for ODIN training scenarios.
    Loads JSON scenarios and encodes inputs.
    """
    
    def __init__(
        self,
        scenario_dir: str,
        max_steps: int = 10,
        use_transformer: bool = False,  # Use hash encoding by default (faster)
        device: str = 'cpu',
    ):
        self.scenario_dir = Path(scenario_dir)
        self.max_steps = max_steps
        
        # Initialize encoder
        self.encoder = FullInputEncoder(use_transformer=use_transformer, device=device)
        
        # Find all scenario files
        self.scenario_files = []
        for subdir in ['planning', 'revision', 'adversarial']:
            subdir_path = self.scenario_dir / subdir
            if subdir_path.exists():
                self.scenario_files.extend(list(subdir_path.glob('*.json')))
        
        print(f"Found {len(self.scenario_files)} scenarios in {scenario_dir}")
        
        # Preload scenarios (for small datasets)
        self.scenarios = []
        for f in self.scenario_files:
            try:
                with open(f, 'r') as fp:
                    self.scenarios.append(json.load(fp))
            except Exception as e:
                print(f"Error loading {f}: {e}")
    
    def __len__(self) -> int:
        return len(self.scenarios)
    
    def _encode_step(self, step: Dict) -> Tuple[np.ndarray, Dict]:
        """Encode a single step's input and extract target."""
        input_state = step['input']
        expected = step['expected_output']
        
        # Encode input to 1024-dim vector
        input_vec = self.encoder.encode(input_state)
        
        # Extract targets
        stance_map = {'EXPLORE': 0, 'PLAN': 1, 'EXECUTE': 2, 'REFLECT': 3}
        action_map = {
            'EXPLORE': 0, 'PLAN': 1, 'DECOMPOSE_TASK': 2, 'QUERY_USER': 3,
            'DELEGATE': 4, 'EXECUTE': 5, 'COMMIT': 6, 'WRITE_MT': 7,
            'BACKTRACK': 8, 'WAIT': 9,
        }
        
        targets = {
            'stance': stance_map.get(expected['stance'], 0),
            'confidence': expected['confidence'],
            'action': action_map.get(expected['action'], 0),
            'reward': step['reward'],
        }
        
        return input_vec, targets
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a scenario as padded tensors.
        
        Returns:
            Dict with:
            - inputs: (max_steps, 1024)
            - stance_targets: (max_steps,)
            - action_targets: (max_steps,)
            - confidence_targets: (max_steps,)
            - rewards: (max_steps,)
            - mask: (max_steps,) - 1 where valid, 0 for padding
        """
        scenario = self.scenarios[idx]
        steps = scenario['steps']
        num_steps = min(len(steps), self.max_steps)
        
        # Initialize tensors
        inputs = np.zeros((self.max_steps, 1024), dtype=np.float32)
        stance_targets = np.zeros(self.max_steps, dtype=np.int64)
        action_targets = np.zeros(self.max_steps, dtype=np.int64)
        confidence_targets = np.zeros(self.max_steps, dtype=np.float32)
        rewards = np.zeros(self.max_steps, dtype=np.float32)
        mask = np.zeros(self.max_steps, dtype=np.float32)
        
        # Encode each step
        for t in range(num_steps):
            input_vec, targets = self._encode_step(steps[t])
            
            inputs[t] = input_vec
            stance_targets[t] = targets['stance']
            action_targets[t] = targets['action']
            confidence_targets[t] = targets['confidence']
            rewards[t] = targets['reward']
            mask[t] = 1.0
        
        return {
            'inputs': torch.from_numpy(inputs),
            'stance_targets': torch.from_numpy(stance_targets),
            'action_targets': torch.from_numpy(action_targets),
            'confidence_targets': torch.from_numpy(confidence_targets),
            'rewards': torch.from_numpy(rewards),
            'mask': torch.from_numpy(mask),
            'scenario_id': scenario['id'],
            'difficulty': scenario['difficulty'],
        }


def create_dataloader(
    scenario_dir: str,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    max_steps: int = 10,
    use_transformer: bool = False,
) -> DataLoader:
    """
    Create DataLoader for training.
    
    Args:
        scenario_dir: Path to scenarios directory
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers (0 for main process)
        max_steps: Maximum steps per scenario
        use_transformer: Whether to use transformer for intent encoding
    """
    dataset = ODINDataset(
        scenario_dir=scenario_dir,
        max_steps=max_steps,
        use_transformer=use_transformer,
    )
    
    # Custom collate function
    def collate_fn(batch):
        return {
            'inputs': torch.stack([b['inputs'] for b in batch]),
            'stance_targets': torch.stack([b['stance_targets'] for b in batch]),
            'action_targets': torch.stack([b['action_targets'] for b in batch]),
            'confidence_targets': torch.stack([b['confidence_targets'] for b in batch]),
            'rewards': torch.stack([b['rewards'] for b in batch]),
            'mask': torch.stack([b['mask'] for b in batch]),
        }
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return loader


if __name__ == '__main__':
    # Test the dataloader
    import sys
    scenario_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/scenarios'
    
    loader = create_dataloader(scenario_dir, batch_size=4, max_steps=10)
    
    print(f"Dataset size: {len(loader.dataset)}")
    print(f"Num batches: {len(loader)}")
    
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
