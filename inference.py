"""
ODIN Inference / Testing
------------------------
Test a trained ODIN model interactively.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from odin.core import ODINModel, ODINConfig
from data.scripts.encoders import FullInputEncoder


# Action names for readable output
ACTION_NAMES = [
    "EXPLORE",      # 0: Need more info
    "PLAN",         # 1: Ready to structure
    "DECOMPOSE",    # 2: Break into steps
    "QUERY_USER",   # 3: Ask for clarification
    "DELEGATE",     # 4: Pass to specialist
    "EXECUTE",      # 5: Do it
    "COMMIT",       # 6: Finalize
    "WRITE_MT",     # 7: Save to memory
    "BACKTRACK",    # 8: Undo/rethink
    "WAIT",         # 9: Need external result
]

STANCE_NAMES = ["EXPLORE", "PLAN", "EXECUTE", "REFLECT"]


class ODINRunner:
    """
    Interactive runner for testing ODIN model.
    """
    
    def __init__(self, model_path: str, config_name: str = 'tiny', device: str = None):
        """
        Load a trained model.
        
        Args:
            model_path: Path to .pt checkpoint
            config_name: 'tiny', 'small', 'medium'
            device: 'cuda' or 'cpu'
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        config_map = {
            'tiny': ODINConfig.tiny,
            'small': ODINConfig.small,
            'medium': ODINConfig.medium,
        }
        self.config = config_map[config_name]()
        
        # Create model
        self.model = ODINModel(self.config).to(self.device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded ODIN-{config_name} from {model_path}")
        print(f"Device: {self.device}")
        
        # Initialize encoder
        self.encoder = FullInputEncoder(use_transformer=False, device='cpu')
        
        # Initialize state
        self.h = self.model.init_state(1, self.device)
        self.step = 0
    
    def reset(self):
        """Reset state for new episode."""
        self.h = self.model.init_state(1, self.device)
        self.step = 0
        print("State reset.")
    
    def process(
        self,
        intent: str,
        mt_hash: str = "0" * 32,
        feedback: dict = None,
        urgency: float = 0.5,
    ) -> dict:
        """
        Process an input and get ODIN's decision.
        
        Args:
            intent: What the user wants (text)
            mt_hash: Memory thread hash (optional)
            feedback: Previous action result (optional)
            urgency: 0-1 urgency level
            
        Returns:
            Decision dict with stance, action, confidence, etc.
        """
        # Encode input
        input_state = {
            "intent": intent,
            "mt_hash": mt_hash,
            "mt_delta": [],
            "feedback": feedback,
            "system": {"urgency": urgency, "timestamp": 0},
        }
        
        input_vec = self.encoder.encode(input_state)
        u = torch.from_numpy(input_vec).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            self.h, y, info = self.model(self.h, u)
            _, outputs = self.model.heads(self.h)
        
        # Parse output
        stance_logits = outputs['cognitive']['stance_logits'][0]
        action_logits = outputs['action']['action_logits'][0]
        confidence = outputs['cognitive']['confidence'][0].item()
        
        stance_probs = torch.softmax(stance_logits, dim=-1)
        action_probs = torch.softmax(action_logits, dim=-1)
        
        stance_idx = stance_probs.argmax().item()
        action_idx = action_probs.argmax().item()
        
        self.step += 1
        
        decision = {
            'step': self.step,
            'stance': STANCE_NAMES[stance_idx],
            'stance_confidence': stance_probs[stance_idx].item(),
            'action': ACTION_NAMES[action_idx],
            'action_confidence': action_probs[action_idx].item(),
            'overall_confidence': confidence,
            'h_norm': info['h_norm'],
        }
        
        return decision
    
    def print_decision(self, decision: dict):
        """Pretty print a decision."""
        print(f"\n{'='*50}")
        print(f"Step {decision['step']}")
        print(f"{'='*50}")
        print(f"Stance:     {decision['stance']} ({decision['stance_confidence']:.1%})")
        print(f"Action:     {decision['action']} ({decision['action_confidence']:.1%})")
        print(f"Confidence: {decision['overall_confidence']:.1%}")
        print(f"State norm: {decision['h_norm']:.4f}")
    
    def interactive(self):
        """Run interactive session."""
        print("\n" + "="*60)
        print("ODIN Interactive Session")
        print("="*60)
        print("Commands:")
        print("  Type any task/intent to process")
        print("  'reset' - Reset state")
        print("  'quit' - Exit")
        print("="*60 + "\n")
        
        while True:
            try:
                intent = input("You: ").strip()
                
                if not intent:
                    continue
                if intent.lower() == 'quit':
                    print("Goodbye!")
                    break
                if intent.lower() == 'reset':
                    self.reset()
                    continue
                
                decision = self.process(intent)
                self.print_decision(decision)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break


def run_test_scenarios(runner: ODINRunner):
    """Run some test scenarios to see ODIN in action."""
    
    print("\n" + "="*60)
    print("ODIN Test Scenarios")
    print("="*60)
    
    scenarios = [
        # Clear task
        ("Build a REST API for user authentication", None),
        ("Proceeding with step 1", {"success": True}),
        ("Complete", {"success": True}),
        
        # Ambiguous
        ("Fix it", None),
        
        # After reset - new task
        ("RESET", None),
        ("I need to debug the login issue", None),
        ("Error: Connection timeout", {"success": False, "error": "timeout"}),
    ]
    
    for intent, feedback in scenarios:
        if intent == "RESET":
            runner.reset()
            continue
            
        print(f"\nInput: \"{intent}\"")
        if feedback:
            print(f"Feedback: {feedback}")
            
        decision = runner.process(intent, feedback=feedback)
        runner.print_decision(decision)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ODIN model')
    parser.add_argument('model', type=str, help='Path to model .pt file')
    parser.add_argument('--config', choices=['tiny', 'small', 'medium'], 
                       default='tiny', help='Model size')
    parser.add_argument('--test', action='store_true', 
                       help='Run test scenarios instead of interactive')
    
    args = parser.parse_args()
    
    runner = ODINRunner(args.model, args.config)
    
    if args.test:
        run_test_scenarios(runner)
    else:
        runner.interactive()
