"""
ODIN Intent Encoder
-------------------
Encodes text intents to 384-dim vectors using DistilBERT.
"""

import torch
from typing import List, Union
import numpy as np


class IntentEncoder:
    """
    Encodes text intents to fixed-dimension vectors.
    Uses DistilBERT or falls back to simple hash-based encoding.
    """
    
    def __init__(self, use_transformer: bool = True, device: str = "cpu"):
        self.use_transformer = use_transformer
        self.device = device
        self.model = None
        self.tokenizer = None
        self.output_dim = 384
        
        if use_transformer:
            try:
                self._load_transformer()
            except Exception as e:
                print(f"Warning: Could not load transformer, using fallback: {e}")
                self.use_transformer = False
    
    def _load_transformer(self):
        """Load DistilBERT model."""
        from transformers import DistilBertTokenizer, DistilBertModel
        
        print("Loading DistilBERT...")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.model.to(self.device)
        self.model.eval()
        
        # DistilBERT outputs 768, we project to 384
        self.projection = torch.nn.Linear(768, self.output_dim).to(self.device)
        print("DistilBERT loaded.")
    
    def _hash_encode(self, text: str) -> np.ndarray:
        """Simple hash-based encoding fallback."""
        import hashlib
        
        # Create deterministic hash
        h = hashlib.sha256(text.encode()).digest()
        
        # Convert to floats
        embedding = np.zeros(self.output_dim)
        for i in range(self.output_dim):
            byte_idx = i % len(h)
            embedding[i] = (h[byte_idx] / 255.0) * 2 - 1  # Normalize to [-1, 1]
        
        # Add some structure based on words
        words = text.lower().split()
        for i, word in enumerate(words[:10]):  # First 10 words
            word_hash = hashlib.md5(word.encode()).digest()
            offset = (i * 38) % self.output_dim
            for j in range(min(38, self.output_dim - offset)):
                embedding[offset + j] += (word_hash[j % len(word_hash)] / 255.0 - 0.5) * 0.1
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    @torch.no_grad()
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text(s) to vectors.
        
        Args:
            texts: Single string or list of strings
            
        Returns:
            Array of shape (384,) for single text or (N, 384) for list
        """
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False
        
        if self.use_transformer:
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            # Encode
            outputs = self.model(**inputs)
            
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Project to 384
            embeddings = self.projection(embeddings)
            
            result = embeddings.cpu().numpy()
        else:
            result = np.array([self._hash_encode(text) for text in texts])
        
        if single:
            return result[0]
        return result
    
    def encode_scenario(self, scenario_dict: dict) -> dict:
        """
        Encode all intents in a scenario.
        
        Returns scenario with 'intent_embedding' added to each step.
        """
        for step in scenario_dict["steps"]:
            intent = step["input"]["intent"]
            step["input"]["intent_embedding"] = self.encode(intent).tolist()
        
        return scenario_dict


class MTEncoder:
    """
    Encodes MT (Memory Thread) state to fixed-dimension vectors.
    - Hash: 128-dim
    - Delta: 128-dim
    """
    
    def __init__(self):
        self.hash_dim = 128
        self.delta_dim = 128
    
    def encode_hash(self, mt_hash: str) -> np.ndarray:
        """Convert hex hash to embedding."""
        # Pad or truncate to 32 chars (128 bits)
        mt_hash = (mt_hash + "0" * 32)[:32]
        
        embedding = np.zeros(self.hash_dim)
        for i in range(32):
            val = int(mt_hash[i], 16) / 15.0  # Normalize hex digit
            embedding[i * 4:(i + 1) * 4] = val
        
        return embedding.astype(np.float32)
    
    def encode_delta(self, mt_delta: list) -> np.ndarray:
        """Encode MT changes."""
        embedding = np.zeros(self.delta_dim)
        
        for i, change in enumerate(mt_delta[:4]):  # Max 4 changes
            change_type = change.get("type", "unknown")
            
            # Type embedding
            type_map = {
                "progress": 0, "complete": 1, "error": 2,
                "goal_revision": 3, "unknown": 4
            }
            type_idx = type_map.get(change_type, 4)
            offset = i * 32
            embedding[offset + type_idx * 6:offset + (type_idx + 1) * 6] = 1.0
        
        return embedding.astype(np.float32)


class FeedbackEncoder:
    """Encodes execution feedback to 192-dim."""
    
    def __init__(self):
        self.output_dim = 192
    
    def encode(self, feedback: dict) -> np.ndarray:
        """Encode feedback dict."""
        if feedback is None:
            return np.zeros(self.output_dim, dtype=np.float32)
        
        embedding = np.zeros(self.output_dim)
        
        # Success/failure (first 64 dims)
        if feedback.get("success", False):
            embedding[:64] = 1.0
        else:
            embedding[:64] = -1.0
        
        # Error encoding (next 64 dims)
        error = feedback.get("error", "")
        if error:
            import hashlib
            error_hash = hashlib.md5(error.encode()).digest()
            for i in range(64):
                embedding[64 + i] = error_hash[i % len(error_hash)] / 255.0
        
        # Step info (last 64 dims)
        step = feedback.get("step", 0)
        embedding[128:132] = step / 20.0  # Normalize step number
        
        return embedding.astype(np.float32)


class SystemEncoder:
    """Encodes system metadata to 192-dim."""
    
    def __init__(self):
        self.output_dim = 192
    
    def encode(self, system: dict) -> np.ndarray:
        """Encode system metadata."""
        embedding = np.zeros(self.output_dim)
        
        # Urgency (first 64 dims)
        urgency = system.get("urgency", 0.5)
        embedding[:64] = urgency
        
        # Timestamp encoding (next 64 dims) - cyclical
        timestamp = system.get("timestamp", 0)
        hour = (timestamp // 3600) % 24
        embedding[64:96] = np.sin(2 * np.pi * hour / 24)
        embedding[96:128] = np.cos(2 * np.pi * hour / 24)
        
        # Reserved (last 64 dims)
        
        return embedding.astype(np.float32)


class FullInputEncoder:
    """
    Complete input encoder combining all components.
    Output: 1024-dim vector
    """
    
    def __init__(self, use_transformer: bool = True, device: str = "cpu"):
        self.intent_encoder = IntentEncoder(use_transformer, device)
        self.mt_encoder = MTEncoder()
        self.feedback_encoder = FeedbackEncoder()
        self.system_encoder = SystemEncoder()
        
        self.output_dim = 1024  # 384 + 256 + 192 + 192
    
    def encode(self, input_state: dict) -> np.ndarray:
        """
        Encode full input state.
        
        Args:
            input_state: Dict with intent, mt_hash, mt_delta, feedback, system
            
        Returns:
            1024-dim vector
        """
        intent_emb = self.intent_encoder.encode(input_state["intent"])  # 384
        
        hash_emb = self.mt_encoder.encode_hash(input_state["mt_hash"])  # 128
        delta_emb = self.mt_encoder.encode_delta(input_state["mt_delta"])  # 128
        mt_emb = np.concatenate([hash_emb, delta_emb])  # 256
        
        feedback_emb = self.feedback_encoder.encode(input_state["feedback"])  # 192
        system_emb = self.system_encoder.encode(input_state["system"])  # 192
        
        full_emb = np.concatenate([intent_emb, mt_emb, feedback_emb, system_emb])
        
        assert full_emb.shape[0] == self.output_dim, f"Expected {self.output_dim}, got {full_emb.shape[0]}"
        
        return full_emb.astype(np.float32)


if __name__ == "__main__":
    # Test encoders
    print("Testing encoders...")
    
    # Test intent encoder (without transformer)
    intent_enc = IntentEncoder(use_transformer=False)
    test_intent = "Build a REST API for user management"
    intent_emb = intent_enc.encode(test_intent)
    print(f"Intent embedding shape: {intent_emb.shape}")
    
    # Test full encoder
    full_enc = FullInputEncoder(use_transformer=False)
    test_input = {
        "intent": "Build a REST API",
        "mt_hash": "a1b2c3d4e5f6",
        "mt_delta": [{"type": "progress", "step": 1}],
        "feedback": {"success": True, "step": 0},
        "system": {"urgency": 0.5, "timestamp": 1704825600}
    }
    full_emb = full_enc.encode(test_input)
    print(f"Full embedding shape: {full_emb.shape}")
    print("Encoders working!")
