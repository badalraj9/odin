"""
ODIN Dataset Schema
-------------------
Data classes for scenario-based training data.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import json


class Stance(Enum):
    """Cognitive modes"""
    EXPLORE = 0
    PLAN = 1
    EXECUTE = 2
    REFLECT = 3


class Action(Enum):
    """Action primitives"""
    EXPLORE = 0
    PLAN = 1
    DECOMPOSE_TASK = 2
    QUERY_USER = 3
    DELEGATE = 4
    EXECUTE = 5
    COMMIT = 6
    WRITE_MT = 7
    BACKTRACK = 8
    WAIT = 9


class ScenarioType(Enum):
    """Training scenario types"""
    PLANNING = "planning"
    REVISION = "revision"
    ADVERSARIAL = "adversarial"


@dataclass
class InputState:
    """
    Input to ODIN at each timestep.
    Total dimension: 1024
    """
    intent: str                          # Will be encoded to 384-dim
    mt_hash: str                          # 128-dim hash
    mt_delta: List[Dict[str, Any]]        # Changes, encoded to 128-dim
    feedback: Optional[Dict[str, Any]]    # 192-dim (null if no prior action)
    system: Dict[str, Any]                # 192-dim metadata
    
    def to_dict(self) -> Dict:
        return {
            "intent": self.intent,
            "mt_hash": self.mt_hash,
            "mt_delta": self.mt_delta,
            "feedback": self.feedback,
            "system": self.system,
        }


@dataclass
class ExpectedOutput:
    """
    Expected ODIN output at each timestep.
    """
    stance: Stance
    confidence: float                     # 0-1
    action: Action
    uncertainty: List[str]                # What's uncertain
    parameters: Optional[Dict[str, Any]] = None  # Action params
    
    def to_dict(self) -> Dict:
        return {
            "stance": self.stance.name,
            "confidence": self.confidence,
            "action": self.action.name,
            "uncertainty": self.uncertainty,
            "parameters": self.parameters,
        }


@dataclass
class ScenarioStep:
    """Single timestep in a scenario."""
    t: int
    input_state: InputState
    expected_output: ExpectedOutput
    reward: float
    
    def to_dict(self) -> Dict:
        return {
            "t": self.t,
            "input": self.input_state.to_dict(),
            "expected_output": self.expected_output.to_dict(),
            "reward": self.reward,
        }


@dataclass
class ScenarioMetadata:
    """Scenario metadata for curriculum."""
    domain: str
    complexity: str  # "simple", "medium", "complex"
    required_pivots: int
    error_recovery: bool = False
    ambiguity_level: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "domain": self.domain,
            "complexity": self.complexity,
            "required_pivots": self.required_pivots,
            "error_recovery": self.error_recovery,
            "ambiguity_level": self.ambiguity_level,
        }


@dataclass
class Scenario:
    """
    Complete training scenario.
    """
    id: str
    type: ScenarioType
    difficulty: float  # 0-1
    steps: List[ScenarioStep]
    metadata: ScenarioMetadata
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "difficulty": self.difficulty,
            "steps": [s.to_dict() for s in self.steps],
            "metadata": self.metadata.to_dict(),
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Scenario':
        steps = []
        for step_data in data["steps"]:
            input_state = InputState(
                intent=step_data["input"]["intent"],
                mt_hash=step_data["input"]["mt_hash"],
                mt_delta=step_data["input"]["mt_delta"],
                feedback=step_data["input"]["feedback"],
                system=step_data["input"]["system"],
            )
            expected_output = ExpectedOutput(
                stance=Stance[step_data["expected_output"]["stance"]],
                confidence=step_data["expected_output"]["confidence"],
                action=Action[step_data["expected_output"]["action"]],
                uncertainty=step_data["expected_output"]["uncertainty"],
                parameters=step_data["expected_output"].get("parameters"),
            )
            steps.append(ScenarioStep(
                t=step_data["t"],
                input_state=input_state,
                expected_output=expected_output,
                reward=step_data["reward"],
            ))
        
        metadata = ScenarioMetadata(**data["metadata"])
        
        return cls(
            id=data["id"],
            type=ScenarioType(data["type"]),
            difficulty=data["difficulty"],
            steps=steps,
            metadata=metadata,
        )
    
    @classmethod
    def load(cls, path: str) -> 'Scenario':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# Reward calculation helpers
def calculate_reward(
    action: Action,
    expected_action: Action,
    confidence: float,
    was_honest_uncertain: bool = False,
    caused_pivot: bool = False,
    caught_error: bool = False,
) -> float:
    """Calculate reward based on action correctness and confidence calibration."""
    
    reward = 0.0
    
    # Base reward for correct action
    if action == expected_action:
        reward = 1.0 if confidence > 0.7 else 0.7
    else:
        # Wrong action penalty scaled by confidence
        if confidence > 0.7:
            reward = -1.0  # Overconfident and wrong
        else:
            reward = -0.3  # At least was uncertain
    
    # Bonuses
    if was_honest_uncertain:
        reward += 0.3
    if caused_pivot and action == expected_action:
        reward += 0.5
    if caught_error:
        reward += 0.8
    
    # Penalties
    if action == Action.QUERY_USER and expected_action != Action.QUERY_USER:
        reward -= 0.2  # Unnecessary query
    
    return max(-1.0, min(1.0, reward))  # Clamp to [-1, 1]
