"""
Interventions for memory experiments.

Implements different intervention strategies:
- None: No intervention (baseline)
- Injection: Seed false high-confidence hypotheses at episode start
- Surgery: Suppress memory context after certain step threshold
- Noise: Add observation noise (flip bits with probability p)
"""

from typing import Optional
import random


class Intervention:
    """Base class for interventions."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
    
    def apply_at_start(self, memory_system, env) -> None:
        """Apply intervention at episode start."""
        pass
    
    def apply_to_observation(self, obs: str, step: int) -> str:
        """Apply intervention to observation."""
        return obs
    
    def should_suppress_memory(self, step: int) -> bool:
        """Check if memory should be suppressed at this step."""
        return False


class NoIntervention(Intervention):
    """No intervention - baseline."""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)


class InjectionIntervention(Intervention):
    """
    Injection intervention - seed false high-confidence hypotheses at episode start.
    
    Injects likely-false hypotheses like:
    - H(1->2): "Toggling bulb 1 flips bulb 2"
    - H(2->4): "Toggling bulb 2 flips bulb 4"
    - H(3->5): "Toggling bulb 3 flips bulb 5"
    
    These are seeded with high confidence (0.95), support=3, contradict=0.
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        inject_confidence: float = 0.95,
        inject_support: int = 3
    ):
        super().__init__(seed)
        self.inject_confidence = inject_confidence
        self.inject_support = inject_support
        
        # False hypothesis patterns: (action, flip_idx)
        self.false_patterns = [
            (1, 2),
            (2, 4),
            (3, 5)
        ]
    
    def apply_at_start(self, memory_system, env) -> None:
        """
        Inject false hypotheses into memory at episode start.
        
        Args:
            memory_system: Memory system to inject into
            env: Environment (to check valid indices)
        """
        num_actions = env.get_num_actions()
        
        # Only inject patterns that are valid for this environment size
        for action, flip_idx in self.false_patterns:
            if action < num_actions and flip_idx < num_actions:
                # Inject as natural language hypothesis (new format)
                if hasattr(memory_system, 'hypotheses'):
                    false_hypothesis = {
                        'text': f"Toggling bulb {action} affects bulb {flip_idx}",
                        'support': self.inject_support,
                        'contradict': 0,
                        'confidence': self.inject_confidence,
                        'source': 'injection',  # Mark as injected
                        'created_at': -1  # Before episode starts
                    }
                    memory_system.hypotheses.append(false_hypothesis)
                
                # Backward compatibility with old hyp_store format
                elif hasattr(memory_system, 'hyp_store'):
                    key = (action, flip_idx)
                    memory_system.hyp_store[key] = {
                        'support': self.inject_support,
                        'contradict': 0,
                        'confidence': self.inject_confidence
                    }


class SurgeryIntervention(Intervention):
    """
    Surgery intervention - suppress memory context after step threshold.
    
    After step >= threshold, memory context is not included in prompts.
    Memory continues to be updated, but not used for decision-making.
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        suppress_after_step: int = 80
    ):
        super().__init__(seed)
        self.suppress_after_step = suppress_after_step
    
    def should_suppress_memory(self, step: int) -> bool:
        """Check if memory should be suppressed at this step."""
        return step >= self.suppress_after_step


class NoiseIntervention(Intervention):
    """
    Noise intervention - flip each observed bit independently with probability p.
    
    Only applies to the agent's observation (not the actual environment state).
    This simulates noisy/unreliable perception.
    
    Supports both bitstring and emoji observation formats.
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        noise_prob: float = 0.05,
        obs_format: str = 'bitstring'
    ):
        super().__init__(seed)
        self.noise_prob = noise_prob
        self.obs_format = obs_format
    
    def apply_to_observation(self, obs: str, step: int) -> str:
        """
        Apply noise to observation based on format.
        
        Args:
            obs: Original observation string
            step: Current step (unused)
            
        Returns:
            Noisy observation string
        """
        if self.obs_format == 'emoji':
            return self._apply_noise_emoji(obs)
        else:  # bitstring
            return self._apply_noise_bitstring(obs)
    
    def _apply_noise_bitstring(self, obs: str) -> str:
        """Apply bit-flip noise to bitstring format (e.g., "010101")."""
        noisy_bits = []
        for bit in obs:
            if self.rng.random() < self.noise_prob:
                # Flip bit
                noisy_bits.append('0' if bit == '1' else '1')
            else:
                noisy_bits.append(bit)
        return ''.join(noisy_bits)
    
    def _apply_noise_emoji(self, obs: str) -> str:
        """Apply noise to emoji format (e.g., "💡 ○ 💡 ○")."""
        symbols = obs.split()
        noisy_symbols = []
        for symbol in symbols:
            if self.rng.random() < self.noise_prob:
                # Flip emoji
                noisy_symbols.append('○' if symbol == '💡' else '💡')
            else:
                noisy_symbols.append(symbol)
        return ' '.join(noisy_symbols)


def create_intervention(
    intervention_type: str,
    seed: Optional[int] = None,
    obs_format: str = 'bitstring',
    **kwargs
) -> Intervention:
    """
    Factory function to create intervention.
    
    Args:
        intervention_type: Type of intervention ('none', 'injection', 'surgery', 'noise')
        seed: Random seed for intervention
        obs_format: Observation format ('bitstring' or 'emoji') - used by noise intervention
        **kwargs: Additional arguments for intervention
        
    Returns:
        Intervention instance
    """
    if intervention_type == 'none':
        return NoIntervention(seed=seed)
    elif intervention_type == 'injection':
        return InjectionIntervention(seed=seed, **kwargs)
    elif intervention_type == 'surgery':
        return SurgeryIntervention(seed=seed, **kwargs)
    elif intervention_type == 'noise':
        return NoiseIntervention(seed=seed, obs_format=obs_format, **kwargs)
    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")
