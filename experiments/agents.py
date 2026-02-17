"""
Agent implementations for Odyssey-Arena tasks.

Implements baseline agents:
- LLMAgent: Uses language model to generate actions
- RandomAgent: Random valid actions
- HeuristicAgent: Simple rule-based agent (toggle first 0-bit)
"""

from typing import Optional, Dict, Any, List
import random
from .prompts import PromptBuilder


class Agent:
    """Base class for agents."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.name = "base"
    
    def select_action(
        self,
        obs: str,
        num_actions: int,
        step: int,
        memory_context: Optional[str] = None,
        **kwargs
    ) -> tuple:
        """
        Select action given observation.
        
        Returns:
            (action, raw_output) tuple
            - action: integer action or None if invalid
            - raw_output: raw output from agent (for logging)
        """
        raise NotImplementedError


class RandomAgent(Agent):
    """Random agent - selects actions uniformly at random."""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.name = "random"
    
    def select_action(
        self,
        obs: str,
        num_actions: int,
        step: int,
        memory_context: Optional[str] = None,
        **kwargs
    ) -> tuple:
        """Select random action."""
        action = self.rng.randint(0, num_actions - 1)
        return action, f"random:{action}"


class HeuristicAgent(Agent):
    """
    Heuristic agent - toggle first 0-bit (first OFF bulb).
    
    Simple rule: scan from left to right, toggle first bulb that is OFF.
    If all bulbs are ON, toggle bulb 0.
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.name = "heuristic"
    
    def select_action(
        self,
        obs: str,
        num_actions: int,
        step: int,
        memory_context: Optional[str] = None,
        **kwargs
    ) -> tuple:
        """Select action by toggling first 0-bit."""
        # Find first 0-bit
        for i, bit in enumerate(obs):
            if bit == '0':
                return i, f"heuristic:toggle_first_zero->bulb_{i}"
        
        # If all bits are 1 (shouldn't happen in normal gameplay), toggle 0
        return 0, "heuristic:all_on->bulb_0"


class LLMAgent(Agent):
    """
    LLM agent - uses language model to select actions.
    
    Constructs prompts with task description, memory context, and history.
    Parses LLM output to extract action integer.
    """
    
    def __init__(
        self,
        llm_client,
        env_type: str = "light",
        seed: Optional[int] = None,
        max_history: int = 3
    ):
        """
        Args:
            llm_client: LLM client for inference
            env_type: Environment type ('light', etc.)
            seed: Random seed
            max_history: Number of recent steps to include in prompt
        """
        super().__init__(seed)
        self.llm_client = llm_client
        self.env_type = env_type
        self.name = "llm"
        self.max_history = max_history
        
        # Prompt builder
        self.prompt_builder = PromptBuilder(env_type=env_type)
        
        # Episode history for context
        self.history = []
    
    def reset_episode(self):
        """Reset episode-specific state."""
        self.history = []
    
    def select_action(
        self,
        obs: str,
        num_actions: int,
        step: int,
        memory_context: Optional[str] = None,
        **kwargs
    ) -> tuple:
        """
        Select action using LLM.
        
        Args:
            obs: Current observation
            num_actions: Number of valid actions
            step: Current step number
            memory_context: Optional memory context string
            **kwargs: Additional arguments (unused)
            
        Returns:
            (action, raw_output) tuple
        """
        # Build prompt
        prompt = self.prompt_builder.build_prompt(
            observation=obs,
            num_actions=num_actions,
            step=step,
            memory_context=memory_context,
            history=self.history[-self.max_history:] if self.history else None
        )
        
        # Generate with LLM
        try:
            raw_output = self.llm_client.generate(prompt)
        except Exception as e:
            print(f"LLM generation error: {e}")
            # Fallback to random action
            action = self.rng.randint(0, num_actions - 1)
            return action, f"error:fallback_random:{action}"
        
        # Parse action
        action = self.prompt_builder.parse_action(raw_output, num_actions)
        
        return action, raw_output
    
    def record_step(
        self,
        step: int,
        obs: str,
        action: int,
        obs_after: str,
        feedback: str,
        done: bool
    ):
        """
        Record step in history for future context.
        
        Args:
            step: Step number
            obs: Observation before action
            action: Action taken
            obs_after: Observation after action
            feedback: Feedback from environment
            done: Whether episode ended
        """
        result = "Success!" if done else feedback
        self.history.append({
            'step': step,
            'obs': obs,
            'action': action,
            'obs_after': obs_after,
            'result': result
        })


def create_agent(
    agent_type: str,
    llm_client=None,
    env_type: str = "light",
    seed: Optional[int] = None,
    **kwargs
) -> Agent:
    """
    Factory function to create agent.
    
    Args:
        agent_type: Type of agent ('llm', 'random', 'heuristic')
        llm_client: LLM client (required for LLMAgent)
        env_type: Environment type
        seed: Random seed
        **kwargs: Additional arguments for agent
        
    Returns:
        Agent instance
    """
    if agent_type == 'llm':
        if llm_client is None:
            raise ValueError("LLMAgent requires llm_client")
        return LLMAgent(llm_client, env_type=env_type, seed=seed, **kwargs)
    elif agent_type == 'random':
        return RandomAgent(seed=seed)
    elif agent_type == 'heuristic':
        return HeuristicAgent(seed=seed)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
