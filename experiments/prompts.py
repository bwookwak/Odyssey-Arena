"""
Prompt builder for LLM agents.

Constructs prompts with task description, observations, and memory context.
"""

from typing import List, Dict, Any, Optional


class PromptBuilder:
    """
    Builds prompts for LightEnv tasks with optional memory context.
    """
    
    def __init__(self, env_type: str = "light"):
        """
        Args:
            env_type: Type of environment ('light', 'energy', etc.)
        """
        self.env_type = env_type
        
        # Task-specific instructions
        self.task_instructions = {
            "light": """You are solving a light bulb puzzle. Your goal is to turn on all bulbs.

- Observation format: A string of 0s and 1s (e.g., "010101"), where 1 means ON and 0 means OFF.
- Action: Choose a bulb index to toggle (integer from 0 to num_bulbs-1).
- Rules: Each bulb can only be toggled if certain conditions are met based on other bulbs' states.
- Strategy: Experiment to learn which actions work in which states, then use that knowledge.

You must respond with ONLY a single integer (the action index). Do not include any explanation or other text.
"""
        }
    
    def build_prompt(
        self,
        observation: str,
        num_actions: int,
        step: int,
        memory_context: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build prompt for current step.
        
        Args:
            observation: Current observation string
            num_actions: Number of valid actions
            step: Current step number
            memory_context: Optional memory context to include
            history: Optional list of recent (obs, action, result) tuples
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Task instructions
        prompt_parts.append(self.task_instructions[self.env_type])
        prompt_parts.append("")
        
        # Memory context (if available)
        if memory_context:
            prompt_parts.append("=== LEARNED KNOWLEDGE ===")
            prompt_parts.append(memory_context)
            prompt_parts.append("")
        
        # Recent history (optional, last 3 steps)
        if history and len(history) > 0:
            prompt_parts.append("=== RECENT HISTORY ===")
            for i, h in enumerate(history[-3:]):
                prompt_parts.append(
                    f"Step {h['step']}: State={h['obs']}, Action={h['action']} -> {h['result']}"
                )
            prompt_parts.append("")
        
        # Current situation
        prompt_parts.append("=== CURRENT SITUATION ===")
        prompt_parts.append(f"Step: {step}")
        prompt_parts.append(f"Current state: {observation}")
        prompt_parts.append(f"Valid actions: 0 to {num_actions - 1}")
        prompt_parts.append("")
        prompt_parts.append("Your action (single integer only):")
        
        return "\n".join(prompt_parts)
    
    def parse_action(self, llm_output: str, num_actions: int) -> Optional[int]:
        """
        Parse LLM output to extract action integer.
        
        Args:
            llm_output: Raw LLM output text
            num_actions: Number of valid actions
            
        Returns:
            Parsed action integer, or None if invalid
        """
        import re
        
        # Try to extract first integer from output
        matches = re.findall(r'\b\d+\b', llm_output)
        if not matches:
            return None
        
        try:
            action = int(matches[0])
            if 0 <= action < num_actions:
                return action
            else:
                return None
        except (ValueError, IndexError):
            return None


def format_memory_context(hypotheses: List[Dict[str, Any]], max_items: int = 10) -> str:
    """
    Format memory hypotheses into readable context string.
    
    Args:
        hypotheses: List of hypothesis dicts with keys:
            - 'statement': hypothesis text
            - 'confidence': confidence score [0, 1]
            - 'support': support count
            - 'contradict': contradiction count
        max_items: Maximum number of items to include
        
    Returns:
        Formatted context string
    """
    if not hypotheses:
        return ""
    
    lines = []
    for i, hyp in enumerate(hypotheses[:max_items], 1):
        conf = hyp.get('confidence', 0.0)
        statement = hyp.get('statement', '')
        lines.append(f"{i}. {statement} (confidence: {conf:.2f})")
    
    return "\n".join(lines)
