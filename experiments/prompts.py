"""
Prompt builder for LLM agents.

Constructs prompts with task description, observations, and memory context.
"""

from typing import List, Dict, Any, Optional


class PromptBuilder:
    """
    Builds prompts for LightEnv tasks with optional memory context.
    
    Supports multiple prompt templates for compatibility:
    - 'research': Concise format optimized for memory experiments
    - 'original': Original Odyssey-Arena format (compatible with existing baselines)
    """
    
    def __init__(self, env_type: str = "light", template: str = "research"):
        """
        Args:
            env_type: Type of environment ('light', 'energy', etc.)
            template: Prompt template ('research' or 'original')
        """
        self.env_type = env_type
        self.template = template
        
        # Task-specific instructions - Research template (default)
        self.task_instructions_research = {
            "light": """You are solving a light bulb puzzle. Your goal is to turn on all bulbs.

- Observation format: A string of 0s and 1s (e.g., "010101"), where 1 means ON and 0 means OFF.
- Action: Choose a bulb index to toggle (integer from 0 to num_bulbs-1).
- Rules: Each bulb can only be toggled if certain conditions are met based on other bulbs' states.
- Strategy: Experiment to learn which actions work in which states, then use that knowledge.

You must respond with ONLY a single integer (the action index). Do not include any explanation or other text.
"""
        }
        
        # Task-specific instructions - Original Odyssey-Arena template
        self.task_instructions_original = {
            "light": """You are an intelligent agent.

### Goal:
Your mission is to light on all the bulbs.
However, the accessibility of the bulbs is based on the current condition of other bulbs.
You need to learn the hidden rule behind the environment and complete the task.

### Action Space:
The action space is based on the index of bulbs. For example, you would like to light on / off the first bulb, you should output <action>0</action> to toggle the state of the bulb.
"""
        }
        
        # Select template
        if template == "original":
            self.task_instructions = self.task_instructions_original
        else:
            self.task_instructions = self.task_instructions_research
    
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
        if self.template == "original":
            return self._build_prompt_original(observation, num_actions, step, 
                                               memory_context, history)
        else:
            return self._build_prompt_research(observation, num_actions, step, 
                                               memory_context, history)
    
    def _build_prompt_research(
        self,
        observation: str,
        num_actions: int,
        step: int,
        memory_context: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Build research-style prompt (concise, structured)."""
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
    
    def _build_prompt_original(
        self,
        observation: str,
        num_actions: int,
        step: int,
        memory_context: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Build original Odyssey-Arena style prompt (compatible with baselines)."""
        prompt_parts = []
        
        # Task instructions
        prompt_parts.append(self.task_instructions[self.env_type])
        prompt_parts.append("")
        
        # Learned Knowledge (if available) - inserted after goal
        if memory_context:
            prompt_parts.append("### Learned Knowledge:")
            prompt_parts.append(memory_context)
            prompt_parts.append("")
        
        # History Action and Feedback
        prompt_parts.append("### History Action and Feedback:")
        if history and len(history) > 0:
            for h in history:
                # Original format: "Action: {action}, Feedback: {feedback}, State: {obs}"
                prompt_parts.append(
                    f"Action: {h['action']}, Feedback: {h['result']}, State: {h['obs']}"
                )
        else:
            prompt_parts.append("(No history yet)")
        prompt_parts.append("")
        
        # Current State
        prompt_parts.append("### Current State:")
        prompt_parts.append(observation)
        prompt_parts.append("")
        
        # Instructions
        prompt_parts.append("Now think step by step and choose the next action to act in the environment.")
        prompt_parts.append("You are encouraged to act actively to derive the environment dynamics.")
        prompt_parts.append("Output ONLY one action in the format: <action>n</action>")
        
        return "\n".join(prompt_parts)
    
    def parse_action(self, llm_output: str, num_actions: int, 
                    output_format: str = None) -> Optional[int]:
        """
        Parse LLM output to extract action integer.
        
        Supports two formats:
        - 'xml': Extracts from <action>n</action> tags (original Odyssey-Arena)
        - 'integer': Extracts first integer (research format)
        
        Args:
            llm_output: Raw LLM output text
            num_actions: Number of valid actions
            output_format: Output format ('xml' or 'integer'). 
                          If None, auto-detects based on template.
            
        Returns:
            Parsed action integer, or None if invalid
        """
        import re
        
        # Auto-detect format based on template if not specified
        if output_format is None:
            output_format = 'xml' if self.template == 'original' else 'integer'
        
        action_str = None
        
        if output_format == 'xml':
            # Extract from <action>n</action> tags
            m = re.search(r"<action>(.*?)</action>", llm_output, re.IGNORECASE | re.DOTALL)
            if m:
                action_str = m.group(1).strip()
        else:
            # Extract first integer
            matches = re.findall(r'\b\d+\b', llm_output)
            if matches:
                action_str = matches[0]
        
        # Parse and validate
        if action_str is None:
            return None
        
        try:
            action = int(action_str)
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
