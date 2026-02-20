"""
Prompt builder for LLM agents.

Constructs prompts with task description, observations, and memory context.
"""

from typing import List, Dict, Any, Optional


LATENT_RULE_GUIDE = {
    "original": """\
### Latent Rule Guide:
Each bulb has a hidden boolean condition. Key patterns and strategies:
- Always available (True)     : Toggle anytime, no restriction.
- Requires B_x ON  (B_x)     : Turn B_x on first, then toggle this bulb.
- Requires B_x OFF (not B_x) : Handle THIS bulb BEFORE turning B_x on.
                                At the start (all OFF), "not B_x" is already satisfied.
- Requires all ON  (A and B) : Turn all dependencies on first.
- Requires any ON  (A or B)  : Turn any one dependency on first.
If a toggle fails ("remains inactive"), the condition is not yet satisfied — try other bulbs first.\
""",
    "research": """\
Latent Rule Guide:
  True      → Toggle anytime, no restriction.
  B_x       → Turn B_x ON first.
  not B_x   → Handle BEFORE turning B_x on (at start all=OFF, so "not B_x" is already True).
  A and B   → Turn ALL dependencies on first.
  A or B    → Turn ANY ONE dependency on first.
If a toggle fails ("remains inactive"), the condition is not yet satisfied — try other bulbs first.\
""",
}


class PromptBuilder:
    """
    Builds prompts for LightEnv tasks with optional memory context.
    
    Supports multiple prompt templates for compatibility:
    - 'research': Concise format optimized for memory experiments
    - 'original': Original Odyssey-Arena format (compatible with existing baselines)
    """
    
    def __init__(self, env_type: str = "light", template: str = "original",
                 latent_rule_guide: bool = True, include_raw_output: bool = False):
        """
        Args:
            env_type: Type of environment ('light', 'energy', etc.)
            template: Prompt template ('research' or 'original')
            latent_rule_guide: Whether to include the Latent Rule Guide section in the prompt (default: True)
            include_raw_output: Whether to include the previous step's full raw output
                                (think text + action tag) in the prompt. Default: False.
        """
        self.env_type = env_type
        self.template = template
        self.latent_rule_guide = latent_rule_guide
        self.include_raw_output = include_raw_output
        # goal_state_str: 에피소드 시작 시 run.py에서 설정. None이면 "all ON" 설명 사용.
        self.goal_state_str: Optional[str] = None
        
        # Select template
        if template == "original":
            self.task_instructions = self._task_instructions_original
        else:
            self.task_instructions = self._task_instructions_research

    def _task_instructions_research(self, env_type: str) -> str:
        if env_type == "light":
            goal_desc = (
                f'reach the target state "{self.goal_state_str}" '
                f'(1=ON, 0=OFF)'
                if self.goal_state_str is not None
                else "turn on all bulbs"
            )
            return (
                f"You are solving a light bulb puzzle. Your goal is to {goal_desc}.\n\n"
                "- Observation format: A string of 0s and 1s (e.g., \"010101\"), where 1 means ON and 0 means OFF.\n"
                "- Action: Choose a bulb index to toggle (integer from 0 to num_bulbs-1).\n"
                "- Rules: Each bulb can only be toggled if certain conditions are met based on other bulbs' states.\n"
                "- Strategy: Experiment to learn which actions work in which states, then use that knowledge.\n"
            )
        return ""

    def _task_instructions_original(self, env_type: str) -> str:
        if env_type == "light":
            if self.goal_state_str is not None:
                goal_desc = (
                    f"Your mission is to reach the target bulb state: \"{self.goal_state_str}\" "
                    f"(1=ON, 0=OFF).\n"
                    "However, the accessibility of the bulbs is based on the current condition of other bulbs.\n"
                    "You need to learn the hidden rule behind the environment and complete the task."
                )
            else:
                goal_desc = (
                    "Your mission is to light on all the bulbs.\n"
                    "However, the accessibility of the bulbs is based on the current condition of other bulbs.\n"
                    "You need to learn the hidden rule behind the environment and complete the task."
                )
            return (
                "You are an intelligent agent.\n\n"
                f"### Goal:\n{goal_desc}\n\n"
                "### Action Space:\n"
                "The action space is based on the index of bulbs. For example, you would like to light on / off the first bulb, you should output <action>0</action> to toggle the state of the bulb.\n"
            )
        return ""
    
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
        
        # Task instructions (callable로 변경됨)
        prompt_parts.append(self.task_instructions(self.env_type))
        prompt_parts.append("")

        # Latent Rule Guide (optional)
        if self.latent_rule_guide and self.env_type in LATENT_RULE_GUIDE:
            prompt_parts.append(LATENT_RULE_GUIDE["research"])
            prompt_parts.append("")
        
        # Memory context (if available)
        if memory_context:
            prompt_parts.append("=== LEARNED KNOWLEDGE ===")
            prompt_parts.append(memory_context)
            prompt_parts.append("")
        
        # Recent history (optional)
        if history and len(history) > 0:
            prompt_parts.append("=== RECENT HISTORY ===")
            for h in history:
                think = h.get('think', '').strip()
                if think:
                    prompt_parts.append(f"[Reasoning] {think}")
                if h.get('invalid_output'):
                    inv_preview = h['invalid_output'][:120].replace('\n', ' ')
                    line = (f"Step {h['step']}: State={h['obs']}, "
                            f"Action=INVALID (parse failed, random fallback={h['action']}) -> {h['result']}"
                            f"\n  [UNPARSEABLE OUTPUT]: \"{inv_preview}\"")
                else:
                    line = f"Step {h['step']}: State={h['obs']}, Action={h['action']} -> {h['result']}"
                prompt_parts.append(line)
            prompt_parts.append("")

            # Previous step full output (opt-in via include_raw_output, default off)
            if self.include_raw_output:
                last = history[-1]
                if last.get('raw_output'):
                    prompt_parts.append("=== PREVIOUS STEP FULL OUTPUT ===")
                    prompt_parts.append(last['raw_output'])
                    prompt_parts.append("")

        # Current situation
        prompt_parts.append("=== CURRENT SITUATION ===")
        prompt_parts.append(f"Step: {step}")
        prompt_parts.append(f"Current state: {observation}")
        prompt_parts.append(f"Valid actions: 0 to {num_actions - 1}")
        prompt_parts.append("")
        prompt_parts.append("Think step by step about the best action, then output your action in the format: <action>n</action>.\n")
        
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
        
        # Task instructions (callable로 변경됨)
        prompt_parts.append(self.task_instructions(self.env_type))
        prompt_parts.append("")

        # Latent Rule Guide (optional)
        if self.latent_rule_guide and self.env_type in LATENT_RULE_GUIDE:
            prompt_parts.append(LATENT_RULE_GUIDE["original"])
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
                think = h.get('think', '').strip()
                if think:
                    prompt_parts.append(f"[Reasoning] {think}")
                if h.get('invalid_output'):
                    inv_preview = h['invalid_output'][:120].replace('\n', ' ')
                    prompt_parts.append(
                        f"Action: INVALID (parse failed, random fallback={h['action']}), "
                        f"Feedback: {h['result']}, State: {h['obs']}"
                        f"\n  [UNPARSEABLE OUTPUT]: \"{inv_preview}\""
                    )
                else:
                    prompt_parts.append(
                        f"Action: {h['action']}, Feedback: {h['result']}, State: {h['obs']}"
                    )
            prompt_parts.append("")

            # Previous step full output (opt-in via include_raw_output, default off)
            if self.include_raw_output:
                last = history[-1]
                if last.get('raw_output'):
                    prompt_parts.append("### Previous Step Full Output:")
                    prompt_parts.append(last['raw_output'])
                    prompt_parts.append("")
        else:
            prompt_parts.append("(No history yet)")
        prompt_parts.append("")
        
        # Current State
        prompt_parts.append("### Current State:")
        prompt_parts.append(observation)
        prompt_parts.append("")
        
        # Instructions
        prompt_parts.append("Now THINK step by step and choose the next ACTION to act in the environment.")
        prompt_parts.append("You are encouraged to act actively to derive the environment dynamics.")
        prompt_parts.append("To act, output ONLY one action in the format: <action>n</action>. You should strictly follow the format to act.")
        
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
        
        # Auto-detect format: both templates now use <action> xml tags
        if output_format is None:
            output_format = 'xml'
        
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


    def parse_think_and_action(self, llm_output: str, num_actions: int) -> tuple:
        """
        Parse LLM output into (think_text, action_int).

        Splits on the first <action>n</action> tag:
        - think_text: everything before the tag (stripped)
        - action_int: parsed integer, or None if invalid/missing
        Falls back to integer extraction if no tag found.
        """
        import re
        m = re.search(r"<action>(.*?)</action>", llm_output, re.IGNORECASE | re.DOTALL)
        if m:
            think_text = llm_output[:m.start()].strip()
            action_str = m.group(1).strip()
            try:
                action_int = int(action_str)
                if not (0 <= action_int < num_actions):
                    action_int = None
            except (ValueError, IndexError):
                action_int = None
            return think_text, action_int
        # Fallback: no tag found — treat entire output as think, try to find an integer
        matches = re.findall(r'\b\d+\b', llm_output)
        for m_str in matches:
            try:
                a = int(m_str)
                if 0 <= a < num_actions:
                    return llm_output.strip(), a
            except ValueError:
                continue
        return llm_output.strip(), None


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
