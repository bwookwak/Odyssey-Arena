"""
Agent implementations for Odyssey-Arena tasks.

Implements baseline agents:
- LLMAgent: Uses language model to generate actions
- RandomAgent: Random valid actions
- HeuristicAgent: Simple rule-based agent (toggle first 0-bit)
"""

from typing import Optional, Dict, Any, List
import asyncio
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

    async def async_select_action(
        self,
        obs: str,
        num_actions: int,
        step: int,
        memory_context: Optional[str] = None,
        **kwargs
    ) -> tuple:
        """Async version of select_action. Default: runs sync version in thread."""
        return await asyncio.to_thread(
            self.select_action, obs, num_actions, step, memory_context, **kwargs
        )


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
        max_history: Optional[int] = None,
        prompt_template: str = "research",
        think_in_history: str = "last",
        latent_rule_guide: bool = True,
        include_raw_output: bool = False,
        parse_action_retries: int = 10,
    ):
        """
        Args:
            llm_client: LLM client for inference
            env_type: Environment type ('light', etc.)
            seed: Random seed
            max_history: Number of recent steps to include in prompt (None = all history)
            prompt_template: Prompt template ('research' or 'original')
            think_in_history: How to include model reasoning in history context.
                'all'  - every step's think is shown
                'last' - only the most recent step's think is shown (default)
                'none' - think is never shown in history
            latent_rule_guide: Whether to include the Latent Rule Guide section in prompts (default: True)
            include_raw_output: Whether to include the previous step's full raw output
                                (think text + action tag) in the prompt. Default: False.
            parse_action_retries: Max number of LLM call retries when action parsing fails.
                                  If all retries are exhausted, returns (None, ...) so the
                                  caller can abort the episode. Default: 10.
        """
        super().__init__(seed)
        self.llm_client = llm_client
        self.env_type = env_type
        self.name = "llm"
        self.max_history = max_history
        self.prompt_template = prompt_template
        self.think_in_history = think_in_history  # 'all' | 'last' | 'none'
        self.parse_action_retries = parse_action_retries

        # Prompt builder
        self.prompt_builder = PromptBuilder(
            env_type=env_type,
            template=prompt_template,
            latent_rule_guide=latent_rule_guide,
            include_raw_output=include_raw_output,
        )
        
        # Episode history for context
        self.history = []
    
    def reset_episode(self):
        """Reset episode-specific state."""
        self.history = []
    
    def _build_prompt_for_step(
        self,
        obs: str,
        num_actions: int,
        step: int,
        memory_context: Optional[str],
    ) -> str:
        """Build the prompt for the current step (extracted for async reuse)."""
        if self.history:
            history_window = (
                self.history[-self.max_history:]
                if self.max_history is not None
                else list(self.history)
            )
            if self.think_in_history == 'none':
                history_window = [
                    {k: v for k, v in h.items() if k != 'think'}
                    for h in history_window
                ]
            elif self.think_in_history == 'last':
                history_window = [
                    h if i == len(history_window) - 1
                    else {k: v for k, v in h.items() if k != 'think'}
                    for i, h in enumerate(history_window)
                ]
            # 'all': keep as-is
        else:
            history_window = None
        return self.prompt_builder.build_prompt(
            observation=obs,
            num_actions=num_actions,
            step=step,
            memory_context=memory_context,
            history=history_window,
        )

    def select_action(
        self,
        obs: str,
        num_actions: int,
        step: int,
        memory_context: Optional[str] = None,
        **kwargs
    ) -> tuple:
        """
        Select action using LLM with automatic retry.

        Retries up to self.parse_action_retries times on:
          - LLM API/network errors
          - Empty or unparseable LLM responses
        If all retries are exhausted, returns (None, last_raw_output, {...})
        so the caller can abort the episode.

        Returns:
            3-tuple: (action_or_None, raw_output, {"model_input": ..., "think": ...})
        """
        prompt = self._build_prompt_for_step(obs, num_actions, step, memory_context)
        raw_output = ""
        think_text = ""

        for attempt in range(1, self.parse_action_retries + 1):
            # ── LLM call ────────────────────────────────────────────────
            try:
                raw_output = self.llm_client.generate(prompt)
            except Exception as e:
                print(f"  [LLM_RETRY {attempt}/{self.parse_action_retries}] "
                      f"API/network error: {e}")
                continue  # retry

            # ── Empty response ────────────────────────────────────────
            if not raw_output or not raw_output.strip():
                print(f"  [LLM_RETRY {attempt}/{self.parse_action_retries}] "
                      f"Empty response received.")
                continue  # retry

            # ── Parse ─────────────────────────────────────────────────
            think_text, action = self.prompt_builder.parse_think_and_action(raw_output, num_actions)
            if action is not None:
                return action, raw_output, {"model_input": prompt, "think": think_text}

            print(f"  [LLM_RETRY {attempt}/{self.parse_action_retries}] "
                  f"Could not parse action from: {raw_output[:120]!r}")

        # All retries exhausted — signal failure to caller (no fallback)
        return None, raw_output, {"model_input": prompt, "think": think_text}

    async def async_select_action(
        self,
        obs: str,
        num_actions: int,
        step: int,
        memory_context: Optional[str] = None,
        **kwargs
    ) -> tuple:
        """
        Async version of select_action with automatic retry.

        Retries up to self.parse_action_retries times on:
          - LLM API/network errors
          - Empty or unparseable LLM responses
        Returns (None, last_raw_output, {...}) when all retries are exhausted.
        """
        prompt = self._build_prompt_for_step(obs, num_actions, step, memory_context)
        raw_output = ""
        think_text = ""

        for attempt in range(1, self.parse_action_retries + 1):
            # ── LLM call ────────────────────────────────────────────────
            try:
                if hasattr(self.llm_client, 'async_generate'):
                    raw_output = await self.llm_client.async_generate(prompt)
                else:
                    raw_output = await asyncio.to_thread(self.llm_client.generate, prompt)
            except Exception as e:
                print(f"  [LLM_RETRY {attempt}/{self.parse_action_retries}] "
                      f"API/network error: {e}")
                continue  # retry

            # ── Empty response ────────────────────────────────────────
            if not raw_output or not raw_output.strip():
                print(f"  [LLM_RETRY {attempt}/{self.parse_action_retries}] "
                      f"Empty response received.")
                continue  # retry

            # ── Parse ─────────────────────────────────────────────────
            think_text, action = self.prompt_builder.parse_think_and_action(raw_output, num_actions)
            if action is not None:
                return action, raw_output, {"model_input": prompt, "think": think_text}

            print(f"  [LLM_RETRY {attempt}/{self.parse_action_retries}] "
                  f"Could not parse action from: {raw_output[:120]!r}")

        # All retries exhausted — signal failure to caller (no fallback)
        return None, raw_output, {"model_input": prompt, "think": think_text}
    
    def record_step(
        self,
        step: int,
        obs: str,
        action: int,
        obs_after: str,
        feedback: str,
        done: bool,
        think: str = "",
        invalid_output: str = "",
        raw_output: str = "",
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
            think: Model reasoning text (included in history if include_think_in_history=True)
            invalid_output: Raw LLM output that failed to parse (non-empty when action was invalid)
            raw_output: Full raw LLM output (think + action tag, the complete response)
        """
        result = "Success!" if done else feedback
        entry: Dict[str, Any] = {
            'step': step,
            'obs': obs,
            'action': action,
            'obs_after': obs_after,
            'result': result,
        }
        if think:
            entry['think'] = think  # always stored internally; filtering done at prompt-build time
        if invalid_output:
            entry['invalid_output'] = invalid_output
        if raw_output:
            entry['raw_output'] = raw_output
        self.history.append(entry)


def create_agent(
    agent_type: str,
    llm_client=None,
    env_type: str = "light",
    seed: Optional[int] = None,
    prompt_template: str = "research",
    max_history: Optional[int] = None,
    think_in_history: str = "last",
    latent_rule_guide: bool = True,
    **kwargs
) -> Agent:
    """
    Factory function to create agent.

    Args:
        agent_type: Type of agent ('llm', 'random', 'heuristic')
        llm_client: LLM client (required for LLMAgent)
        env_type: Environment type
        seed: Random seed
        prompt_template: Prompt template ('research' or 'original')
        max_history: Max history steps in prompt (None = all)
        think_in_history: How to include model reasoning in history context
        latent_rule_guide: Whether to include the Latent Rule Guide in prompts (default: True)
        **kwargs: Additional arguments for agent

    Returns:
        Agent instance
    """
    if agent_type == 'llm':
        if llm_client is None:
            raise ValueError("LLMAgent requires llm_client")
        return LLMAgent(
            llm_client, env_type=env_type, seed=seed,
            prompt_template=prompt_template,
            max_history=max_history,
            think_in_history=think_in_history,
            latent_rule_guide=latent_rule_guide,
            **kwargs
        )
    elif agent_type == 'random':
        return RandomAgent(seed=seed)
    elif agent_type == 'heuristic':
        return HeuristicAgent(seed=seed)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
