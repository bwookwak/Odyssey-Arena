"""
Domain adapters for environment-specific memory and verification logic.

Decouples reflection prompts, oracle verification, and hypothesis parsing
from the Light (bulb) environment so other envs can plug in their own adapters.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple
import re


class DomainAdapter(ABC):
    """
    Abstract adapter for environment-specific behavior in memory systems.
    
    Used for: reflection prompts, curation format, oracle verification,
    and parsing hypothesis text into structured form for verification.
    """
    
    def task_description(self) -> str:
        """Short description of the task, used as context in reflection prompts."""
        return "Reach the goal state through sequential actions."

    def build_midep_reflection_prompt(self, raw_experiences: list) -> str:
        """
        Build the mid-episode (every-k-steps) reflector prompt.

        Includes: reasoning trace, last predicted action, env feedback, recent trajectory.
        Excludes: ground truth, bullet_tags.
        Output format: JSON with reasoning/error_identification/root_cause_analysis/
                        correct_approach/key_insight.

        Default implementation uses the old plain-text format via reflection_instructions().
        Override in concrete domains for the structured JSON format.
        """
        exp_lines = []
        for i, exp in enumerate(raw_experiences, 1):
            obs_change = f"{exp['obs_before']} -> {exp['obs_after']}"
            exp_lines.append(f"{i}. Action {exp['action']}: {obs_change}")
        experiences_text = "\n".join(exp_lines)
        instructions = self.reflection_instructions()
        return (
            "You are the Reflector: extract concrete insights from trajectory.\n\n"
            f"Recent experiences:\n{experiences_text}\n\n{instructions}"
        )

    def build_episode_reflection_prompt(
        self, trajectory: list, success: bool, task_context: str = ""
    ) -> str:
        """
        Build the post-episode reflector prompt.

        Includes: env feedback, full trajectory, task outcome (after trajectory).
        Excludes: reasoning trace, predicted answer, ground truth, bullet_tags.
        Output format: JSON with reasoning/error_identification/root_cause_analysis/
                        correct_approach/key_insights (list, 2-5 items).

        Default implementation produces a minimal prompt.
        Override in concrete domains for richer format.
        """
        traj_lines = [
            f"{i+1}. before={s['obs_before']} action={s['action']} after={s['obs_after']}"
            for i, s in enumerate(trajectory)
        ]
        outcome = "SUCCESS" if success else "FAILURE"
        ctx = task_context or self.task_description()
        return (
            f"Task: {ctx}\n\n"
            f"Trajectory:\n" + "\n".join(traj_lines) + "\n\n"
            f"Outcome: {outcome}\n\n"
            'Output JSON: {"key_insights": ["insight 1", "insight 2", ...]}'
        )

    @abstractmethod
    def reflection_instructions(self) -> str:
        """
        Instructions and format examples for the Reflector (what to extract from experiences).
        Used as fallback in the default build_midep_reflection_prompt().
        """
        pass
    
    @abstractmethod
    def curation_format_hint(self) -> str:
        """Hint for Curator on how to format new insights (for parsability)."""
        pass
    
    @abstractmethod
    def build_oracle_verification_prompt(self, hypothesis_text: str, ground_truth: Any) -> str:
        """
        Build LLM prompt to verify one hypothesis against ground truth.
        ground_truth is environment-specific (e.g. custom_logic dict for light).
        """
        pass
    
    @abstractmethod
    def parse_dependency(self, hypothesis_text: str) -> Optional[Tuple[int, int]]:
        """
        Parse hypothesis text into (subject_index, object_index) for oracle verification.
        E.g. light: (action_bulb, affected_bulb). Returns None if unparseable.
        """
        pass
    
    def verify_dependency_against_ground_truth(
        self, dependency: Tuple[int, int], ground_truth: Any
    ) -> Optional[bool]:
        """
        Verify a parsed dependency against ground truth without LLM.
        Returns True=confirmed, False=contradicted, None=unable to verify.
        Default: return None (use LLM or skip).
        """
        return None
    
    @abstractmethod
    def parse_effects_from_text(self, hypothesis_text: str, action: int) -> Optional[Set[int]]:
        """
        Parse which entity indices are predicted to change for the given action.
        E.g. light: set of bulb indices that flip. Returns None if hypothesis doesn't mention action.
        """
        pass
    
    @abstractmethod
    def get_actual_effects(
        self, obs_before: str, obs_after: str, obs_format: str
    ) -> Set[int]:
        """
        Return set of indices that actually changed between obs_before and obs_after.
        E.g. light: indices where bit flipped. obs_format is 'bitstring' or 'emoji'.
        """
        pass
    
    def get_injection_false_patterns(self) -> List[Tuple[int, int]]:
        """
        Optional: (action, target) pairs for injection intervention. Default: empty (intervention may use its own).
        """
        return []
    
    def format_injection_hypothesis(self, action: int, target: int) -> str:
        """
        Optional: format a single false hypothesis for injection. Default: generic "action X affects Y".
        """
        return f"Action {action} affects {target}"


class LightDomain(DomainAdapter):
    """
    Domain adapter for Light (light bulb) environment.
    
    - Ground truth: custom_logic Dict[str, str] e.g. {"B0": "True", "B1": "B0"}
    - Hypotheses: "Toggling bulb X affects bulb Y" style
    """

    def task_description(self) -> str:
        return (
            "Light Bulb Puzzle: toggle bulbs to reach the target state. "
            "Each action is a bulb index (0-based integer). "
            "Bulbs have hidden toggle conditions — a bulb can only be toggled "
            "when its hidden condition (which depends on other bulbs' states) is met. "
            "Discover these rules through experimentation."
        )

    # ------------------------------------------------------------------ #
    #  Mid-episode (k-step) reflector prompt                              #
    # ------------------------------------------------------------------ #

    def build_midep_reflection_prompt(self, raw_experiences: list) -> str:
        """
        K-step reflector prompt (called every reflection_frequency steps during episode).

        Includes : reasoning trace, model's last action (predicted answer), env feedback,
                   recent trajectory.
        Excludes : ground truth answer, bullet_tags.
        Output   : JSON with key_insight (singular).
        """
        # ── Trajectory ────────────────────────────────────────────────
        traj_lines = []
        for i, exp in enumerate(raw_experiences, 1):
            traj_lines.append(
                f"  {i}. before={exp['obs_before']}  action={exp['action']}  "
                f"after={exp['obs_after']}  feedback={exp.get('feedback', '')}"
            )
        trajectory_text = "\n".join(traj_lines) or "  (none)"

        # ── Reasoning trace (think text per step) ─────────────────────
        think_parts = []
        for i, exp in enumerate(raw_experiences, 1):
            think = (exp.get('think') or '').strip()
            if think:
                think_parts.append(f"  Step {i}: {think[:300]}")
        reasoning_trace = "\n".join(think_parts) or "  (not available)"

        # ── Predicted answer: last action and resulting state ──────────
        if raw_experiences:
            last = raw_experiences[-1]
            predicted_answer = (
                f"action={last['action']} → state={last['obs_after']}"
            )
        else:
            predicted_answer = "(none)"

        # ── Environment feedback ───────────────────────────────────────
        fb_lines = [exp.get('feedback', '') for exp in raw_experiences if exp.get('feedback')]
        env_feedback = "\n".join(f"  {fb}" for fb in fb_lines) or "  (none)"

        return f"""You are a Reflector.

Your job is to analyze recent trajectory data from a light bulb puzzle and extract a concrete insight about the hidden toggle rules.

Task Context:
  {self.task_description()}

Model's Reasoning Trace (recent steps):
{reasoning_trace}

Model's Last Action:
  {predicted_answer}

Environment Feedback:
{env_feedback}

RECENT TRAJECTORY:
{trajectory_text}

Answer in this exact JSON format:
{{
  "reasoning": "[Your analysis of which actions worked or failed and why]",
  "error_identification": "[What went wrong or was inefficient in the recent steps?]",
  "root_cause_analysis": "[Why? Which hidden rule or dependency was missed?]",
  "correct_approach": "[What should the model try next?]",
  "key_insight": "[One concrete, specific rule discovered, e.g. \\"Toggling bulb X only works when bulb Y is ON\\"]"
}}

Output only the JSON object. No additional text."""

    # ------------------------------------------------------------------ #
    #  Post-episode reflector prompt                                      #
    # ------------------------------------------------------------------ #

    def build_episode_reflection_prompt(
        self, trajectory: list, success: bool, task_context: str = ""
    ) -> str:
        """
        Post-episode reflector prompt (called once after episode ends).

        Includes : env feedback, full trajectory, task outcome (placed AFTER trajectory).
        Excludes : reasoning trace, predicted answer, ground truth answer, bullet_tags.
        Output   : JSON with key_insights (list, 2-5 items).
        """
        # ── Environment feedback (step-level messages) ─────────────────
        fb_lines = []
        for s in trajectory:
            fb = (s.get('feedback') or '').strip()
            if fb:
                fb_lines.append(f"  step {s['step']}: {fb}")
        env_feedback = "\n".join(fb_lines) or "  (none)"

        # ── Full trajectory ────────────────────────────────────────────
        traj_lines = []
        for s in trajectory:
            loop_flag = " [LOOP]" if s.get('is_loop') else ""
            traj_lines.append(
                f"  step={s['step']}  before={s['obs_before']}  "
                f"action={s['action']}  after={s['obs_after']}{loop_flag}"
            )
        trajectory_text = "\n".join(traj_lines) or "  (none)"

        # ── Task outcome (AFTER trajectory, as specified) ──────────────
        last = trajectory[-1] if trajectory else {}
        final_state = last.get('obs_after', '?')
        outcome = (
            f"SUCCESS — reached goal state (final: {final_state})"
            if success
            else f"FAILURE — did not reach goal within budget (final: {final_state})"
        )

        ctx = task_context or self.task_description()

        return f"""You are a Reflector.

Your job is to analyze a complete episode trajectory from a light bulb puzzle and extract multiple concrete insights about the hidden toggle rules.

Task Context:
  {ctx}

Environment Feedback:
{env_feedback}

FULL TRAJECTORY:
{trajectory_text}

Task Outcome:
  {outcome}

Answer in this exact JSON format:
{{
  "reasoning": "[Your analysis of patterns across the full episode — which sequences worked, which failed, and why]",
  "error_identification": "[What were the main inefficiencies or wrong moves?]",
  "root_cause_analysis": "[Which hidden rules or dependencies were misunderstood or unknown?]",
  "correct_approach": "[What sequence of actions would have been more efficient?]",
  "key_insights": [
    "[Concrete rule 1, e.g. \\"Toggling bulb X only succeeds when bulb Y is ON\\"]",
    "[Concrete rule 2, ...]",
    "[Concrete rule 3, ...]"
  ]
}}

Provide 2-5 key_insights. Each must be a specific, concrete rule about bulb toggle dependencies (e.g. which bulbs must be ON/OFF for an action to take effect).
Output only the JSON object. No additional text."""

    # ------------------------------------------------------------------ #
    #  Existing methods (unchanged)                                       #
    # ------------------------------------------------------------------ #

    def reflection_instructions(self) -> str:
        return """Analyze and distill concrete insights about which actions affect which bulbs.
Use EXPLICIT format: "Toggling bulb X affects bulb Y" or "Action X flips bulb Y".

Example good insights:
- "Toggling bulb 0 affects bulb 1"
- "Action 2 flips bulb 3 and bulb 4"
- "Bulb 5 depends on bulbs 2 and 3"

Provide 1-3 specific insights (one per line):"""
    
    def curation_format_hint(self) -> str:
        return 'Important: If adding, use format "Toggling bulb X affects bulb Y" for parsability.'
    
    def build_oracle_verification_prompt(self, hypothesis_text: str, ground_truth: Any) -> str:
        if not isinstance(ground_truth, dict):
            return ""
        rules_text = "\n".join([f"{k}: {v}" for k, v in sorted(ground_truth.items())])
        return f"""You are verifying a hypothesis against ground truth boolean rules.

GROUND TRUTH RULES:
{rules_text}

EXPLANATION:
- Each bulb can be toggled ONLY when its rule evaluates to True
- "B1: B0" means B1 can be toggled when B0 is ON
- "B2: not B0" means B2 can be toggled when B0 is OFF
- "B3: (B0 and B1)" means B3 can be toggled when both B0 and B1 are ON
- Therefore, toggling B0 AFFECTS B1, B2, and B3 (they depend on B0's state)

HYPOTHESIS TO VERIFY:
{hypothesis_text}

QUESTION:
Does toggling the mentioned action bulb affect the mentioned target bulb?
In other words: Does the target bulb's rule include the action bulb?

Example analysis:
- Hypothesis: "Toggling bulb 2 affects bulb 5"
- B5 rule: "not B2"
- Analysis: B5's rule CONTAINS B2 → YES, B5 depends on B2's state
- Answer: CONFIRMED

Answer with ONE word:
- "CONFIRMED" - if target bulb's rule mentions action bulb
- "CONTRADICTED" - if target bulb's rule does NOT mention action bulb

Your answer:"""
    
    def parse_dependency(self, hypothesis_text: str) -> Optional[Tuple[int, int]]:
        patterns = [
            r'(?:toggling|action)\s+(?:bulb\s+)?(\d+)\s+(?:affects?|flips?|influences?)\s+(?:bulb\s+)?(\d+)',
            r'bulb\s+(\d+)\s+(?:affects?|flips?)\s+bulb\s+(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, hypothesis_text, re.IGNORECASE)
            if match:
                return (int(match.group(1)), int(match.group(2)))
        return None
    
    def verify_dependency_against_ground_truth(
        self, dependency: Tuple[int, int], ground_truth: Any
    ) -> Optional[bool]:
        if not isinstance(ground_truth, dict):
            return None
        action_bulb, affected_bulb = dependency
        affected_rule = ground_truth.get(f"B{affected_bulb}", "")
        if not affected_rule:
            return None
        action_bulb_name = f"B{action_bulb}"
        return action_bulb_name in affected_rule
    
    def parse_effects_from_text(self, hypothesis_text: str, action: int) -> Optional[Set[int]]:
        action_patterns = [
            rf'\b(?:action|bulb|toggling)\s+{action}\b',
            rf'\b{action}\s+(?:affects?|flips?)',
        ]
        if not any(re.search(p, hypothesis_text, re.IGNORECASE) for p in action_patterns):
            return None
        affected = set()
        effect_patterns = [
            r'(?:affects?|flips?|influences?|toggles?)\s+bulb\s+(\d+)',
            r'(?:affects?|flips?|influences?)\s+(\d+)',
        ]
        for pattern in effect_patterns:
            for match in re.findall(pattern, hypothesis_text, re.IGNORECASE):
                affected.add(int(match))
        return affected if affected else None
    
    def get_actual_effects(self, obs_before: str, obs_after: str, obs_format: str) -> Set[int]:
        if obs_format == 'emoji':
            before_list = [s == '💡' for s in obs_before.split()]
            after_list = [s == '💡' for s in obs_after.split()]
        else:
            before_list = [b == '1' for b in obs_before]
            after_list = [b == '1' for b in obs_after]
        changed = set()
        for i, (b1, b2) in enumerate(zip(before_list, after_list)):
            if b1 != b2:
                changed.add(i)
        return changed
    
    def get_injection_false_patterns(self) -> List[Tuple[int, int]]:
        return [(1, 2), (2, 4), (3, 5)]
    
    def format_injection_hypothesis(self, action: int, target: int) -> str:
        return f"Toggling bulb {action} affects bulb {target}"


def get_domain_adapter(env_type: str) -> DomainAdapter:
    """
    Return the domain adapter for the given environment type.
    Default is Light for backward compatibility.
    """
    if env_type == 'light':
        return LightDomain()
    # Future: elif env_type == 'energy': return EnergyDomain(); ...
    return LightDomain()
