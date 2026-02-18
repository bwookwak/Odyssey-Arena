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
    
    @abstractmethod
    def reflection_instructions(self) -> str:
        """
        Instructions and format examples for the Reflector (what to extract from experiences).
        Appended after "Recent experiences: ..." in the reflection prompt.
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
