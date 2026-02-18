"""
Memory systems for LLM agents.

Memory Architecture:
1. Reflector-Curator: Common process for all memories (generates text insights)
   - Reflector: Extracts concrete insights from trajectory (domain-specific prompts)
   - Curator: Decides what to add/merge/skip (with deduplication)

2. Memory Types:
   - NoMemory: No memory (baseline)
   - TextMemory: Raw text list (general approach)
   - HypothesisMemory: Parses text into structured hypotheses with scores (user's method)
   - GatedHypothesisMemory: Only uses verified hypotheses

Domain-specific logic (reflection prompts, oracle verification, hypothesis parsing)
is delegated to a DomainAdapter (see experiments.domain).
"""

from typing import List, Dict, Any, Optional
import math
import re

from experiments.domain import DomainAdapter, get_domain_adapter


class MemorySystem:
    """
    Base class for memory systems.
    
    All memory systems can optionally use Reflector-Curator process
    to generate text-based memories from experiences.
    """
    
    def __init__(self, use_reflector: bool = False, llm_client=None, 
                 reflection_frequency: int = 5,
                 max_text_memories: int = 20,
                 curation_context_size: int = 10,
                 min_insight_length: int = 10,
                 domain: Optional[DomainAdapter] = None,
                 **kwargs):
        """
        Args:
            use_reflector: Whether to use Reflector-Curator process
            llm_client: LLM client (required if use_reflector=True)
            reflection_frequency: Trigger reflection after N experiences
            max_text_memories: Max number of text memories to keep
            curation_context_size: Number of recent memories in curation prompt
            min_insight_length: Min length of insight string to accept
            domain: Environment-specific adapter for prompts and verification (default: light)
        """
        self.use_reflector = use_reflector
        self.llm_client = llm_client
        self.reflection_frequency = reflection_frequency
        self.max_text_memories = max_text_memories
        self.curation_context_size = curation_context_size
        self.min_insight_length = min_insight_length
        self.domain = domain if domain is not None else get_domain_adapter('light')
        
        # For Reflector-Curator
        self.raw_experiences = []
        self.text_memories = []  # Text-based memories from Reflector-Curator
        self.reflection_count = 0
        self._recent_ops = []  # Per-step memory ops for dashboard (reflection, curation, verify)
        
        if use_reflector and llm_client is None:
            raise ValueError("Reflector-Curator requires llm_client")
    
    def update(self, obs_before: str, action: int, obs_after: str, feedback: str):
        """Update memory based on transition."""
        if self.use_reflector:
            self._update_with_reflector(obs_before, action, obs_after, feedback)
        else:
            self._update_direct(obs_before, action, obs_after, feedback)
    
    def _update_direct(self, obs_before: str, action: int, obs_after: str, feedback: str):
        """Direct update without Reflector-Curator (to be overridden)."""
        pass
    
    def get_recent_ops(self) -> List[Dict[str, Any]]:
        """Return and clear per-step memory ops (reflection, curation, verify) for dashboard."""
        out = list(getattr(self, '_recent_ops', []))
        self._recent_ops.clear()
        return out
    
    def _update_with_reflector(self, obs_before: str, action: int, obs_after: str, feedback: str):
        """
        Update using Reflector-Curator process.
        
        1. Accumulate experiences
        2. When threshold reached: Reflect → Curate → Add to text_memories
        3. Process text_memories (subclass-specific)
        """
        # Accumulate experience
        self.raw_experiences.append({
            'obs_before': obs_before,
            'action': action,
            'obs_after': obs_after,
            'feedback': feedback
        })
        
        # Trigger reflection when threshold reached
        if len(self.raw_experiences) >= self.reflection_frequency:
            try:
                self._reflect_and_curate()
            except Exception as e:
                print(f"Warning: Reflection failed: {e}")
            finally:
                self.raw_experiences = []
    
    def _reflect_and_curate(self):
        """
        Execute Reflector-Curator pipeline.
        
        Stage 1 (Reflector): Extract concrete insights from experiences
        Stage 2 (Curator): Decide what to add/skip (deduplication)
        """
        # Stage 1: Reflection
        reflection_prompt = self._build_reflection_prompt()
        insight = self.llm_client.generate(
            reflection_prompt,
            max_tokens=200,
            temperature=0.7
        )
        
        if not insight or len(insight.strip()) < self.min_insight_length:
            return
        
        # Stage 2: Curation
        curation_prompt = self._build_curation_prompt(insight)
        curator_decision = self.llm_client.generate(
            curation_prompt,
            max_tokens=200,
            temperature=0.5
        )
        
        # Parse curator decision
        action = self._parse_curator_action(curator_decision)
        
        self._recent_ops.append({
            'op': 'reflection_curation',
            'insight': insight.strip()[:1500],
            'curator_decision': curator_decision.strip()[:1000],
            'action': action['type'],
            'memory_added': action.get('memory', '')[:500] if action['type'] == 'ADD' else None,
        })
        
        if action['type'] == 'ADD':
            self.text_memories.append(action['memory'])
            self.reflection_count += 1
        elif action['type'] == 'SKIP':
            pass  # Don't add
        
        # Limit memory size
        if len(self.text_memories) > self.max_text_memories:
            self.text_memories = self.text_memories[-self.max_text_memories:]
    
    def _build_reflection_prompt(self) -> str:
        """Build Reflector prompt; domain-specific instructions from self.domain."""
        exp_lines = []
        for i, exp in enumerate(self.raw_experiences, 1):
            obs_change = f"{exp['obs_before']} -> {exp['obs_after']}"
            exp_lines.append(f"{i}. Action {exp['action']}: {obs_change}")
        
        experiences_text = "\n".join(exp_lines)
        instructions = self.domain.reflection_instructions()
        
        return f"""You are the Reflector: extract concrete insights from trajectory.

Recent experiences:
{experiences_text}

{instructions}"""
    
    def _build_curation_prompt(self, insight: str) -> str:
        """Build Curator prompt with action-based decision."""
        existing_text = "\n".join([
            f"{i+1}. {mem}" for i, mem in enumerate(self.text_memories[-self.curation_context_size:])
        ]) if self.text_memories else "(No existing memories)"
        
        return f"""You are the Curator: decide what to add to memory.

EXISTING MEMORIES:
{existing_text}

NEW INSIGHT:
{insight}

DECISION - Choose ONE action:
1. "ADD: [memory text]" - if insight is new or provides additional value
2. "SKIP: duplicate" - if already covered in existing memories
3. "SKIP: low quality" - if insight is too vague or not actionable

{self.domain.curation_format_hint()}

Your decision:"""
    
    def _parse_curator_action(self, decision: str) -> Dict[str, Any]:
        """Parse curator decision into action."""
        decision_upper = decision.upper().strip()
        
        # Check for ADD
        add_match = re.search(r'ADD:\s*(.+)', decision, re.IGNORECASE)
        if add_match:
            return {
                'type': 'ADD',
                'memory': add_match.group(1).strip()
            }
        
        # Check for SKIP
        if 'SKIP' in decision_upper or 'DUPLICATE' in decision_upper:
            return {'type': 'SKIP'}
        
        # Default: try to add the whole text if it's meaningful
        if len(decision.strip()) > self.min_insight_length and 'SKIP' not in decision_upper:
            return {
                'type': 'ADD',
                'memory': decision.strip()
            }
        
        return {'type': 'SKIP'}
    
    def get_context(self, current_obs: str, suppressed: bool = False) -> str:
        """Get memory context for prompt."""
        return ""
    
    def is_using_memory(self, current_obs: str, suppressed: bool = False) -> bool:
        """Check if memory is being used this step."""
        return len(self.get_context(current_obs, suppressed)) > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'num_memories': len(self.text_memories) if hasattr(self, 'text_memories') else 0,
            'memory_size': 0
        }


class NoMemory(MemorySystem):
    """No memory baseline - empty context."""
    
    def __init__(self):
        super().__init__(use_reflector=False)
    
    def _update_direct(self, obs_before: str, action: int, obs_after: str, feedback: str):
        """No-op."""
        pass
    
    def get_context(self, current_obs: str, suppressed: bool = False) -> str:
        """Always return empty context."""
        return ""


class TextMemory(MemorySystem):
    """
    Simple text-based memory.
    
    Stores insights as text list and uses them directly as context.
    Uses Reflector-Curator to generate memories.
    """
    
    def __init__(self, llm_client, reflection_frequency: int = 5, 
                 max_context_items: Optional[int] = None,
                 max_text_memories: int = 20,
                 curation_context_size: int = 10,
                 min_insight_length: int = 10,
                 **kwargs):
        """
        Args:
            llm_client: LLM client for Reflector-Curator
            reflection_frequency: Trigger reflection after N experiences
            max_context_items: Maximum memories to include in context
        """
        super().__init__(use_reflector=True, llm_client=llm_client,
                        reflection_frequency=reflection_frequency,
                        max_text_memories=max_text_memories,
                        curation_context_size=curation_context_size,
                        min_insight_length=min_insight_length,
                        **kwargs)
        self.max_context_items = max_context_items
    
    def get_context(self, current_obs: str, suppressed: bool = False) -> str:
        """Return text memories as context."""
        if suppressed or not self.text_memories:
            return ""
        
        # Get recent memories (all if max_context_items is None)
        recent = self.text_memories[-self.max_context_items:] if self.max_context_items is not None else self.text_memories
        return "\n".join([f"{i+1}. {mem}" for i, mem in enumerate(recent)])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'num_memories': len(self.text_memories),
            'num_reflections': self.reflection_count,
            'memory_size': len(self.text_memories)
        }


class HypothesisMemory(MemorySystem):
    """
    Hypothesis-based memory (user's method).
    
    Natural language hypotheses with support/contradict tracking.
    
    Process:
    1. Reflector-Curator generates text hypotheses
    2. Store as natural language with scores
    3. Optionally verify against observations (oracle/llm/hybrid)
    
    Hypothesis structure:
    {
        'text': natural language hypothesis (format depends on domain),
        'support': int, 'contradict': int, 'confidence': float, 'source': str
    }
    """
    
    def __init__(self, llm_client, reflection_frequency: int = 5,
                 max_context_items: Optional[int] = None, obs_format: str = 'bitstring',
                 verification_mode: str = 'oracle',
                 initial_confidence: float = 0.73,
                 support_weight: float = 1.0,
                 contradict_weight: float = 2.0,
                 partial_match_support_delta: float = 0.5,
                 max_text_memories: int = 20,
                 curation_context_size: int = 10,
                 min_insight_length: int = 10,
                 domain: Optional[DomainAdapter] = None,
                 **kwargs):
        """
        Args:
            llm_client: LLM client for Reflector-Curator
            reflection_frequency: Trigger reflection after N experiences
            max_context_items: Maximum hypotheses to include in context
            obs_format: Observation format ('bitstring' or 'emoji')
            verification_mode: How to verify hypotheses ('oracle', 'llm', 'hybrid', 'none')
            initial_confidence: Initial confidence for new LLM hypotheses
            support_weight: Weight for support in confidence formula
            contradict_weight: Weight for contradict in confidence formula
            partial_match_support_delta: Support delta for partial observation match
        """
        super().__init__(use_reflector=True, llm_client=llm_client,
                        reflection_frequency=reflection_frequency,
                        max_text_memories=max_text_memories,
                        curation_context_size=curation_context_size,
                        min_insight_length=min_insight_length,
                        domain=domain,
                        **kwargs)
        self.max_context_items = max_context_items
        self.obs_format = obs_format
        self.verification_mode = verification_mode
        self.initial_confidence = initial_confidence
        self.support_weight = float(support_weight)
        self.contradict_weight = float(contradict_weight)
        self.partial_match_support_delta = partial_match_support_delta
        
        # Store hypotheses as list of dicts (natural language based)
        self.hypotheses = []
    
    def add_hypothesis(self, hypothesis: Dict[str, Any]) -> None:
        """
        Add a hypothesis (e.g. from injection). Use this instead of mutating .hypotheses directly.
        hypothesis: dict with keys text, support, contradict, confidence, source, (optional created_at)
        """
        self.hypotheses.append({
            'text': hypothesis['text'],
            'support': hypothesis.get('support', 1),
            'contradict': hypothesis.get('contradict', 0),
            'confidence': hypothesis.get('confidence', self.initial_confidence),
            'source': hypothesis.get('source', 'injection'),
            'created_at': hypothesis.get('created_at', len(self.hypotheses))
        })
    
    def _update_with_reflector(self, obs_before: str, action: int, obs_after: str, feedback: str):
        """
        Hybrid update: Reflector-Curator + optional verification.
        
        1. Accumulate experiences and trigger reflection
        2. Parse text into natural language hypotheses
        3. Optionally verify existing hypotheses
        """
        # Store experience for reflection
        super()._update_with_reflector(obs_before, action, obs_after, feedback)
        
        # Observation-based verification (empirical)
        if self.verification_mode in ['observation', 'hybrid']:
            self._verify_with_observation(obs_before, action, obs_after)
        
        # LLM verification
        if self.verification_mode in ['llm', 'hybrid']:
            # LLM verification happens less frequently (expensive)
            if len(self.raw_experiences) == 0:  # Just after reflection
                self._verify_with_llm()
        
        # Note: Oracle verification (ground truth) is called externally from run.py
        # because it needs custom_logic from environment
    
    def verify_with_oracle(self, ground_truth: Any, use_llm: bool = True):
        """
        Verify hypotheses against environment ground truth (e.g. custom_logic for light).
        Uses domain adapter for prompt/parse/verify. use_llm: use LLM when True, else domain's simple verify.
        """
        if use_llm and self.llm_client:
            self._verify_with_oracle_llm(ground_truth)
        else:
            self._verify_with_oracle_simple(ground_truth)
    
    def _verify_with_oracle_llm(self, ground_truth: Any):
        """
        LLM-based oracle verification.
        Uses domain adapter to build env-specific verification prompt.
        """
        for hyp in self.hypotheses:
            verify_prompt = self.domain.build_oracle_verification_prompt(hyp['text'], ground_truth)
            if not verify_prompt:
                continue
            try:
                response = self.llm_client.generate(verify_prompt, max_tokens=20, temperature=0.0)
                response_upper = response.upper().strip()
                if 'CONFIRM' in response_upper:
                    hyp['support'] += 1
                    if hyp.get('source') != 'injection':
                        hyp['source'] = 'oracle_verified'
                elif 'CONTRADICT' in response_upper:
                    hyp['contradict'] += 1
                self._update_confidence(hyp)
                self._recent_ops.append({
                    'op': 'oracle_verify',
                    'mode': 'llm',
                    'hypothesis': hyp.get('text', '')[:200],
                    'response': response.strip()[:100],
                })
            except Exception as e:
                print(f"Warning: Oracle LLM verification failed: {e}")
    
    def _verify_with_oracle_simple(self, ground_truth: Any):
        """
        Non-LLM oracle verification using domain adapter (parse + verify).
        """
        for hyp in self.hypotheses:
            dependency = self.domain.parse_dependency(hyp['text'])
            if dependency is None:
                continue
            result = self.domain.verify_dependency_against_ground_truth(dependency, ground_truth)
            if result is None:
                continue
            if result:
                hyp['support'] += 1
                if hyp.get('source') != 'injection':
                    hyp['source'] = 'oracle_verified'
            else:
                hyp['contradict'] += 1
            self._update_confidence(hyp)
            self._recent_ops.append({
                'op': 'oracle_verify',
                'mode': 'simple',
                'hypothesis': hyp.get('text', '')[:200],
                'result': result,
            })
    
    def _reflect_and_curate(self):
        """
        Override to convert text into natural language hypotheses after Reflector-Curator.
        """
        # Call parent's Reflector-Curator process
        super()._reflect_and_curate()
        
        # Convert new text memories into hypothesis objects
        self._add_text_as_hypotheses()
    
    def _add_text_as_hypotheses(self):
        """
        Add new text memories as natural language hypotheses.
        
        Each text becomes a hypothesis with initial scores.
        No parsing into (action, flip) tuples - keep as natural language.
        """
        # Only process new memories
        start_idx = getattr(self, '_last_added_idx', 0)
        new_memories = self.text_memories[start_idx:]
        
        for memory_text in new_memories:
            # Check if already exists (by text similarity)
            exists = any(h['text'].lower() == memory_text.lower() 
                        for h in self.hypotheses)
            
            if not exists:
                # Add as new hypothesis
                self.hypotheses.append({
                    'text': memory_text,
                    'support': 1,           # LLM mentioned it
                    'contradict': 0,        # No contradictions yet
                    'confidence': self.initial_confidence,
                    'source': 'llm',
                    'created_at': len(self.hypotheses)
                })
        
        # Update index
        self._last_added_idx = len(self.text_memories)
    
    def _verify_with_observation(self, obs_before: str, action: int, obs_after: str):
        """
        Verify hypotheses against observed transition (empirical, not oracle).
        Uses domain adapter for parsing predicted effects and getting actual effects.
        """
        actual_effects = self.domain.get_actual_effects(obs_before, obs_after, self.obs_format)
        for hyp in self.hypotheses:
            predicted_effects = self.domain.parse_effects_from_text(hyp['text'], action)
            if predicted_effects is None:
                continue
            if predicted_effects == actual_effects:
                hyp['support'] += 1
            elif len(predicted_effects & actual_effects) > 0:
                hyp['support'] += self.partial_match_support_delta
            else:
                hyp['contradict'] += 1
            self._update_confidence(hyp)
    
    def _verify_with_llm(self):
        """
        Ask LLM to verify hypotheses based on recent experiences.
        
        LLM verification:
        - Show hypothesis + recent experiences
        - Ask if confirmed or contradicted
        - Update scores based on LLM judgment
        """
        if not self.hypotheses or not self.raw_experiences:
            return
        
        # Build verification prompt
        exp_summary = "\n".join([
            f"Action {e['action']}: {e['obs_before']} -> {e['obs_after']}"
            for e in self.raw_experiences[-5:]  # Last 5
        ])
        
        for hyp in self.hypotheses[:5]:  # Verify top 5 hypotheses
            verify_prompt = f"""Verify this hypothesis against observations.

HYPOTHESIS:
{hyp['text']}

RECENT OBSERVATIONS:
{exp_summary}

QUESTION:
Based on the observations, is this hypothesis:
1. "CONFIRMED" - observations support it
2. "CONTRADICTED" - observations contradict it  
3. "UNCLEAR" - not enough evidence

Your answer (one word):"""
            
            try:
                response = self.llm_client.generate(verify_prompt, max_tokens=20)
                
                if 'CONFIRM' in response.upper():
                    hyp['support'] += 1
                elif 'CONTRADICT' in response.upper():
                    hyp['contradict'] += 1
                
                # Recalculate confidence
                self._update_confidence(hyp)
            except Exception as e:
                print(f"Warning: LLM verification failed: {e}")
    
    def _update_confidence(self, hypothesis: Dict):
        """Recalculate confidence based on support/contradict (configurable weights)."""
        support = hypothesis['support']
        contradict = hypothesis['contradict']
        score = self.support_weight * support - self.contradict_weight * contradict
        hypothesis['confidence'] = 1.0 / (1.0 + math.exp(-score))
    
    def _obs_to_list(self, obs: str) -> List[bool]:
        """Convert observation to boolean list."""
        if self.obs_format == 'emoji':
            return [symbol == '💡' for symbol in obs.split()]
        else:
            return [bit == '1' for bit in obs]
    
    def get_context(self, current_obs: str, suppressed: bool = False) -> str:
        """
        Get memory context with top-N hypotheses by confidence.
        
        Args:
            current_obs: Current observation (unused)
            suppressed: If True, return empty context (for surgery intervention)
        """
        if suppressed or not self.hypotheses:
            return ""
        
        # Sort by confidence (descending)
        sorted_hyps = sorted(
            self.hypotheses,
            key=lambda x: x.get('confidence', 0.0),
            reverse=True
        )
        
        hyps_to_show = sorted_hyps[:self.max_context_items] if self.max_context_items is not None else sorted_hyps
        lines = []
        for i, hyp in enumerate(hyps_to_show, 1):
            conf = hyp.get('confidence', 0.0)
            supp = hyp.get('support', 0)
            cont = hyp.get('contradict', 0)
            text = hyp.get('text', '')
            lines.append(
                f"{i}. {text} (conf: {conf:.2f}, +{supp}/-{cont})"
            )
        
        return "\n".join(lines)
    
    def is_using_memory(self, current_obs: str, suppressed: bool = False) -> bool:
        """Check if memory context is non-empty."""
        return len(self.get_context(current_obs, suppressed)) > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'num_hypotheses': len(self.hypotheses),
            'memory_size': len(self.hypotheses),
            'verification_mode': self.verification_mode
        }


class GatedHypothesisMemory(HypothesisMemory):
    """
    Gated hypothesis memory - verification-lite version.
    
    Same as HypothesisMemory but only includes verified hypotheses in context.
    Thresholds are configurable (min_confidence, min_support, max_contradictions;
    for LLM-sourced: min_confidence_llm, min_support_llm).
    """
    
    def __init__(
        self,
        llm_client,
        reflection_frequency: int = 5,
        max_context_items: Optional[int] = None,
        obs_format: str = 'bitstring',
        verification_mode: str = 'oracle',
        min_confidence: float = 0.75,
        min_support: int = 2,
        max_contradictions: int = 0,
        min_confidence_llm: float = 0.70,
        min_support_llm: int = 1,
        **kwargs
    ):
        super().__init__(llm_client, reflection_frequency, max_context_items, 
                        obs_format, verification_mode, **kwargs)
        self.min_confidence = min_confidence
        self.min_support = min_support
        self.max_contradictions = max_contradictions
        self.min_confidence_llm = min_confidence_llm
        self.min_support_llm = min_support_llm
    
    def get_context(self, current_obs: str, suppressed: bool = False) -> str:
        """
        Get memory context with only verified high-quality hypotheses.
        
        Filtering criteria:
        - contradict <= max_contradictions (default: 0)
        - confidence >= min_confidence (default: 0.75)  
        - support >= min_support (default: 2, or 1 for LLM)
        
        Args:
            current_obs: Current observation (unused)
            suppressed: If True, return empty context (for surgery intervention)
        """
        if suppressed or not self.hypotheses:
            return ""
        
        # Filter hypotheses by quality criteria
        verified_hyps = []
        for hyp in self.hypotheses:
            source = hyp.get('source', 'observation')
            contradict = hyp.get('contradict', 0)
            confidence = hyp.get('confidence', 0.0)
            support = hyp.get('support', 0)
            
            # Source-aware verification
            if source == 'llm':
                # LLM-generated: more lenient (configurable)
                if (contradict <= self.max_contradictions and
                    confidence >= self.min_confidence_llm and
                    support >= self.min_support_llm):
                    verified_hyps.append(hyp)
            else:
                # Observation-based: stricter
                if (contradict <= self.max_contradictions and
                    confidence >= self.min_confidence and
                    support >= self.min_support):
                    verified_hyps.append(hyp)
        
        if not verified_hyps:
            return ""
        
        # Sort by confidence (descending)
        verified_hyps.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
        
        to_show = verified_hyps[:self.max_context_items] if self.max_context_items is not None else verified_hyps
        lines = []
        for i, hyp in enumerate(to_show, 1):
            conf = hyp.get('confidence', 0.0)
            supp = hyp.get('support', 0)
            cont = hyp.get('contradict', 0)
            text = hyp.get('text', '')
            lines.append(
                f"{i}. {text} (verified: conf={conf:.2f}, +{supp}/-{cont})"
            )
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics including verified count."""
        verified_count = sum(
            1 for hyp in self.hypotheses
            if (hyp.get('contradict', 0) <= self.max_contradictions and
                hyp.get('confidence', 0.0) >= self.min_confidence and
                hyp.get('support', 0) >= self.min_support)
        )
        return {
            'num_hypotheses': len(self.hypotheses),
            'num_verified': verified_count,
            'memory_size': len(self.hypotheses),
            'verification_mode': self.verification_mode
        }


def create_memory_system(memory_type: str, llm_client=None, obs_format='bitstring', 
                        verification_mode='oracle',
                        reflection_frequency=1,
                        max_text_memories=20,
                        curation_context_size=10,
                        min_insight_length=10,
                        max_context_items=None,
                        initial_confidence=0.73,
                        confidence_support_weight=1.0,
                        confidence_contradict_weight=2.0,
                        partial_match_support_delta=0.5,
                        min_confidence=0.75,
                        min_support=2,
                        max_contradictions=0,
                        min_confidence_llm=0.70,
                        min_support_llm=1,
                        domain=None,
                        env_type='light',
                        **kwargs) -> MemorySystem:
    """
    Factory function to create memory system.
    
    Memory Types:
    - 'nomem': No memory (baseline)
    - 'text': Simple text-based memory (Reflector-Curator → text list)
    - 'hypothesis': Hypothesis memory with verification (Reflector-Curator → natural language hypotheses)
    - 'gated': Gated hypothesis memory (hypothesis + quality filter)
    
    All numeric thresholds and sizes are configurable via explicit args or kwargs.
    domain: optional DomainAdapter; if None, get_domain_adapter(env_type) is used.
    """
    if domain is None:
        domain = get_domain_adapter(env_type)
    mem_kw = {
        'reflection_frequency': reflection_frequency,
        'max_text_memories': max_text_memories,
        'curation_context_size': curation_context_size,
        'min_insight_length': min_insight_length,
        'max_context_items': max_context_items,
        'domain': domain,
        **kwargs
    }
    # Avoid passing hypothesis-specific args twice (explicit params above; also aliases from callers)
    for key in ('initial_confidence', 'confidence_support_weight', 'confidence_contradict_weight',
                'partial_match_support_delta', 'min_confidence', 'min_support', 'max_contradictions',
                'min_confidence_llm', 'min_support_llm', 'support_weight', 'contradict_weight'):
        mem_kw.pop(key, None)
    if memory_type == 'nomem':
        return NoMemory()
    elif memory_type == 'text':
        if llm_client is None:
            raise ValueError("TextMemory requires llm_client parameter")
        return TextMemory(llm_client=llm_client, **mem_kw)
    elif memory_type == 'hypothesis':
        if llm_client is None:
            raise ValueError("HypothesisMemory requires llm_client parameter")
        return HypothesisMemory(
            llm_client=llm_client, obs_format=obs_format, 
            verification_mode=verification_mode,
            initial_confidence=initial_confidence,
            support_weight=confidence_support_weight,
            contradict_weight=confidence_contradict_weight,
            partial_match_support_delta=partial_match_support_delta,
            **mem_kw
        )
    elif memory_type == 'gated':
        if llm_client is None:
            raise ValueError("GatedHypothesisMemory requires llm_client parameter")
        return GatedHypothesisMemory(
            llm_client=llm_client, obs_format=obs_format,
            verification_mode=verification_mode,
            min_confidence=min_confidence,
            min_support=min_support,
            max_contradictions=max_contradictions,
            min_confidence_llm=min_confidence_llm,
            min_support_llm=min_support_llm,
            initial_confidence=initial_confidence,
            support_weight=confidence_support_weight,
            contradict_weight=confidence_contradict_weight,
            partial_match_support_delta=partial_match_support_delta,
            **mem_kw
        )
    # Backward compatibility aliases
    elif memory_type == 'naive':
        if llm_client is None:
            raise ValueError("HypothesisMemory (naive) requires llm_client parameter")
        return HypothesisMemory(
            llm_client=llm_client, obs_format=obs_format,
            verification_mode=verification_mode,
            initial_confidence=initial_confidence,
            support_weight=confidence_support_weight,
            contradict_weight=confidence_contradict_weight,
            partial_match_support_delta=partial_match_support_delta,
            **mem_kw
        )
    elif memory_type == 'reflector':
        if llm_client is None:
            raise ValueError("TextMemory (reflector) requires llm_client parameter")
        return TextMemory(llm_client=llm_client, **mem_kw)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")
