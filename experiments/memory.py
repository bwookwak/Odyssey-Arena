"""
Memory systems for LLM agents.

Memory Architecture:
1. Reflector-Curator: Common process for all memories (generates text insights)
   - Reflector: Extracts concrete insights from trajectory
   - Curator: Decides what to add/merge/skip (with deduplication)

2. Memory Types:
   - NoMemory: No memory (baseline)
   - TextMemory: Raw text list (general approach)
   - HypothesisMemory: Parses text into structured hypotheses with scores (user's method)
   - GatedHypothesisMemory: Only uses verified hypotheses
"""

from typing import List, Dict, Any, Optional
import math
import re


class MemorySystem:
    """
    Base class for memory systems.
    
    All memory systems can optionally use Reflector-Curator process
    to generate text-based memories from experiences.
    """
    
    def __init__(self, use_reflector: bool = False, llm_client=None, 
                 reflection_frequency: int = 5):
        """
        Args:
            use_reflector: Whether to use Reflector-Curator process
            llm_client: LLM client (required if use_reflector=True)
            reflection_frequency: Trigger reflection after N experiences
        """
        self.use_reflector = use_reflector
        self.llm_client = llm_client
        self.reflection_frequency = reflection_frequency
        
        # For Reflector-Curator
        self.raw_experiences = []
        self.text_memories = []  # Text-based memories from Reflector-Curator
        self.reflection_count = 0
        
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
        
        if not insight or len(insight.strip()) < 10:
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
        
        if action['type'] == 'ADD':
            self.text_memories.append(action['memory'])
            self.reflection_count += 1
        elif action['type'] == 'SKIP':
            pass  # Don't add
        
        # Limit memory size
        if len(self.text_memories) > 20:
            self.text_memories = self.text_memories[-20:]
    
    def _build_reflection_prompt(self) -> str:
        """Build Reflector prompt with explicit format guidance."""
        exp_lines = []
        for i, exp in enumerate(self.raw_experiences, 1):
            obs_change = f"{exp['obs_before']} -> {exp['obs_after']}"
            exp_lines.append(f"{i}. Action {exp['action']}: {obs_change}")
        
        experiences_text = "\n".join(exp_lines)
        
        return f"""You are the Reflector: extract concrete insights from trajectory.

Recent experiences:
{experiences_text}

Analyze and distill concrete insights about which actions affect which bulbs.
Use EXPLICIT format: "Toggling bulb X affects bulb Y" or "Action X flips bulb Y".

Example good insights:
- "Toggling bulb 0 affects bulb 1"
- "Action 2 flips bulb 3 and bulb 4"
- "Bulb 5 depends on bulbs 2 and 3"

Provide 1-3 specific insights (one per line):"""
    
    def _build_curation_prompt(self, insight: str) -> str:
        """Build Curator prompt with action-based decision."""
        existing_text = "\n".join([
            f"{i+1}. {mem}" for i, mem in enumerate(self.text_memories[-10:])
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

Important: If adding, use format "Toggling bulb X affects bulb Y" for parsability.

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
        if len(decision.strip()) > 10 and 'SKIP' not in decision_upper:
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
                 max_context_items: int = 10):
        """
        Args:
            llm_client: LLM client for Reflector-Curator
            reflection_frequency: Trigger reflection after N experiences
            max_context_items: Maximum memories to include in context
        """
        super().__init__(use_reflector=True, llm_client=llm_client,
                        reflection_frequency=reflection_frequency)
        self.max_context_items = max_context_items
    
    def get_context(self, current_obs: str, suppressed: bool = False) -> str:
        """Return text memories as context."""
        if suppressed or not self.text_memories:
            return ""
        
        # Get recent memories
        recent = self.text_memories[-self.max_context_items:]
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
        'text': "Toggling bulb 0 affects bulb 1",
        'support': 3,
        'contradict': 1, 
        'confidence': 0.73,
        'source': 'llm'
    }
    """
    
    def __init__(self, llm_client, reflection_frequency: int = 5,
                 max_context_items: int = 10, obs_format: str = 'bitstring',
                 verification_mode: str = 'oracle'):
        """
        Args:
            llm_client: LLM client for Reflector-Curator
            reflection_frequency: Trigger reflection after N experiences
            max_context_items: Maximum hypotheses to include in context
            obs_format: Observation format ('bitstring' or 'emoji')
            verification_mode: How to verify hypotheses ('oracle', 'llm', 'hybrid', 'none')
        """
        super().__init__(use_reflector=True, llm_client=llm_client,
                        reflection_frequency=reflection_frequency)
        self.max_context_items = max_context_items
        self.obs_format = obs_format
        self.verification_mode = verification_mode
        
        # Store hypotheses as list of dicts (natural language based)
        self.hypotheses = []
    
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
    
    def verify_with_oracle(self, custom_logic: Dict[str, str]):
        """
        Verify hypotheses against ground truth rules (custom_logic).
        
        This is TRUE oracle verification - uses actual latent rules from environment.
        
        Example:
            Hypothesis: "Toggling bulb 0 affects bulb 1"
            custom_logic: {"B1": "B0"} → B1 depends on B0
            → CONFIRMED (support++)
            
            Hypothesis: "Toggling bulb 0 affects bulb 2"  
            custom_logic: {"B2": "not B3"} → B2 doesn't depend on B0
            → CONTRADICTED (contradict++)
        
        Args:
            custom_logic: Ground truth rules like {"B0": "True", "B1": "B0", ...}
        """
        for hyp in self.hypotheses:
            # Extract predicted dependency from hypothesis text
            dependency = self._extract_dependency_from_text(hyp['text'])
            
            if dependency is None:
                continue  # Can't parse hypothesis
            
            action_bulb, affected_bulb = dependency
            
            # Get the rule for affected_bulb
            affected_rule = custom_logic.get(f"B{affected_bulb}", "")
            
            if not affected_rule:
                continue  # No rule found
            
            # Check if action_bulb appears in the rule
            # This indicates a dependency (direct or indirect)
            action_bulb_name = f"B{action_bulb}"
            
            if action_bulb_name in affected_rule:
                # Oracle confirms: dependency exists
                hyp['support'] += 1
                if hyp.get('source') != 'injection':  # Don't overwrite injection source
                    hyp['source'] = 'oracle_verified'
            else:
                # Oracle contradicts: no dependency in ground truth
                hyp['contradict'] += 1
            
            # Update confidence
            self._update_confidence(hyp)
    
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
                    'confidence': 0.73,     # Initial from LLM
                    'source': 'llm',
                    'created_at': len(self.hypotheses)
                })
        
        # Update index
        self._last_added_idx = len(self.text_memories)
    
    def _verify_with_observation(self, obs_before: str, action: int, obs_after: str):
        """
        Verify hypotheses against observed transition (NOT oracle - empirical).
        
        Observation-based verification:
        - Extract expected effects from hypothesis text
        - Check if observed transition matches
        - Update support/contradict accordingly
        
        Note: This is empirical, not oracle. Can be noisy.
        """
        obs1_list = self._obs_to_list(obs_before)
        obs2_list = self._obs_to_list(obs_after)
        
        # Detect actual changes
        actual_flips = set()
        for i, (b1, b2) in enumerate(zip(obs1_list, obs2_list)):
            if b1 != b2:
                actual_flips.add(i)
        
        # Verify each hypothesis
        for hyp in self.hypotheses:
            # Extract predicted effects from hypothesis text
            predicted_effects = self._extract_effects_from_text(hyp['text'], action)
            
            if predicted_effects is None:
                continue  # Not about this action
            
            # Check if prediction matches reality
            if predicted_effects == actual_flips:
                hyp['support'] += 1  # Confirmed!
            elif len(predicted_effects & actual_flips) > 0:
                hyp['support'] += 0.5  # Partial match
            else:
                hyp['contradict'] += 1  # Contradicted!
            
            # Recalculate confidence
            self._update_confidence(hyp)
    
    def _extract_dependency_from_text(self, hypothesis_text: str) -> Optional[tuple]:
        """
        Extract dependency from hypothesis text for oracle verification.
        
        "Toggling bulb 0 affects bulb 1" → (0, 1)
        "Action 2 flips bulb 3" → (2, 3)
        
        Returns:
            (action_bulb, affected_bulb) or None
        """
        patterns = [
            r'(?:toggling|action)\s+(?:bulb\s+)?(\d+)\s+(?:affects?|flips?|influences?)\s+(?:bulb\s+)?(\d+)',
            r'bulb\s+(\d+)\s+(?:affects?|flips?)\s+bulb\s+(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, hypothesis_text, re.IGNORECASE)
            if match:
                return (int(match.group(1)), int(match.group(2)))
        
        return None
    
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
    
    def _extract_effects_from_text(self, hypothesis_text: str, action: int) -> Optional[set]:
        """
        Extract predicted effects from hypothesis text.
        
        Args:
            hypothesis_text: "Toggling bulb 0 affects bulb 1 and bulb 2"
            action: Current action taken
            
        Returns:
            Set of bulb indices expected to flip, or None if not about this action
        """
        # Check if hypothesis mentions this action
        action_patterns = [
            rf'\b(?:action|bulb|toggling)\s+{action}\b',
            rf'\b{action}\s+(?:affects?|flips?)',
        ]
        
        mentions_action = any(re.search(p, hypothesis_text, re.IGNORECASE) 
                             for p in action_patterns)
        
        if not mentions_action:
            return None
        
        # Extract affected bulb indices
        affected = set()
        effect_patterns = [
            r'(?:affects?|flips?|influences?|toggles?)\s+bulb\s+(\d+)',
            r'(?:affects?|flips?|influences?)\s+(\d+)',
        ]
        
        for pattern in effect_patterns:
            matches = re.findall(pattern, hypothesis_text, re.IGNORECASE)
            for match in matches:
                affected.add(int(match))
        
        return affected if affected else None
    
    def _update_confidence(self, hypothesis: Dict):
        """Recalculate confidence based on support/contradict."""
        support = hypothesis['support']
        contradict = hypothesis['contradict']
        score = support - 2 * contradict
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
        
        lines = []
        for i, hyp in enumerate(sorted_hyps[:self.max_context_items], 1):
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
    
    Same as HypothesisMemory but only includes verified hypotheses in context:
    - contradict == 0 (no contradictions observed)
    - confidence >= 0.75
    - support >= 2 (or >= 1 for LLM-sourced)
    """
    
    def __init__(
        self,
        llm_client,
        reflection_frequency: int = 5,
        max_context_items: int = 10,
        obs_format: str = 'bitstring',
        verification_mode: str = 'oracle',
        min_confidence: float = 0.75,
        min_support: int = 2,
        max_contradictions: int = 0
    ):
        super().__init__(llm_client, reflection_frequency, max_context_items, 
                        obs_format, verification_mode)
        self.min_confidence = min_confidence
        self.min_support = min_support
        self.max_contradictions = max_contradictions
    
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
                # LLM-generated: more lenient
                if (contradict <= self.max_contradictions and
                    confidence >= 0.70 and
                    support >= 1):
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
        
        lines = []
        for i, hyp in enumerate(verified_hyps[:self.max_context_items], 1):
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
                        verification_mode='oracle', **kwargs) -> MemorySystem:
    """
    Factory function to create memory system.
    
    Memory Types:
    - 'nomem': No memory (baseline)
    - 'text': Simple text-based memory (Reflector-Curator → text list)
    - 'hypothesis': Hypothesis memory with verification (Reflector-Curator → natural language hypotheses)
    - 'gated': Gated hypothesis memory (hypothesis + quality filter)
    
    Args:
        memory_type: Type of memory
        llm_client: LLM client (required for text/hypothesis/gated)
        obs_format: Observation format ('bitstring' or 'emoji')
        verification_mode: How to verify hypotheses ('oracle', 'llm', 'hybrid', 'none')
        **kwargs: Additional arguments for memory system
        
    Returns:
        Memory system instance
    """
    if memory_type == 'nomem':
        return NoMemory()
    elif memory_type == 'text':
        if llm_client is None:
            raise ValueError("TextMemory requires llm_client parameter")
        return TextMemory(llm_client=llm_client, **kwargs)
    elif memory_type == 'hypothesis':
        if llm_client is None:
            raise ValueError("HypothesisMemory requires llm_client parameter")
        return HypothesisMemory(llm_client=llm_client, obs_format=obs_format, 
                               verification_mode=verification_mode, **kwargs)
    elif memory_type == 'gated':
        if llm_client is None:
            raise ValueError("GatedHypothesisMemory requires llm_client parameter")
        return GatedHypothesisMemory(llm_client=llm_client, obs_format=obs_format,
                                    verification_mode=verification_mode, **kwargs)
    # Backward compatibility aliases
    elif memory_type == 'naive':
        if llm_client is None:
            raise ValueError("HypothesisMemory (naive) requires llm_client parameter")
        return HypothesisMemory(llm_client=llm_client, obs_format=obs_format,
                               verification_mode=verification_mode, **kwargs)
    elif memory_type == 'reflector':
        # Alias for 'text' memory
        if llm_client is None:
            raise ValueError("TextMemory (reflector) requires llm_client parameter")
        return TextMemory(llm_client=llm_client, **kwargs)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")
