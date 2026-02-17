"""
Memory systems for LLM agents.

Implements different strategies for storing and using experiential memory:
- NoMemory: No memory (baseline)
- NaiveHypMemory: Store all hypotheses without verification
- GatedHypMemory: Only use high-confidence, non-contradicted hypotheses
"""

from typing import List, Dict, Any, Optional
import math


class MemorySystem:
    """Base class for memory systems."""
    
    def __init__(self):
        self.hypotheses = []
        
    def update(self, obs_before: str, action: int, obs_after: str, feedback: str):
        """Update memory based on transition."""
        pass
    
    def get_context(self, current_obs: str, suppressed: bool = False) -> str:
        """Get memory context for prompt."""
        return ""
    
    def is_using_memory(self, current_obs: str, suppressed: bool = False) -> bool:
        """Check if memory is being used this step."""
        return len(self.get_context(current_obs, suppressed)) > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'num_hypotheses': len(self.hypotheses),
            'memory_size': 0
        }


class NoMemory(MemorySystem):
    """No memory baseline - empty context."""
    
    def __init__(self):
        super().__init__()
    
    def update(self, obs_before: str, action: int, obs_after: str, feedback: str):
        """No-op."""
        pass
    
    def get_context(self, current_obs: str, suppressed: bool = False) -> str:
        """Always return empty context."""
        return ""


class NaiveHypMemory(MemorySystem):
    """
    Naive hypothesis memory - stores all hypotheses without verification.
    
    Tracks hypotheses like "toggling bulb i flips bulb j" with support/contradict counts.
    Includes all hypotheses in context (even low-confidence ones).
    """
    
    def __init__(self, max_context_items: int = 10):
        super().__init__()
        self.max_context_items = max_context_items
        # Store hypotheses as dict: (action, flip_idx) -> {support, contradict, conf}
        self.hyp_store = {}
    
    def update(self, obs_before: str, action: int, obs_after: str, feedback: str):
        """
        Update hypotheses based on observed transition.
        
        Logic:
        1. Detect which bits flipped from obs_before to obs_after
        2. For flipped bits j: support hypothesis (action -> j)
        3. For non-flipped bits k: contradict hypothesis (action -> k) if it exists
        """
        if len(obs_before) != len(obs_after):
            return
        
        flipped_indices = []
        for i, (b1, b2) in enumerate(zip(obs_before, obs_after)):
            if b1 != b2:
                flipped_indices.append(i)
        
        # Update support for observed flips
        for flip_idx in flipped_indices:
            key = (action, flip_idx)
            if key not in self.hyp_store:
                self.hyp_store[key] = {'support': 0, 'contradict': 0}
            self.hyp_store[key]['support'] += 1
        
        # Update contradictions for non-flips (if hypothesis exists)
        for bit_idx in range(len(obs_before)):
            if bit_idx not in flipped_indices:
                key = (action, bit_idx)
                if key in self.hyp_store:
                    self.hyp_store[key]['contradict'] += 1
        
        # Recalculate confidences
        for key in self.hyp_store:
            support = self.hyp_store[key]['support']
            contradict = self.hyp_store[key]['contradict']
            # Confidence: sigmoid(support - 2*contradict)
            score = support - 2 * contradict
            conf = 1.0 / (1.0 + math.exp(-score))
            self.hyp_store[key]['confidence'] = conf
    
    def get_context(self, current_obs: str, suppressed: bool = False) -> str:
        """
        Get memory context with top-N hypotheses by confidence.
        
        Args:
            current_obs: Current observation (unused in naive version)
            suppressed: If True, return empty context (for surgery intervention)
        """
        if suppressed or not self.hyp_store:
            return ""
        
        # Sort by confidence (descending)
        sorted_hyps = sorted(
            self.hyp_store.items(),
            key=lambda x: x[1].get('confidence', 0.0),
            reverse=True
        )
        
        lines = []
        for i, ((act, flip_idx), stats) in enumerate(sorted_hyps[:self.max_context_items], 1):
            conf = stats.get('confidence', 0.0)
            supp = stats.get('support', 0)
            cont = stats.get('contradict', 0)
            lines.append(
                f"{i}. Toggling bulb {act} flips bulb {flip_idx} "
                f"(conf: {conf:.2f}, +{supp}/-{cont})"
            )
        
        return "\n".join(lines)
    
    def is_using_memory(self, current_obs: str, suppressed: bool = False) -> bool:
        """Check if memory context is non-empty."""
        return len(self.get_context(current_obs, suppressed)) > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'num_hypotheses': len(self.hyp_store),
            'memory_size': len(self.hyp_store)
        }


class GatedHypMemory(NaiveHypMemory):
    """
    Gated hypothesis memory - verification-lite version.
    
    Only includes hypotheses in context that meet quality criteria:
    - contradict == 0 (no contradictions observed)
    - confidence >= 0.75
    - support >= 2 (at least 2 observations)
    """
    
    def __init__(
        self,
        max_context_items: int = 10,
        min_confidence: float = 0.75,
        min_support: int = 2,
        max_contradictions: int = 0
    ):
        super().__init__(max_context_items)
        self.min_confidence = min_confidence
        self.min_support = min_support
        self.max_contradictions = max_contradictions
    
    def get_context(self, current_obs: str, suppressed: bool = False) -> str:
        """
        Get memory context with only verified high-quality hypotheses.
        
        Args:
            current_obs: Current observation (unused)
            suppressed: If True, return empty context (for surgery intervention)
        """
        if suppressed or not self.hyp_store:
            return ""
        
        # Filter hypotheses by quality criteria
        verified_hyps = []
        for key, stats in self.hyp_store.items():
            if (stats.get('contradict', 0) <= self.max_contradictions and
                stats.get('confidence', 0.0) >= self.min_confidence and
                stats.get('support', 0) >= self.min_support):
                verified_hyps.append((key, stats))
        
        if not verified_hyps:
            return ""
        
        # Sort by confidence (descending)
        verified_hyps.sort(key=lambda x: x[1].get('confidence', 0.0), reverse=True)
        
        lines = []
        for i, ((act, flip_idx), stats) in enumerate(verified_hyps[:self.max_context_items], 1):
            conf = stats.get('confidence', 0.0)
            supp = stats.get('support', 0)
            cont = stats.get('contradict', 0)
            lines.append(
                f"{i}. Toggling bulb {act} flips bulb {flip_idx} "
                f"(verified: conf={conf:.2f}, +{supp}/-{cont})"
            )
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics including verified count."""
        verified_count = sum(
            1 for stats in self.hyp_store.values()
            if (stats.get('contradict', 0) <= self.max_contradictions and
                stats.get('confidence', 0.0) >= self.min_confidence and
                stats.get('support', 0) >= self.min_support)
        )
        return {
            'num_hypotheses': len(self.hyp_store),
            'num_verified': verified_count,
            'memory_size': len(self.hyp_store)
        }


class ReflectorCuratorMemory(MemorySystem):
    """
    Two-stage memory system with reflection and curation.
    
    Stage 1 (Reflector): Analyzes recent experiences and extracts insights
    Stage 2 (Curator): Curates memories with deduplication and consolidation
    
    This approach helps prevent mis-evolved memory by:
    1. Reflecting on patterns before storing
    2. Deduplicating similar insights
    3. Consolidating overlapping knowledge
    """
    
    def __init__(
        self,
        llm_client,
        reflection_frequency: int = 5,
        max_memories: int = 20,
        max_context_items: int = 10
    ):
        """
        Args:
            llm_client: LLM client for reflection and curation
            reflection_frequency: Trigger reflection after N experiences
            max_memories: Maximum number of curated memories to keep
            max_context_items: Maximum memories to include in context
        """
        super().__init__()
        self.llm_client = llm_client
        self.reflection_frequency = reflection_frequency
        self.max_memories = max_memories
        self.max_context_items = max_context_items
        
        # Storage
        self.raw_experiences = []  # Recent raw experiences
        self.curated_memories = []  # Curated memory items
        self.reflection_count = 0
    
    def update(self, obs_before: str, action: int, obs_after: str, feedback: str):
        """
        Update memory with new experience.
        
        Accumulates experiences and triggers reflection when threshold is reached.
        """
        # Store raw experience
        self.raw_experiences.append({
            'obs_before': obs_before,
            'action': action,
            'obs_after': obs_after,
            'feedback': feedback
        })
        
        # Trigger reflection when enough experiences accumulated
        if len(self.raw_experiences) >= self.reflection_frequency:
            try:
                self._reflect_and_curate()
            except Exception as e:
                print(f"Warning: Reflection failed: {e}")
            finally:
                self.raw_experiences = []  # Clear regardless of success
    
    def _reflect_and_curate(self):
        """
        Execute Reflector -> Curator pipeline.
        
        1. Reflector: Analyze experiences and extract insights
        2. Curator: Deduplicate and consolidate into memory
        """
        # Stage 1: Reflection
        reflection_prompt = self._build_reflection_prompt()
        reflection = self.llm_client.generate(
            reflection_prompt,
            max_tokens=200,
            temperature=0.7
        )
        
        if not reflection or len(reflection.strip()) < 10:
            return  # Skip if reflection is too short
        
        # Stage 2: Curation
        curation_prompt = self._build_curation_prompt(reflection)
        curated_memory = self.llm_client.generate(
            curation_prompt,
            max_tokens=150,
            temperature=0.5
        )
        
        if not curated_memory or len(curated_memory.strip()) < 5:
            return  # Skip if curation failed
        
        # Store curated memory
        self.curated_memories.append({
            'memory': curated_memory.strip(),
            'reflection': reflection.strip(),
            'timestamp': self.reflection_count
        })
        self.reflection_count += 1
        
        # Limit memory size
        if len(self.curated_memories) > self.max_memories:
            self.curated_memories = self.curated_memories[-self.max_memories:]
    
    def _build_reflection_prompt(self) -> str:
        """
        Build prompt for Reflector stage.
        
        Asks LLM to analyze recent experiences and identify patterns.
        """
        # Summarize experiences
        exp_lines = []
        for i, exp in enumerate(self.raw_experiences, 1):
            obs_change = f"{exp['obs_before']} -> {exp['obs_after']}"
            exp_lines.append(f"{i}. Action {exp['action']}: {obs_change}")
        
        experiences_text = "\n".join(exp_lines)
        
        prompt = f"""You are analyzing experiences from a light bulb puzzle game.

Recent experiences (state changes after toggling bulbs):
{experiences_text}

Analyze these experiences and answer:
1. What patterns did you observe? (Which actions caused which state changes?)
2. What relationships between bulbs did you discover?
3. What strategy insights can you extract?

Provide a concise analysis (2-3 sentences focusing on actionable patterns):"""
        
        return prompt
    
    def _build_curation_prompt(self, reflection: str) -> str:
        """
        Build prompt for Curator stage.
        
        Asks LLM to deduplicate and consolidate new insight with existing memories.
        """
        # Format existing memories
        if self.curated_memories:
            existing_text = "\n".join([
                f"{i+1}. {mem['memory']}"
                for i, mem in enumerate(self.curated_memories[-10:])  # Last 10
            ])
        else:
            existing_text = "(No existing memories)"
        
        prompt = f"""You are a memory curator managing knowledge about a light bulb puzzle.

EXISTING MEMORIES:
{existing_text}

NEW INSIGHT:
{reflection}

TASK:
1. Check if this insight is ALREADY COVERED by existing memories (deduplication)
2. If duplicate: respond with "DUPLICATE"
3. If new or partially overlapping: consolidate and return ONE concise memory item (max 1 sentence)
4. Focus on actionable patterns like "Toggling bulb X affects bulb Y"

Your response (either "DUPLICATE" or a new memory item):"""
        
        return prompt
    
    def get_context(self, current_obs: str, suppressed: bool = False) -> str:
        """
        Get curated memory context for prompt.
        
        Args:
            current_obs: Current observation (unused)
            suppressed: If True, return empty context
            
        Returns:
            Formatted memory context string
        """
        if suppressed or not self.curated_memories:
            return ""
        
        # Get most recent memories (up to max_context_items)
        recent_memories = self.curated_memories[-self.max_context_items:]
        
        # Filter out duplicates
        filtered_memories = []
        for mem in recent_memories:
            if mem['memory'].upper() != "DUPLICATE":
                filtered_memories.append(mem['memory'])
        
        if not filtered_memories:
            return ""
        
        # Format as numbered list
        lines = [f"{i+1}. {mem}" for i, mem in enumerate(filtered_memories)]
        return "\n".join(lines)
    
    def is_using_memory(self, current_obs: str, suppressed: bool = False) -> bool:
        """Check if memory context is non-empty."""
        return len(self.get_context(current_obs, suppressed)) > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        non_duplicate = sum(
            1 for m in self.curated_memories
            if m['memory'].upper() != "DUPLICATE"
        )
        return {
            'num_hypotheses': len(self.curated_memories),
            'num_curated': non_duplicate,
            'num_reflections': self.reflection_count,
            'memory_size': len(self.curated_memories)
        }


def create_memory_system(memory_type: str, llm_client=None, **kwargs) -> MemorySystem:
    """
    Factory function to create memory system.
    
    Args:
        memory_type: Type of memory ('nomem', 'naive', 'gated', 'reflector')
        llm_client: LLM client (required for 'reflector' type)
        **kwargs: Additional arguments for memory system
        
    Returns:
        Memory system instance
    """
    if memory_type == 'nomem':
        return NoMemory()
    elif memory_type == 'naive':
        return NaiveHypMemory(**kwargs)
    elif memory_type == 'gated':
        return GatedHypMemory(**kwargs)
    elif memory_type == 'reflector':
        if llm_client is None:
            raise ValueError("ReflectorCuratorMemory requires llm_client parameter")
        return ReflectorCuratorMemory(llm_client=llm_client, **kwargs)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")
