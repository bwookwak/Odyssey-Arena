"""
Odyssey-Arena Mis-evolved Memory Research Framework

Research Context:
- Agents learn from interaction by storing experiential memory. In noisy/partial/dynamic
  settings, agents may store incorrect "insights" as facts (mis-evolved memory), which
  then biases future decisions and causes cumulative failures (loops/stagnation) and
  a performance ceiling.
- We test whether treating new insights as tentative hypotheses (with simple evidence
  tracking) and only using high-confidence, non-contradicted items improves robustness.

MVP: Odyssey-Arena Turn On Lights (LightEnv) - extensible to other environments.
"""

__version__ = "0.1.0"
