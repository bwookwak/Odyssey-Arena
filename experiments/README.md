# Mis-evolved Memory Experiments

Extensible experimental codebase for graduate research on **mis-evolved memory** in LLM agents.

## Research Context

Agents learn from interaction by storing experiential memory. In noisy/partial/dynamic settings, agents may store incorrect "insights" as facts (mis-evolved memory), which then biases future decisions and causes cumulative failures (loops/stagnation) and a performance ceiling.

We test whether treating new insights as tentative hypotheses (with simple evidence tracking) and only using high-confidence, non-contradicted items improves robustness.

**MVP Target**: Odyssey-Arena Turn On Lights (LightEnv), extensible to other Odyssey-Arena environments.

## Setup

### 1. Environment Setup

Create and activate conda environment:

```bash
conda create -n ody python=3.10
conda activate ody
```

### 2. Install Dependencies

```bash
cd Odyssey-Arena/experiments
pip install -r requirements.txt
```

**Note**: 
- For **vLLM** (local GPU): Requires CUDA-compatible GPU
- For **OpenAI/OpenRouter** (cloud API): Only needs `openai` package (no GPU)
- For **testing**: Use `--use_dummy_llm` flag (no installation required)

### 3. Verify Installation

Check that the LightEnv tasks are accessible:

```bash
ls ../test_data/turnonlights/test_turnonlights_lite_251030.json
```

## Running Experiments

### Basic Usage

**With vLLM (local GPU):**
```bash
python -m experiments.run \
  --env light --agent llm --provider vllm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --memory nomem --intervention none \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/nomem
```

**With OpenAI API:**
```bash
export OPENAI_API_KEY="your-key"
python -m experiments.run \
  --env light --agent llm --provider openai \
  --model gpt-4 \
  --memory reflector --intervention none \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/reflector_gpt4
```

**With OpenRouter:**
```bash
export OPENROUTER_API_KEY="your-key"
python -m experiments.run \
  --env light --agent llm --provider openrouter \
  --model anthropic/claude-3.5-sonnet \
  --memory reflector --intervention none \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/reflector_claude
```

### Key Arguments

**Environment & Agent:**
- `--env`: Environment type (`light`, `energy`, `repo`, `trade`)
- `--agent`: Agent type (`llm`, `random`, `heuristic`)

**LLM Configuration:**
- `--provider`: LLM provider (`vllm`, `openai`, `openrouter`, `dummy`)
- `--model`: Model name (e.g., `gpt-4`, `Qwen/Qwen2.5-7B-Instruct`, `anthropic/claude-3.5-sonnet`)
- `--api_key`: API key for cloud providers (or use environment variables)
- `--use_dummy_llm`: Use dummy LLM for testing (no GPU/API required)

**Memory & Intervention:**
- `--memory`: Memory system (`nomem`, `naive`, `gated`, `reflector`)
- `--intervention`: Intervention type (`none`, `injection`, `surgery`, `noise`)
- `--noise_p`: Noise probability for noise intervention (default: 0.05)
- `--surgery_step`: Step threshold for surgery intervention (default: 80)

**Experiment Settings:**
- `--num_episodes`: Number of episodes to run
- `--max_steps`: Maximum steps per episode
- `--seed`: Random seed
- `--output_dir`: Output directory for results

### Five Core Experiments

Run these five configurations to test the main hypotheses:

#### 1. No Memory Baseline

```bash
python -m experiments.run \
  --env light --agent llm --memory nomem --intervention none \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/nomem
```

#### 2. Naive Hypothesis Memory

```bash
python -m experiments.run \
  --env light --agent llm --memory naive --intervention none \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/naive
```

#### 3. Gated Hypothesis Memory (Verification)

```bash
python -m experiments.run \
  --env light --agent llm --memory gated --intervention none \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/gated
```

#### 4. Injection Intervention (False Memory Seeding)

```bash
python -m experiments.run \
  --env light --agent llm --memory naive --intervention injection \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/injection
```

#### 5. Surgery Intervention (Memory Suppression)

```bash
python -m experiments.run \
  --env light --agent llm --memory naive --intervention surgery \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/surgery
```

### Testing Without GPU

For quick testing without GPU:

```bash
python -m experiments.run \
  --env light --agent random --memory nomem --intervention none \
  --num_episodes 5 --max_steps 200 --seed 42 \
  --output_dir output/test_random
```

Or with dummy LLM:

```bash
python -m experiments.run \
  --env light --agent llm --memory nomem --intervention none \
  --use_dummy_llm \
  --num_episodes 5 --max_steps 200 --seed 42 \
  --output_dir output/test_dummy
```

## Visualizing Results

After running experiments, generate comparison plots:

```bash
python -m experiments.plot_quick \
  --output_dirs output/nomem output/naive output/gated output/injection output/surgery \
  --output_prefix comparison
```

This generates three PNG files:
- `comparison_success_rate.png`: Success rate comparison
- `comparison_steps.png`: Average steps to success
- `comparison_loop_memory.png`: Loop ratio and memory usage

## Output Files

Each experiment run creates three JSON files in the output directory:

- **`episodes.json`**: Detailed step-by-step logs for all episodes
- **`summary.json`**: Aggregated metrics (success rate, loop ratio, etc.)
- **`config.json`**: Full configuration used for the run

### Key Features

### 🤖 Multiple LLM Providers
- **vLLM**: Local GPU inference with HuggingFace models (fast, free)
- **OpenAI**: GPT-4, GPT-3.5-turbo via API (high quality, cost per token)
- **OpenRouter**: Access to Claude, Llama, and more (flexible, various pricing)
- **Dummy**: Testing without any LLM (instant, free)

### 🧠 4 Memory Systems
1. **NoMemory**: No memory (reactive baseline)
2. **NaiveHypMemory**: Store all hypotheses without verification
3. **GatedHypMemory**: Only use verified high-confidence hypotheses
4. **ReflectorCuratorMemory** (NEW!): 
   - Reflector stage: Analyzes experiences and extracts insights
   - Curator stage: Deduplicates and consolidates memories
   - Prevents mis-evolved memory through meta-cognitive reflection

### 📊 Key Metrics

- **Success rate**: Fraction of episodes where all bulbs were turned on
- **Loop ratio**: Fraction of actions that repeat (state, action) without progress
- **Avg steps**: Average number of steps taken (overall and for successful episodes)
- **Memory usage rate**: Fraction of steps where memory context was used
- **Invalid action rate**: Fraction of invalid actions generated

## Architecture

### Module Overview

- **`run.py`**: Main runner and CLI interface
- **`envs.py`**: Environment wrappers (LightEnv with normalized observations)
- **`llm.py`**: vLLM client for language model inference
- **`prompts.py`**: Prompt construction and action parsing
- **`agents.py`**: Agent implementations (LLM, Random, Heuristic)
- **`memory.py`**: Memory systems (NoMemory, NaiveHypMemory, GatedHypMemory)
- **`interventions.py`**: Intervention strategies (Injection, Surgery, Noise)
- **`metrics.py`**: Metrics computation and aggregation
- **`plot_quick.py`**: Quick plotting script for visualization

### Memory Systems

#### NoMemory (Baseline)
- No memory context provided to agent
- Pure reactive decision-making

#### NaiveHypMemory
- Stores hypotheses like "toggling bulb i flips bulb j"
- Tracks support/contradict counts
- Confidence: `sigmoid(support - 2*contradict)`
- Includes **all** hypotheses in prompt (even low-confidence)

#### GatedHypMemory (Verification-lite)
- Same tracking as NaiveHypMemory
- Only includes verified hypotheses in prompt:
  - `contradict == 0` (no contradictions)
  - `confidence >= 0.75`
  - `support >= 2` (at least 2 observations)

### Interventions

#### Injection
- Seeds false high-confidence hypotheses at episode start
- Example: H(1→2), H(2→4), H(3→5)
- Tests robustness to initially incorrect beliefs

#### Surgery
- Suppresses memory context after step 80
- Memory still updates but isn't used
- Tests recovery from memory dependence

#### Noise
- Flips each observation bit with probability `p` (default: 0.05)
- Only affects agent's perception, not true environment state
- Tests robustness to noisy observations

## Extending to Other Environments

To add support for other Odyssey-Arena environments:

1. **Add environment wrapper** in `envs.py`:
   ```python
   class EnergyEnvWrapper:
       def __init__(self, ...):
           # Initialize environment
       def reset(self): ...
       def step(self, action): ...
       def _normalize_obs(self, obs): ...
   ```

2. **Update factory function** in `envs.py`:
   ```python
   def create_env(env_type, task_data, seed=None):
       if env_type == 'energy':
           return EnergyEnvWrapper(...)
   ```

3. **Add task instructions** in `prompts.py`:
   ```python
   self.task_instructions = {
       "light": "...",
       "energy": "New instructions for energy env..."
   }
   ```

4. **Update CLI** in `run.py` to support new environment type.

## Citation

If you use this codebase in your research, please cite:

```
[Your paper citation here]
```

## License

[Your license here]
