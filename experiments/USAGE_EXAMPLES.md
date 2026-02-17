# 사용 예제: Multi-Provider LLM + Reflector Memory

## 🚀 다양한 LLM Provider로 실험하기

### 1. vLLM (로컬 GPU 추론)

```bash
# 기본 설정 - vLLM with Qwen2.5-7B
python -m experiments.run \
  --env light \
  --agent llm \
  --provider vllm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --memory reflector \
  --intervention none \
  --num_episodes 30 \
  --max_steps 200 \
  --seed 42 \
  --output_dir output/reflector_vllm
```

### 2. OpenAI API

```bash
# GPT-4를 사용한 실험
export OPENAI_API_KEY="your-api-key"

python -m experiments.run \
  --env light \
  --agent llm \
  --provider openai \
  --model gpt-4 \
  --memory reflector \
  --intervention none \
  --num_episodes 30 \
  --max_steps 200 \
  --seed 42 \
  --output_dir output/reflector_gpt4

# 또는 API key를 직접 전달
python -m experiments.run \
  --provider openai \
  --model gpt-4 \
  --api_key "your-api-key" \
  --agent llm --memory reflector \
  --num_episodes 30 --output_dir output/gpt4_test
```

### 3. OpenRouter (다양한 모델 접근)

```bash
# Claude 3.5 Sonnet 사용
export OPENROUTER_API_KEY="your-api-key"

python -m experiments.run \
  --env light \
  --agent llm \
  --provider openrouter \
  --model anthropic/claude-3.5-sonnet \
  --memory reflector \
  --intervention none \
  --num_episodes 30 \
  --max_steps 200 \
  --seed 42 \
  --output_dir output/reflector_claude

# Llama 3.1 사용 (OpenRouter를 통해)
python -m experiments.run \
  --provider openrouter \
  --model meta-llama/llama-3.1-70b-instruct \
  --agent llm --memory reflector \
  --num_episodes 30 --output_dir output/llama31_test
```

## 🧠 Reflector-Curator 메모리 시스템

### 특징

1. **Reflector**: 5개의 경험마다 패턴을 분석하고 통찰 추출
2. **Curator**: 중복 제거 및 메모리 통합
3. **Deduplication**: 이미 알고 있는 내용은 "DUPLICATE"로 필터링

### 사용 예제

```bash
# Reflector memory with OpenAI
python -m experiments.run \
  --env light \
  --agent llm \
  --provider openai \
  --model gpt-3.5-turbo \
  --memory reflector \
  --intervention none \
  --num_episodes 10 \
  --max_steps 150 \
  --seed 42 \
  --output_dir output/reflector_openai

# Reflector memory with injection intervention
python -m experiments.run \
  --provider openai \
  --model gpt-4 \
  --agent llm \
  --memory reflector \
  --intervention injection \
  --num_episodes 30 \
  --output_dir output/reflector_injection
```

## 📊 6가지 핵심 실험 (확장 버전)

### 1. No Memory Baseline (nomem)

```bash
python -m experiments.run \
  --provider openai --model gpt-4 \
  --agent llm --memory nomem --intervention none \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/nomem_gpt4
```

### 2. Naive Hypothesis Memory

```bash
python -m experiments.run \
  --provider openai --model gpt-4 \
  --agent llm --memory naive --intervention none \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/naive_gpt4
```

### 3. Gated Hypothesis Memory

```bash
python -m experiments.run \
  --provider openai --model gpt-4 \
  --agent llm --memory gated --intervention none \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/gated_gpt4
```

### 4. Reflector-Curator Memory (NEW!)

```bash
python -m experiments.run \
  --provider openai --model gpt-4 \
  --agent llm --memory reflector --intervention none \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/reflector_gpt4
```

### 5. Injection Intervention

```bash
python -m experiments.run \
  --provider openai --model gpt-4 \
  --agent llm --memory reflector --intervention injection \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/reflector_injection
```

### 6. Surgery Intervention

```bash
python -m experiments.run \
  --provider openai --model gpt-4 \
  --agent llm --memory reflector --intervention surgery \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/reflector_surgery
```

## 🧪 빠른 테스트 (비용 절감)

### Dummy LLM으로 구조 테스트

```bash
python -m experiments.run \
  --use_dummy_llm \
  --agent llm --memory reflector \
  --num_episodes 3 --max_steps 50 \
  --output_dir output/test_reflector_dummy
```

### GPT-3.5-turbo로 저비용 테스트

```bash
python -m experiments.run \
  --provider openai --model gpt-3.5-turbo \
  --agent llm --memory reflector \
  --num_episodes 5 --max_steps 100 \
  --output_dir output/test_reflector_gpt35
```

## 🔬 고급 설정

### Reflection 빈도 조정

Reflector memory는 기본적으로 5개 경험마다 반성합니다. 이를 조정하려면 코드 수정이 필요:

```python
# experiments/run.py에서
memory_kwargs = {
    'llm_client': self.llm_client,
    'reflection_frequency': 3,  # 3개 경험마다 반성
    'max_memories': 30,         # 최대 30개 메모리 유지
    'max_context_items': 15     # 프롬프트에 15개까지 포함
}
```

### 모델별 비용/성능 비교

| Provider | Model | Cost/1M tokens | Speed | Quality |
|----------|-------|----------------|-------|---------|
| vLLM | Qwen2.5-7B | Free (local) | Fast | Good |
| OpenAI | gpt-3.5-turbo | $0.50-$1.50 | Fast | Good |
| OpenAI | gpt-4 | $30-$60 | Slow | Excellent |
| OpenRouter | claude-3.5-sonnet | $3-$15 | Medium | Excellent |
| OpenRouter | llama-3.1-70b | $0.50-$1.00 | Medium | Very Good |

## 📈 결과 비교

모든 실험 완료 후:

```bash
python -m experiments.plot_quick \
  --output_dirs \
    output/nomem_gpt4 \
    output/naive_gpt4 \
    output/gated_gpt4 \
    output/reflector_gpt4 \
    output/reflector_injection \
    output/reflector_surgery \
  --output_prefix comparison_all
```

## 🎯 연구 가설 검증

1. **Reflector vs Naive/Gated**: Reflector의 메타인지적 반성이 더 강건한 메모리 형성
2. **Deduplication 효과**: 중복 제거가 메모리 품질 향상
3. **Provider 비교**: GPT-4 vs Claude vs Llama의 반성 능력 차이
4. **Injection 저항성**: Reflector가 잘못된 메모리 주입에 더 강건한지 검증

## 💡 팁

1. **비용 관리**: 먼저 gpt-3.5-turbo나 dummy로 테스트 후 gpt-4 실행
2. **API 제한**: OpenAI rate limit에 유의 (TPM/RPM)
3. **로그 확인**: 반성 내용은 episodes.json에서 확인 가능
4. **에러 처리**: Reflector는 LLM 호출 실패 시 자동으로 스킵
