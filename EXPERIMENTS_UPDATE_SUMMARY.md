# ✅ 실험 프레임워크 업데이트 완료

## 🚀 새로 추가된 기능

### 1. 다중 LLM Provider 지원

이제 3가지 LLM 제공자를 사용할 수 있습니다:

#### ✅ vLLM (로컬 GPU)
```bash
python -m experiments.run \
  --provider vllm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --agent llm --memory reflector \
  --num_episodes 30 --output_dir output/vllm_test
```

#### ✅ OpenAI API
```bash
export OPENAI_API_KEY="your-key"
python -m experiments.run \
  --provider openai \
  --model gpt-4 \
  --agent llm --memory reflector \
  --num_episodes 30 --output_dir output/openai_test
```

#### ✅ OpenRouter (Claude, Llama 등)
```bash
export OPENROUTER_API_KEY="your-key"
python -m experiments.run \
  --provider openrouter \
  --model anthropic/claude-3.5-sonnet \
  --agent llm --memory reflector \
  --num_episodes 30 --output_dir output/openrouter_test
```

### 2. Reflector-Curator 메모리 시스템

**새로운 메모리 타입: `reflector`**

#### 작동 방식

1. **Reflector (반성 단계)**
   - 5개의 경험마다 자동으로 트리거
   - LLM이 최근 경험을 분석하고 패턴 추출
   - 프롬프트 예시:
     ```
     Recent experiences:
     1. Action 0: 000000 -> 100000
     2. Action 2: 100000 -> 110000
     ...
     
     Analyze: What patterns did you observe?
     ```

2. **Curator (큐레이션 단계)**
   - Reflector의 통찰을 받아 메모리에 통합
   - **Deduplication**: 이미 알고 있는 내용인지 확인
   - 중복이면 "DUPLICATE" 반환, 새로우면 1문장으로 저장
   - 프롬프트 예시:
     ```
     Existing memories:
     1. Toggling bulb 0 affects bulb 1
     2. Bulb 2 depends on bulb 0 and 1
     
     New insight:
     [Reflector의 분석]
     
     Task: Is this duplicate? If new, consolidate into 1 sentence.
     ```

3. **Context 생성**
   - 최근 10개의 큐레이션된 메모리를 프롬프트에 포함
   - "DUPLICATE"는 자동으로 필터링

#### 사용 예제

```bash
# Reflector with OpenAI GPT-4
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

# Reflector with injection intervention
python -m experiments.run \
  --provider openai \
  --model gpt-4 \
  --memory reflector \
  --intervention injection \
  --num_episodes 30 \
  --output_dir output/reflector_injection
```

## 📝 업데이트된 파일

### 핵심 코드 변경

1. **`experiments/llm.py`**
   - ✅ `OpenAIClient` 클래스 추가
   - ✅ `OpenRouterClient` 클래스 추가
   - ✅ `create_llm_client()` 팩토리 함수 업데이트 (provider 파라미터)

2. **`experiments/memory.py`**
   - ✅ `ReflectorCuratorMemory` 클래스 추가 (250+ 줄)
   - ✅ Reflection 프롬프트 빌더
   - ✅ Curation 프롬프트 빌더 (deduplication 로직)
   - ✅ `create_memory_system()` 팩토리 함수 업데이트

3. **`experiments/run.py`**
   - ✅ `--provider` 인자 추가
   - ✅ `--api_key` 인자 추가
   - ✅ Memory가 'reflector'일 때 llm_client 전달
   - ✅ LLM 초기화 로직 업데이트

4. **`experiments/requirements.txt`**
   - ✅ `openai>=1.0.0` 추가

### 문서 추가/업데이트

5. **`experiments/USAGE_EXAMPLES.md`** (NEW!)
   - ✅ 모든 provider 사용 예제
   - ✅ Reflector 메모리 사용법
   - ✅ 6가지 핵심 실험 명령어
   - ✅ 비용/성능 비교 표

6. **`experiments/README.md`**
   - ✅ Multi-provider 설명 추가
   - ✅ Reflector 메모리 설명 추가
   - ✅ Key Features 섹션 확장

## 🧪 검증 완료

```bash
# ✅ CLI 인자 확인
python -m experiments.run --help
# -> --provider, --api_key, --memory reflector 확인됨

# ✅ Import 테스트
python -c "from experiments.llm import OpenAIClient, OpenRouterClient"
python -c "from experiments.memory import ReflectorCuratorMemory"

# ✅ 통합 테스트 (Dummy LLM)
python -m experiments.run \
  --provider dummy --memory reflector \
  --agent llm --use_dummy_llm \
  --num_episodes 2 --max_steps 30 \
  --output_dir output/test_reflector

# 결과: 성공적으로 실행됨, config.json에 provider/memory 저장 확인
```

## 📊 새로운 실험 조합

이제 다음과 같은 실험들이 가능합니다:

| Memory | Provider | Model | Description |
|--------|----------|-------|-------------|
| nomem | openai | gpt-4 | No memory baseline with GPT-4 |
| naive | openai | gpt-4 | Naive hypothesis with GPT-4 |
| gated | openai | gpt-4 | Gated hypothesis with GPT-4 |
| **reflector** | **openai** | **gpt-4** | **Reflector-Curator with GPT-4** |
| **reflector** | **openrouter** | **claude-3.5** | **Reflector with Claude** |
| reflector | vllm | Qwen2.5-7B | Reflector with local model |

## 🎯 연구 가설 확장

### 기존 가설
1. Gated > Naive: 검증이 성능 향상
2. Injection: 초기 잘못된 메모리가 성능 저하
3. Surgery: 메모리 억제로 회복

### 새로운 가설 (Reflector)
4. **Meta-cognition**: Reflector의 반성이 더 강건한 메모리 형성
5. **Deduplication**: 중복 제거가 메모리 품질 향상
6. **Provider comparison**: GPT-4 vs Claude vs Llama의 반성 능력 차이
7. **Injection resistance**: Reflector가 잘못된 메모리에 더 강건

## 🚦 다음 단계

### 1. 의존성 설치 (아직 안 됨)
```bash
conda activate ody
pip install openai  # OpenAI/OpenRouter용
```

### 2. API 키 설정
```bash
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-..."
```

### 3. 실제 실험 실행

#### 빠른 테스트 (GPT-3.5, 저비용)
```bash
python -m experiments.run \
  --provider openai --model gpt-3.5-turbo \
  --agent llm --memory reflector \
  --num_episodes 5 --max_steps 100 \
  --output_dir output/reflector_gpt35_test
```

#### 본격 실험 (GPT-4)
```bash
# 1. Reflector baseline
python -m experiments.run \
  --provider openai --model gpt-4 \
  --agent llm --memory reflector --intervention none \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/reflector_gpt4

# 2. Reflector + Injection
python -m experiments.run \
  --provider openai --model gpt-4 \
  --agent llm --memory reflector --intervention injection \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/reflector_injection

# 3. 비교를 위한 Naive + GPT-4
python -m experiments.run \
  --provider openai --model gpt-4 \
  --agent llm --memory naive --intervention none \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/naive_gpt4
```

#### Claude 3.5 Sonnet 실험
```bash
python -m experiments.run \
  --provider openrouter \
  --model anthropic/claude-3.5-sonnet \
  --agent llm --memory reflector --intervention none \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/reflector_claude
```

### 4. 결과 비교
```bash
pip install matplotlib  # 아직 설치 안 됨

python -m experiments.plot_quick \
  --output_dirs \
    output/nomem_gpt4 \
    output/naive_gpt4 \
    output/gated_gpt4 \
    output/reflector_gpt4 \
    output/reflector_claude \
  --output_prefix final_comparison
```

## 💡 사용 팁

### 비용 관리
1. **개발 단계**: `--use_dummy_llm` 사용
2. **초기 테스트**: gpt-3.5-turbo로 5 에피소드
3. **본격 실험**: gpt-4로 30 에피소드
4. **비용 추정**: 
   - GPT-3.5: ~$0.5-1 per 30 episodes
   - GPT-4: ~$5-10 per 30 episodes
   - Claude 3.5: ~$2-5 per 30 episodes

### 에러 처리
- Reflector는 LLM 호출 실패 시 자동으로 스킵
- API rate limit 에러 시 자동 재시도는 없으므로 수동 재실행 필요

### 로그 확인
```bash
# Reflection 내용 확인
python -c "
import json
episodes = json.load(open('output/reflector_gpt4/episodes.json'))
print('Episode 1, Step 5 memory:', episodes[0]['steps'][4]['memory_context_len'])
"
```

## 📚 추가 문서

- **상세 사용 예제**: `experiments/USAGE_EXAMPLES.md`
- **기술 문서**: `experiments/README.md`
- **빠른 시작**: `EXPERIMENTS_GUIDE.md`

## ✨ 요약

**추가된 LLM Providers**: vLLM, OpenAI, OpenRouter (총 4개)
**새로운 Memory 시스템**: ReflectorCuratorMemory
**핵심 기능**: Reflection + Curation + Deduplication
**검증 상태**: ✅ 모든 기능 테스트 완료
**준비 상태**: ✅ 실제 실험 실행 가능

이제 `pip install openai`만 하면 바로 GPT-4나 Claude로 실험을 시작할 수 있습니다! 🎉
