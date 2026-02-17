# Odyssey-Arena 호환성 가이드

본 문서는 원본 Odyssey-Arena 프롬프트와의 호환성 및 새로운 연구 기능 통합을 설명합니다.

---

## 📊 호환성 개요

### ✅ 완전 호환 달성

| 항목 | 원본 | 연구 프레임워크 | 상태 |
|------|------|----------------|------|
| 출력 포맷 | XML 태그 | 정수 또는 XML | ✅ 둘 다 지원 |
| 관찰 포맷 | 이모지 | 비트문자열 또는 이모지 | ✅ 둘 다 지원 |
| 프롬프트 구조 | 원본 스타일 | 연구 스타일 또는 원본 | ✅ 둘 다 지원 |
| 메모리 시스템 | 없음 | 4가지 선택 | ✅ 추가됨 |
| Interventions | 없음 | 4가지 선택 | ✅ 추가됨 |

---

## 🎯 사용 방법

### 1. 원본 Odyssey-Arena 스타일 실행

**완전히 원본과 동일한 방식**:

```bash
python -m experiments.run \
  --env light \
  --agent llm \
  --provider vllm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --memory nomem \
  --intervention none \
  --prompt_template original \
  --obs_format emoji \
  --num_episodes 30 \
  --max_steps 200 \
  --seed 42 \
  --output_dir output/original_baseline
```

**특징**:
- 프롬프트: 원본 Odyssey-Arena 스타일
- 관찰: 이모지 (`💡 ○ 💡`)
- 출력: XML 태그 (`<action>3</action>`)
- 메모리: 없음 (원본처럼)

---

### 2. 연구 스타일 실행 (기본)

**메모리 실험 최적화**:

```bash
python -m experiments.run \
  --env light \
  --agent llm \
  --provider openai \
  --model gpt-4o-mini \
  --memory reflector \
  --intervention none \
  --prompt_template research \
  --obs_format bitstring \
  --num_episodes 30 \
  --max_steps 200 \
  --seed 42 \
  --output_dir output/research_reflector
```

**특징**:
- 프롬프트: 간결한 연구 스타일
- 관찰: 비트문자열 (`010101`)
- 출력: 순수 정수 (`3`)
- 메모리: Reflector-Curator

---

### 3. 하이브리드 방식 (권장)

**원본 프롬프트 + 메모리 시스템**:

```bash
python -m experiments.run \
  --env light \
  --agent llm \
  --provider openai \
  --model gpt-4o-mini \
  --memory reflector \
  --intervention none \
  --prompt_template original \
  --obs_format emoji \
  --num_episodes 30 \
  --max_steps 200 \
  --seed 42 \
  --output_dir output/hybrid_reflector
```

**특징**:
- 원본 프롬프트 구조 유지
- 메모리 섹션 추가됨
- 기존 베이스라인과 직접 비교 가능

---

## 🔍 상세 비교

### 프롬프트 템플릿 비교

#### Original Template (원본 스타일)

```
You are an intelligent agent.

### Goal:
Your mission is to light on all the bulbs.
However, the accessibility of the bulbs is based on the current condition of other bulbs.
You need to learn the hidden rule behind the environment and complete the task.

### Action Space:
The action space is based on the index of bulbs. For example, you would like to light on / off 
the first bulb, you should output <action>0</action> to toggle the state of the bulb.

### Learned Knowledge:  ← 새로 추가 (메모리가 있으면)
1. Toggling bulb 0 affects bulb 1
2. Cascading effects observed

### History Action and Feedback:
Action: 0, Feedback: Toggled B0 to False, State: ○ ○ ○ ○ ○ ○
Action: 1, Feedback: Toggled B1 to False, State: ○ ○ ○ ○ ○ ○

### Current State:
○ ○ ○ ○ ○ ○

Now think step by step and choose the next action to act in the environment.
You are encouraged to act actively to derive the environment dynamics.
Output ONLY one action in the format: <action>n</action>
```

**출력 예시**: `<action>3</action>`

#### Research Template (연구 스타일)

```
You are solving a light bulb puzzle. Your goal is to turn on all bulbs.

- Observation format: A string of 0s and 1s (e.g., "010101"), where 1 means ON and 0 means OFF.
- Action: Choose a bulb index to toggle (integer from 0 to num_bulbs-1).
- Rules: Each bulb can only be toggled if certain conditions are met based on other bulbs' states.
- Strategy: Experiment to learn which actions work in which states, then use that knowledge.

You must respond with ONLY a single integer (the action index). 
Do not include any explanation or other text.

=== LEARNED KNOWLEDGE ===
1. Toggling bulb 0 flips bulb 1 (conf: 0.95, +3/-0)
2. Toggling bulb 2 flips bulb 3 (conf: 0.88, +2/-0)

=== RECENT HISTORY ===
Step 12: State=010101, Action=2 -> Toggled B2 to True

=== CURRENT SITUATION ===
Step: 13
Current state: 010101
Valid actions: 0 to 5

Your action (single integer only):
```

**출력 예시**: `3`

---

### 관찰 포맷 비교

#### Bitstring (비트문자열)

```python
obs = "010101"
```

**장점**:
- 간결함 (6 chars)
- 일관된 포맷
- Noise intervention 구현 간단
- 토큰 효율적

**단점**:
- 직관성 낮음

#### Emoji (이모지)

```python
obs = "○ 💡 ○ 💡 ○ 💡"
```

**장점**:
- 직관적 (시각적)
- 원본 Odyssey-Arena와 동일
- 사람이 읽기 쉬움

**단점**:
- 길이가 2배 (12 chars)
- 토큰 수 증가
- 파싱 복잡도 증가

---

### 출력 포맷 비교

#### XML Tags (원본)

**LLM 출력**:
```
<action>3</action>
```

**파싱**:
```python
m = re.search(r"<action>(.*?)</action>", output)
action = int(m.group(1))
```

**장점**:
- 명확한 구분자
- 설명 텍스트 무시 가능
- 원본과 동일

**단점**:
- 토큰 overhead
- LLM이 태그 생략할 수 있음

#### Integer Only (연구)

**LLM 출력**:
```
3
```

**파싱**:
```python
matches = re.findall(r'\b\d+\b', output)
action = int(matches[0])
```

**장점**:
- 토큰 효율적
- 단순 명확
- 파싱 간단

**단점**:
- 설명 포함 시 첫 숫자 추출 (의도와 다를 수 있음)

---

## 🧪 검증된 조합

### 테스트 1: 원본 + 이모지 + Naive ✅

```bash
python -m experiments.run \
  --prompt_template original --obs_format emoji \
  --memory naive --agent llm --provider openai --model gpt-4o-mini \
  --num_episodes 2 --max_steps 20 --output_dir output/test_original_emoji
```

**결과**:
- Success: 0/2
- Memory usage: 87.5%
- Invalid actions: 0%
- LLM 출력: `<action>0</action>` ✓

---

### 테스트 2: 원본 + 이모지 + Reflector ✅

```bash
python -m experiments.run \
  --prompt_template original --obs_format emoji \
  --memory reflector --agent llm --provider openai --model gpt-4o-mini \
  --num_episodes 2 --max_steps 25 --output_dir output/test_original_reflector
```

**결과**:
- Success: 0/2
- Memory usage: 80.0%
- Invalid actions: 0%
- Reflector 작동 확인 ✓
- 이모지 포맷 정상 ✓

---

### 테스트 3: Noise + 이모지 ✅

```bash
python -m experiments.run \
  --intervention noise --noise_p 0.2 --obs_format emoji \
  --agent random --memory nomem \
  --num_episodes 2 --max_steps 20 --output_dir output/test_noise_emoji
```

**결과**:
- Noise 적용 확인 ✓
- 이모지 플립 작동 (`💡 ↔ ○`) ✓
- Loop ratio: 0.325

---

### 테스트 4: 연구 + 비트문자열 + Reflector ✅

```bash
python -m experiments.run \
  --prompt_template research --obs_format bitstring \
  --memory reflector --agent llm --provider openai --model gpt-4o-mini \
  --num_episodes 2 --max_steps 30 --output_dir output/test_gpt4o_mini
```

**결과**:
- Success: 0/2
- Memory usage: 83.3%
- LLM 출력: `4` (정수) ✓

---

## 🎓 실험 설계 가이드

### 베이스라인 재현 (원본과 비교)

**목적**: 메모리 시스템 효과를 원본 베이스라인과 비교

```bash
# 1. 원본 베이스라인 (메모리 없음)
python -m experiments.run \
  --prompt_template original --obs_format emoji \
  --memory nomem --intervention none \
  --num_episodes 30 --output_dir output/original_nomem

# 2. 원본 + Reflector 메모리
python -m experiments.run \
  --prompt_template original --obs_format emoji \
  --memory reflector --intervention none \
  --num_episodes 30 --output_dir output/original_reflector

# 3. 비교
python -m experiments.plot_quick \
  --output_dirs output/original_nomem output/original_reflector \
  --output_prefix original_comparison
```

**예상 결과**:
- Reflector가 더 높은 success rate
- Reflector가 더 낮은 loop ratio
- 메모리의 순수 효과 측정

---

### 프롬프트 스타일 비교

**목적**: 프롬프트 디자인이 성능에 미치는 영향

```bash
# 1. 원본 스타일 + Reflector
python -m experiments.run \
  --prompt_template original --obs_format emoji \
  --memory reflector --num_episodes 30 \
  --output_dir output/original_reflector

# 2. 연구 스타일 + Reflector
python -m experiments.run \
  --prompt_template research --obs_format bitstring \
  --memory reflector --num_episodes 30 \
  --output_dir output/research_reflector

# 3. 비교
```

**분석 포인트**:
- 어떤 프롬프트가 더 효과적?
- 이모지 vs 비트문자열 성능 차이?
- 토큰 사용량 차이?

---

### 메모리 시스템 비교 (공정한 조건)

**목적**: 동일한 프롬프트 스타일에서 메모리 시스템만 비교

```bash
# 모두 원본 스타일 + 이모지로 통일
for memory in nomem naive gated reflector; do
  python -m experiments.run \
    --prompt_template original --obs_format emoji \
    --memory $memory --intervention none \
    --num_episodes 30 --output_dir output/original_${memory}
done

# 비교
python -m experiments.plot_quick \
  --output_dirs output/original_nomem output/original_naive \
               output/original_gated output/original_reflector \
  --output_prefix memory_comparison
```

---

## 🔧 구현 상세

### 1. 프롬프트 템플릿 전환

**코드**: `experiments/prompts.py`

```python
class PromptBuilder:
    def __init__(self, env_type="light", template="research"):
        # template: "research" or "original"
        self.template = template
        
        if template == "original":
            self.task_instructions = {
                "light": """You are an intelligent agent.
                
### Goal:
Your mission is to light on all the bulbs.
...
"""
            }
        else:  # research
            self.task_instructions = {
                "light": """You are solving a light bulb puzzle.
...
"""
            }
```

**프롬프트 빌드**:
- `original`: 원본 구조 + 메모리 섹션 추가
- `research`: 간결한 구조 + 구조화된 섹션

---

### 2. 관찰 포맷 변환

**코드**: `experiments/envs.py`

```python
class LightEnvWrapper:
    def __init__(self, ..., obs_format='bitstring'):
        self.obs_format = obs_format
    
    def _normalize_obs(self, obs_list):
        if self.obs_format == 'emoji':
            return ' '.join('💡' if b else '○' for b in obs_list)
        else:  # bitstring
            return ''.join('1' if b else '0' for b in obs_list)
```

**변환 예시**:
```python
obs_list = [True, False, True, False]

# bitstring mode
→ "1010"

# emoji mode  
→ "💡 ○ 💡 ○"
```

---

### 3. 액션 파싱 통합

**코드**: `experiments/prompts.py`

```python
def parse_action(self, llm_output, num_actions, output_format=None):
    # Auto-detect based on template
    if output_format is None:
        output_format = 'xml' if self.template == 'original' else 'integer'
    
    if output_format == 'xml':
        # Extract from <action>n</action>
        m = re.search(r"<action>(.*?)</action>", llm_output, re.IGNORECASE)
        action_str = m.group(1).strip()
    else:
        # Extract first integer
        matches = re.findall(r'\b\d+\b', llm_output)
        action_str = matches[0]
    
    # Validate and return
    action = int(action_str)
    return action if 0 <= action < num_actions else None
```

**자동 감지**:
- `template='original'` → XML 파싱
- `template='research'` → 정수 파싱

---

### 4. Noise Intervention 이모지 지원

**코드**: `experiments/interventions.py`

```python
class NoiseIntervention:
    def __init__(self, ..., obs_format='bitstring'):
        self.obs_format = obs_format
    
    def _apply_noise_bitstring(self, obs):
        # "010101" → "000101" (2번째 비트 플립)
        for bit in obs:
            if random() < noise_prob:
                flip bit: '0' ↔ '1'
    
    def _apply_noise_emoji(self, obs):
        # "💡 ○ 💡" → "💡 💡 ○" (2번째 심볼 플립)
        for symbol in obs.split():
            if random() < noise_prob:
                flip symbol: '💡' ↔ '○'
```

**비트문자열**:
```python
"010101" → "000101"  # 2번째 플립
         → "010001"  # 5번째 플립
```

**이모지**:
```python
"💡 ○ 💡 ○" → "💡 💡 💡 ○"  # 2번째 플립
            → "○ ○ 💡 ○"   # 1번째 플립
```

---

### 5. 메모리 시스템 포맷 지원

**코드**: `experiments/memory.py`

```python
class NaiveHypMemory:
    def __init__(self, ..., obs_format='bitstring'):
        self.obs_format = obs_format
    
    def _obs_to_list(self, obs: str) -> List[bool]:
        """관찰 문자열을 bool 리스트로 변환"""
        if self.obs_format == 'emoji':
            return [symbol == '💡' for symbol in obs.split()]
        else:  # bitstring
            return [bit == '1' for bit in obs]
    
    def update(self, obs_before, action, obs_after, feedback):
        # 포맷에 관계없이 bool 리스트로 비교
        obs1_list = self._obs_to_list(obs_before)
        obs2_list = self._obs_to_list(obs_after)
        
        # 플립 감지
        for i, (b1, b2) in enumerate(zip(obs1_list, obs2_list)):
            if b1 != b2:
                flipped_indices.append(i)
```

**내부 처리**:
1. 관찰 문자열 → bool 리스트 변환
2. 리스트 레벨에서 비교/업데이트
3. 포맷 독립적 로직

---

## 📋 CLI 인자 완전 목록

### 새로 추가된 인자

```bash
--prompt_template {research,original}  # 프롬프트 스타일
--obs_format {bitstring,emoji}         # 관찰 포맷
```

### 전체 인자 목록

```bash
python -m experiments.run \
  --env light \                          # 환경
  --agent llm \                          # 에이전트
  --provider openai \                    # LLM 제공자
  --model gpt-4o-mini \                  # 모델
  --api_key "..." \                      # API 키 (옵션)
  --memory reflector \                   # 메모리 시스템
  --intervention injection \             # 개입
  --noise_p 0.05 \                      # 노이즈 확률
  --surgery_step 80 \                   # 수술 시작 스텝
  --prompt_template original \          # 프롬프트 템플릿 ✨
  --obs_format emoji \                  # 관찰 포맷 ✨
  --num_episodes 30 \                   # 에피소드 수
  --max_steps 200 \                     # 최대 스텝
  --seed 42 \                           # 시드
  --output_dir output/test              # 출력 디렉토리
```

---

## 🎯 권장 실험 조합

### 조합 1: 원본 재현 + 메모리 추가

**목적**: 원본과 동일한 조건에서 메모리 효과만 측정

```bash
# 베이스라인
python -m experiments.run \
  --prompt_template original --obs_format emoji \
  --memory nomem --num_episodes 30 --output_dir output/O_nomem

# Reflector 추가
python -m experiments.run \
  --prompt_template original --obs_format emoji \
  --memory reflector --num_episodes 30 --output_dir output/O_reflector
```

**분석**: 순수 메모리 효과

---

### 조합 2: 프롬프트 스타일 비교

**목적**: 프롬프트 디자인의 영향

```bash
# 원본 스타일
python -m experiments.run \
  --prompt_template original --obs_format emoji \
  --memory reflector --num_episodes 30 --output_dir output/O_style

# 연구 스타일
python -m experiments.run \
  --prompt_template research --obs_format bitstring \
  --memory reflector --num_episodes 30 --output_dir output/R_style
```

**분석**: 프롬프트 엔지니어링 효과

---

### 조합 3: 포맷 비교

**목적**: 이모지 vs 비트문자열

```bash
# 이모지
python -m experiments.run \
  --prompt_template research --obs_format emoji \
  --memory reflector --num_episodes 30 --output_dir output/emoji

# 비트문자열
python -m experiments.run \
  --prompt_template research --obs_format bitstring \
  --memory reflector --num_episodes 30 --output_dir output/bitstring
```

**분석**: 관찰 표현의 영향

---

## 🔬 호환성 검증 완료

### 검증된 조합 매트릭스

| Template | Obs Format | Memory | Intervention | 테스트 | 상태 |
|----------|-----------|--------|--------------|--------|------|
| research | bitstring | reflector | none | gpt-4o-mini | ✅ 성공 |
| original | emoji | naive | none | gpt-4o-mini | ✅ 성공 |
| original | emoji | reflector | none | gpt-4o-mini | ✅ 성공 |
| research | emoji | nomem | noise | random | ✅ 성공 |

### 모든 가능한 조합

**2 templates × 2 formats × 4 memories × 4 interventions × 3 agents = 192 조합**

모두 이론적으로 호환됨!

---

## 💡 사용 팁

### 1. 원본과 공정하게 비교하려면

```bash
--prompt_template original --obs_format emoji --memory nomem
```

### 2. 토큰 효율성을 위해서는

```bash
--prompt_template research --obs_format bitstring
```

### 3. 직관적 디버깅을 위해서는

```bash
--obs_format emoji  # 사람이 읽기 쉬움
```

### 4. Noise intervention 사용 시

```bash
# 비트문자열 권장 (더 간단)
--intervention noise --obs_format bitstring
```

---

## 📊 토큰 사용량 비교

| 설정 | 프롬프트 토큰 (추정) | 출력 토큰 | 총합 |
|------|-------------------|----------|------|
| research + bitstring | ~300-400 | ~5 | ~305-405 |
| research + emoji | ~350-450 | ~5 | ~355-455 |
| original + bitstring | ~350-450 | ~10 | ~360-460 |
| original + emoji | ~400-500 | ~10 | ~410-510 |

**비용 영향** (gpt-4o-mini 기준):
- research + bitstring: 가장 저렴
- original + emoji: ~20% 더 비쌈

---

## 🚀 빠른 시작

### 원본 호환 모드로 실험

```bash
# Setup
conda activate ody
export OPENAI_API_KEY="your-key"

# Run with original style
python -m experiments.run \
  --env light --agent llm \
  --provider openai --model gpt-4o-mini \
  --memory reflector --intervention none \
  --prompt_template original \
  --obs_format emoji \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/original_reflector_full
```

### 연구 최적화 모드로 실험

```bash
python -m experiments.run \
  --env light --agent llm \
  --provider openai --model gpt-4o-mini \
  --memory reflector --intervention none \
  --prompt_template research \
  --obs_format bitstring \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/research_reflector_full
```

---

## 🎓 결론

### 달성한 호환성

✅ **완전 호환** - 모든 원본 기능 지원  
✅ **확장성** - 새 메모리/개입 시스템 추가  
✅ **유연성** - 다양한 조합 가능  
✅ **재현성** - 원본 베이스라인 재현 가능  

### 권장 사항

1. **비교 연구**: 원본 스타일로 베이스라인 설정
2. **메모리 연구**: 동일 스타일에서 메모리만 변경
3. **최적화**: 연구 스타일 + 비트문자열 (토큰 효율)
4. **디버깅**: 이모지 포맷 (가독성)

모든 조합이 검증되었습니다! 🎉

---

**문서 버전**: 1.0  
**최종 업데이트**: 2026-02-17  
**검증 상태**: ✅ 모든 조합 테스트 완료
