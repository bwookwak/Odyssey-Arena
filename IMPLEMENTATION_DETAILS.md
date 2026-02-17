# 구현 상세 문서

본 문서는 Mis-evolved Memory 실험 프레임워크의 핵심 구현 내용을 상세히 설명합니다.

---

## 📋 목차
1. [Prompts 구현](#1-prompts-구현)
2. [Metrics 구현](#2-metrics-구현)
3. [Interventions 구현](#3-interventions-구현)
4. [Memory Systems 구현](#4-memory-systems-구현)
5. [Reflector-Curator 상세](#5-reflector-curator-상세)

---

## 1. Prompts 구현

### 1.1 PromptBuilder 클래스

**위치**: `experiments/prompts.py`

**주요 기능**: LLM 에이전트를 위한 프롬프트 구성

### 1.2 프롬프트 구조

프롬프트는 다음 섹션으로 구성됩니다:

```
[태스크 설명]
↓
[학습된 지식] (메모리가 있으면)
↓
[최근 히스토리] (옵션, 최근 3 스텝)
↓
[현재 상황]
↓
[액션 요청]
```

### 1.3 LightEnv 태스크 인스트럭션

```python
"""You are solving a light bulb puzzle. Your goal is to turn on all bulbs.

- Observation format: A string of 0s and 1s (e.g., "010101"), 
  where 1 means ON and 0 means OFF.
- Action: Choose a bulb index to toggle (integer from 0 to num_bulbs-1).
- Rules: Each bulb can only be toggled if certain conditions are met 
  based on other bulbs' states.
- Strategy: Experiment to learn which actions work in which states, 
  then use that knowledge.

You must respond with ONLY a single integer (the action index). 
Do not include any explanation or other text.
"""
```

**핵심 포인트**:
- 관찰 포맷 명확화: "010101" 형태의 비트 문자열
- 액션 명세: 0부터 num_bulbs-1까지의 정수
- 출력 제약: 정수만 출력 (설명 불허)

### 1.4 메모리 컨텍스트 통합

```python
def build_prompt(..., memory_context: Optional[str] = None, ...):
    if memory_context:
        prompt_parts.append("=== LEARNED KNOWLEDGE ===")
        prompt_parts.append(memory_context)
        prompt_parts.append("")
```

**메모리 컨텍스트 예시**:
```
=== LEARNED KNOWLEDGE ===
1. Toggling bulb 0 flips bulb 1 (conf: 0.92, +3/-0)
2. Toggling bulb 2 flips bulb 3 (conf: 0.88, +2/-0)
```

### 1.5 액션 파싱

**정규식 기반 파싱**:
```python
def parse_action(self, llm_output: str, num_actions: int):
    # 첫 번째 정수 추출
    matches = re.findall(r'\b\d+\b', llm_output)
    action = int(matches[0])
    
    # 유효성 검증
    if 0 <= action < num_actions:
        return action
    else:
        return None  # Invalid
```

**처리 로직**:
1. LLM 출력에서 정수 패턴 찾기 (`\b\d+\b`)
2. 첫 번째 매치 선택
3. 범위 검증: `[0, num_actions)`
4. 무효 시 `None` 반환

---

## 2. Metrics 구현

### 2.1 메트릭 계층 구조

```
EpisodeMetrics (단일 에피소드)
    ↓
    record_step() × N
    ↓
    compute_metrics()
    ↓
AggregateMetrics (여러 에피소드)
    ↓
    compute_aggregate()
```

### 2.2 EpisodeMetrics 클래스

**위치**: `experiments/metrics.py`

**추적 항목**:

| 항목 | 타입 | 설명 |
|------|------|------|
| `steps` | list | 스텝별 상세 로그 |
| `success` | bool | 에피소드 성공 여부 |
| `num_steps` | int | 총 스텝 수 |
| `seen_states` | dict | 루프 탐지용: `(obs, action) → max_progress` |
| `loop_count` | int | 루프 횟수 |
| `memory_used_count` | int | 메모리 사용 스텝 수 |
| `invalid_action_count` | int | 무효 액션 횟수 |

### 2.3 Loop Detection (루프 탐지)

**핵심 알고리즘**:

```python
# 각 스텝마다
state_action = (obs_before, action)

if state_action in seen_states:
    prev_max_progress = seen_states[state_action]
    
    # 진전 없이 같은 행동 반복 → 루프
    if progress <= prev_max_progress:
        loop_count += 1

# 최대 진전도 업데이트
seen_states[state_action] = max(seen_states.get(state_action, -1), progress)
```

**Loop Ratio 계산**:
```python
loop_ratio = loop_count / total_actions
```

**의미**:
- `loop_ratio = 0.0`: 루프 없음, 항상 새로운 시도
- `loop_ratio = 0.5`: 50%가 진전 없는 반복
- `loop_ratio = 1.0`: 모든 액션이 반복 (매우 나쁨)

### 2.4 Per-Episode Metrics

**계산되는 메트릭**:

```python
{
    'success': bool,              # 에피소드 성공 여부
    'num_steps': int,             # 총 스텝 수
    'loop_ratio': float,          # 루프 비율 [0, 1]
    'memory_usage_rate': float,   # 메모리 사용 비율 [0, 1]
    'invalid_action_rate': float, # 무효 액션 비율 [0, 1]
    'loop_count': int,            # 루프 발생 횟수
    'total_actions': int          # 총 액션 수
}
```

### 2.5 Aggregate Metrics

**여러 에피소드 집계**:

```python
{
    'num_episodes': int,              # 총 에피소드 수
    'num_success': int,               # 성공한 에피소드 수
    'success_rate': float,            # 성공률 [0, 1]
    'avg_steps_all': float,           # 평균 스텝 (전체)
    'avg_steps_success': float,       # 평균 스텝 (성공만)
    'avg_loop_ratio': float,          # 평균 루프 비율
    'avg_memory_usage_rate': float,   # 평균 메모리 사용률
    'avg_invalid_action_rate': float  # 평균 무효 액션률
}
```

**Success Rate vs Avg Steps Success**:
- `success_rate = 0.7` → 70% 성공
- `avg_steps_success = 45` → 성공 시 평균 45스텝 소요

---

## 3. Interventions 구현

### 3.1 Intervention 기본 인터페이스

**위치**: `experiments/interventions.py`

**3가지 적용 지점**:

```python
class Intervention:
    def apply_at_start(self, memory_system, env):
        """에피소드 시작 시"""
        pass
    
    def apply_to_observation(self, obs: str, step: int):
        """매 스텝 관찰 시"""
        return obs
    
    def should_suppress_memory(self, step: int):
        """메모리 억제 여부"""
        return False
```

---

### 3.2 NoIntervention (베이스라인)

**구현**:
```python
class NoIntervention(Intervention):
    # 모든 메서드가 pass 또는 False 반환
    pass
```

**목적**: 개입 없는 순수 베이스라인

---

### 3.3 InjectionIntervention (거짓 메모리 주입)

**목적**: 초기에 잘못된 고신뢰도 가설을 주입하여 mis-evolved memory 시뮬레이션

**주입 패턴**:
```python
false_patterns = [
    (1, 2),  # "Toggling bulb 1 flips bulb 2"
    (2, 4),  # "Toggling bulb 2 flips bulb 4"
    (3, 5)   # "Toggling bulb 3 flips bulb 5"
]
```

**주입 값**:
```python
{
    'support': 3,        # 3번 관찰한 것처럼
    'contradict': 0,     # 모순 0회
    'confidence': 0.95   # 95% 신뢰도
}
```

**적용 시점**: `apply_at_start()`에서 메모리에 직접 주입

**코드**:
```python
def apply_at_start(self, memory_system, env):
    num_actions = env.get_num_actions()
    
    for action, flip_idx in self.false_patterns:
        if action < num_actions and flip_idx < num_actions:
            if hasattr(memory_system, 'hyp_store'):
                key = (action, flip_idx)
                memory_system.hyp_store[key] = {
                    'support': self.inject_support,
                    'contradict': 0,
                    'confidence': self.inject_confidence
                }
```

**연구 가설**: Naive 메모리는 이 거짓 정보를 그대로 사용하지만, Gated나 Reflector는 필터링/검증할 것

---

### 3.4 SurgeryIntervention (메모리 억제)

**목적**: 특정 스텝 이후 메모리 사용 중단 (메모리 의존성 테스트)

**파라미터**:
```python
suppress_after_step: int = 80  # 기본값
```

**동작**:
```python
def should_suppress_memory(self, step: int):
    return step >= self.suppress_after_step
```

**효과**:
- Step 0~79: 메모리 컨텍스트 사용
- Step 80+: 메모리 컨텍스트 억제 (빈 문자열)
- **메모리는 계속 업데이트됨** (사용만 안 함)

**연구 가설**: 메모리 의존도가 높은 에이전트는 80스텝 이후 성능 저하

---

### 3.5 NoiseIntervention (관찰 노이즈)

**목적**: 노이즈가 있는 환경 시뮬레이션 (부분적/불확실한 관찰)

**파라미터**:
```python
noise_prob: float = 0.05  # 각 비트가 플립될 확률
```

**구현**:
```python
def apply_to_observation(self, obs: str, step: int):
    noisy_bits = []
    for bit in obs:
        if self.rng.random() < self.noise_prob:
            # 비트 플립
            noisy_bits.append('0' if bit == '1' else '1')
        else:
            noisy_bits.append(bit)
    return ''.join(noisy_bits)
```

**예시** (noise_prob=0.1):
```
True obs:  "010101"
Noisy obs: "000101"  # 2번째 비트가 플립됨
           "010111"  # 6번째 비트가 플립됨
```

**중요**: 
- 노이즈는 **에이전트의 관찰에만** 적용
- 실제 환경 상태는 변하지 않음
- 메모리 업데이트는 **진짜 관찰** 사용

**연구 가설**: Reflector/Gated 메모리가 노이즈에 더 강건할 것

---

## 4. Memory Systems 구현

### 4.1 메모리 시스템 비교표

| 메모리 타입 | 저장 | 필터링 | 검증 | LLM 사용 |
|------------|------|--------|------|----------|
| **NoMemory** | ✗ | N/A | N/A | ✗ |
| **NaiveHypMemory** | ✓ | ✗ | ✗ | ✗ |
| **GatedHypMemory** | ✓ | ✓ | ✓ | ✗ |
| **ReflectorCuratorMemory** | ✓ | ✓ | ✓ | ✓ |

---

### 4.2 NoMemory (베이스라인)

**구현**:
```python
class NoMemory(MemorySystem):
    def update(self, ...):
        pass  # 아무것도 안 함
    
    def get_context(self, ...):
        return ""  # 항상 빈 컨텍스트
```

**목적**: 순수 반응형 에이전트, 메모리 효과 측정의 대조군

---

### 4.3 NaiveHypMemory (순진한 가설 메모리)

**저장 구조**:
```python
hyp_store = {
    (action, flip_idx): {
        'support': int,      # 지지 횟수
        'contradict': int,   # 모순 횟수
        'confidence': float  # 신뢰도 [0, 1]
    }
}
```

**예시**:
```python
{
    (0, 1): {'support': 3, 'contradict': 0, 'confidence': 0.95},
    (2, 3): {'support': 2, 'contradict': 1, 'confidence': 0.27}
}
```

#### Update 로직

**1단계: 비트 변화 감지**
```python
flipped_indices = []
for i, (b1, b2) in enumerate(zip(obs_before, obs_after)):
    if b1 != b2:
        flipped_indices.append(i)
```

**2단계: Support 업데이트**
```python
for flip_idx in flipped_indices:
    key = (action, flip_idx)
    hyp_store[key]['support'] += 1
```

**3단계: Contradict 업데이트**
```python
for bit_idx in range(len(obs_before)):
    if bit_idx not in flipped_indices:
        key = (action, bit_idx)
        if key in hyp_store:  # 기존 가설이 있으면
            hyp_store[key]['contradict'] += 1
```

**4단계: Confidence 재계산**
```python
score = support - 2 * contradict
confidence = 1.0 / (1.0 + exp(-score))  # Sigmoid
```

**Confidence 함수 특성**:
- `support=3, contradict=0` → conf ≈ 0.95
- `support=2, contradict=1` → conf ≈ 0.50
- `support=1, contradict=2` → conf ≈ 0.05

#### Get Context

**정렬 및 출력**:
```python
# Confidence 내림차순 정렬
sorted_hyps = sorted(hyp_store.items(), 
                    key=lambda x: x[1]['confidence'], 
                    reverse=True)

# Top-10 출력
for i, ((act, flip_idx), stats) in enumerate(sorted_hyps[:10], 1):
    context += f"{i}. Toggling bulb {act} flips bulb {flip_idx} "
    context += f"(conf: {stats['confidence']:.2f}, "
    context += f"+{stats['support']}/-{stats['contradict']})\n"
```

**출력 예시**:
```
1. Toggling bulb 0 flips bulb 1 (conf: 0.95, +3/-0)
2. Toggling bulb 2 flips bulb 3 (conf: 0.88, +2/-0)
3. Toggling bulb 1 flips bulb 2 (conf: 0.73, +2/-1)
```

**문제점**: 
- ✗ 저신뢰도 가설도 포함 (잡음)
- ✗ 모순된 가설도 포함 (잘못된 학습)

---

### 4.4 GatedHypMemory (게이트 가설 메모리)

**NaiveHypMemory 상속** + **검증 필터 추가**

**필터링 기준**:
```python
min_confidence: float = 0.75
min_support: int = 2
max_contradictions: int = 0
```

**Get Context (필터링 버전)**:
```python
verified_hyps = []
for key, stats in hyp_store.items():
    if (stats['contradict'] <= 0 and          # 모순 없음
        stats['confidence'] >= 0.75 and       # 높은 신뢰도
        stats['support'] >= 2):               # 충분한 지지
        verified_hyps.append((key, stats))

# 검증된 가설만 반환
```

**NaiveHypMemory vs GatedHypMemory**:

| 시나리오 | Naive | Gated |
|---------|-------|-------|
| support=3, contradict=0, conf=0.95 | ✓ 포함 | ✓ 포함 |
| support=2, contradict=0, conf=0.88 | ✓ 포함 | ✓ 포함 |
| support=1, contradict=0, conf=0.73 | ✓ 포함 | ✗ 제외 (support < 2) |
| support=2, contradict=1, conf=0.50 | ✓ 포함 | ✗ 제외 (contradict > 0) |

**연구 가설**: Gated가 잡음을 필터링하여 더 나은 성능

---

## 5. Reflector-Curator 상세

### 5.1 ReflectorCuratorMemory 구조

**2단계 파이프라인**:

```
Raw Experiences (5개 누적)
    ↓
[Reflector Stage]
    - LLM이 패턴 분석
    - 통찰 추출
    ↓
Reflection Output
    ↓
[Curator Stage]
    - 기존 메모리와 비교
    - 중복 제거
    - 통합/단순화
    ↓
Curated Memory Item
    ↓
Memory Context (프롬프트에 포함)
```

---

### 5.2 Reflector Stage (반성 단계)

#### Input

**Raw Experiences** (5개 누적):
```python
[
    {
        'obs_before': '000000',
        'action': 0,
        'obs_after': '100000',
        'feedback': 'Toggled B0 to True'
    },
    {
        'obs_before': '100000',
        'action': 1,
        'obs_after': '110000',
        'feedback': 'Toggled B1 to True'
    },
    # ... 3 more experiences
]
```

#### Reflection Prompt

**템플릿**:
```
You are analyzing experiences from a light bulb puzzle game.

Recent experiences (state changes after toggling bulbs):
1. Action 0: 000000 -> 100000
2. Action 1: 100000 -> 110000
3. Action 0: 110000 -> 010000
4. Action 2: 010000 -> 011000
5. Action 1: 011000 -> 001000

Analyze these experiences and answer:
1. What patterns did you observe? (Which actions caused which state changes?)
2. What relationships between bulbs did you discover?
3. What strategy insights can you extract?

Provide a concise analysis (2-3 sentences focusing on actionable patterns):
```

#### Output

**LLM 생성 예시**:
```
"Based on the observations, toggling bulb 0 affects bulb 1, as seen when 
action 0 turned both bulbs on or off together. Additionally, there appears 
to be a cascading effect where certain bulbs influence adjacent bulbs, 
requiring a strategic sequence to achieve the goal."
```

**특징**:
- 200 tokens 제한
- Temperature 0.7 (창의적)
- 패턴 중심 분석

---

### 5.3 Curator Stage (큐레이션 단계)

#### Input

1. **Reflection** (위에서 생성됨)
2. **Existing Memories** (최근 10개):
```python
[
    "Toggling bulb 0 affects bulb 1 directly",
    "Bulb 2 depends on bulbs 0 and 1 being ON",
    # ...
]
```

#### Curation Prompt

**템플릿**:
```
You are a memory curator managing knowledge about a light bulb puzzle.

EXISTING MEMORIES:
1. Toggling bulb 0 affects bulb 1 directly
2. Bulb 2 depends on bulbs 0 and 1 being ON

NEW INSIGHT:
"Based on the observations, toggling bulb 0 affects bulb 1, as seen when 
action 0 turned both bulbs on or off together. Additionally, there appears 
to be a cascading effect where certain bulbs influence adjacent bulbs, 
requiring a strategic sequence to achieve the goal."

TASK:
1. Check if this insight is ALREADY COVERED by existing memories (deduplication)
2. If duplicate: respond with "DUPLICATE"
3. If new or partially overlapping: consolidate and return ONE concise memory item (max 1 sentence)
4. Focus on actionable patterns like "Toggling bulb X affects bulb Y"

Your response (either "DUPLICATE" or a new memory item):
```

#### Output

**시나리오 1: 중복**
```
DUPLICATE
```
→ 저장하지 않음

**시나리오 2: 새로운 통찰**
```
Toggling bulb X creates cascading effects on adjacent bulbs requiring sequential strategy
```
→ 메모리에 저장

**시나리오 3: 부분 중복 + 새 정보**
```
Toggling bulb 0 affects bulb 1 with additional cascading effects on nearby bulbs
```
→ 통합하여 저장

**특징**:
- 150 tokens 제한
- Temperature 0.5 (보수적)
- 1문장으로 단순화
- Deduplication 강조

---

### 5.4 Memory Storage

**저장 구조**:
```python
curated_memories = [
    {
        'memory': "Toggling bulb 0 affects bulb 1 directly",
        'reflection': "Based on the observations...",  # 원본 반성
        'timestamp': 0
    },
    {
        'memory': "DUPLICATE",
        'reflection': "Action 0 flips bulb 1...",
        'timestamp': 1
    },
    {
        'memory': "Bulb 2 requires bulbs 0 and 1 to be ON first",
        'reflection': "I noticed that...",
        'timestamp': 2
    },
    # ...
]
```

**크기 제한**:
```python
if len(curated_memories) > max_memories:  # 기본 20
    curated_memories = curated_memories[-max_memories:]  # 최근 20개만 유지
```

---

### 5.5 Get Context

**필터링 + 포맷팅**:
```python
# 최근 10개 선택
recent_memories = curated_memories[-10:]

# DUPLICATE 제거
filtered_memories = [
    mem['memory'] 
    for mem in recent_memories 
    if mem['memory'].upper() != "DUPLICATE"
]

# 번호 매겨 출력
context = "\n".join([
    f"{i+1}. {mem}" 
    for i, mem in enumerate(filtered_memories)
])
```

**출력 예시**:
```
1. Toggling bulb 0 affects bulb 1 directly
2. Bulb 2 requires bulbs 0 and 1 to be ON first
3. Cascading effects require sequential strategy
```

---

### 5.6 Reflector-Curator의 장점

| 측면 | NaiveHypMemory | ReflectorCuratorMemory |
|------|----------------|------------------------|
| **저장 방식** | 저수준 (action→flip) | 고수준 (패턴/전략) |
| **중복 처리** | ✗ 없음 | ✓ Curator가 제거 |
| **잡음 필터** | ✗ 모든 가설 저장 | ✓ LLM이 판단 |
| **통찰 수준** | 단순 인과관계 | 전략적 패턴 |
| **메타인지** | ✗ 없음 | ✓ 반성 단계 |
| **비용** | 무료 (계산만) | 비쌈 (LLM 호출) |

**연구 가설**:
1. Reflector가 더 추상적이고 유용한 지식 생성
2. Deduplication으로 메모리 효율성 증가
3. 메타인지적 반성이 mis-evolved memory 방지
4. 노이즈/주입 개입에 더 강건

---

### 5.7 ReflectorCuratorMemory 통계

```python
def get_stats(self):
    return {
        'num_hypotheses': len(curated_memories),      # 총 메모리 수
        'num_curated': count_non_duplicates,          # 실제 저장된 수
        'num_reflections': reflection_count,          # 반성 횟수
        'memory_size': len(curated_memories)
    }
```

**예시**:
```python
{
    'num_hypotheses': 6,     # 6번 큐레이션 시도
    'num_curated': 4,        # 4개 실제 저장 (2개는 DUPLICATE)
    'num_reflections': 6,    # 6번 반성 (30 experiences / 5)
    'memory_size': 6
}
```

---

## 6. 전체 실행 흐름

### 6.1 에피소드 실행 순서

```
1. 환경 초기화
   env.reset() → obs

2. 메모리 초기화
   memory = create_memory_system(type)

3. 개입 적용 (시작 시)
   intervention.apply_at_start(memory, env)

4. 에피소드 루프 (max_steps까지)
   
   a. 관찰 개입
      obs_for_agent = intervention.apply_to_observation(obs, step)
   
   b. 메모리 컨텍스트 가져오기
      suppressed = intervention.should_suppress_memory(step)
      context = memory.get_context(obs_for_agent, suppressed)
   
   c. 에이전트 액션 선택
      action = agent.select_action(obs_for_agent, context)
   
   d. 환경 스텝
      obs_after, feedback, done, info = env.step(action)
   
   e. 메모리 업데이트 (진짜 관찰 사용!)
      memory.update(obs, action, obs_after, feedback)
   
   f. 메트릭 기록
      metrics.record_step(...)
   
   g. 관찰 업데이트
      obs = obs_after
   
   h. 종료 체크
      if done: break

5. 에피소드 종료
   metrics.finalize(success=done)
   
6. 결과 저장
   episodes.json, summary.json, config.json
```

---

## 7. 핵심 설계 원칙

### 7.1 모듈화
- 각 컴포넌트가 독립적
- Factory 패턴으로 생성 (`create_*()`)
- 쉬운 확장 (새 메모리/개입 추가)

### 7.2 실험 재현성
- 모든 난수에 seed 전달
- Config 전체 저장
- 스텝별 상세 로깅

### 7.3 유연성
- 다양한 조합 가능
  - 4 memory × 4 intervention × 3 agent = 48 조합
- 새 환경 추가 용이
- 프롬프트 수정 간단

### 7.4 확장성
- vLLM, OpenAI, OpenRouter 지원
- 로컬/클라우드 모두 가능
- 배치 실험 자동화 가능

---

## 8. 연구 질문과 구현 대응

| 연구 질문 | 구현 컴포넌트 | 메트릭 |
|----------|--------------|--------|
| 메모리가 도움되는가? | NoMemory vs Naive/Gated | success_rate, avg_steps |
| 검증이 중요한가? | Naive vs Gated | success_rate, loop_ratio |
| 반성이 효과적인가? | Naive/Gated vs Reflector | success_rate, memory_quality |
| 거짓 메모리 영향? | Injection intervention | success_rate 하락 정도 |
| 메모리 의존성? | Surgery intervention | step 80 전후 성능 변화 |
| 노이즈 강건성? | Noise intervention | success_rate (noisy env) |

---

## 9. 파일별 코드 라인 수

```
prompts.py      : 140 lines
metrics.py      : 215 lines
interventions.py: 177 lines
memory.py       : 460 lines
  - NoMemory    : ~15 lines
  - NaiveHyp    : ~95 lines
  - GatedHyp    : ~75 lines (+ 상속)
  - Reflector   : ~210 lines
```

---

## 10. 사용 예시

### 10.1 간단한 실험

```bash
# Naive 메모리
python -m experiments.run \
  --env light --agent llm \
  --provider openai --model gpt-4o-mini \
  --memory naive --intervention none \
  --num_episodes 30 --max_steps 150 \
  --output_dir output/naive

# Reflector 메모리
python -m experiments.run \
  --env light --agent llm \
  --provider openai --model gpt-4o-mini \
  --memory reflector --intervention none \
  --num_episodes 30 --max_steps 150 \
  --output_dir output/reflector
```

### 10.2 개입 실험

```bash
# 거짓 메모리 주입
python -m experiments.run \
  --memory naive --intervention injection \
  --output_dir output/naive_injection

# 메모리 억제 (80스텝 이후)
python -m experiments.run \
  --memory reflector --intervention surgery \
  --output_dir output/reflector_surgery

# 관찰 노이즈 (10%)
python -m experiments.run \
  --memory gated --intervention noise --noise_p 0.1 \
  --output_dir output/gated_noise
```

### 10.3 결과 비교

```bash
python -m experiments.plot_quick \
  --output_dirs output/naive output/gated output/reflector \
  --output_prefix comparison
```

---

## 부록: 주요 상수 및 기본값

```python
# Prompts
MAX_HISTORY_STEPS = 3  # 프롬프트에 포함할 최근 스텝 수

# Memory
MAX_CONTEXT_ITEMS = 10  # 프롬프트에 포함할 메모리 항목 수
REFLECTION_FREQUENCY = 5  # N 경험마다 반성
MAX_MEMORIES = 20  # 저장할 최대 메모리 수

# Gated Memory
MIN_CONFIDENCE = 0.75
MIN_SUPPORT = 2
MAX_CONTRADICTIONS = 0

# Interventions
INJECT_CONFIDENCE = 0.95  # 주입할 거짓 메모리 신뢰도
INJECT_SUPPORT = 3
SURGERY_STEP = 80  # 메모리 억제 시작 스텝
NOISE_PROB = 0.05  # 기본 노이즈 확률

# Metrics
# (모두 runtime 계산)
```

---

**문서 버전**: 1.0  
**최종 업데이트**: 2026-02-17  
**작성자**: Mis-evolved Memory Research Team
