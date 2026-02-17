# 새로운 메모리 아키텍처

## 🏗️ 재설계된 구조

### 핵심 개념

**메모리는 기본적으로 텍스트(text)**이며, Reflector-Curator는 모든 메모리에 적용 가능한 **공통 프로세스**입니다.

```
┌────────────────────────────────────────────────┐
│  Reflector-Curator (Common Process)           │
│  ┌──────────────┐      ┌──────────────┐       │
│  │  Reflector   │  →   │   Curator    │       │
│  │ (Extract     │      │ (Decide      │       │
│  │  insights)   │      │  ADD/SKIP)   │       │
│  └──────────────┘      └──────────────┘       │
└────────────────┬───────────────────────────────┘
                 │
         Text Memories
         (string list)
                 │
     ┌───────────┴───────────┐
     │                       │
┌────▼──────────┐  ┌────────▼─────────────┐
│  TextMemory   │  │  HypothesisMemory    │
│  (그대로 사용) │  │  (파싱 + 점수화)      │
└───────────────┘  └──────────────────────┘
                            │
                   ┌────────▼─────────┐
                   │ GatedHypothesis  │
                   │ (+ 검증 필터)     │
                   └──────────────────┘
```

---

## 📋 메모리 타입

### 1. NoMemory (베이스라인)

**특징**:
- 메모리 없음
- Reflector-Curator 사용 안 함

**용도**: 대조군

---

### 2. TextMemory (일반적 접근)

**프로세스**:
```
Experiences → Reflector → Insight (text)
                         ↓
                    Curator → ADD/SKIP
                         ↓
                  Text Memories (list)
                         ↓
                  Context (그대로 출력)
```

**예시**:
```python
text_memories = [
    "Toggling bulb 0 affects bulb 1",
    "Bulb 2 depends on bulbs 0 and 1",
    "Cascading effects require sequential strategy"
]

# Context 생성
context = """
1. Toggling bulb 0 affects bulb 1
2. Bulb 2 depends on bulbs 0 and 1
3. Cascading effects require sequential strategy
"""
```

**장점**:
- 단순함
- LLM 출력을 그대로 사용
- 고수준 통찰 보존

**단점**:
- 구조화되지 않음
- 점수/신뢰도 없음

---

### 3. HypothesisMemory (사용자 방법론)

**프로세스**:
```
Experiences → Reflector → Insight (text)
                         ↓
                    Curator → ADD/SKIP
                         ↓
                  Text Memories
                         ↓
                    Parse Text
                         ↓
              hyp_store[(action, flip)]
              {support, contradict, conf, source}
                         ↓
                  Context (점수 포함)
```

**Reflector Prompt 개선**:
```
Analyze and distill concrete insights about which actions affect which bulbs.
Use EXPLICIT format: "Toggling bulb X affects bulb Y" or "Action X flips bulb Y".

Example good insights:
- "Toggling bulb 0 affects bulb 1"
- "Action 2 flips bulb 3 and bulb 4"

Provide 1-3 specific insights (one per line):
```

**Curator Prompt**:
```
DECISION - Choose ONE action:
1. "ADD: [memory text]" - if insight is new or provides additional value
2. "SKIP: duplicate" - if already covered
3. "SKIP: low quality" - if too vague

Important: If adding, use format "Toggling bulb X affects bulb Y" for parsability.
```

**파싱 로직**:
```python
patterns = [
    r'bulb\s+(\d+)\s+(?:affects?|flips?|influences?)\s+bulb\s+(\d+)',
    r'toggling\s+(\d+)\s+(?:affects?|flips?)\s+(?:bulb\s+)?(\d+)',
    r'action\s+(\d+)\s+(?:affects?|flips?)\s+(?:bulb\s+)?(\d+)',
]

# "Toggling bulb 0 affects bulb 1" → (0, 1)
for pattern in patterns:
    matches = re.findall(pattern, text, re.IGNORECASE)
    for (act, flip_idx) in matches:
        hyp_store[(act, flip_idx)] = {
            'support': 1,
            'contradict': 0,
            'confidence': 0.73,  # Initial from LLM
            'source': 'llm'
        }
```

**Context 생성**:
```python
# hyp_store를 confidence 순으로 정렬
context = """
1. Toggling bulb 0 flips bulb 1 (conf: 0.95, +3/-0)
2. Toggling bulb 2 flips bulb 3 (conf: 0.88, +2/-0)
"""
```

**장점**:
- 구조화된 가설
- 신뢰도 점수
- Support/Contradict 추적
- 파싱 가능

**단점**:
- 파싱 의존적 (LLM이 포맷 안 지키면 실패)
- 저수준 표현 (action→bulb)

---

### 4. GatedHypothesisMemory (검증 버전)

**HypothesisMemory + 필터링**:

```python
# Get context에서만 차이
verified = [h for h in hyp_store 
            if contradict == 0 
            and confidence >= 0.75 
            and support >= 2]
```

**문제**: 초기 가설은 `support=1`이므로 필터링됨
→ 여러 번 반복 관찰되어야 사용됨

---

## 🧪 테스트 결과 비교

### 메모리 시스템 비교 (2 episodes, 25-30 steps, seed 변화)

| Memory Type | Loop Ratio | Memory Usage | 특징 |
|-------------|-----------|--------------|------|
| **nomem** | - | 0% | 베이스라인 |
| **text** | 0.700 | 80.0% | 고수준 텍스트 |
| **hypothesis** | 0.633 | 83.3% | 구조화 가설 |
| **gated** | 0.300 | 0% | 필터 너무 엄격 |

**관찰**:
1. **TextMemory**: 텍스트를 그대로 사용, 합리적 성능
2. **HypothesisMemory**: 파싱 후 사용, 약간 더 나은 loop ratio
3. **GatedHypothesisMemory**: 필터링이 너무 엄격해서 메모리 사용 못함

---

## 🔬 Reflector-Curator 작동 예시

### 실제 생성된 메모리 (테스트에서)

#### Reflection 1:
**Input** (3 experiences):
```
1. Action 0: 000000 -> 100000
2. Action 0: 100000 -> 000000  
3. Action 1: 000000 -> 010000
```

**Reflector Output**:
```
"Toggling bulb 0 affects bulb 1."
```

**Curator Decision**:
```
"ADD: Toggling bulb 0 affects bulb 1"
```

**Parsed Hypothesis**:
```python
hyp_store[(0, 1)] = {
    'support': 1,
    'contradict': 0,
    'confidence': 0.73,
    'source': 'llm'
}
```

**Context for Agent**:
```
1. Toggling bulb 0 flips bulb 1 (conf: 0.73, +1/-0)
```

---

### Reflection 2:
**Input** (3 more experiences):
```
4. Action 2: 010000 -> 011000
5. Action 1: 011000 -> 001000
6. Action 3: 001000 -> 001100
```

**Reflector Output**:
```
"Action 2 flips bulb 3. Action 3 affects bulb 4."
```

**Curator Decision**:
```
"ADD: Action 2 flips bulb 3 and action 3 affects bulb 4"
```

**Parsed Hypotheses**:
```python
hyp_store[(2, 3)] = {'support': 1, 'contradict': 0, 'confidence': 0.73, 'source': 'llm'}
hyp_store[(3, 4)] = {'support': 1, 'contradict': 0, 'confidence': 0.73, 'source': 'llm'}
```

---

## 🎯 메모리 타입별 사용 시나리오

### NoMemory
- **언제**: 베이스라인, 메모리 없는 순수 반응형
- **장점**: 빠름, 간단
- **단점**: 학습 없음

### TextMemory
- **언제**: LLM이 고수준 전략을 생성하길 원할 때
- **장점**: 추상적 통찰, 파싱 불필요
- **단점**: 구조화 안 됨
- **예시**: "Sequential strategy needed", "Dependencies exist"

### HypothesisMemory (권장)
- **언제**: 구체적 인과관계 학습이 중요할 때
- **장점**: 구조화, 점수화, 추적 가능
- **단점**: 파싱 의존적
- **예시**: "Toggling 0 affects 1 (conf: 0.95)"

### GatedHypothesisMemory
- **언제**: 고품질 가설만 사용하고 싶을 때
- **장점**: 검증된 것만 사용
- **단점**: 초기에는 메모리 비어있음 (cold start)
- **개선 필요**: 초기 support 기준 완화

---

## 🔧 개선 사항

### 1. Gated Memory Cold Start 문제

**현재**:
- 초기 가설: `support=1` (LLM에서 생성)
- Gated 기준: `support>=2`
- 결과: 초기 가설 전부 필터링

**해결책 A**: 초기 가설에 높은 support
```python
if stats.get('source') == 'llm':
    # LLM-generated hypotheses get initial bonus
    hyp_store[key] = {'support': 2, ...}  # 2로 시작
```

**해결책 B**: Gated 기준 완화
```python
min_support = 1  # LLM 가설은 1번만 나와도 OK
```

**해결책 C**: Source별 다른 기준
```python
if stats['source'] == 'llm' and stats['support'] >= 1:
    verified.append(...)
elif stats['source'] == 'observation' and stats['support'] >= 2:
    verified.append(...)
```

---

### 2. Reflector 프롬프트 개선

**현재**: 명시적 포맷 요청 추가 ✓

**추가 개선**:
```python
prompt = f"""
...
Format your insights as:
- "Toggling bulb X affects bulb Y"
- "Action X flips bulb Y"

DO NOT use vague terms like "some bulbs" or "certain patterns".
BE SPECIFIC with bulb numbers.

Your insights:
"""
```

---

### 3. Curator 파싱 로직 강화

**현재**: regex 패턴 3개

**추가 패턴**:
```python
patterns = [
    r'bulb\s+(\d+)\s+(?:affects?|flips?|influences?)\s+bulb\s+(\d+)',
    r'toggling\s+(\d+)\s+(?:affects?|flips?)\s+(?:bulb\s+)?(\d+)',
    r'action\s+(\d+)\s+(?:affects?|flips?)\s+(?:bulb\s+)?(\d+)',
    r'(\d+)\s*->\s*(\d+)',  # "0 -> 1" 형태
    r'(\d+)\s+to\s+(\d+)',  # "0 to 1" 형태
]
```

---

## 📊 실험 결과 (초기 테스트)

### 설정
- Model: gpt-4o-mini
- Episodes: 2
- Max steps: 25-30
- Seed: 42, 50

### 결과

| Memory | Loop Ratio | Memory Usage | 분석 |
|--------|-----------|--------------|------|
| **text** | 0.700 | 80% | 텍스트 그대로 사용, 합리적 |
| **hypothesis** | 0.633 | 83% | 파싱 작동, 약간 더 나음 |
| **gated** | 0.300 | 0% | ⚠️ 필터 너무 엄격, cold start |

**핵심 발견**:
1. ✅ Reflector-Curator 작동 확인
2. ✅ Text → Hypothesis 파싱 작동
3. ⚠️ Gated는 초기 가설 필터링 (개선 필요)

---

## 🎯 권장 사용법

### 연구 초기 (탐색)
```bash
# TextMemory로 빠른 테스트
python -m experiments.run \
  --memory text \
  --num_episodes 5 --max_steps 50 \
  --output_dir output/explore_text
```

### 본격 실험 (구조화)
```bash
# HypothesisMemory로 정밀 분석
python -m experiments.run \
  --memory hypothesis \
  --num_episodes 30 --max_steps 200 \
  --output_dir output/main_hypothesis
```

### 고품질 가설만 (주의: cold start)
```bash
# GatedHypothesisMemory (초기 필터링 주의)
python -m experiments.run \
  --memory gated \
  --num_episodes 30 --max_steps 200 \
  --output_dir output/gated_verify
```

---

## 🔍 상세 구현

### Reflector Input

```python
{
    'obs_before': '000000',
    'action': 0,
    'obs_after': '100000',
    'feedback': 'Toggled B0 to True'
}
```

### Reflector Output

```
"Toggling bulb 0 affects bulb 1 directly.
Action 2 flips bulb 3."
```

### Curator Input

```
EXISTING MEMORIES:
1. Toggling bulb 0 affects bulb 1

NEW INSIGHT:
Toggling bulb 0 affects bulb 1 directly.
Action 2 flips bulb 3.
```

### Curator Output

```
SKIP: duplicate (first part)
ADD: Action 2 flips bulb 3
```

### Parse Result (HypothesisMemory)

```python
# From "Action 2 flips bulb 3"
hyp_store[(2, 3)] = {
    'support': 1,
    'contradict': 0,
    'confidence': 0.73,
    'source': 'llm'
}
```

### Final Context

```
1. Toggling bulb 0 flips bulb 1 (conf: 0.73, +1/-0)
2. Toggling bulb 2 flips bulb 3 (conf: 0.73, +1/-0)
```

---

## 💡 핵심 차별점

### 기존 NaiveHypMemory (삭제됨)

```python
# 직접 관찰로 가설 생성
obs_before = "000000"
obs_after = "100000"
action = 0

# 비트 비교
flipped = [0]  # 0번 비트 플립

# 가설 생성
hyp_store[(0, 0)] = {'support': 1, ...}  # action 0 → bulb 0
```

**문제**: 저수준, 노이즈 많음

---

### 새 HypothesisMemory (현재)

```python
# LLM 분석으로 가설 생성
experiences = [...]  # 5개 누적

# Reflector
insight = llm.generate("Analyze patterns...")
# → "Toggling bulb 0 affects bulb 1"

# Curator
decision = llm.generate("Check duplicate...")
# → "ADD: Toggling bulb 0 affects bulb 1"

# Parse
hyp_store[(0, 1)] = {'support': 1, 'source': 'llm', ...}
```

**장점**: 
- 고수준 추론
- 노이즈 필터링
- Deduplication
- 메타인지

---

## 🚀 다음 단계

### 1. Gated Memory 개선

```python
# experiments/memory.py - GatedHypothesisMemory

def get_context(self, ...):
    verified = []
    for key, stats in self.hyp_store.items():
        # Source별 다른 기준
        if stats.get('source') == 'llm':
            if stats['confidence'] >= 0.70 and stats['support'] >= 1:
                verified.append((key, stats))
        else:
            if stats['contradict'] == 0 and stats['support'] >= 2:
                verified.append((key, stats))
```

### 2. 전체 비교 실험

```bash
# 3가지 메모리 비교
for mem in text hypothesis gated; do
  python -m experiments.run \
    --memory $mem \
    --provider openai --model gpt-4o-mini \
    --num_episodes 30 --max_steps 200 \
    --output_dir output/final_${mem}
done

python -m experiments.plot_quick \
  --output_dirs output/final_* \
  --output_prefix final_memory_comparison
```

### 3. 논문용 실험

```bash
# Hypothesis (user's method)
python -m experiments.run \
  --memory hypothesis --intervention none \
  --num_episodes 50 --output_dir output/paper_hypothesis

# + Injection
python -m experiments.run \
  --memory hypothesis --intervention injection \
  --num_episodes 50 --output_dir output/paper_hypothesis_inject

# + Surgery
python -m experiments.run \
  --memory hypothesis --intervention surgery \
  --num_episodes 50 --output_dir output/paper_hypothesis_surgery
```

---

## 📚 문서 업데이트 필요

다음 문서들을 새 아키텍처에 맞게 업데이트해야 합니다:

1. ✅ `NEW_MEMORY_ARCHITECTURE.md` (현재 문서)
2. ⏳ `IMPLEMENTATION_DETAILS.md` - Reflector-Curator 섹션 업데이트
3. ⏳ `experiments/README.md` - 메모리 타입 설명 업데이트
4. ⏳ `USAGE_EXAMPLES.md` - 명령어 예시 업데이트

---

## ✅ 현재 상태

**구현**: ✅ 완료  
**테스트**: ✅ 작동 확인  
**이슈**: ⚠️ Gated cold start (개선 필요)  
**문서**: 🔄 일부 업데이트 필요

**준비 상태**: 🚀 실험 가능!
