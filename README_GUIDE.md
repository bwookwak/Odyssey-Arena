# 📚 문서 읽기 가이드

프로젝트에 11개의 문서가 있습니다. **읽기 순서**를 추천드립니다!

---

## 🎯 빠른 시작 (5분)

바로 실험을 시작하고 싶다면:

### 1️⃣ `PROJECT_COMPLETE.md` (필독!)
- **크기**: ~9KB
- **내용**: 전체 프로젝트 요약, 바로 실행 가능한 명령어
- **읽는 시간**: 3분
- **왜 읽어야**: 전체 구조 파악, 빠른 시작

**→ 이것만 읽고 바로 실험 시작 가능!**

---

## 📖 완전한 이해 (30분)

전체 시스템을 이해하고 싶다면:

### 2️⃣ `COMPLETE_IMPLEMENTATION.md` (핵심!)
- **크기**: ~12KB
- **내용**: 
  - Q1: 메모리 주입 가능한가? ✅
  - Q2: Hypothesis 점수는? ✅
  - Q3: Memory usage는? ✅
  - 검증 모드 비교
- **읽는 시간**: 8분
- **왜 읽어야**: 핵심 질문 3가지 답변

### 3️⃣ `NEW_MEMORY_ARCHITECTURE.md`
- **크기**: ~10KB
- **내용**: 재설계된 메모리 구조, Reflector-Curator 프로세스
- **읽는 시간**: 7분
- **왜 읽어야**: 아키텍처 이해, 텍스트 기반 메모리 개념

### 4️⃣ `VERIFICATION_MODES_GUIDE.md` (최신!)
- **크기**: ~10KB
- **내용**: 
  - Oracle (ground truth)
  - Observation (empirical)
  - LLM (meta-cognitive)
  - 각각의 의미와 사용법
- **읽는 시간**: 8분
- **왜 읽어야**: 검증 모드 선택 기준

### 5️⃣ `IMPLEMENTATION_DETAILS.md`
- **크기**: ~14KB
- **내용**: Prompts, Metrics, Interventions, Memory 구현 코드
- **읽는 시간**: 10분
- **왜 읽어야**: 세부 구현 이해, 코드 수정 시 필요

---

## 🔧 특정 목적별

### 호환성이 궁금하다면

**6️⃣ `COMPATIBILITY_GUIDE.md`**
- 원본 Odyssey-Arena vs 연구 프레임워크
- 프롬프트/포맷 선택
- 192가지 조합 설명

---

### 실험 명령어가 필요하다면

**7️⃣ `experiments/USAGE_EXAMPLES.md`**
- vLLM, OpenAI, OpenRouter 사용법
- 6가지 핵심 실험 명령어
- Provider별 비용/성능

---

### 빠른 참조

**8️⃣ `EXPERIMENTS_GUIDE.md`**
- 파일 트리
- 5가지 핵심 실험 명령어 (간단 버전)
- 빠른 시작 가이드

---

## 🗂️ 전체 문서 목록

### 필수 (읽기 순서대로)

1. ⭐⭐⭐ `PROJECT_COMPLETE.md` - **여기서 시작!**
2. ⭐⭐⭐ `COMPLETE_IMPLEMENTATION.md` - **핵심 Q&A**
3. ⭐⭐ `NEW_MEMORY_ARCHITECTURE.md` - 아키텍처
4. ⭐⭐ `VERIFICATION_MODES_GUIDE.md` - 검증 모드 (최신!)
5. ⭐ `IMPLEMENTATION_DETAILS.md` - 구현 상세

### 참고 (필요 시)

6. `COMPATIBILITY_GUIDE.md` - 호환성
7. `experiments/USAGE_EXAMPLES.md` - 명령어 예시
8. `experiments/README.md` - 상세 사용법
9. `EXPERIMENTS_GUIDE.md` - 빠른 참조

### 로그/히스토리

10. `TEST_RESULTS.md` - 초기 테스트 기록
11. `COMPATIBILITY_RESULTS.md` - 호환성 테스트

---

## 🎓 상황별 추천

### "지금 바로 실험 돌리고 싶어!"

→ **`PROJECT_COMPLETE.md`** 읽고 명령어 복사

```bash
# 3분 읽기 → 바로 실행
python -m experiments.run --memory hypothesis ...
```

---

### "메모리 시스템이 어떻게 작동하는지 이해하고 싶어"

→ **순서대로**:
1. `PROJECT_COMPLETE.md` (전체 흐름)
2. `NEW_MEMORY_ARCHITECTURE.md` (구조)
3. `COMPLETE_IMPLEMENTATION.md` (Q&A)

**총 25분**

---

### "Oracle이 뭔지, 어떻게 쓰는지 알고 싶어"

→ **`VERIFICATION_MODES_GUIDE.md`** (8분)

핵심:
- Oracle = custom_logic (ground truth)
- 100% 정확한 검증
- Upper bound 측정용

---

### "코드를 수정하고 싶어"

→ **순서대로**:
1. `IMPLEMENTATION_DETAILS.md` (구현 상세)
2. 해당 `.py` 파일 직접 읽기

---

### "원본 Odyssey-Arena와 비교하고 싶어"

→ **`COMPATIBILITY_GUIDE.md`**

핵심:
- 원본 프롬프트 지원
- 이모지 포맷 지원
- 베이스라인 재현 가능

---

## 💡 추천 읽기 순서 (완전 학습)

### 초급 (30분)

```
1. PROJECT_COMPLETE.md        (9KB, 5분)  ← 시작!
2. COMPLETE_IMPLEMENTATION.md (12KB, 10분) ← Q&A
3. VERIFICATION_MODES_GUIDE.md (10KB, 8분) ← 검증
4. experiments/USAGE_EXAMPLES.md (7분)    ← 명령어
```

**목표**: 전체 이해 + 실험 가능

---

### 중급 (1시간)

```
초급 문서 (30분)
    +
5. NEW_MEMORY_ARCHITECTURE.md (10KB, 10분) ← 구조
6. COMPATIBILITY_GUIDE.md     (12KB, 12분) ← 호환성
7. experiments/README.md       (8분)       ← 상세
```

**목표**: 깊은 이해 + 커스터마이징

---

### 고급 (2시간)

```
중급 문서 (1시간)
    +
8. IMPLEMENTATION_DETAILS.md  (14KB, 15분) ← 코드 레벨
9. 실제 코드 읽기             (30분)
10. 테스트 직접 실행          (15분)
```

**목표**: 완전 숙달 + 확장 개발

---

## 🎯 TL;DR (30초)

**바로 시작**: `PROJECT_COMPLETE.md` 읽고 명령어 복사  
**완전 이해**: 위에서 5개 문서 순서대로  
**특정 주제**: 상황별 추천 섹션 참고

---

## 📊 문서 크기 및 중요도

| 문서 | 크기 | 중요도 | 읽는 시간 |
|------|------|--------|----------|
| PROJECT_COMPLETE.md | 9KB | ⭐⭐⭐ | 5분 |
| COMPLETE_IMPLEMENTATION.md | 12KB | ⭐⭐⭐ | 10분 |
| VERIFICATION_MODES_GUIDE.md | 10KB | ⭐⭐⭐ | 8분 |
| NEW_MEMORY_ARCHITECTURE.md | 10KB | ⭐⭐ | 10분 |
| IMPLEMENTATION_DETAILS.md | 14KB | ⭐⭐ | 15분 |
| COMPATIBILITY_GUIDE.md | 12KB | ⭐ | 12분 |
| experiments/README.md | 8KB | ⭐ | 8분 |
| others | - | - | - |

**핵심 3개** (23분):
1. PROJECT_COMPLETE.md
2. COMPLETE_IMPLEMENTATION.md  
3. VERIFICATION_MODES_GUIDE.md

**이 3개만 읽으면 실험 가능합니다!** ✅

---

**가이드 버전**: 1.0  
**최종 업데이트**: 2026-02-17
