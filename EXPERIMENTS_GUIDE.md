# Mis-evolved Memory 실험 프레임워크 가이드

## 📁 파일 트리

```
Odyssey-Arena/
├── experiments/
│   ├── __init__.py          # 패키지 초기화 및 연구 컨텍스트
│   ├── __main__.py          # 모듈 실행 엔트리 포인트
│   ├── run.py               # 메인 러너 + CLI
│   ├── envs.py              # LightEnv 래퍼 (확장 가능)
│   ├── llm.py               # vLLM 클라이언트
│   ├── prompts.py           # 프롬프트 빌더
│   ├── agents.py            # LLM/Random/Heuristic 에이전트
│   ├── memory.py            # NoMemory/NaiveHypMemory/GatedHypMemory
│   ├── interventions.py     # Injection/Surgery/Noise 개입
│   ├── metrics.py           # 메트릭 계산 및 집계
│   ├── plot_quick.py        # 시각화 스크립트
│   ├── requirements.txt     # 의존성 패키지
│   └── README.md            # 상세 문서
├── test_data/
│   └── turnonlights/
│       └── test_turnonlights_lite_251030.json  # 테스트 태스크 (30개)
└── output/                  # 실험 결과 출력 디렉토리
```

## 🚀 빠른 시작

### 1. 환경 설정 (이미 완료됨)

```bash
# conda 환경 활성화
conda activate ody

# experiments 디렉토리로 이동
cd /home/bwoo/workspace/Odyssey-Arena
```

### 2. 의존성 설치

실제 LLM 실험을 위해 (GPU 필요):

```bash
pip install -r experiments/requirements.txt
```

테스트용 (GPU 불필요):

```bash
pip install matplotlib numpy  # 시각화를 위한 최소 패키지
```

## 📋 5가지 핵심 실험 명령어

### 1. No Memory Baseline (nomem)

```bash
python -m experiments.run \
  --env light \
  --agent llm \
  --memory nomem \
  --intervention none \
  --num_episodes 30 \
  --max_steps 200 \
  --seed 42 \
  --output_dir output/nomem
```

### 2. Naive Hypothesis Memory (naive)

```bash
python -m experiments.run \
  --env light \
  --agent llm \
  --memory naive \
  --intervention none \
  --num_episodes 30 \
  --max_steps 200 \
  --seed 42 \
  --output_dir output/naive
```

### 3. Gated Hypothesis Memory (gated)

```bash
python -m experiments.run \
  --env light \
  --agent llm \
  --memory gated \
  --intervention none \
  --num_episodes 30 \
  --max_steps 200 \
  --seed 42 \
  --output_dir output/gated
```

### 4. Injection Intervention (injection)

```bash
python -m experiments.run \
  --env light \
  --agent llm \
  --memory naive \
  --intervention injection \
  --num_episodes 30 \
  --max_steps 200 \
  --seed 42 \
  --output_dir output/injection
```

### 5. Surgery Intervention (surgery)

```bash
python -m experiments.run \
  --env light \
  --agent llm \
  --memory naive \
  --intervention surgery \
  --num_episodes 30 \
  --max_steps 200 \
  --seed 42 \
  --output_dir output/surgery
```

## 📊 시각화

모든 실험 완료 후:

```bash
python -m experiments.plot_quick \
  --output_dirs output/nomem output/naive output/gated output/injection output/surgery \
  --output_prefix comparison
```

생성된 파일:
- `comparison_success_rate.png` - 성공률 비교
- `comparison_steps.png` - 성공까지 평균 스텝 수
- `comparison_loop_memory.png` - 루프 비율 및 메모리 사용률

## 🧪 빠른 테스트 (검증 완료)

### Random Agent 테스트 (GPU 불필요)

```bash
python -m experiments.run \
  --env light --agent random --memory nomem --intervention none \
  --num_episodes 5 --max_steps 50 --seed 42 \
  --output_dir output/test_random
```

### Dummy LLM 테스트 (GPU 불필요)

```bash
python -m experiments.run \
  --env light --agent llm --memory naive --intervention none \
  --use_dummy_llm \
  --num_episodes 5 --max_steps 50 --seed 42 \
  --output_dir output/test_dummy
```

## 📈 주요 메트릭

각 실험은 `summary.json`에 다음 메트릭을 저장합니다:

- **success_rate**: 성공한 에피소드 비율
- **avg_steps_success**: 성공 에피소드의 평균 스텝 수
- **avg_loop_ratio**: 진전 없는 반복 행동 비율
- **avg_memory_usage_rate**: 메모리 컨텍스트 사용 비율
- **avg_invalid_action_rate**: 무효 액션 비율

## 🔧 고급 옵션

### 모델 변경

```bash
--model meta-llama/Llama-2-7b-chat-hf
```

### Noise 개입 확률 조정

```bash
--intervention noise --noise_p 0.1  # 10% 비트 플립 확률
```

### Surgery 개입 시점 조정

```bash
--intervention surgery --surgery_step 50  # 50 스텝 이후 메모리 억제
```

## 📝 출력 파일

각 실험 디렉토리에는 3개의 JSON 파일이 생성됩니다:

1. **`episodes.json`**: 모든 에피소드의 스텝별 상세 로그
2. **`summary.json`**: 집계된 메트릭
3. **`config.json`**: 실험 설정

## 🎯 연구 가설

1. **Naive vs Gated**: Gated 메모리가 잘못된 가설을 필터링하여 더 나은 성능 제공
2. **Injection**: 초기 잘못된 메모리가 성능 저하 유발 (Naive > Gated 효과 검증)
3. **Surgery**: 메모리 의존성이 높을 때 억제가 회복 지원
4. **Noise**: 노이즈 환경에서 Gated 메모리의 강건성 검증

## 📚 추가 문서

자세한 내용은 `experiments/README.md`를 참조하세요.

## ✅ 검증 상태

다음 기능들이 검증되었습니다:

- ✅ LightEnv 태스크 로딩 (30개 태스크)
- ✅ Random Agent 실행
- ✅ Heuristic Agent 실행
- ✅ Dummy LLM Agent 실행
- ✅ NoMemory 시스템
- ✅ NaiveHypMemory 시스템
- ✅ GatedHypMemory 시스템
- ✅ Injection 개입
- ✅ Surgery 개입
- ✅ Noise 개입
- ✅ JSON 출력 생성
- ✅ 메트릭 계산

## 🚦 다음 단계

1. **실제 LLM 실험 실행**:
   ```bash
   # vLLM 설치 후
   python -m experiments.run --env light --agent llm --memory nomem --intervention none \
     --num_episodes 30 --max_steps 200 --seed 42 --output_dir output/nomem
   ```

2. **5가지 핵심 실험 완료**

3. **결과 시각화**:
   ```bash
   python -m experiments.plot_quick --output_dirs output/* --output_prefix final
   ```

4. **결과 분석 및 논문 작성**
