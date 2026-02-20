# OpenAI 모델 검증 실험

## 목적

슬라이드/논문용 실험에서 **베이스라인 성능**과 **메모리·인과성 데모에 필요한 요소**가 잘 나오는지,  
OpenAI API의 **저렴·빠른 모델**과 **똑똑·균형 모델** 두 가지로 짧게 검증하기 위함.

## 선정 모델

| 구분 | 모델 | 특징 | 용도 |
|------|------|------|------|
| **저렴·빠름** | `gpt-4o-mini` | 비용·속도 우수, 성능 양호 | 대량 실험, 빠른 체크 |
| **똑똑·균형** | `gpt-4o` | 추론·이해 우수, 비용·속도 무난 | 메인 결과, 질적 사례 |

- 둘 다 최신 API 모델명 기준. 가격·스펙은 [OpenAI Pricing](https://openai.com/api/pricing) 참고.

## 실험 조건 (5개 베이스라인)

1. **NoMem** – LLM + 메모리 없음  
2. **Naive** – LLM + NaiveMem (`verification_mode=none`, 검증 없이 규칙을 사실처럼 사용)  
3. **Gated** – LLM + GatedMem (oracle로 검증된 가설만 사용)  
4. **Injection** – LLM + NaiveMem + 시작 시 고확신 거짓 가설 주입  
5. **Surgery** – LLM + NaiveMem + N스텝 이후 메모리 차단  

공통: `--env light`, `--agent llm`, `--provider openai`, `--max_steps 200`, `--seed 42`,  
`--reflection_frequency 1`, `--max_context_items` 제한 없음(None).

## 확인할 것

- **성능 (슬라이드 3)**  
  - Success Rate, Steps-to-success가 조건별로 구분되는지  
  - 기대: Naive ≤ NoMem 또는 plateau, Gated ≥ Naive  
- **실패 모드 (슬라이드 4)**  
  - Loop Ratio가 Naive/Injection에서 높고, Gated/Surgery에서 상대적으로 낮은지  
- **인과성 (슬라이드 5)**  
  - Injection: 거짓만 넣었는데 성능/루프 악화  
  - Surgery: 메모리 끄니 루프 감소·회복  
- **질적 사례 (슬라이드 6)**  
  - `episodes.json`의 step별 `memory_context`로 "틀린 기억 인용 → 반복", "contradiction 후 회복" 구간을 골라 스크린샷 가능한지  

## 출력 구조

스크립트 실행 후 `experiments/scripts/openai_validation/results/` 아래에 생성됨.

```
experiments/scripts/openai_validation/
  EXPERIMENT.md          # 이 문서
  run_validation.sh      # 실행 스크립트
  results/               # 스크립트 실행 후 생성
    gpt-4o-mini/
      nomem/
      naive/
      gated/
      injection/
      surgery/
    gpt-4o/
      nomem/
      naive/
      gated/
      injection/
      surgery/
```

각 하위 폴더에는 `episodes.json`, `summary.json`, `config.json`이 저장됨.

## 실행 방법

1. **API 키**  
   ```bash
   export OPENAI_API_KEY="your-key"
   ```

2. **테스트 (기본: 에피소드 3)**  
   ```bash
   cd /path/to/Odyssey-Arena
   bash experiments/scripts/openai_validation/run_validation.sh
   ```
   - 실행 시 **EXPERIMENT SETUP** 이 먼저 출력되고, 에피소드 중에는 **일정 step마다** 진행 로그가 출력됨 (기본 25 step마다, `--log_step_interval N`으로 변경 가능).

3. **API 없이 스크립트/설정만 검증**  
   ```bash
   USE_DUMMY_LLM=1 NUM_EPISODES=2 bash experiments/scripts/openai_validation/run_validation.sh
   ```
   - dummy LLM으로 gpt-4o-mini의 nomem 한 조건만 돌려서 스크립트·설정·로깅이 정상인지 확인.

4. **전체 30 에피소드로 실행**  
   ```bash
   NUM_EPISODES=30 bash experiments/scripts/openai_validation/run_validation.sh
   ```

5. **결과 요약·플롯**  
   ```bash
   cd /path/to/Odyssey-Arena
   RB=experiments/scripts/openai_validation/results
   python -m experiments.plot_quick \
     --output_dirs $RB/gpt-4o-mini/nomem $RB/gpt-4o-mini/naive $RB/gpt-4o-mini/gated $RB/gpt-4o-mini/injection $RB/gpt-4o-mini/surgery \
     --output_prefix gpt4o-mini
   python -m experiments.plot_quick \
     --output_dirs $RB/gpt-4o/nomem $RB/gpt-4o/naive $RB/gpt-4o/gated $RB/gpt-4o/injection $RB/gpt-4o/surgery \
     --output_prefix gpt4o
   ```

## 비고

- `run_validation.sh`는 프로젝트 **루트**에서 실행하는 것을 가정함 (`python -m experiments.run`).  
- Gated 조건은 `--verification_mode oracle` 사용. 오라클을 LLM으로 해석하려면 `--oracle_use_llm`을 스크립트에 추가하면 됨.
