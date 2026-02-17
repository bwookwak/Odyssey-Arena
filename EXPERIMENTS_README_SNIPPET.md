# Experiments: Mis-evolved Memory Research

## 연구 컨텍스트

에이전트는 상호작용을 통해 경험적 메모리를 저장하며 학습합니다. 노이즈가 있거나 부분적이거나 동적인 환경에서 에이전트는 잘못된 "통찰"을 사실로 저장할 수 있으며(mis-evolved memory), 이는 이후 의사결정을 편향시키고 누적 실패(루프/정체)와 성능 한계를 초래합니다.

우리는 새로운 통찰을 잠정적 가설로 다루고(간단한 증거 추적) 높은 신뢰도의 모순되지 않은 항목만 사용하는 것이 강건성을 개선하는지 테스트합니다.

## 빠른 시작

```bash
# 환경 설정
conda activate ody
cd /home/bwoo/workspace/Odyssey-Arena

# 베이스라인 실험 실행
python -m experiments.run \
  --env light --agent llm --memory nomem --intervention none \
  --num_episodes 30 --max_steps 200 --seed 42 \
  --output_dir output/nomem

# 결과 시각화
python -m experiments.plot_quick \
  --output_dirs output/nomem output/naive output/gated \
  --output_prefix comparison
```

## 주요 기능

- **3가지 메모리 시스템**: NoMemory (베이스라인), NaiveHypMemory (모든 가설), GatedHypMemory (검증된 가설만)
- **4가지 개입 방식**: None, Injection (거짓 메모리 주입), Surgery (메모리 억제), Noise (관찰 노이즈)
- **3가지 에이전트**: LLM (vLLM), Random, Heuristic
- **완전한 메트릭**: success rate, loop ratio, memory usage, invalid actions
- **확장 가능**: 다른 Odyssey-Arena 환경으로 쉽게 확장 가능

## 문서

- 전체 가이드: `EXPERIMENTS_GUIDE.md`
- 상세 문서: `experiments/README.md`
- 모듈 구조: `experiments/` 디렉토리

## 검증된 5가지 핵심 실험

1. **nomem**: LLM + NoMemory + None
2. **naive**: LLM + NaiveHypMemory + None
3. **gated**: LLM + GatedHypMemory + None
4. **injection**: LLM + NaiveHypMemory + Injection
5. **surgery**: LLM + NaiveHypMemory + Surgery

각 실험은 `episodes.json`, `summary.json`, `config.json`을 생성합니다.
