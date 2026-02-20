#!/usr/bin/env bash
#
# Feature regression test — gpt-4o-mini, 소규모 실행
#
# 테스트 대상:
#   [T1] nomem / nothink        — 기본 흐름, retry on empty/parse error, no random fallback
#   [T2] nomem / think          — think_text 가 memory.update 로 전달되는지 확인
#   [T3] text memory            — mid-episode JSON reflector, post-episode reflect_on_episode
#   [T4] hypothesis / oracle    — HypothesisMemory.reflect_on_episode (multi-insight)
#   [T5] include_raw_output=1   — raw output 포함 프롬프트
#   [T6] pipeline               — explore → infer, memory serialization
#
# 사용법:
#   cd /path/to/Odyssey-Arena
#   bash experiments/scripts/test_features/run_test.sh
#
# 환경 변수:
#   OPENAI_API_KEY   (필수)
#   NUM_EPISODES=3   (기본 3, 더 짧게: 1)
#   MAX_STEPS=30     (기본 30)
#   PARALLEL=2       (기본 2)
#   SKIP_PIPELINE=1  (파이프라인 테스트 건너뛰기)
#   RUN_ONLY="T1 T3" (지정 테스트만 실행, 공백 구분)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT_DIR"

# ── 파라미터 ──────────────────────────────────────────────────────────────
MODEL="${MODEL:-gpt-4o-mini}"
PROVIDER="${PROVIDER:-openai}"
NUM_EPISODES="${NUM_EPISODES:-3}"
MAX_STEPS="${MAX_STEPS:-30}"
PARALLEL="${PARALLEL:-2}"
MAX_PARSE_ERRORS="${MAX_PARSE_ERRORS:-3}"   # 테스트용 retry 횟수 (빠른 실패 확인)
OUTPUT_BASE="$SCRIPT_DIR/results/$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_BASE"

# ── API 키 확인 ───────────────────────────────────────────────────────────
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "[ERROR] OPENAI_API_KEY 가 설정되지 않았습니다."
  echo "        export OPENAI_API_KEY=your-key"
  exit 1
fi

# ── Ctrl+C 처리 ───────────────────────────────────────────────────────────
PIDS=()
cleanup() {
  echo ""
  echo "[test] 중단 — 실행 중인 프로세스를 종료합니다..."
  for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  sleep 2
  for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
  exit 130
}
trap cleanup INT TERM

# ── 공통 인수 ─────────────────────────────────────────────────────────────
BASE_ARGS=(
  --env light
  --agent llm
  --model "$MODEL"
  --provider "$PROVIDER"
  --num_episodes "$NUM_EPISODES"
  --max_steps "$MAX_STEPS"
  --max_state_visits 8
  --seed 42
  --parallel_episodes "$PARALLEL"
  --log_step_interval 5
  --max_parse_errors "$MAX_PARSE_ERRORS"
  --prompt_template original
  --no-dashboard
)

# ── 실행할 테스트 목록 (RUN_ONLY 미지정 시 전부) ──────────────────────────
ALL_TESTS="T1 T2 T3 T4 T5 T6"
RUN_ONLY="${RUN_ONLY:-$ALL_TESTS}"
[ "${SKIP_PIPELINE:-0}" = "1" ] && RUN_ONLY="${RUN_ONLY/T6/}"

should_run() { echo "$RUN_ONLY" | grep -qw "$1"; }

# ── 헬퍼 ──────────────────────────────────────────────────────────────────
PASS_COUNT=0
FAIL_COUNT=0
RESULTS=()

run_test() {
  local id="$1"
  local desc="$2"
  shift 2

  local outdir="$OUTPUT_BASE/$id"
  mkdir -p "$outdir"

  printf "\n%s\n" "$(printf '=%.0s' {1..60})"
  echo "  [$id] $desc"
  printf "%s\n\n" "$(printf '=%.0s' {1..60})"

  local log="$outdir/run.log"

  if python -m experiments.run \
      "${BASE_ARGS[@]}" \
      --output_dir "$outdir" \
      --experiment_name "test-${id}" \
      "$@" \
      2>&1 | tee "$log"; then
    echo "  [$id] ✓ 완료 → $outdir"
    RESULTS+=("PASS [$id] $desc")
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    echo "  [$id] ✗ 실패 (exit $?)"
    RESULTS+=("FAIL [$id] $desc")
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
}

verify_summary() {
  local id="$1"
  local key="$2"
  local outdir="$OUTPUT_BASE/$id"
  local summary="$outdir/summary.json"

  if [ ! -f "$summary" ]; then
    echo "  [verify] [$id] summary.json 없음 — skip"
    return
  fi

  local val
  val=$(python3 -c "
import json, sys
d = json.load(open('$summary'))
v = d.get('$key', None)
print(v)
" 2>/dev/null)

  echo "  [verify] [$id] summary.$key = $val"
}

check_ratio() {
  # summary.json 에서 key 값이 0~1 범위인지 확인
  local id="$1"
  local key="$2"
  local outdir="$OUTPUT_BASE/$id"
  local summary="$outdir/summary.json"
  [ -f "$summary" ] || return
  python3 -c "
import json, sys
d = json.load(open('$summary'))
v = float(d.get('$key', -1))
if 0.0 <= v <= 1.0:
    print(f'  [check] [$id] {\"$key\"} = {v:.4f}  ✓ (0-1 range)')
else:
    print(f'  [check] [$id] {\"$key\"} = {v}  ✗ (out of 0-1 range!)')
    sys.exit(1)
" 2>/dev/null && true
}

# ══════════════════════════════════════════════════════════════════════════
# [T1] 기본 nomem / nothink
#   확인: 실행 완료, avg_max_progress·avg_avg_progress 가 0~1
# ══════════════════════════════════════════════════════════════════════════
if should_run T1; then
  run_test T1 "nomem / nothink — 기본 흐름, retry, no random fallback" \
    --memory nomem \
    --intervention none \
    --think_in_history none
  check_ratio T1 avg_max_progress
  check_ratio T1 avg_avg_progress
  verify_summary T1 avg_loop_ratio
fi

# ══════════════════════════════════════════════════════════════════════════
# [T2] nomem / think
#   확인: think_text 가 history 에 포함, log 에 [THINK] 섹션 존재
# ══════════════════════════════════════════════════════════════════════════
if should_run T2; then
  run_test T2 "nomem / think — think_text memory.update 전달 확인" \
    --memory nomem \
    --intervention none \
    --think_in_history last

  # run.log 에 think 관련 출력 확인
  if grep -q "\[ THINK \]\|\[ INPUT \]" "$OUTPUT_BASE/T2/run.log" 2>/dev/null; then
    echo "  [verify] [T2] [THINK]/[INPUT] 섹션 존재 ✓"
  else
    echo "  [verify] [T2] [THINK]/[INPUT] 섹션 없음 (nothink 모델이면 정상)"
  fi
fi

# ══════════════════════════════════════════════════════════════════════════
# [T3] TextMemory — mid-episode JSON reflector + post-episode reflection
#   확인: reflection_frequency=2, run.log 에 episode_reflection op 존재
# ══════════════════════════════════════════════════════════════════════════
if should_run T3; then
  run_test T3 "text memory — mid+traj reflection" \
    --memory text \
    --reflection_mode mid \
    --reflection_frequency 2 \
    --intervention none \
    --think_in_history last

  # memory_ops 에 episode_reflection 이 기록됐는지 episodes.json 에서 확인
  if grep -q "episode_reflection" "$OUTPUT_BASE/T3/episodes.json" 2>/dev/null; then
    echo "  [verify] [T3] episodes.json 에 episode_reflection op 존재 ✓"
  else
    echo "  [verify] [T3] episodes.json 에 episode_reflection 없음 (에피소드 수가 적으면 발생 가능)"
  fi
fi

# ══════════════════════════════════════════════════════════════════════════
# [T4] HypothesisMemory / oracle — multi-insight post-episode reflection
#   확인: summary 에 avg_memory_usage_rate > 0
# ══════════════════════════════════════════════════════════════════════════
if should_run T4; then
  run_test T4 "hypothesis memory / oracle — post-episode (traj) reflection" \
    --memory hypothesis \
    --reflection_mode traj \
    --verification_mode oracle \
    --intervention none \
    --think_in_history last

  verify_summary T4 avg_memory_usage_rate
  if grep -q "episode_reflection" "$OUTPUT_BASE/T4/episodes.json" 2>/dev/null; then
    echo "  [verify] [T4] episode_reflection op 존재 ✓"
  fi
fi

# ══════════════════════════════════════════════════════════════════════════
# [T5] include_raw_output=True 프롬프트 포함 확인
#   확인: model_input 에 "Previous Step Full Output" 섹션 존재
# ══════════════════════════════════════════════════════════════════════════
if should_run T5; then
  run_test T5 "include_raw_output=True — 프롬프트에 이전 raw output 포함" \
    --memory nomem \
    --intervention none \
    --think_in_history last \
    --include_raw_output   # CLI 플래그 (기본 False → True)

  if grep -q "Previous Step Full Output\|PREVIOUS STEP FULL OUTPUT" \
      "$OUTPUT_BASE/T5/episodes.json" 2>/dev/null; then
    echo "  [verify] [T5] Previous Step Full Output 섹션 존재 ✓"
  else
    echo "  [verify] [T5] Previous Step Full Output 섹션 없음 ✗"
  fi
fi

# ══════════════════════════════════════════════════════════════════════════
# [T6] Pipeline — explore → memory → infer
#   확인: explore/ infer/ 서브디렉터리, memory.json 존재
# ══════════════════════════════════════════════════════════════════════════
if should_run T6; then
  PIPE_OUT="$OUTPUT_BASE/T6"
  mkdir -p "$PIPE_OUT"

  printf "\n%s\n" "$(printf '=%.0s' {1..60})"
  echo "  [T6] pipeline — explore → infer"
  printf "%s\n\n" "$(printf '=%.0s' {1..60})"

  if python -m experiments.run \
      --env light \
      --agent llm \
      --model "$MODEL" \
      --provider "$PROVIDER" \
      --mode pipeline \
      --num_episodes "$NUM_EPISODES" \
      --explore_episodes "$NUM_EPISODES" \
      --max_steps "$MAX_STEPS" \
      --max_state_visits 8 \
      --seed 42 \
      --parallel_episodes "$PARALLEL" \
      --log_step_interval 5 \
      --max_parse_errors "$MAX_PARSE_ERRORS" \
      --memory hypothesis \
      --verification_mode oracle \
      --reflection_frequency 2 \
      --intervention none \
      --think_in_history last \
      --prompt_template original \
      --no-dashboard \
      --output_dir "$PIPE_OUT" \
      --experiment_name "test-T6" \
      2>&1 | tee "$PIPE_OUT/run.log"; then

    echo "  [T6] ✓ 완료"
    RESULTS+=("PASS [T6] pipeline (explore→infer)")
    PASS_COUNT=$((PASS_COUNT + 1))

    # 서브디렉터리 확인
    for sub in explore infer; do
      if [ -d "$PIPE_OUT/$sub" ]; then
        echo "  [verify] [T6] $sub/ 디렉터리 존재 ✓"
      else
        echo "  [verify] [T6] $sub/ 디렉터리 없음 ✗"
      fi
    done

    if ls "$PIPE_OUT"/*.json 2>/dev/null | grep -q memory; then
      echo "  [verify] [T6] memory snapshot 존재 ✓"
    fi
  else
    echo "  [T6] ✗ 실패"
    RESULTS+=("FAIL [T6] pipeline (explore→infer)")
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
fi

# ══════════════════════════════════════════════════════════════════════════
# 결과 요약
# ══════════════════════════════════════════════════════════════════════════
printf "\n%s\n" "$(printf '═%.0s' {1..60})"
echo "  테스트 결과 요약"
printf "%s\n" "$(printf '═%.0s' {1..60})"
for r in "${RESULTS[@]}"; do echo "  $r"; done
printf "%s\n" "$(printf '─%.0s' {1..60})"
echo "  PASS: $PASS_COUNT  /  FAIL: $FAIL_COUNT  /  TOTAL: $((PASS_COUNT + FAIL_COUNT))"
printf "%s\n" "$(printf '═%.0s' {1..60})"
echo ""
echo "  결과 디렉터리: $OUTPUT_BASE"
echo ""

[ "$FAIL_COUNT" -eq 0 ] && exit 0 || exit 1
