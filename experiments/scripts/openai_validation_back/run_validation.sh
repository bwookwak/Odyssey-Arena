#!/usr/bin/env bash
#
# OpenAI 모델 검증 (두 모델 동시 병렬 실행):
#   gpt-4.1-mini: nomem_think, nomem_nothink, naive, gated_oracle, gated_llm, gated_observation
#   gpt-4.1:      nomem_think, nomem_nothink, naive
# 대시보드 보고 기본 켜짐(실험 등록 + run.log). 끄려면 DASHBOARD=0
# 프로젝트 루트에서 실행: bash experiments/scripts/openai_validation/run_validation.sh
# 테스트용 기본 3 에피소드. 전체 30 에피소드: NUM_EPISODES=30 bash ...
#

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PARALLEL_EPISODES=6

cd "$ROOT_DIR"


# Ctrl+C 또는 TERM 시 모든 백그라운드 실험 프로세스 종료
PIDS=()
cleanup() {
  echo ""
  echo "[run_validation] Interrupted — stopping all experiment processes..."
  for pid in "${PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  # SIGTERM 무시 시를 대비해 2초 후 SIGKILL
  sleep 2
  for pid in "${PIDS[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done
  echo "[run_validation] All processes stopped."
  exit 130
}
trap cleanup INT TERM

OUTPUT_BASE="$SCRIPT_DIR/results"
mkdir -p "$OUTPUT_BASE"

# 대시보드 보고(실험 등록 + run.log): run.py 기본 켜짐. 끄려면 DASHBOARD=0
DASHBOARD_ARGS=()
[ "$DASHBOARD" = "0" ] && DASHBOARD_ARGS=(--no-dashboard)

# 테스트 모드: USE_DUMMY_LLM=1 이면 API 없이 dummy LLM으로 1개 조건만 실행
USE_DUMMY_LLM="${USE_DUMMY_LLM:-0}"
if [ "$USE_DUMMY_LLM" = "1" ]; then
  echo "[TEST MODE] USE_DUMMY_LLM=1: running one condition (nomem) with dummy LLM for verification."
  EXTRA_ARGS=(--use_dummy_llm)
else
  if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Warning: OPENAI_API_KEY not set. Set it with: export OPENAI_API_KEY=your-key"
  fi
  EXTRA_ARGS=()
fi

# NUM_EPISODES 미지정 시 run.py 기본값(30)이 사용됨
EPISODES_ARGS=()
[ -n "${NUM_EPISODES:-}" ] && EPISODES_ARGS=(--num_episodes "$NUM_EPISODES")

BASE_ARGS=(
  --env light
  --agent llm
  "${EPISODES_ARGS[@]}"
  --max_steps 200
  --seed 42
  --reflection_frequency 1
  --parallel_episodes "$PARALLEL_EPISODES"
  --max_state_visits 10
  "${EXTRA_ARGS[@]}"
)

run_one() {
  local model="$1"
  local condition="$2"
  shift 2
  local outdir="$OUTPUT_BASE/$model/$condition"
  mkdir -p "$outdir"
  echo "=============================================="
  echo "Running: model=$model condition=$condition -> $outdir"
  echo "=============================================="
  python -m experiments.run \
    "${BASE_ARGS[@]}" \
    --model "$model" \
    --provider openai \
    --output_dir "$outdir" \
    --experiment_name "${model}-${condition}" \
    "${DASHBOARD_ARGS[@]}" \
    "$@"
  echo ""
}

# run_two() {
#   local model="$1"
#   local condition="$2"
#   shift 2
#   local outdir="$OUTPUT_BASE/$model/$condition"
#   mkdir -p "$outdir"
#   echo "=============================================="
#   echo "Running: model=$model condition=$condition -> $outdir"
#   echo "=============================================="
#   python -m experiments.run \
#     "${BASE_ARGS[@]}" \
#     --model "$model" \
#     --provider gemini \
#     --output_dir "$outdir" \
#     --experiment_name "${model}-${condition}" \
#     "${DASHBOARD_ARGS[@]}" \
#     "$@"
#   echo ""
# }

# --- gpt-4.1-mini (6개) + gpt-4.1 (3개) 동시 병렬 실행 ---
PIDS=()
# # gpt-4.1-mini
# run_one "gpt-4.1-mini" "nomem_think"       --memory nomem --intervention none &
# PIDS+=($!)
# run_one "gpt-4.1-mini" "nomem_nothink"     --memory nomem --intervention none --think_in_history none &
# PIDS+=($!)
# if [ "$USE_DUMMY_LLM" != "1" ]; then
#   run_one "gpt-4.1-mini" "naive"             --memory naive --verification_mode none --intervention none &
#   PIDS+=($!)
#   run_one "gpt-4.1-mini" "gated_oracle"      --memory gated --verification_mode oracle --intervention none &
#   PIDS+=($!)
#   run_one "gpt-4.1-mini" "gated_llm"         --memory gated --verification_mode llm --intervention none &
#   PIDS+=($!)
#   run_one "gpt-4.1-mini" "gated_observation" --memory gated --verification_mode observation --intervention none &
#   PIDS+=($!)
# fi

# gpt-5
# run_one "gpt-5" "nomem_think"   --memory nomem --intervention none &
# PIDS+=($!)
run_one "gpt-5" "nomem_nothink" --memory nomem --intervention none --think_in_history none &
PIDS+=($!)
run_two "gemini-2.5-pro" "nomem_nothink" --memory nomem --intervention none --think_in_history none &
PIDS+=($!)
# if [ "$USE_DUMMY_LLM" != "1" ]; then
#   run_one "gpt-5" "naive"         --memory naive --verification_mode none --intervention none &
#   PIDS+=($!)
# fi

for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done

echo "Done. Results under $OUTPUT_BASE"
# echo "Summaries: $OUTPUT_BASE/gpt-4.1-mini/*/summary.json and $OUTPUT_BASE/gpt-4.1/*/summary.json"
