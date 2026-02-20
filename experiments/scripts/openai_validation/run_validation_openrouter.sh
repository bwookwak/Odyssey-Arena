#!/usr/bin/env bash
#
# OpenRouter를 통한 gemini-2.5-pro 검증 (gpt-5 동시 병렬 실행):
#   gpt-5:                  nomem_nothink  (OpenAI)
#   google/gemini-2.5-pro:  nomem_nothink  (OpenRouter)
# 대시보드 보고 기본 켜짐. 끄려면 DASHBOARD=0
# 프로젝트 루트에서 실행: bash experiments/scripts/openai_validation/run_validation_openrouter.sh
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
  echo "[run_validation_openrouter] Interrupted — stopping all experiment processes..."
  for pid in "${PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "${PIDS[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done
  echo "[run_validation_openrouter] All processes stopped."
  exit 130
}
trap cleanup INT TERM

OUTPUT_BASE="$SCRIPT_DIR/results"
mkdir -p "$OUTPUT_BASE"

# 대시보드 보고: 끄려면 DASHBOARD=0
DASHBOARD_ARGS=()
[ "$DASHBOARD" = "0" ] && DASHBOARD_ARGS=(--no-dashboard)

# 테스트 모드: USE_DUMMY_LLM=1 이면 API 없이 dummy LLM으로 실행
USE_DUMMY_LLM="${USE_DUMMY_LLM:-0}"
if [ "$USE_DUMMY_LLM" = "1" ]; then
  echo "[TEST MODE] USE_DUMMY_LLM=1: running with dummy LLM for verification."
  EXTRA_ARGS=(--use_dummy_llm)
else
  if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Warning: OPENAI_API_KEY not set. Set it with: export OPENAI_API_KEY=your-key"
  fi
  if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "Warning: OPENROUTER_API_KEY not set. Set it with: export OPENROUTER_API_KEY=your-key"
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

# OpenAI provider (gpt-5 등)
run_openai() {
  local model="$1"
  local condition="$2"
  shift 2
  local outdir="$OUTPUT_BASE/$model/$condition"
  mkdir -p "$outdir"
  echo "=============================================="
  echo "Running: model=$model condition=$condition [openai] -> $outdir"
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

# OpenRouter provider (gemini-2.5-pro 등)
run_openrouter() {
  local model="$1"
  local condition="$2"
  shift 2
  # 결과 디렉토리명에서 '/' 를 '-' 로 치환 (google/gemini-2.5-pro → google-gemini-2.5-pro)
  local safe_model="${model//\//-}"
  local outdir="$OUTPUT_BASE/$safe_model/$condition"
  mkdir -p "$outdir"
  echo "=============================================="
  echo "Running: model=$model condition=$condition [openrouter] -> $outdir"
  echo "=============================================="
  python -m experiments.run \
    "${BASE_ARGS[@]}" \
    --model "$model" \
    --provider openrouter \
    --output_dir "$outdir" \
    --experiment_name "${safe_model}-${condition}" \
    "${DASHBOARD_ARGS[@]}" \
    "$@"
  echo ""
}

# --- gpt-5 (OpenAI) + google/gemini-2.5-pro (OpenRouter) 동시 병렬 실행 ---
PIDS=()

run_openai "gpt-5" "nomem_nothink" --memory nomem --intervention none --think_in_history none &
PIDS+=($!)

run_openrouter "google/gemini-2.5-pro" "nomem_nothink" --memory nomem --intervention none --think_in_history none &
PIDS+=($!)

for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done

echo "Done. Results under $OUTPUT_BASE"
