#!/usr/bin/env bash
# 프로젝트 루트에서 실행: bash experiments/scripts/start_dashboard.sh
# 대시보드 + 외부 접속 터널(cloudflared 우선, 없으면 ngrok). 터널 없으면 로컬만.
# 터널 끄고 로컬만: TUNNEL=0 bash experiments/scripts/start_dashboard.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

PORT="${PORT:-5000}"
TUNNEL="${TUNNEL:-1}"

# 포트 사용 중이면 기존 프로세스 종료
if command -v lsof >/dev/null 2>&1; then
  OLD_PIDS=$(lsof -ti ":$PORT" 2>/dev/null || true)
  if [ -n "$OLD_PIDS" ]; then
    echo "Port $PORT in use, stopping existing process(es): $OLD_PIDS"
    echo "$OLD_PIDS" | xargs kill 2>/dev/null || true
    sleep 1
  fi
elif command -v fuser >/dev/null 2>&1; then
  if fuser "$PORT/tcp" >/dev/null 2>&1; then
    echo "Port $PORT in use, stopping with fuser."
    fuser -k "$PORT/tcp" 2>/dev/null || true
    sleep 1
  fi
fi

cleanup() {
  if [ -n "${DASH_PID:-}" ] && kill -0 "$DASH_PID" 2>/dev/null; then
    kill "$DASH_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting dashboard on 0.0.0.0:$PORT (use PORT=8080 for custom port)"
python -m experiments.dashboard.app --host 0.0.0.0 --port "$PORT" &
DASH_PID=$!
sleep 2
if ! kill -0 $DASH_PID 2>/dev/null; then
  echo "Dashboard failed to start."
  exit 1
fi

if [ "$TUNNEL" = "0" ]; then
  echo "Local: http://localhost:$PORT  |  Same network: http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT"
  wait $DASH_PID
  exit 0
fi

if command -v cloudflared >/dev/null 2>&1; then
  echo "Starting Cloudflare Tunnel (external URL will appear below). Ctrl+C to stop."
  cloudflared tunnel --url "http://127.0.0.1:$PORT"
elif command -v ngrok >/dev/null 2>&1; then
  echo "Starting ngrok tunnel (external URL will appear below). Ctrl+C to stop."
  ngrok http "$PORT"
else
  echo "No tunnel (cloudflared/ngrok) found. Local only."
  echo "  Local: http://localhost:$PORT  |  Same network: http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT"
  echo "  Install cloudflared or ngrok for external access."
  wait $DASH_PID
fi
