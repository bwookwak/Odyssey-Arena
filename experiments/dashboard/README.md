# Experiment Dashboard

실험 목록·상태·로그·결과를 브라우저에서 확인하는 웹 대시보드입니다. **대시보드는 그냥 계속 띄워 두고**, 실험을 이 대시보드에 보고하는지는 각 실행에서 선택하면 됩니다.

## 실행 방법

프로젝트 루트에서 대시보드만 켜 두면 됩니다:

```bash
# 기본 (localhost:5000)
python -m experiments.dashboard.app

# 외부 접속 허용 (같은 네트워크에서 접근)
python -m experiments.dashboard.app --host 0.0.0.0 --port 5000
```

스크립트로 띄우기(대시보드 + 외부 터널 자동): `bash experiments/scripts/start_dashboard.sh`  
- cloudflared 또는 ngrok이 있으면 터널을 띄워 외부 URL을 출력합니다.  
- 터널 없이 로컬만: `TUNNEL=0 bash experiments/scripts/start_dashboard.sh`

## 대시보드에 보고(등록) — 선택 사항

- 실험을 대시보드에 올리려면 실행 시 **보고 옵션**을 켜면 됩니다.
- `experiments.run`: `--dashboard` 옵션 또는 환경변수 `DASHBOARD=1`
- `run_validation.sh`: 기본으로 보고함. 끄려면 `DASHBOARD=0 bash ... run_validation.sh`

등록 시 `experiments/dashboard/data/registry.json`에 실험 메타데이터가 쌓이고, 각 실험의 `output_dir`에 `run.log`, `experiment_meta.json`이 생성됩니다.

## 외부에서 접속 (호스팅/터널)

로컬이 아닌 다른 기기나 인터넷에서 접속하려면 터널 서비스를 쓰면 됩니다.

### ngrok (예시)

1. [ngrok](https://ngrok.com/) 가입 후 설치.
2. 대시보드를 `0.0.0.0`으로 실행한 뒤, 다른 터미널에서:

   ```bash
   ngrok http 5000
   ```
3. 출력된 `https://xxxx.ngrok.io` 주소로 접속.

### Cloudflare Tunnel (cloudflared)

1. [Cloudflare Zero Trust](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/) 또는 [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/) 설치.
2. 대시보드 실행 후:

   ```bash
   cloudflared tunnel --url http://localhost:5000
   ```
3. 표시되는 `*.trycloudflare.com` URL로 접속.

### Railway / Render 등

- Flask 앱을 배포할 수 있는 호스팅에 올려서 서비스할 수 있습니다.
- `experiments/dashboard/app.py`의 `main()`이 `app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))` 형태로 포트를 환경변수에서 읽도록 되어 있으면, 대부분의 PaaS에서 그대로 동작합니다.

## 화면 구성

- **목록 (`/`)**: 실험 ID, 상태(running/completed/failed), 시작·종료 시각, 설정 요약.
- **상세 (`/experiment/<id>`)**: 해당 실험의 config, summary, run.log(최근 512KB) 보기.

진행 중인 실험은 상태가 `running`이며, 상세 페이지에서 로그를 새로고침하면 최신 run.log가 반영됩니다.
