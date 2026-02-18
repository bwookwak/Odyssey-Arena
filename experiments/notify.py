"""
Email notification helper for experiment end.

Usage (environment variables — recommended):
    export NOTIFY_TO=you@gmail.com
    export NOTIFY_FROM=you@gmail.com
    export NOTIFY_PASSWORD=your_app_password
    # optionally:
    export NOTIFY_SMTP_HOST=smtp.gmail.com  (default)
    export NOTIFY_SMTP_PORT=587              (default)

Or pass kwargs directly to send_experiment_email().

Gmail: use App Password (2FA 필요).
Outlook/Office365: smtp-mail.outlook.com, port 587.
Custom SMTP: set NOTIFY_SMTP_HOST / NOTIFY_SMTP_PORT.
"""

import os
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any


def send_experiment_email(
    *,
    to: Optional[str] = None,
    from_addr: Optional[str] = None,
    password: Optional[str] = None,
    smtp_host: Optional[str] = None,
    smtp_port: Optional[int] = None,
    experiment_name: Optional[str] = None,
    experiment_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    success: bool = True,
    summary: Optional[Dict[str, Any]] = None,
    config_summary: Optional[Dict[str, Any]] = None,
):
    """Send experiment completion email. Falls back to env vars for any unset param."""
    to = to or os.environ.get("NOTIFY_TO", "")
    from_addr = from_addr or os.environ.get("NOTIFY_FROM", to)
    password = password or os.environ.get("NOTIFY_PASSWORD", "")
    smtp_host = smtp_host or os.environ.get("NOTIFY_SMTP_HOST", "smtp.gmail.com")
    smtp_port = smtp_port or int(os.environ.get("NOTIFY_SMTP_PORT", "587"))

    if not to or not password:
        print("[notify] Skipping email: NOTIFY_TO or NOTIFY_PASSWORD not set.")
        return

    status_str = "✅ 완료 (completed)" if success else "❌ 실패 (failed)"
    name = experiment_name or experiment_id or "unknown"

    # Truncate experiment_id to everything before the last '_' (strip microseconds suffix)
    # e.g. exp_20260218_174739_606859 → exp_20260218_174739
    def _short_id(eid: Optional[str]) -> str:
        if not eid:
            return ""
        idx = eid.rfind("_")
        return eid[:idx] if idx > 0 else eid

    id_suffix = _short_id(experiment_id)
    subject = f"[Experiment] {id_suffix} {name} — {status_str}" if id_suffix else f"[Experiment] {name} — {status_str}"

    # Build HTML body
    parts = [
        f"<h2>실험 종료 알림</h2>",
        f"<p><b>이름:</b> {name}</p>",
        f"<p><b>ID:</b> {experiment_id or '-'}</p>",
        f"<p><b>상태:</b> {status_str}</p>",
        f"<p><b>출력 디렉토리:</b> <code>{output_dir or '-'}</code></p>",
    ]

    if config_summary:
        lines = "\n".join(f"  {k}: {v}" for k, v in config_summary.items() if v is not None)
        parts.append(f"<h3>설정 요약</h3><pre>{lines}</pre>")

    if summary:
        parts.append(f"<h3>실험 결과 요약</h3><pre>{json.dumps(summary, indent=2, ensure_ascii=False)}</pre>")

    html_body = "\n".join(parts)
    text_body = (
        f"실험 종료 알림\n"
        f"이름: {name}\n"
        f"ID: {experiment_id or '-'}\n"
        f"상태: {status_str}\n"
        f"출력: {output_dir or '-'}\n"
    )
    if config_summary:
        text_body += "\n[설정 요약]\n" + "\n".join(f"  {k}: {v}" for k, v in config_summary.items() if v is not None)
    if summary:
        text_body += "\n\n[결과 요약]\n" + json.dumps(summary, indent=2, ensure_ascii=False)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to
    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(from_addr, password)
            server.sendmail(from_addr, [to], msg.as_string())
        print(f"[notify] Email sent to {to}: {subject}")
    except Exception as e:
        print(f"[notify] Email failed: {e}")
