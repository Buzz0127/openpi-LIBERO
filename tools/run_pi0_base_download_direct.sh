#!/usr/bin/env bash
set -euo pipefail

# HISTORICAL C0 NOTE: this launcher records the completed 2026-09-02 direct
# download configuration and its superseded 200 GB budget. It is intentionally
# fail-closed and must not be reused for a future download.
printf '%s\n' 'historical launcher: pi0_base is complete; do not reuse this 200 GB configuration' >&2
exit 64

PY=/home/wengzr/projects/openpi/.venv/bin/python
TOOLS=/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora
EVIDENCE_ROOT="$TOOLS/evidence/b1-pi0-base"
SOURCE_MANIFEST="$EVIDENCE_ROOT/attempt-20260902T-B1-MANIFEST-3SAqO00x/source_manifest.json"
TARGET_ROOT=/home/wengzr/.cache/openpi/openpi-assets/checkpoints/pi0_base

mkdir -p "$EVIDENCE_ROOT"
launcher_log=$(mktemp "$EVIDENCE_ROOT/launcher-20260902T-B1-DIRECT-XXXXXXXX.log")
exec >>"$launcher_log" 2>&1
printf 'mode=direct\nlauncher_pid=%s\nstart_utc=%s\n' "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export PYTHONDONTWRITEBYTECODE=1
exec "$PY" "$TOOLS/automate_pi0_base_download.py" \
  --python "$PY" \
  --guard "$TOOLS/storage_budget_guard.py" \
  --downloader "$TOOLS/resumable_gcs_prefix_download.py" \
  --source-manifest "$SOURCE_MANIFEST" \
  --evidence-root "$EVIDENCE_ROOT" \
  --monitor-root /home/wengzr/projects/openpi \
  --monitor-root /home/wengzr/projects/openpi-eval-tools \
  --monitor-root /home/wengzr/.cache/openpi \
  --monitor-root /home/wengzr/.cache/uv \
  --monitor-root /home/wengzr/projects/openpi-lora-cache/huggingface \
  --monitor-root /home/wengzr/projects/openpi-lora-assets \
  --existing-billed-bytes 81650000000 \
  --soft-limit-bytes 190000000000 \
  --hard-limit-bytes 200000000000 \
  --scratch "$TARGET_ROOT/params.partial" \
  --final "$TARGET_ROOT/params" \
  --lock "$TARGET_ROOT/params.lock" \
  --timeout-seconds 86400
