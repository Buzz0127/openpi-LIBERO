#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 ATTEMPT_DIR [OFFLINE_INPUT_DIR]" >&2
  exit 2
fi

attempt=$1
offline_input_dir=${2:-}
revision=a4336d589d589045d1c56423ffdf3b88a0e19b1f
python_bin=/home/wengzr/projects/openpi/.venv/bin/python
worktree=/home/wengzr/projects/openpi-worktrees/pi0-libero-pure-lora
audit_script="$attempt/audit_libero_dataset_metadata.py"

umask 077
test -d "$attempt"
test -f "$audit_script"
test ! -e "$attempt/dataset_audit"
test ! -e "$attempt/command.txt"

offline_args=()
acquisition=online_huggingface_hub
if [[ -n "$offline_input_dir" ]]; then
  test -d "$offline_input_dir"
  expected_inputs=(episodes.jsonl info.json repo_info.json stats.json tasks.jsonl)
  [[ $(find "$offline_input_dir" -maxdepth 1 -type f | wc -l) -eq ${#expected_inputs[@]} ]]
  total_input_bytes=0
  for name in "${expected_inputs[@]}"; do
    test -f "$offline_input_dir/$name"
    total_input_bytes=$((total_input_bytes + $(stat -c %s "$offline_input_dir/$name")))
  done
  [[ $total_input_bytes -le 1048576 ]]
  [[ -z $(find "$offline_input_dir" -maxdepth 1 -type f \( -name '*.parquet' -o -name '*.mp4' \) -print -quit) ]]
  sha256sum "$offline_input_dir"/* >"$attempt/offline_input_hashes.sha256"
  offline_args=(--offline-input-dir "$offline_input_dir")
  acquisition=offline_verified_inputs
fi

printf '%s\n' \
  "stage=D1a" \
  "repo=physical-intelligence/libero" \
  "revision=$revision" \
  "python=$python_bin" \
  "acquisition=$acquisition" \
  "policy=metadata-only; no parquet/model/GPU" \
  >"$attempt/command.txt"

{
  date -u +start=%Y-%m-%dT%H:%M:%SZ
  hostname
  printf 'worktree='
  git -C "$worktree" rev-parse HEAD
  git -C "$worktree" status --short --branch
  "$python_bin" -c 'import sys,huggingface_hub,requests; print(sys.version); print("huggingface_hub="+huggingface_hub.__version__); print("requests="+requests.__version__)'
} >"$attempt/identity_pre.log" 2>&1

meter() {
  local path
  for path in \
    /home/wengzr/projects/openpi \
    /home/wengzr/projects/openpi-worktrees/pi0-libero-pure-lora \
    /home/wengzr/projects/openpi-eval-tools \
    /home/wengzr/.cache/openpi \
    /home/wengzr/.cache/uv \
    /home/wengzr/projects/openpi-lora-cache \
    /home/wengzr/projects/openpi-lora-assets \
    /home/wengzr/projects/openpi-lora-runs
  do
    if [[ -e "$path" ]]; then
      du -sB1 "$path"
    else
      printf '0\t%s\n' "$path"
    fi
  done
}

meter >"$attempt/storage_pre.tsv" 2>"$attempt/storage_pre.stderr.log"

env \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX="$attempt/pycache" \
  "$python_bin" -m py_compile "$audit_script" \
  >"$attempt/py_compile.stdout.log" \
  2>"$attempt/py_compile.stderr.log"
compile_rc=$?
printf '%s\n' "$compile_rc" >"$attempt/py_compile.exit_code"

audit_rc=$compile_rc
if [[ $compile_rc -eq 0 ]]; then
  env \
    -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
    -u ALL_PROXY -u all_proxy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX="$attempt/pycache" \
    HF_HOME=/home/wengzr/projects/openpi-lora-cache/huggingface \
    HF_LEROBOT_HOME=/home/wengzr/projects/openpi-lora-cache/huggingface/lerobot \
    HF_HUB_DISABLE_TELEMETRY=1 \
    CUDA_VISIBLE_DEVICES='' \
    JAX_PLATFORM_NAME=cpu \
    timeout --signal=TERM --kill-after=10s 300s \
    "$python_bin" "$audit_script" \
      --revision "$revision" \
      --output-dir "$attempt/dataset_audit" \
      --timeout-seconds 60 \
      "${offline_args[@]}" \
    >"$attempt/audit.stdout.log" \
    2>"$attempt/audit.stderr.log"
  audit_rc=$?
fi
printf '%s\n' "$audit_rc" >"$attempt/audit.exit_code"

{
  date -u +end=%Y-%m-%dT%H:%M:%SZ
  git -C "$worktree" rev-parse HEAD
  git -C "$worktree" status --short --branch
} >"$attempt/identity_post.log" 2>&1
meter >"$attempt/storage_post.tsv" 2>"$attempt/storage_post.stderr.log"

find "$attempt" -type f ! -name evidence_hashes.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  >"$attempt/evidence_hashes.sha256"

exit "$audit_rc"
