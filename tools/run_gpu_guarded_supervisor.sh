#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 6 ]]; then
  echo "usage: $0 PHYSICAL_GPU E_MIB MAX_RUNTIME_SECONDS CONTROL_DIR -- COMMAND..." >&2
  exit 2
fi

physical_gpu=$1
e_mib=$2
max_runtime_seconds=$3
control_dir=$4
shift 4
if [[ $1 != -- ]]; then
  echo "missing -- before command" >&2
  exit 2
fi
shift
if [[ $# -eq 0 ]]; then
  echo "missing guarded command" >&2
  exit 2
fi
command_to_run=("$@")

guard="$HOME/projects/openpi-eval-tools/gpu_utilization_guard.py"
guard_sha=ba9f7c74b843a61dc7cdf2f48b182390edb59022b5a2e76a489ce9d7048c4594

mkdir -p "$control_dir"
rm -f "$control_dir/workload.start"
actual_guard_sha=$(sha256sum "$guard" | cut -d' ' -f1)
if [[ $actual_guard_sha != "$guard_sha" ]]; then
  echo "guard SHA-256 mismatch" >&2
  exit 10
fi

{
  echo "physical_gpu=$physical_gpu"
  echo "E_mib=$e_mib"
  echo "E_is_report_only=true"
  echo "minimum_launch_free_memory_percent=15"
  echo "pause_free_memory_percent=15"
  echo "resume_free_memory_percent=20"
  echo "terminate_free_memory_percent=10"
} > "$control_dir/capacity_plan.txt"

guard_log="$control_dir/guard.jsonl"
guard_stdout="$control_dir/guard.stdout.log"
guard_command=(
  python3 "$guard"
  --physical-gpu "$physical_gpu"
  --pause-at 95
  --resume-at 85
  --min-free-memory-percent 15
  --resume-free-memory-percent 20
  --terminate-free-memory-percent 10
  --resume-samples 5
  --interval-seconds 1
  --monitor-error-limit 3
  --max-prelaunch-wait-seconds 300
  --max-runtime-seconds "$max_runtime_seconds"
  --terminate-grace-seconds 15
  --log "$guard_log"
  --
  "${command_to_run[@]}"
)
printf '%q ' "${guard_command[@]}" > "$control_dir/guard_command.txt"
printf '\n' >> "$control_dir/guard_command.txt"

"${guard_command[@]}" > "$guard_stdout" 2>&1 &
guard_pid=$!
echo "$guard_pid" > "$control_dir/guard.pid"

cleanup() {
  if kill -0 "$guard_pid" 2>/dev/null; then
    kill -TERM "$guard_pid"
    wait "$guard_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

child_deadline=$((SECONDS + 310))
while ! grep -q '"event": "child_started"' "$guard_log" 2>/dev/null; do
  if ! kill -0 "$guard_pid" 2>/dev/null; then
    wait "$guard_pid" || true
    echo "guard exited before child start" >&2
    exit 11
  fi
  if (( SECONDS >= child_deadline )); then
    echo "guard child start timeout" >&2
    kill -TERM "$guard_pid"
    wait "$guard_pid" || true
    exit 12
  fi
  sleep 0.2
done

touch "$control_dir/workload.start"
set +e
wait "$guard_pid"
guard_rc=$?
set -e
trap - EXIT INT TERM
echo "$guard_rc" > "$control_dir/guard.exit_code"
exit "$guard_rc"
