#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PHYSICAL_GPU OUTPUT_ROOT CONTROL_DIR" >&2
  exit 2
fi

physical_gpu=$1
output_root=$2
control_dir=$3

openpi_root="${OPENPI_ROOT:-$HOME/projects/openpi}"
model_python="$openpi_root/.venv/bin/python"
libero_python="$openpi_root/examples/libero/.venv/bin/python"
evaluator="${EVALUATOR_PATH:-$HOME/projects/openpi-eval-tools/eval_libero_bounded.py}"
checkpoint="${CHECKPOINT_DIR:-$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_libero}"
egl_vendor="${EGL_VENDOR_FILE:-$HOME/tmp/openpi-setup/egl-vendor/10_nvidia.json}"
libero_config="${LIBERO_CONFIG_PATH:-$HOME/projects/openpi-eval-tools/config/libero}"

dry_run="${BENCHMARK_DRY_RUN:-0}"
seed=7
replan_steps=5
episodes_per_task=50
per_task_output_limit=$((64 * 1024 * 1024))
global_output_limit=$((4 * 1024 * 1024 * 1024))
expected_openpi_commit=15a9616a00943ada6c20a0f158e3adb39df2ccac
expected_libero_commit=f78abd68ee283de9f9be3c8f7e2a9ad60246e95c

if [[ $dry_run != 0 && $dry_run != 1 ]]; then
  echo "BENCHMARK_DRY_RUN must be 0 or 1" >&2
  exit 3
fi
if [[ ! $physical_gpu =~ ^[0-9]+$ ]]; then
  echo "PHYSICAL_GPU must be a non-negative integer" >&2
  exit 4
fi

mkdir -p "$control_dir/logs"
server_log="$control_dir/server.log"
progress_file="$control_dir/progress.tsv"
manifest="$output_root/benchmark_config.txt"
expected_manifest="$control_dir/expected_benchmark_config.txt"
server_pid=

cat > "$expected_manifest" <<EOF
schema_version=1
scope=official_four_suites_remaining_after_libero_spatial_task0
suite_tasks=libero_spatial:1-9,libero_object:0-9,libero_goal:0-9,libero_10:0-9
episodes_per_task=50
initial_states=0:50
new_episode_total=1950
seed=$seed
replan_steps=$replan_steps
policy_config=pi0_libero
checkpoint=$checkpoint
openpi_commit=$expected_openpi_commit
libero_commit=$expected_libero_commit
physical_gpu=$physical_gpu
per_task_output_limit_bytes=$per_task_output_limit
global_output_limit_bytes=$global_output_limit
EOF

directory_bytes() {
  local path=$1
  if [[ -d $path ]]; then
    du -sb "$path" | cut -f1
  else
    echo 0
  fi
}

task_state() {
  local task_output=$1
  "$libero_python" -c '
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    print("fresh")
elif not (path / "run_config.json").is_file():
    print("invalid")
else:
    completed = set()
    results = path / "results.jsonl"
    if results.is_file():
        for line in results.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(int(json.loads(line)["initial_state_index"]))
    print("complete" if completed == set(range(50)) else "resume")
' "$task_output"
}

append_progress() {
  local status=$1
  local suite=$2
  local task_id=$3
  local completed=$4
  local bytes=$5
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" "$status" "$suite" "$task_id" "$completed" "$bytes" >> "$progress_file"
}

stop_server() {
  if [[ -n ${server_pid:-} ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid"
    for _ in {1..15}; do
      if ! kill -0 "$server_pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -KILL "$server_pid"
    fi
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap stop_server EXIT INT TERM

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="$physical_gpu"
export __EGL_VENDOR_LIBRARY_FILENAMES="$egl_vendor"
export PYTHONPATH="$openpi_root/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$libero_config"

if [[ $dry_run == 1 ]]; then
  export JAX_PLATFORMS=cpu
  export CUDA_VISIBLE_DEVICES=
fi

if [[ $dry_run == 0 ]]; then
  for _ in {1..1800}; do
    if [[ -f "$control_dir/workload.start" ]]; then
      break
    fi
    sleep 0.1
  done
  if [[ ! -f "$control_dir/workload.start" ]]; then
    echo "workload start gate timeout" >&2
    exit 12
  fi

  if [[ -e $output_root ]]; then
    if [[ ! -f $manifest ]]; then
      echo "existing output root has no benchmark_config.txt: $output_root" >&2
      exit 13
    fi
    if ! cmp -s "$expected_manifest" "$manifest"; then
      echo "existing benchmark configuration does not match this run" >&2
      diff -u "$manifest" "$expected_manifest" >&2 || true
      exit 14
    fi
  else
    mkdir -p "$output_root"
    cp "$expected_manifest" "$manifest"
  fi

  if "$model_python" -c 'import socket; s=socket.socket(); s.settimeout(0.2); raise SystemExit(0 if s.connect_ex(("127.0.0.1", 8000)) == 0 else 1)'; then
    echo "port 8000 is already in use" >&2
    exit 15
  fi

  cd "$openpi_root"
  "$model_python" scripts/serve_policy.py \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi0_libero \
    --policy.dir "$checkpoint" \
    > "$server_log" 2>&1 &
  server_pid=$!
  echo "$server_pid" > "$control_dir/server.pid"

  server_ready=0
  for _ in {1..240}; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "policy server exited before readiness" >&2
      wait "$server_pid" || true
      exit 20
    fi
    if grep -q "server listening on 0.0.0.0:8000" "$server_log"; then
      server_ready=1
      break
    fi
    sleep 1
  done
  if [[ $server_ready -ne 1 ]]; then
    echo "policy server readiness timeout" >&2
    exit 21
  fi
fi

if [[ ! -f $progress_file ]]; then
  printf 'timestamp\tstatus\tsuite\ttask_id\tcompleted_episodes\toutput_bytes\n' > "$progress_file"
fi

run_task() {
  local suite=$1
  local task_id=$2
  local max_control_steps=$3
  local task_label
  task_label=$(printf '%s_task%02d' "$suite" "$task_id")
  local task_output="$output_root/$suite/task_$(printf '%02d' "$task_id")/evaluation"
  local task_log="$control_dir/logs/$task_label.log"
  local state
  state=$(task_state "$task_output")

  if [[ $state == invalid ]]; then
    echo "invalid partial output directory: $task_output" >&2
    return 30
  fi
  if [[ $state == complete ]]; then
    append_progress skipped_complete "$suite" "$task_id" 50 "$(directory_bytes "$task_output")"
    echo "TASK_SKIP suite=$suite task=$task_id reason=already_complete"
    return 0
  fi

  local current_bytes
  current_bytes=$(directory_bytes "$output_root")
  if (( current_bytes >= global_output_limit )); then
    echo "global output limit reached before $task_label: $current_bytes" >&2
    return 31
  fi

  local resume_args=()
  if [[ $state == resume ]]; then
    resume_args=(--resume)
  fi
  local dry_args=()
  if [[ $dry_run == 1 ]]; then
    dry_args=(--dry-run)
  fi

  echo "TASK_START suite=$suite task=$task_id state=$state max_steps=$max_control_steps"
  set +e
  "$libero_python" "$evaluator" \
    --suite "$suite" \
    --task-id "$task_id" \
    --initial-states 0:50 \
    --max-episodes "$episodes_per_task" \
    --seed "$seed" \
    --host 127.0.0.1 \
    --port 8000 \
    --server-wait-seconds 30 \
    --replan-steps "$replan_steps" \
    --max-control-steps "$max_control_steps" \
    --physical-gpu "$physical_gpu" \
    --mujoco-egl-device-id "$physical_gpu" \
    --egl-vendor-file "$egl_vendor" \
    --gpu-sample-interval 1 \
    --max-gpu-memory-fraction 0.90 \
    --max-baseline-gpu-utilization 100 \
    --max-baseline-gpu-memory-fraction 1.0 \
    --policy-config pi0_libero \
    --openpi-root "$openpi_root" \
    --checkpoint-dir "$checkpoint" \
    --expected-openpi-commit "$expected_openpi_commit" \
    --expected-libero-commit "$expected_libero_commit" \
    --output-dir "$task_output" \
    --max-output-bytes "$per_task_output_limit" \
    "${resume_args[@]}" \
    "${dry_args[@]}" \
    > "$task_log" 2>&1
  local evaluator_rc=$?
  set -e

  if [[ $evaluator_rc -ne 0 ]]; then
    append_progress failed "$suite" "$task_id" -1 "$(directory_bytes "$task_output")"
    echo "TASK_FAILED suite=$suite task=$task_id rc=$evaluator_rc log=$task_log" >&2
    return "$evaluator_rc"
  fi
  if [[ $dry_run == 1 ]]; then
    append_progress dry_run_ok "$suite" "$task_id" 0 0
    echo "TASK_DRY_RUN_OK suite=$suite task=$task_id"
    return 0
  fi

  state=$(task_state "$task_output")
  if [[ $state != complete ]]; then
    append_progress incomplete "$suite" "$task_id" -1 "$(directory_bytes "$task_output")"
    echo "evaluator exited successfully but task is not complete: $task_label" >&2
    return 32
  fi
  current_bytes=$(directory_bytes "$output_root")
  if (( current_bytes > global_output_limit )); then
    append_progress output_limit_exceeded "$suite" "$task_id" 50 "$current_bytes"
    echo "global output limit exceeded: $current_bytes" >&2
    return 33
  fi
  append_progress completed "$suite" "$task_id" 50 "$(directory_bytes "$task_output")"
  echo "TASK_COMPLETE suite=$suite task=$task_id global_output_bytes=$current_bytes"
}

for task_id in {1..9}; do
  run_task libero_spatial "$task_id" 220
done
for task_id in {0..9}; do
  run_task libero_object "$task_id" 280
done
for task_id in {0..9}; do
  run_task libero_goal "$task_id" 300
done
for task_id in {0..9}; do
  run_task libero_10 "$task_id" 520
done

if [[ $dry_run == 0 ]]; then
  stop_server
  server_pid=
  {
    echo "completed_at=$(date -Is)"
    echo "completed_tasks=39"
    echo "completed_new_episodes=1950"
    echo "output_bytes=$(directory_bytes "$output_root")"
  } > "$control_dir/benchmark_complete.txt"
fi
echo "BENCHMARK_WORKLOAD_COMPLETE dry_run=$dry_run"
