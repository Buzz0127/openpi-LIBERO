#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PHYSICAL_GPU OUTPUT_DIR CONTROL_DIR" >&2
  exit 2
fi

physical_gpu=$1
output_dir=$2
control_dir=$3

openpi_root="${OPENPI_ROOT:-$HOME/projects/openpi}"
model_python="$openpi_root/.venv/bin/python"
libero_python="$openpi_root/examples/libero/.venv/bin/python"
evaluator="${EVALUATOR_PATH:-$HOME/projects/openpi-eval-tools/eval_libero_bounded.py}"
checkpoint="${CHECKPOINT_DIR:-$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_libero}"
egl_vendor="${EGL_VENDOR_FILE:-$HOME/tmp/openpi-setup/egl-vendor/10_nvidia.json}"
libero_config="${LIBERO_CONFIG_PATH:-$HOME/projects/openpi-eval-tools/config/libero}"

# Evaluation scope defaults to the already validated 10-state run. Override
# these variables to stage a smaller smoke run before expanding the scope.
suite="${LIBERO_SUITE:-libero_spatial}"
task_id="${LIBERO_TASK_ID:-0}"
initial_states="${LIBERO_INITIAL_STATES:-0:10}"
max_episodes="${LIBERO_MAX_EPISODES:-10}"
seed="${LIBERO_SEED:-7}"
replan_steps="${LIBERO_REPLAN_STEPS:-5}"
max_control_steps="${LIBERO_MAX_CONTROL_STEPS:-220}"
max_output_bytes="${LIBERO_MAX_OUTPUT_BYTES:-1073741824}"
resume="${LIBERO_RESUME:-0}"

if [[ $resume != 0 && $resume != 1 ]]; then
  echo "LIBERO_RESUME must be 0 or 1" >&2
  exit 3
fi
resume_args=()
if [[ $resume == 1 ]]; then
  resume_args=(--resume)
fi

mkdir -p "$control_dir"
server_log="$control_dir/server.log"
evaluator_log="$control_dir/evaluator.log"
server_pid=

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
if [[ -e "$output_dir" && $resume != 1 ]]; then
  echo "output directory already exists: $output_dir" >&2
  exit 13
fi
if [[ ! -e "$output_dir" && $resume == 1 ]]; then
  echo "resume output directory does not exist: $output_dir" >&2
  exit 14
fi

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

{
  echo "started_at=$(date -Is)"
  echo "physical_gpu=$physical_gpu"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE"
  echo "MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
  echo "suite=$suite"
  echo "task_id=$task_id"
  echo "initial_states=$initial_states"
  echo "max_episodes=$max_episodes"
  echo "seed=$seed"
  echo "replan_steps=$replan_steps"
  echo "max_control_steps=$max_control_steps"
  echo "max_output_bytes=$max_output_bytes"
  echo "resume=$resume"
  echo "workload_pid=$$"
  echo "workload_pgid=$(ps -o pgid= -p $$ | tr -d ' ')"
} > "$control_dir/workload_identity.txt"

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

set +e
"$libero_python" "$evaluator" \
  --suite "$suite" \
  --task-id "$task_id" \
  --initial-states "$initial_states" \
  --max-episodes "$max_episodes" \
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
  --expected-openpi-commit 15a9616a00943ada6c20a0f158e3adb39df2ccac \
  --expected-libero-commit f78abd68ee283de9f9be3c8f7e2a9ad60246e95c \
  --output-dir "$output_dir" \
  --max-output-bytes "$max_output_bytes" \
  "${resume_args[@]}" \
  > "$evaluator_log" 2>&1
evaluator_rc=$?
set -e

echo "$evaluator_rc" > "$control_dir/evaluator.exit_code"
stop_server
server_pid=
echo "completed_at=$(date -Is)" >> "$control_dir/workload_identity.txt"
exit "$evaluator_rc"
