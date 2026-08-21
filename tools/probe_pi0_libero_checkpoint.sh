#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PHYSICAL_GPU CONTROL_DIR" >&2
  exit 2
fi

physical_gpu=$1
control_dir=$2
openpi_root="${OPENPI_ROOT:-$HOME/projects/openpi}"
model_python="$openpi_root/.venv/bin/python"
checkpoint="${CHECKPOINT_DIR:-$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_libero}"

mkdir -p "$control_dir"
for _ in {1..1200}; do
  if [[ -f "$control_dir/workload.start" ]]; then
    break
  fi
  sleep 0.1
done
if [[ ! -f "$control_dir/workload.start" ]]; then
  echo "checkpoint probe start gate timeout" >&2
  exit 12
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

server_pid=
stop_server() {
  if [[ -n ${server_pid:-} ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid"
    for _ in {1..30}; do
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

server_log="$control_dir/checkpoint_server.log"
{
  echo "physical_gpu=$physical_gpu"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE"
  echo "probe_pid=$$"
  echo "probe_pgid=$(ps -o pgid= -p $$ | tr -d ' ')"
  echo "checkpoint=$checkpoint"
  echo "policy_config=pi0_libero"
} > "$control_dir/checkpoint_probe_identity.txt"

cd "$openpi_root"
"$model_python" scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi0_libero \
  --policy.dir "$checkpoint" \
  > "$server_log" 2>&1 &
server_pid=$!
echo "$server_pid" > "$control_dir/checkpoint_server.pid"

ready=0
for _ in {1..240}; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid" || true
    echo "policy server exited before readiness" >&2
    exit 20
  fi
  if grep -q "server listening on 0.0.0.0:8000" "$server_log"; then
    ready=1
    break
  fi
  sleep 1
done
if [[ $ready -ne 1 ]]; then
  echo "policy server readiness timeout" >&2
  exit 21
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$control_dir/checkpoint_ready_utc.txt"
sleep 5
stop_server
server_pid=
trap - EXIT INT TERM

for _ in {1..30}; do
  if ! ss -H -ltn 'sport = :8000' | grep -q .; then
    echo "checkpoint_load_probe=ok"
    exit 0
  fi
  sleep 1
done

echo "port 8000 remained busy after checkpoint probe cleanup" >&2
exit 22
