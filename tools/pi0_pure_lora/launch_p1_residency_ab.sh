#!/usr/bin/env bash
# P1 only: bounded sequential host-vs-device parameter-residency benchmark.
set -euo pipefail
: "${P1_PHYSICAL_GPU:?set only from this P1 attempt fresh preflight}"
: "${P1_CONTROL:?set to this collision-safe P1 control directory}"
: "${P1_RUN:?set to this collision-safe P1 run directory}"
: "${P1_GUARD:?set to deployed memory-only guard}"
[[ "$P1_PHYSICAL_GPU" =~ ^[0-9]+$ ]]
[[ "$P1_CONTROL" = /* ]]
[[ "$P1_RUN" = /* ]]
[[ "$P1_GUARD" = /* ]]

readonly CONTROL="$P1_CONTROL"
readonly SNAPSHOT=/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora/tool-snapshots/ft4-control-progress-heartbeat-20260911
readonly RUN="$P1_RUN"
readonly PYTHON=/home/wengzr/projects/openpi/.venv/bin/python
readonly POLICY_ROOT=/home/wengzr/projects/openpi-worktrees/pi0-libero-pure-lora
readonly ADAPTER=/home/wengzr/projects/openpi-lora-runs/adapters/pi0-libero-pure-lora-ft0-seed42-v1/step-00025000
readonly MODEL_MANIFEST=/home/wengzr/projects/openpi-lora-runs/evaluations/attempt-20260912T-E1-DEV-R8-B1w9aV/candidates/step-00025000/model_manifest.json
readonly GUARD_SHA256=273012b8e52afa7e5ac54f131ef5049da001732258f850fec9db2dee0f139083

test ! -e "$RUN"
test "$(sha256sum "$P1_GUARD" | awk '{print $1}')" = "$GUARD_SHA256"
export CUDA_VISIBLE_DEVICES="$P1_PHYSICAL_GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$CONTROL/pycache"
export PYTHONPATH="$CONTROL:$POLICY_ROOT/src:$POLICY_ROOT:/home/wengzr/projects/openpi/packages/openpi-client/src"

exec "$PYTHON" "$P1_GUARD" \
  --physical-gpu "$P1_PHYSICAL_GPU" --disable-utilization-gate \
  --min-free-memory-percent 15 --resume-free-memory-percent 20 \
  --terminate-free-memory-percent 10 --resume-samples 5 --interval-seconds 1 \
  --monitor-error-limit 3 --max-prelaunch-wait-seconds 300 \
  --max-runtime-seconds 2700 --terminate-grace-seconds 15 \
  --log "$CONTROL/gpu_guard.jsonl" -- \
  "$PYTHON" "$CONTROL/benchmark_policy_residency.py" \
  --attempt-dir "$RUN" --model-python "$PYTHON" --server "$CONTROL/serve_pure_lora_policy.py" \
  --openpi-root "$POLICY_ROOT" \
  --base-params /home/wengzr/.cache/openpi/openpi-assets/checkpoints/pi0_base/params \
  --base-manifest "$SNAPSHOT/manifests/pi0_pure_lora/base_model_manifest_c0.json" \
  --golden "$SNAPSHOT/manifests/pi0_pure_lora/golden_adapter_paths.json" \
  --norm-stats "$SNAPSHOT/artifacts/pi0-pure-lora/evidence/n1b-r3/attempt-20260902T-N1B-R3-nZfeRdmk/canonical/norm_stats.json" \
  --config-patch-sha256 "$(cat "$SNAPSHOT/manifests/pi0_pure_lora/c0_runtime_identity.sha256")" \
  --adapter "$ADAPTER" --model-manifest "$MODEL_MANIFEST" --rng-seed 0 \
  --warmups 10 --samples 50 --host-port 18031 --device-port 18032 --server-wait-seconds 180
