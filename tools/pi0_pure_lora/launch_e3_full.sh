#!/usr/bin/env bash
# Authorized E3 only: one locked pure-LoRA adapter over the preregistered full-2000 set.
set -euo pipefail
: "${E3_PHYSICAL_GPU:?set from the fresh E3 preflight selected physical GPU}"
[[ "$E3_PHYSICAL_GPU" =~ ^[0-9]+$ ]]

readonly CONTROL=/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora/evidence/e3/attempt-20260913T-E3-RUN-R1
readonly SNAPSHOT=/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora/tool-snapshots/ft4-control-progress-heartbeat-20260911
readonly RUN_PARENT=/home/wengzr/projects/openpi-lora-runs/evaluations
readonly RUN="$RUN_PARENT/attempt-20260913T-E3-FULL-2000-H3q7Ln"
readonly E1=/home/wengzr/projects/openpi-lora-runs/evaluations/attempt-20260912T-E1-DEV-R8-B1w9aV
readonly E2_AUDIT=/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora/evidence/e2-audit/attempt-20260913T-E2A-AUDIT-R1/comparison_summary.json
readonly E3_MANIFEST=/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora/evidence/e3/attempt-20260913T-E3-CPU-R1/full-manifest/e3_full_state_manifest.json
readonly PYTHON=/home/wengzr/projects/openpi/.venv/bin/python
readonly POLICY_ROOT=/home/wengzr/projects/openpi-worktrees/pi0-libero-pure-lora
readonly LIBERO_ROOT=/home/wengzr/projects/openpi

test ! -e "$RUN"
export CUDA_VISIBLE_DEVICES="$E3_PHYSICAL_GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$E3_PHYSICAL_GPU" PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="$SNAPSHOT/config/10_nvidia.json"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$CONTROL/pycache"
export PYTHONPATH="$CONTROL:${LIBERO_ROOT}/third_party/libero:${LIBERO_ROOT}/packages/openpi-client/src"

exec "$PYTHON" "$SNAPSHOT/tools/gpu_utilization_guard.py" \
  --physical-gpu "$E3_PHYSICAL_GPU" --pause-at 95 --resume-at 85 --min-free-memory-percent 15 \
  --resume-free-memory-percent 20 --terminate-free-memory-percent 10 \
  --resume-samples 5 --interval-seconds 1 --monitor-error-limit 3 \
  --max-prelaunch-wait-seconds 300 --max-runtime-seconds 432000 \
  --terminate-grace-seconds 15 --kill-grace-seconds 5 \
  --log "$RUN_PARENT/attempt-20260913T-E3-FULL-2000-H3q7Ln.gpu_guard.jsonl" -- \
  "$PYTHON" "$CONTROL/run_e3_full.py" \
  --authorization authorized-e3-full-2000 --attempt-dir "$RUN" \
  --e3-manifest "$E3_MANIFEST" --e2-audit-summary "$E2_AUDIT" \
  --selection-lock "$E1/selection_lock.json" \
  --openpi-root "$POLICY_ROOT" --libero-openpi-root "$LIBERO_ROOT" \
  --model-python "$PYTHON" --libero-python "$LIBERO_ROOT/examples/libero/.venv/bin/python" \
  --evaluator "$CONTROL/eval_libero_pure_lora_bounded.py" \
  --server "$CONTROL/serve_pure_lora_policy.py" \
  --base-params /home/wengzr/.cache/openpi/openpi-assets/checkpoints/pi0_base/params \
  --base-manifest "$SNAPSHOT/manifests/pi0_pure_lora/base_model_manifest_c0.json" \
  --selected-model-manifest "$E1/candidates/step-00025000/model_manifest.json" \
  --golden "$SNAPSHOT/manifests/pi0_pure_lora/golden_adapter_paths.json" \
  --norm-stats "$SNAPSHOT/artifacts/pi0-pure-lora/evidence/n1b-r3/attempt-20260902T-N1B-R3-nZfeRdmk/canonical/norm_stats.json" \
  --config-identity "$SNAPSHOT/manifests/pi0_pure_lora/c0_runtime_identity.sha256" \
  --adapter-root /home/wengzr/projects/openpi-lora-runs/adapters/pi0-libero-pure-lora-ft0-seed42-v1 \
  --libero-config /home/wengzr/projects/openpi-eval-tools/config/libero \
  --egl-vendor-file "$SNAPSHOT/config/10_nvidia.json" \
  --expected-policy-openpi-commit 3619c35ffdcbfe97ae735de175d91c2fb67a899d \
  --expected-libero-openpi-commit 15a9616a00943ada6c20a0f158e3adb39df2ccac \
  --expected-libero-commit f78abd68ee283de9f9be3c8f7e2a9ad60246e95c \
  --physical-gpu "$E3_PHYSICAL_GPU" --port 18001 --max-task-output-bytes 178571 --max-stage-output-bytes 250000000
