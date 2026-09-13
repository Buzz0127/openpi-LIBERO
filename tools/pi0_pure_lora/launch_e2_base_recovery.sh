#!/usr/bin/env bash
# Authorized E2 recovery only: replace the invalid Base main-200 half.
set -euo pipefail
readonly CONTROL=/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora/evidence/e2-base-repair/attempt-20260913T-E2B-RUN-R2
readonly SNAPSHOT=/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora/tool-snapshots/ft4-control-progress-heartbeat-20260911
readonly RUN_PARENT=/home/wengzr/projects/openpi-lora-runs/evaluations
readonly RUN="$RUN_PARENT/attempt-20260913T-E2-BASE-RECOVERY-200-Q2r5Lm"
readonly PRIOR_E2="$RUN_PARENT/attempt-20260912T-E2-MAIN-400-E4z8Pr"
readonly E1=/home/wengzr/projects/openpi-lora-runs/evaluations/attempt-20260912T-E1-DEV-R8-B1w9aV
readonly PYTHON=/home/wengzr/projects/openpi/.venv/bin/python
readonly POLICY_ROOT=/home/wengzr/projects/openpi-worktrees/pi0-libero-pure-lora
readonly LIBERO_ROOT=/home/wengzr/projects/openpi

test ! -e "$RUN"
export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="$SNAPSHOT/config/10_nvidia.json"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$CONTROL/pycache"
export PYTHONPATH="$CONTROL:${LIBERO_ROOT}/third_party/libero:${LIBERO_ROOT}/packages/openpi-client/src"

exec "$PYTHON" "$SNAPSHOT/tools/gpu_utilization_guard.py" \
  --physical-gpu 1 --pause-at 95 --resume-at 85 --min-free-memory-percent 15 \
  --resume-free-memory-percent 20 --terminate-free-memory-percent 10 \
  --resume-samples 5 --interval-seconds 1 --monitor-error-limit 3 \
  --max-prelaunch-wait-seconds 300 --max-runtime-seconds 43200 \
  --terminate-grace-seconds 15 --kill-grace-seconds 5 \
  --log "$RUN_PARENT/attempt-20260913T-E2-BASE-RECOVERY-200-Q2r5Lm.gpu_guard.jsonl" -- \
  "$PYTHON" "$CONTROL/run_e2_base_recovery.py" \
  --authorization authorized-e2-base-recovery-200 \
  --attempt-dir "$RUN" --prior-e2-attempt "$PRIOR_E2" \
  --selection-lock "$E1/selection_lock.json" \
  --e0-manifest "$SNAPSHOT/manifests/pi0_pure_lora/e0_task_state_manifest.json" \
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
  --libero-config /home/wengzr/projects/openpi-eval-tools/config/libero \
  --egl-vendor-file "$SNAPSHOT/config/10_nvidia.json" \
  --expected-policy-openpi-commit 3619c35ffdcbfe97ae735de175d91c2fb67a899d \
  --expected-libero-openpi-commit 15a9616a00943ada6c20a0f158e3adb39df2ccac \
  --expected-libero-commit f78abd68ee283de9f9be3c8f7e2a9ad60246e95c \
  --physical-gpu 1 --port 18001 --max-task-output-bytes 178571 --max-stage-output-bytes 50000000
