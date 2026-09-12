#!/usr/bin/env bash
# Immutable E1 retry launcher: only the already-authorized 7 x 40 dev run.
set -euo pipefail

readonly EVIDENCE=/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora/evidence/e1/attempt-20260911T-E1-DEV-9fK2mQ
readonly EVIDENCE_ROOT=/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora/evidence
readonly CONTROL="$EVIDENCE/control-source-runtime-r3"
readonly SNAPSHOT=/home/wengzr/projects/openpi-eval-tools/pi0-pure-lora/tool-snapshots/ft4-control-progress-heartbeat-20260911
readonly RUN_PARENT=/home/wengzr/projects/openpi-lora-runs/evaluations
readonly RUN="$RUN_PARENT/attempt-20260912T-E1-DEV-R3-6nQ4vP"
readonly PYTHON=/home/wengzr/projects/openpi/.venv/bin/python
readonly POLICY_ROOT=/home/wengzr/projects/openpi-worktrees/pi0-libero-pure-lora
readonly LIBERO_ROOT=/home/wengzr/projects/openpi
readonly CANONICAL="$SNAPSHOT"

test ! -e "$RUN"
mkdir -p "$RUN_PARENT"
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$CONTROL/pycache"
export PYTHONPATH="$CONTROL:${LIBERO_ROOT}/third_party/libero:${LIBERO_ROOT}/packages/openpi-client/src"

exec "$PYTHON" "$SNAPSHOT/tools/gpu_utilization_guard.py" \
  --physical-gpu 1 --pause-at 95 --resume-at 85 \
  --min-free-memory-percent 15 --resume-free-memory-percent 20 \
  --terminate-free-memory-percent 10 --resume-samples 5 \
  --interval-seconds 1 --monitor-error-limit 3 \
  --max-prelaunch-wait-seconds 300 --max-runtime-seconds 86400 \
  --terminate-grace-seconds 15 --kill-grace-seconds 5 \
  --log "$RUN_PARENT/attempt-20260912T-E1-DEV-R3-6nQ4vP.gpu_guard.jsonl" -- \
  "$PYTHON" "$CONTROL/run_e1_development.py" \
  --authorization authorized-e1-dev-280 --attempt-dir "$RUN" \
  --registration "$EVIDENCE_ROOT/e-prep/attempt-20260911T-EPREP-A1/registration.json" \
  --candidate-index "$EVIDENCE_ROOT/e-prep/attempt-20260911T-EPREP-A1/candidate_index.json" \
  --e1-plan "$EVIDENCE_ROOT/e-prep/attempt-20260911T-EPREP-A1/e1_dev40_plan.json" \
  --readiness "$EVIDENCE_ROOT/e-prep/attempt-20260911T-EPREP-A1/readiness_index_repaired.json" \
  --openpi-root "$POLICY_ROOT" --libero-openpi-root "$LIBERO_ROOT" \
  --model-python "$PYTHON" --libero-python "$LIBERO_ROOT/examples/libero/.venv/bin/python" \
  --evaluator "$CONTROL/eval_libero_pure_lora_bounded.py" --server "$CONTROL/serve_pure_lora_policy.py" \
  --base-params /home/wengzr/.cache/openpi/openpi-assets/checkpoints/pi0_base/params \
  --base-manifest "$CANONICAL/manifests/pi0_pure_lora/base_model_manifest_c0.json" \
  --golden "$CANONICAL/manifests/pi0_pure_lora/golden_adapter_paths.json" \
  --norm-stats "$CANONICAL/artifacts/pi0-pure-lora/evidence/n1b-r3/attempt-20260902T-N1B-R3-nZfeRdmk/canonical/norm_stats.json" \
  --config-identity "$CANONICAL/manifests/pi0_pure_lora/c0_runtime_identity.sha256" \
  --e0-manifest "$CANONICAL/manifests/pi0_pure_lora/e0_task_state_manifest.json" \
  --adapter-root /home/wengzr/projects/openpi-lora-runs/adapters/pi0-libero-pure-lora-ft0-seed42-v1 \
  --libero-config /home/wengzr/projects/openpi-eval-tools/config/libero \
  --egl-vendor-file "$CANONICAL/config/10_nvidia.json" \
  --expected-openpi-commit 3619c35ffdcbfe97ae735de175d91c2fb67a899d \
  --expected-libero-openpi-commit 15a9616a00943ada6c20a0f158e3adb39df2ccac \
  --expected-libero-commit f78abd68ee283de9f9be3c8f7e2a9ad60246e95c \
  --physical-gpu 1 --port 18001 --max-task-output-bytes 178571 --max-stage-output-bytes 50000000
