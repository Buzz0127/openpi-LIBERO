#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PHYSICAL_GPU CONTROL_DIR" >&2
  exit 2
fi

physical_gpu=$1
control_dir=$2
model_python="$HOME/projects/openpi/.venv/bin/python"
libero_python="$HOME/projects/openpi/examples/libero/.venv/bin/python"
egl_vendor="$HOME/tmp/openpi-setup/egl-vendor/10_nvidia.json"

for _ in {1..1200}; do
  if [[ -f "$control_dir/workload.start" ]]; then
    break
  fi
  sleep 0.1
done
if [[ ! -f "$control_dir/workload.start" ]]; then
  echo "mapping start gate timeout" >&2
  exit 12
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="$physical_gpu"
export __EGL_VENDOR_LIBRARY_FILENAMES="$egl_vendor"

"$model_python" -c \
  'import json, jax; d=jax.devices(); print(json.dumps({"backend":jax.default_backend(),"count":len(d),"devices":[str(x) for x in d]}))' \
  > "$control_dir/jax_mapping.json"
"$model_python" -c \
  'import json, torch; print(json.dumps({"available":torch.cuda.is_available(),"count":torch.cuda.device_count(),"device0":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))' \
  > "$control_dir/torch_mapping.json" 2> "$control_dir/torch_mapping.stderr"
"$libero_python" -c \
  'import json, mujoco; c=mujoco.GLContext(32,32); c.make_current(); print(json.dumps({"mujoco_egl_context":"ok"})); c.free()' \
  > "$control_dir/egl_mapping.json" 2> "$control_dir/egl_mapping.stderr"

{
  echo "physical_gpu=$physical_gpu"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
  echo "XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE"
  echo "mapping_pid=$$"
  echo "mapping_pgid=$(ps -o pgid= -p $$ | tr -d ' ')"
} > "$control_dir/mapping_identity.txt"
