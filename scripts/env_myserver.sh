#!/usr/bin/env bash
# Source this file; do not execute it in a child shell.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate future-repro
# CANN 9.1 beta in the conda environment is ABI-incompatible with torch_npu 2.9
# and cannot resolve Conv2D. Remove every beta path before loading CANN 8.5.
_strip_cann_91_paths() {
  local variable value clean="" entry
  for variable in PATH PYTHONPATH LD_LIBRARY_PATH CMAKE_PREFIX_PATH; do
    value="${!variable:-}"
    while IFS= read -r entry; do
      [[ -z "$entry" || "$entry" == *"cann-9.1"* || "$entry" == *"9.1.0-beta"* ]] && continue
      clean="${clean:+$clean:}$entry"
    done < <(printf '%s' "$value" | tr ':' '\n')
    printf -v "$variable" '%s' "$clean"
    export "$variable"
    clean=""
  done
}
_strip_cann_91_paths
unset -f _strip_cann_91_paths
source "$HOME/miniconda3/Ascend/cann-8.5.0/set_env.sh"
export ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID:-0}"

# Optional application-local SpikingJelly-NPU snapshot. The authoritative
# source is maintained separately; record the synced commit before formal use.
# Prefer an installed package, then the stable external-state checkout. Never
# depend on short-lived staging/test directories for a formal queue.
if ! python -c 'import spikingjelly_npu' >/dev/null 2>&1; then
  if [[ -d "$HOME/FedSNN_state/deps/spikingjelly_npu/src" ]]; then
    export PYTHONPATH="$HOME/FedSNN_state/deps/spikingjelly_npu/src${PYTHONPATH:+:$PYTHONPATH}"
  fi
fi

