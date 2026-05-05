#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# partial_alfn / FreeSolv3 sweep runner
#
# Sweep dimensions:
#   1. predictor_type   : mcd, dkl
#   2. mode             : partial, sink_only
#   3. cost setting     : 1_1, 1_3, 1_5, 1_9, 1_49
#   4. trial
#
# Notes:
#   - partial mode: auxiliary observations allowed
#   - sink_only mode: target observation only
#   - current freesolv3_runner implementation effectively supports
#     sink_selector_objective=uncertainty in sink-only mode
# ============================================================

# ------------------------------------------------------------
# macOS / conda OpenMP workaround
# ------------------------------------------------------------
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ----------------------------
# User editable candidate sets
# ----------------------------
PREDICTOR_TYPES=(
  "mcd"
  "dkl"
)

MODES=(
  "partial"
  "sink_only"
)

COST_SETS=(
  "1_1"
  "1_3"
  "1_5"
  "1_9"
  "1_49"
)

TRIALS=(
  0
)

# ----------------------------
# Global experiment settings
# ----------------------------
ALGO="NN_UQ"
BUDGET=10
NOISY=0

# MCD options
MCD_HIDDEN=256
MCD_P_DROP=0.1
MCD_MC_SAMPLES=32

# DKL options
DKL_FEATURE_DIM=32
DKL_KERNEL="rbf"
DKL_INFERENCE="exact"
N_POSTERIOR_SAMPLES=64

# sink-only selector
# NOTE: according to --help, current freesolv3_runner effectively supports
# only "uncertainty" in sink-only mode unless select_next_query.py is extended
SINK_SELECTOR_OBJECTIVE="uncertainty"

BASE_OUTPUT_DIR="outputs/partial_alfn_freesolv3_sweep"

# ----------------------------
# Utility functions
# ----------------------------
log() {
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_freesolv3_cmd() {
  local predictor="$1"
  local mode="$2"
  local costs="$3"
  local trial="$4"

  local out_dir="$5"

  local cmd=(
    python -m partial_alfn.experiments.freesolv3_runner
    --trial "$trial"
    --algo "$ALGO"
    --costs "$costs"
    --budget "$BUDGET"
    --predictor_type "$predictor"
  )

  if [[ "$NOISY" == "1" ]]; then
    cmd+=( --noisy )
  fi

  if [[ "$predictor" == "mcd" ]]; then
    cmd+=(
      --hidden "$MCD_HIDDEN"
      --p_drop "$MCD_P_DROP"
      --mc_samples "$MCD_MC_SAMPLES"
    )
  elif [[ "$predictor" == "dkl" ]]; then
    cmd+=(
      --dkl_inference "$DKL_INFERENCE"
      --dkl_feature_dim "$DKL_FEATURE_DIM"
      --dkl_kernel "$DKL_KERNEL"
      --n_posterior_samples "$N_POSTERIOR_SAMPLES"
    )
  else
    echo "Unknown predictor_type: $predictor" >&2
    exit 1
  fi

  if [[ "$mode" == "sink_only" ]]; then
    cmd+=(
      --sink_only
      --sink_selector_objective "$SINK_SELECTOR_OBJECTIVE"
    )
  fi

  mkdir -p "$out_dir"

  log "Command:"
  printf ' %q' "${cmd[@]}"
  echo

  (
    "${cmd[@]}"
  )
}

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
mkdir -p "$BASE_OUTPUT_DIR"

for predictor in "${PREDICTOR_TYPES[@]}"; do
  for mode in "${MODES[@]}"; do
    for costs in "${COST_SETS[@]}"; do
      for trial in "${TRIALS[@]}"; do

        combo_output_dir="${BASE_OUTPUT_DIR}/pred_${predictor}/mode_${mode}/costs_${costs}/trial_${trial}"

        log "Start combination"
        log "predictor   = ${predictor}"
        log "mode        = ${mode}"
        log "costs       = ${costs}"
        log "trial       = ${trial}"
        log "budget      = ${BUDGET}"
        log "output_dir  = ${combo_output_dir}"

        run_freesolv3_cmd \
          "$predictor" \
          "$mode" \
          "$costs" \
          "$trial" \
          "$combo_output_dir"

        log "Finished combination"
      done
    done
  done
done

log "All partial_alfn FreeSolv3 experiments finished."
