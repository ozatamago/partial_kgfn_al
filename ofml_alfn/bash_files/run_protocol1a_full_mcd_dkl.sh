#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Problem 1_A full sweep (valid combinations only)
#
# Predictor types:
#   - mcd
#   - dkl
#
# Family 1:
#   - scratch_then_sequential_adapt
#   - pretrain_then_sequential_adapt
#
# Family 2:
#   - fantasy_al
#
# Policies:
#   mcd -> random, local_uncertainty, fantasy
#   dkl -> random, fantasy
#
# Sweep dimensions:
#   1. predictor_type
#   2. protocol_costs
#   3. similarities_to_target
# ============================================================

# ----------------------------
# User editable candidate sets
# ----------------------------
PREDICTOR_TYPES=(
#   "mcd"
  "dkl"
)

COST_SETS=(
  "1 1 3"
  "1 1 5"
  "1 1 9"
)

SIMILARITY_SETS=(
  "0.4 0.7 1.0"
  "0.5 0.7 1.0"
  "0.8 0.9 1.0"
)

# ----------------------------
# Global experiment settings
# ----------------------------
TRIAL=0
DEVICE="cpu"

N_PRETRAIN_P1=128
N_PRETRAIN_P2=128
N_ADAPT_P3=32
N_VAL_P3=128
N_TEST_P3=256

N_INIT_P1=8
N_INIT_P2=8
N_INIT_P3=4

TARGET_ADAPT_BUDGET=30
FAMILY2_BUDGET=40

OUTER_TRAIN_STEPS_MCD=500
OUTER_TRAIN_STEPS_DKL=300

FANTASY_MC_SAMPLES=8
FANTASY_TRAIN_STEPS=20

DKL_HIDDEN=128
DKL_FEATURE_DIM=32
DKL_KERNEL="rbf"

BASE_OUTPUT_DIR="outputs/protocol1a_full"

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
log() {
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

make_tag() {
  local triplet="$1"
  echo "$triplet" | tr ' ' '_' | tr '.' 'p'
}

outer_steps_for_predictor() {
  local predictor="$1"
  if [[ "$predictor" == "mcd" ]]; then
    echo "$OUTER_TRAIN_STEPS_MCD"
  elif [[ "$predictor" == "dkl" ]]; then
    echo "$OUTER_TRAIN_STEPS_DKL"
  else
    echo "Unknown predictor_type: $predictor" >&2
    exit 1
  fi
}

run_protocol1a_cmd() {
  local predictor="$1"
  local mode="$2"
  local policy="$3"
  local costs="$4"
  local sims="$5"
  local output_dir="$6"

  local outer_steps
  outer_steps="$(outer_steps_for_predictor "$predictor")"

  local cmd=(
    python -m ofml_alfn.experiments.run_protocol1a_fantasy
    --experiment_mode "$mode"
    --target_acquisition_policy "$policy"
    --protocol_costs ${costs}
    --similarities_to_target ${sims}
    --n_pretrain_p1 "$N_PRETRAIN_P1"
    --n_pretrain_p2 "$N_PRETRAIN_P2"
    --n_adapt_p3 "$N_ADAPT_P3"
    --n_val_p3 "$N_VAL_P3"
    --n_test_p3 "$N_TEST_P3"
    --n_init_p1 "$N_INIT_P1"
    --n_init_p2 "$N_INIT_P2"
    --n_init_p3 "$N_INIT_P3"
    --predictor_type "$predictor"
    --outer_train_steps "$outer_steps"
    --trial "$TRIAL"
    --device "$DEVICE"
    --output_dir "$output_dir"
    --save_json
  )

  if [[ "$predictor" == "dkl" ]]; then
    cmd+=(
      --dkl_hidden "$DKL_HIDDEN"
      --dkl_feature_dim "$DKL_FEATURE_DIM"
      --dkl_kernel "$DKL_KERNEL"
    )
  fi

  if [[ "$mode" == "fantasy_al" ]]; then
    cmd+=(--budget "$FAMILY2_BUDGET")
  else
    cmd+=(--target_adapt_budget "$TARGET_ADAPT_BUDGET")
  fi

  if [[ "$policy" == "fantasy" ]]; then
    cmd+=(
      --fantasy_mc_samples "$FANTASY_MC_SAMPLES"
      --fantasy_train_steps "$FANTASY_TRAIN_STEPS"
    )
  fi

  "${cmd[@]}"
}

# ------------------------------------------------------------
# Main 3-level loop
# ------------------------------------------------------------
mkdir -p "$BASE_OUTPUT_DIR"

for predictor in "${PREDICTOR_TYPES[@]}"; do
  if [[ "$predictor" == "mcd" ]]; then
    POLICIES=("random" "local_uncertainty" "fantasy")
  elif [[ "$predictor" == "dkl" ]]; then
    POLICIES=("random" "fantasy")
  else
    echo "Unknown predictor_type: $predictor" >&2
    exit 1
  fi

  for costs in "${COST_SETS[@]}"; do
    for sims in "${SIMILARITY_SETS[@]}"; do
      cost_tag="$(make_tag "$costs")"
      sim_tag="$(make_tag "$sims")"

      combo_output_dir="${BASE_OUTPUT_DIR}/pred_${predictor}/costs_${cost_tag}__sims_${sim_tag}"
      mkdir -p "$combo_output_dir"

      log "Start combination"
      log "predictor           = ${predictor}"
      log "policies            = ${POLICIES[*]}"
      log "protocol_costs      = ${costs}"
      log "similarities_target = ${sims}"
      log "output_dir          = ${combo_output_dir}"

      # ========================================================
      # Family 1
      # ========================================================
      for policy in "${POLICIES[@]}"; do
        log "Family 1 | ${policy} | scratch_then_sequential_adapt"
        run_protocol1a_cmd \
          "$predictor" \
          "scratch_then_sequential_adapt" \
          "$policy" \
          "$costs" \
          "$sims" \
          "$combo_output_dir"

        log "Family 1 | ${policy} | pretrain_then_sequential_adapt"
        run_protocol1a_cmd \
          "$predictor" \
          "pretrain_then_sequential_adapt" \
          "$policy" \
          "$costs" \
          "$sims" \
          "$combo_output_dir"
      done

      # ========================================================
      # Family 2
      # ========================================================
      for policy in "${POLICIES[@]}"; do
        log "Family 2 | ${policy} | fantasy_al"
        run_protocol1a_cmd \
          "$predictor" \
          "fantasy_al" \
          "$policy" \
          "$costs" \
          "$sims" \
          "$combo_output_dir"
      done

      log "Finished combination"
    done
  done
done

log "All valid full experiments finished."