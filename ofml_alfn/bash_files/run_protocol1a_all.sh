#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Problem 1_A
# Family 1:
#   scratch_then_sequential_adapt vs pretrain_then_sequential_adapt
#   under fixed acquisition policy
#
# Family 2:
#   fantasy_al with random / local_uncertainty / fantasy
#
# This script sweeps over:
#   1. protocol_cost triplets
#   2. similarity_to_target triplets
# using a double loop.
# ============================================================

# ----------------------------
# User editable candidate sets
# ----------------------------
COST_SETS=(
  "1 1 3"
  "1 1 5"
  "1 1 9"
  "1 2 3"
  "1 2 5"
  "1 2 9"
  "1 3 5"
  "1 3 9"
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
PREDICTOR_TYPE="mcd"

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

OUTER_TRAIN_STEPS=500
FANTASY_MC_SAMPLES=8
FANTASY_TRAIN_STEPS=20

BASE_OUTPUT_DIR="outputs/protocol1a"

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

run_protocol1a_cmd() {
  local mode="$1"
  local policy="$2"
  local costs="$3"
  local sims="$4"
  local output_dir="$5"

  if [[ "$mode" == "fantasy_al" ]]; then
    if [[ "$policy" == "fantasy" ]]; then
      python -m ofml_alfn.experiments.run_protocol1a_fantasy \
        --experiment_mode "$mode" \
        --target_acquisition_policy "$policy" \
        --budget "$FAMILY2_BUDGET" \
        --protocol_costs ${costs} \
        --similarities_to_target ${sims} \
        --n_pretrain_p1 "$N_PRETRAIN_P1" \
        --n_pretrain_p2 "$N_PRETRAIN_P2" \
        --n_adapt_p3 "$N_ADAPT_P3" \
        --n_val_p3 "$N_VAL_P3" \
        --n_test_p3 "$N_TEST_P3" \
        --n_init_p1 "$N_INIT_P1" \
        --n_init_p2 "$N_INIT_P2" \
        --n_init_p3 "$N_INIT_P3" \
        --predictor_type "$PREDICTOR_TYPE" \
        --outer_train_steps "$OUTER_TRAIN_STEPS" \
        --fantasy_mc_samples "$FANTASY_MC_SAMPLES" \
        --fantasy_train_steps "$FANTASY_TRAIN_STEPS" \
        --trial "$TRIAL" \
        --device "$DEVICE" \
        --output_dir "$output_dir" \
        --save_json
    else
      python -m ofml_alfn.experiments.run_protocol1a_fantasy \
        --experiment_mode "$mode" \
        --target_acquisition_policy "$policy" \
        --budget "$FAMILY2_BUDGET" \
        --protocol_costs ${costs} \
        --similarities_to_target ${sims} \
        --n_pretrain_p1 "$N_PRETRAIN_P1" \
        --n_pretrain_p2 "$N_PRETRAIN_P2" \
        --n_adapt_p3 "$N_ADAPT_P3" \
        --n_val_p3 "$N_VAL_P3" \
        --n_test_p3 "$N_TEST_P3" \
        --n_init_p1 "$N_INIT_P1" \
        --n_init_p2 "$N_INIT_P2" \
        --n_init_p3 "$N_INIT_P3" \
        --predictor_type "$PREDICTOR_TYPE" \
        --outer_train_steps "$OUTER_TRAIN_STEPS" \
        --trial "$TRIAL" \
        --device "$DEVICE" \
        --output_dir "$output_dir" \
        --save_json
    fi

  else
    if [[ "$policy" == "fantasy" ]]; then
      python -m ofml_alfn.experiments.run_protocol1a_fantasy \
        --experiment_mode "$mode" \
        --target_acquisition_policy "$policy" \
        --target_adapt_budget "$TARGET_ADAPT_BUDGET" \
        --protocol_costs ${costs} \
        --similarities_to_target ${sims} \
        --n_pretrain_p1 "$N_PRETRAIN_P1" \
        --n_pretrain_p2 "$N_PRETRAIN_P2" \
        --n_adapt_p3 "$N_ADAPT_P3" \
        --n_val_p3 "$N_VAL_P3" \
        --n_test_p3 "$N_TEST_P3" \
        --n_init_p1 "$N_INIT_P1" \
        --n_init_p2 "$N_INIT_P2" \
        --n_init_p3 "$N_INIT_P3" \
        --predictor_type "$PREDICTOR_TYPE" \
        --fantasy_mc_samples "$FANTASY_MC_SAMPLES" \
        --fantasy_train_steps "$FANTASY_TRAIN_STEPS" \
        --trial "$TRIAL" \
        --device "$DEVICE" \
        --output_dir "$output_dir" \
        --save_json
    else
      python -m ofml_alfn.experiments.run_protocol1a_fantasy \
        --experiment_mode "$mode" \
        --target_acquisition_policy "$policy" \
        --target_adapt_budget "$TARGET_ADAPT_BUDGET" \
        --protocol_costs ${costs} \
        --similarities_to_target ${sims} \
        --n_pretrain_p1 "$N_PRETRAIN_P1" \
        --n_pretrain_p2 "$N_PRETRAIN_P2" \
        --n_adapt_p3 "$N_ADAPT_P3" \
        --n_val_p3 "$N_VAL_P3" \
        --n_test_p3 "$N_TEST_P3" \
        --n_init_p1 "$N_INIT_P1" \
        --n_init_p2 "$N_INIT_P2" \
        --n_init_p3 "$N_INIT_P3" \
        --predictor_type "$PREDICTOR_TYPE" \
        --trial "$TRIAL" \
        --device "$DEVICE" \
        --output_dir "$output_dir" \
        --save_json
    fi
  fi
}

# ------------------------------------------------------------
# Main double loop
# ------------------------------------------------------------
mkdir -p "$BASE_OUTPUT_DIR"

for costs in "${COST_SETS[@]}"; do
  for sims in "${SIMILARITY_SETS[@]}"; do
    cost_tag="$(make_tag "$costs")"
    sim_tag="$(make_tag "$sims")"

    combo_output_dir="${BASE_OUTPUT_DIR}/costs_${cost_tag}__sims_${sim_tag}"
    mkdir -p "$combo_output_dir"

    log "Start combination"
    log "protocol_costs      = ${costs}"
    log "similarities_target = ${sims}"
    log "output_dir          = ${combo_output_dir}"

    # ========================================================
    # Family 1
    # ========================================================

    log "Family 1 | random | scratch_then_sequential_adapt"
    run_protocol1a_cmd \
      "scratch_then_sequential_adapt" \
      "random" \
      "$costs" \
      "$sims" \
      "$combo_output_dir"

    log "Family 1 | random | pretrain_then_sequential_adapt"
    run_protocol1a_cmd \
      "pretrain_then_sequential_adapt" \
      "random" \
      "$costs" \
      "$sims" \
      "$combo_output_dir"

    log "Family 1 | local_uncertainty | scratch_then_sequential_adapt"
    run_protocol1a_cmd \
      "scratch_then_sequential_adapt" \
      "local_uncertainty" \
      "$costs" \
      "$sims" \
      "$combo_output_dir"

    log "Family 1 | local_uncertainty | pretrain_then_sequential_adapt"
    run_protocol1a_cmd \
      "pretrain_then_sequential_adapt" \
      "local_uncertainty" \
      "$costs" \
      "$sims" \
      "$combo_output_dir"

    log "Family 1 | fantasy | scratch_then_sequential_adapt"
    run_protocol1a_cmd \
      "scratch_then_sequential_adapt" \
      "fantasy" \
      "$costs" \
      "$sims" \
      "$combo_output_dir"

    log "Family 1 | fantasy | pretrain_then_sequential_adapt"
    run_protocol1a_cmd \
      "pretrain_then_sequential_adapt" \
      "fantasy" \
      "$costs" \
      "$sims" \
      "$combo_output_dir"

    # ========================================================
    # Family 2
    # ========================================================

    log "Family 2 | random | fantasy_al"
    run_protocol1a_cmd \
      "fantasy_al" \
      "random" \
      "$costs" \
      "$sims" \
      "$combo_output_dir"

    log "Family 2 | local_uncertainty | fantasy_al"
    run_protocol1a_cmd \
      "fantasy_al" \
      "local_uncertainty" \
      "$costs" \
      "$sims" \
      "$combo_output_dir"

    log "Family 2 | fantasy | fantasy_al"
    run_protocol1a_cmd \
      "fantasy_al" \
      "fantasy" \
      "$costs" \
      "$sims" \
      "$combo_output_dir"

    log "Finished combination: costs=${costs}, sims=${sims}"
  done
done

log "All experiments finished."