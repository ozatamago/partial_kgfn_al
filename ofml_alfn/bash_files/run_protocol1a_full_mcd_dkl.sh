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
#   4. noise setting
#   5. observer scale setting
# ============================================================

# ----------------------------
# User editable candidate sets
# ----------------------------
PREDICTOR_TYPES=(
#   "mcd"
  "dkl"
)

COST_SETS=(
  # "1 2 3"
  # "1 3 5"
  "1 3 9"
)

SIMILARITY_SETS=(
#   "0.4 0.7 1.0"
#   "0.5 0.7 1.0"
#   "0.8 0.9 1.0"
  "1.0 1.0 1.0"
)

# Format:
#   "source_noise_p1 source_noise_p2 target_noise_p3"
NOISE_SETS=(
  "1.0 0.5 0.0"
  "2.0 1.0 0.0"
  "0.5 0.25 0.0"
)

# Format:
#   "scale_p1 scale_p2 scale_p3"
#
# Example interpretations:
#   "1.0 1.0 1.0" -> no scale shift
#   "0.5 1.0 2.0" -> Protocol 1 smaller, Protocol 2 baseline, Protocol 3 larger
#   "2.0 1.5 1.0" -> sources larger than target
SCALE_SETS=(
  "1.0 1.0 1.0"
  "0.5 1.0 2.0"
  "2.0 1.5 1.0"
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
  local src_noise_p1="$6"
  local src_noise_p2="$7"
  local tgt_noise_p3="$8"
  local scale_p1="$9"
  local scale_p2="${10}"
  local scale_p3="${11}"
  local output_dir="${12}"

  local outer_steps
  outer_steps="$(outer_steps_for_predictor "$predictor")"

  local cmd=(
    python -m ofml_alfn.experiments.run_protocol1a_fantasy
    --experiment_mode "$mode"
    --target_acquisition_policy "$policy"
    --protocol_costs ${costs}
    --similarities_to_target ${sims}
    --target_noise_std "$tgt_noise_p3"
    --source_noise_stds "$src_noise_p1" "$src_noise_p2"
    --observer_scales "$scale_p1" "$scale_p2" "$scale_p3"
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
# Main loop
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
      for noise_triplet in "${NOISE_SETS[@]}"; do
        read -r SOURCE_NOISE_STD_P1 SOURCE_NOISE_STD_P2 TARGET_NOISE_STD <<< "$noise_triplet"

        for scale_triplet in "${SCALE_SETS[@]}"; do
          read -r SCALE_P1 SCALE_P2 SCALE_P3 <<< "$scale_triplet"

          cost_tag="$(make_tag "$costs")"
          sim_tag="$(make_tag "$sims")"
          noise_tag="$(make_tag "$noise_triplet")"
          scale_tag="$(make_tag "$scale_triplet")"

          combo_output_dir="${BASE_OUTPUT_DIR}/pred_${predictor}/costs_${cost_tag}__sims_${sim_tag}__noise_${noise_tag}__scale_${scale_tag}"
          mkdir -p "$combo_output_dir"

          log "Start combination"
          log "predictor           = ${predictor}"
          log "policies            = ${POLICIES[*]}"
          log "protocol_costs      = ${costs}"
          log "similarities_target = ${sims}"
          log "source_noise_stds   = ${SOURCE_NOISE_STD_P1} ${SOURCE_NOISE_STD_P2}"
          log "target_noise_std    = ${TARGET_NOISE_STD}"
          log "observer_scales     = ${SCALE_P1} ${SCALE_P2} ${SCALE_P3}"
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
              "$SOURCE_NOISE_STD_P1" \
              "$SOURCE_NOISE_STD_P2" \
              "$TARGET_NOISE_STD" \
              "$SCALE_P1" \
              "$SCALE_P2" \
              "$SCALE_P3" \
              "$combo_output_dir"

            log "Family 1 | ${policy} | pretrain_then_sequential_adapt"
            run_protocol1a_cmd \
              "$predictor" \
              "pretrain_then_sequential_adapt" \
              "$policy" \
              "$costs" \
              "$sims" \
              "$SOURCE_NOISE_STD_P1" \
              "$SOURCE_NOISE_STD_P2" \
              "$TARGET_NOISE_STD" \
              "$SCALE_P1" \
              "$SCALE_P2" \
              "$SCALE_P3" \
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
              "$SOURCE_NOISE_STD_P1" \
              "$SOURCE_NOISE_STD_P2" \
              "$TARGET_NOISE_STD" \
              "$SCALE_P1" \
              "$SCALE_P2" \
              "$SCALE_P3" \
              "$combo_output_dir"
          done

          log "Finished combination"
        done
      done
    done
  done
done

log "All valid full experiments finished."