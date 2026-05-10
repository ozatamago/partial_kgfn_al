#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Problem 1_A Ellipsoid | Family 3 only
#
# Family 3:
#   family3_candidate_pool_ablation
#
# Comparison:
#   all_protocols candidate pool
#     candidate_pool = pool_p1 + pool_p2 + pool_p3
#
#   target_only candidate pool
#     candidate_pool = pool_p3
#
# Policy:
#   fantasy only
#
# Sweep dimensions:
#   1. predictor_type
#   2. protocol_costs
#   3. noise setting
#   4. observer scale setting
#   5. candidate_pool_scope
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
  # "mcd"
  "dkl"
)

COST_SETS=(
  # "1 2 3"
  # "1 3 5"
  "1 3 9"
)

# Format:
#   "source_noise_p1 source_noise_p2 target_noise_p3"
NOISE_SETS=(
  "0.0 0.0 0.0"
  # "1.0 0.5 0.0"
  # "2.0 1.0 0.0"
  # "0.5 0.25 0.0"
)

# Format:
#   "scale_p1 scale_p2 scale_p3"
SCALE_SETS=(
  "1.0 1.0 1.0"
  # "0.5 1.0 2.0"
  # "2.0 1.5 1.0"
)

FAMILY3_POOL_SCOPES=(
  "all_protocols"
  "target_only"
)

FAMILY3_POLICIES=(
  "fantasy"
)

# ----------------------------
# Global experiment settings
# ----------------------------
TRIAL=0
DEVICE="cpu"

INPUT_DIM=3
LATENT_DIM=4
COMMON_OUTPUT_DIM=3

CHART_RADIUS=0.90
ELLIPSOID_CENTER=(0.0 0.0 0.0 0.0)
ELLIPSOID_SCALES=(1.0 1.2 0.8 1.5)
CHART_SIGN=1

N_PRETRAIN_P1=128
N_PRETRAIN_P2=128
N_ADAPT_P3=32
N_VAL_P3=128
N_TEST_P3=256

N_INIT_P1=8
N_INIT_P2=8
N_INIT_P3=8

TARGET_ADAPT_BUDGET=30
FAMILY3_BUDGET=100

OUTER_TRAIN_STEPS_MCD=500
OUTER_TRAIN_STEPS_DKL=300

FANTASY_MC_SAMPLES=8
FANTASY_TRAIN_STEPS=20

DKL_HIDDEN=128
DKL_FEATURE_DIM=32
DKL_KERNEL="rbf"

BASE_OUTPUT_DIR="outputs/protocol1a_ellipsoid_family3_only"

# ----------------------------
# Debug settings
# ----------------------------
DEBUG_PROGRESS=1
DEBUG_TOP_K_CANDIDATES=5
FANTASY_VERBOSE=0

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

run_protocol1a_ellipsoid_family3_cmd() {
  local predictor="$1"
  local policy="$2"

  local cost_p1="$3"
  local cost_p2="$4"
  local cost_p3="$5"

  local src_noise_p1="$6"
  local src_noise_p2="$7"
  local tgt_noise_p3="$8"

  local scale_p1="$9"
  local scale_p2="${10}"
  local scale_p3="${11}"

  local candidate_pool_scope="${12}"
  local output_dir="${13}"

  local outer_steps
  outer_steps="$(outer_steps_for_predictor "$predictor")"

  local cmd=(
    python -m ofml_alfn.experiments.run_protocol1a_ellipsoid_fantasy
    --experiment_mode "family3_candidate_pool_ablation"
    --target_acquisition_policy "$policy"
    --candidate_pool_scope "$candidate_pool_scope"

    --protocol_costs "$cost_p1" "$cost_p2" "$cost_p3"
    --target_noise_std "$tgt_noise_p3"
    --source_noise_stds "$src_noise_p1" "$src_noise_p2"
    --observer_scales "$scale_p1" "$scale_p2" "$scale_p3"

    --input_dim "$INPUT_DIM"
    --latent_dim "$LATENT_DIM"
    --common_output_dim "$COMMON_OUTPUT_DIM"

    --chart_radius "$CHART_RADIUS"
    --ellipsoid_center "${ELLIPSOID_CENTER[@]}"
    --ellipsoid_scales "${ELLIPSOID_SCALES[@]}"
    --chart_sign "$CHART_SIGN"

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

    --budget "$FAMILY3_BUDGET"
    --target_adapt_budget "$TARGET_ADAPT_BUDGET"
    --fantasy_mc_samples "$FANTASY_MC_SAMPLES"
    --fantasy_train_steps "$FANTASY_TRAIN_STEPS"

    --save_json
  )

  if [[ "${DEBUG_PROGRESS}" == "1" ]]; then
    cmd+=(
      --debug_progress
      --debug_top_k_candidates "$DEBUG_TOP_K_CANDIDATES"
    )
  fi

  if [[ "${FANTASY_VERBOSE}" == "1" ]]; then
    cmd+=(
      --fantasy_verbose
    )
  fi

  if [[ "$predictor" == "dkl" ]]; then
    cmd+=(
      --dkl_hidden "$DKL_HIDDEN"
      --dkl_feature_dim "$DKL_FEATURE_DIM"
      --dkl_kernel "$DKL_KERNEL"
    )
  fi

  log "Command:"
  printf ' %q' "${cmd[@]}"
  echo

  "${cmd[@]}"
}

# ------------------------------------------------------------
# Main loop: Family 3 only, fantasy only
# ------------------------------------------------------------
mkdir -p "$BASE_OUTPUT_DIR"

for predictor in "${PREDICTOR_TYPES[@]}"; do
  for cost_triplet in "${COST_SETS[@]}"; do
    read -r COST_P1 COST_P2 COST_P3 <<< "$cost_triplet"

    for noise_triplet in "${NOISE_SETS[@]}"; do
      read -r SOURCE_NOISE_STD_P1 SOURCE_NOISE_STD_P2 TARGET_NOISE_STD <<< "$noise_triplet"

      for scale_triplet in "${SCALE_SETS[@]}"; do
        read -r SCALE_P1 SCALE_P2 SCALE_P3 <<< "$scale_triplet"

        cost_tag="$(make_tag "$cost_triplet")"
        noise_tag="$(make_tag "$noise_triplet")"
        scale_tag="$(make_tag "$scale_triplet")"

        combo_output_dir="${BASE_OUTPUT_DIR}/pred_${predictor}/costs_${cost_tag}__noise_${noise_tag}__scale_${scale_tag}"
        mkdir -p "$combo_output_dir"

        log "Start Family 3 combination"
        log "predictor           = ${predictor}"
        log "policies            = ${FAMILY3_POLICIES[*]}"
        log "protocol_costs      = ${COST_P1} ${COST_P2} ${COST_P3}"
        log "source_noise_stds   = ${SOURCE_NOISE_STD_P1} ${SOURCE_NOISE_STD_P2}"
        log "target_noise_std    = ${TARGET_NOISE_STD}"
        log "observer_scales     = ${SCALE_P1} ${SCALE_P2} ${SCALE_P3}"
        log "candidate_pools     = ${FAMILY3_POOL_SCOPES[*]}"
        log "output_dir          = ${combo_output_dir}"
        log "debug_progress      = ${DEBUG_PROGRESS}"
        log "debug_top_k         = ${DEBUG_TOP_K_CANDIDATES}"
        log "fantasy_verbose     = ${FANTASY_VERBOSE}"

        for policy in "${FAMILY3_POLICIES[@]}"; do
          for pool_scope in "${FAMILY3_POOL_SCOPES[@]}"; do
            log "Family 3 | ${policy} | candidate_pool=${pool_scope}"

            run_protocol1a_ellipsoid_family3_cmd \
              "$predictor" \
              "$policy" \
              "$COST_P1" \
              "$COST_P2" \
              "$COST_P3" \
              "$SOURCE_NOISE_STD_P1" \
              "$SOURCE_NOISE_STD_P2" \
              "$TARGET_NOISE_STD" \
              "$SCALE_P1" \
              "$SCALE_P2" \
              "$SCALE_P3" \
              "$pool_scope" \
              "$combo_output_dir"
          done
        done

        log "Finished Family 3 combination"
      done
    done
  done
done

log "All Family 3 fantasy-only ellipsoid experiments finished."