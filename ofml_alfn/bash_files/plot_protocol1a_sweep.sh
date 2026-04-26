#!/usr/bin/env bash
set -euo pipefail

ROOT_BASE="outputs/protocol1a_full/pred_dkl"
OUTPUT_BASE="outputs/protocol1a_sweep_plots/pred_dkl"
TMP_BASE="outputs/protocol1a_plot_temp/pred_dkl"
METRIC="target_test_loss"

NOISE_SETS=(
  "1.0 0.5 0.0"
  "2.0 1.0 0.0"
  "0.5 0.25 0.0"
)

SCALE_SETS=(
  "1.0 1.0 1.0"
  "0.5 1.0 2.0"
  "2.0 1.5 1.0"
)

make_tag() {
  local triplet="$1"
  echo "$triplet" | tr ' ' '_' | tr '.' 'p'
}

mkdir -p "$OUTPUT_BASE"
mkdir -p "$TMP_BASE"

for noise_triplet in "${NOISE_SETS[@]}"; do
  noise_tag="$(make_tag "$noise_triplet")"

  for scale_triplet in "${SCALE_SETS[@]}"; do
    scale_tag="$(make_tag "$scale_triplet")"

    filtered_root="${TMP_BASE}/noise_${noise_tag}__scale_${scale_tag}"
    filtered_output="${OUTPUT_BASE}/noise_${noise_tag}__scale_${scale_tag}"

    rm -rf "$filtered_root"
    mkdir -p "$filtered_root"
    mkdir -p "$filtered_output"

    found_any=0

    shopt -s nullglob
    for combo_dir in "${ROOT_BASE}"/costs_*__sims_*__noise_"${noise_tag}"__scale_"${scale_tag}"; do
      ln -sfn "$(cd "$combo_dir" && pwd)" "${filtered_root}/$(basename "$combo_dir")"
      found_any=1
    done
    shopt -u nullglob

    if [[ "$found_any" -eq 0 ]]; then
      echo "No matching result directories found for noise=${noise_triplet}, scale=${scale_triplet}"
      continue
    fi

    echo
    echo "=============================================================="
    echo "Plotting DKL results"
    echo "noise           = ${noise_triplet}"
    echo "scale           = ${scale_triplet}"
    echo "filtered_root   = ${filtered_root}"
    echo "filtered_output = ${filtered_output}"
    echo "=============================================================="

    python -m ofml_alfn.analysis.plot_protocol1a_sweep \
      --root_dir "${filtered_root}" \
      --output_dir "${filtered_output}" \
      --metric "${METRIC}" \
      --title_prefix "Protocol 1A DKL noise=[${noise_triplet}] scale=[${scale_triplet}]"
  done
done