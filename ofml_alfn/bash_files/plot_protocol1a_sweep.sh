#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="outputs/protocol1a"
OUTPUT_DIR="outputs/protocol1a_sweep_plots"
METRIC="target_test_loss"

python -m ofml_alfn.analysis.plot_protocol1a_sweep \
  --root_dir "${ROOT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --metric "${METRIC}"