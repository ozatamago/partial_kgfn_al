# Bayesian Optimization of Function Networks with Partial Evaluations

## Software requirements

The entire codebase is written in Python. Package requirements are as follows:

- python=3.9
- botorch==0.8.4
- numpy==1.23.5
- gpytorch==1.10
- scipy==1.10.1
- pandas
- matplotlib
- jupyter

## Setup

Create the conda environment from the repository root:

```bash
conda env create -f pKGFN_env.yml
```

Activate the environment:

```bash
conda activate pKGFN
```

If your environment was created under a different name, replace `pKGFN` with the name shown by `conda env list`.

You can inspect the CLI for the two main runners with:

```bash
python -m partial_alfn.experiments.freesolv3_runner --help
python -m ofml_alfn.experiments.run_protocol1a_fantasy --help
```

## Running `partial_alfn`

### Single run: partial observation, DKL

```bash
python -m partial_alfn.experiments.freesolv3_runner \
  --trial 3 \
  --algo NN_UQ \
  --costs 1_3 \
  --budget 1000 \
  --predictor_type dkl
```

### Single run: target only, DKL

```bash
python -m partial_alfn.experiments.freesolv3_runner \
  --trial 3 \
  --algo NN_UQ \
  --costs 1_3 \
  --budget 1000 \
  --predictor_type dkl \
  --sink_only \
  --sink_selector_objective uncertainty
```

### Sweep run: target only / partial, DKL / MCD, multiple cost settings

If you saved the sweep script at `partial_alfn/scripts/run_partial_alfn_freesolv3_sweep.sh`, run:

```bash
chmod +x partial_alfn/scripts/run_partial_alfn_freesolv3_sweep.sh
./partial_alfn/scripts/run_partial_alfn_freesolv3_sweep.sh
```

## Plotting `partial_alfn` results

Use `compare_two_methods.py` to compare two saved `.pt` result files:

```bash
python compare_two_methods.py \
  --file_a results/freesolv3_1_3/NN_UQ/trial_3.pt \
  --file_b results_sink_only/freesolv3_1_3/NN_UQ_dkl_fantasy/trial_3.pt \
  --name_a dkl_partial \
  --name_b dkl_sink_only \
  --budget_points 50 100 200 500 1000 \
  --thresholds 5.0 4.5 4.0 3.5 3.0 \
  --output_png compare_dkl_1_3_1000_v3.png \
  --output_scatter_png compare_uncertainty_vs_gain.png
```

You can inspect the comparison script CLI with:

```bash
python compare_two_methods.py --help
```

## Running `ofml_alfn`

### Single run: all protocols, DKL, fantasy sampling

```bash
python -m ofml_alfn.experiments.run_protocol1a_fantasy \
  --experiment_mode family3_candidate_pool_ablation \
  --target_acquisition_policy fantasy \
  --candidate_pool_scope all_protocols \
  --predictor_type dkl \
  --protocol_costs 1 3 9 \
  --similarities_to_target 0.5 0.8 1.0 \
  --source_noise_stds 1.0 0.5 \
  --target_noise_std 0.0 \
  --observer_scales 0.5 1.0 2.0 \
  --budget 100 \
  --save_json
```

### Single run: target only, DKL, fantasy sampling

```bash
python -m ofml_alfn.experiments.run_protocol1a_fantasy \
  --experiment_mode family3_candidate_pool_ablation \
  --target_acquisition_policy fantasy \
  --candidate_pool_scope target_only \
  --predictor_type dkl \
  --protocol_costs 1 3 9 \
  --similarities_to_target 0.5 0.8 1.0 \
  --source_noise_stds 1.0 0.5 \
  --target_noise_std 0.0 \
  --observer_scales 0.5 1.0 2.0 \
  --budget 100 \
  --save_json
```

## Plotting `ofml_alfn` results

The repository includes a plotting script at:

```bash
ofml_alfn/bash_files/plot_protocol1a_sweep.sh
```

Run it from the repository root with:

```bash
chmod +x ofml_alfn/bash_files/plot_protocol1a_sweep.sh
./ofml_alfn/bash_files/plot_protocol1a_sweep.sh
```

By default, this script reads results from:

```bash
outputs/protocol1a_family3_only
```

and writes plots to:

```bash
outputs/protocol1a_sweep_plots_family3_only
```