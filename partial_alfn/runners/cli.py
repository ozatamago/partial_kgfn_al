#!/usr/bin/env python3

import argparse


def parse():
    parser = argparse.ArgumentParser(
        description="Run one replication of an AL experiment."
    )

    parser.add_argument("--trial", "-t", type=int, default=0)
    parser.add_argument(
        "--algo",
        "-a",
        type=str,
        default="NN_UQ",
        choices=["Random", "NN_UQ"],
    )
    parser.add_argument("--costs", "-c", type=str, required=True)
    parser.add_argument("--budget", "-b", type=int, default=200)
    parser.add_argument(
        "--noisy",
        action="store_true",
        help="Use the noisy problem variant.",
    )

    # predictor backend
    parser.add_argument(
        "--predictor_type",
        type=str,
        default="mcd",
        choices=["mcd", "dkl"],
    )

    # common model size
    parser.add_argument("--hidden", type=int, default=256)

    # MCD-specific
    parser.add_argument("--p_drop", type=float, default=0.1)
    parser.add_argument("--mc_samples", type=int, default=30)

    # DKL-specific
    parser.add_argument(
        "--dkl_inference",
        type=str,
        default="exact",
        choices=["exact", "svdkl"],
    )
    parser.add_argument("--dkl_feature_dim", type=int, default=32)
    parser.add_argument(
        "--dkl_kernel",
        type=str,
        default="rbf",
        choices=["rbf", "matern"],
    )
    parser.add_argument("--n_posterior_samples", type=int, default=64)

    # runner / selector mode
    parser.add_argument(
        "--sink_only",
        action="store_true",
        help=(
            "Run in sink-only mode. In the current freesolv3_runner design, "
            "this maps to enable_partial_queries=False."
        ),
    )
    parser.add_argument(
        "--sink_selector_objective",
        type=str,
        default="uncertainty",
        choices=["uncertainty", "fantasy_gain"],
        help=(
            "Objective used when --sink_only is enabled. "
            "Note: the current freesolv3_runner implementation only supports "
            "'uncertainty' in sink-only mode unless select_next_query.py is "
            "extended further."
        ),
    )

    return parser.parse_args()