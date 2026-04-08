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

    return parser.parse_args()