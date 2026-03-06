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
    return parser.parse_args()