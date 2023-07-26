"""
Given args for the specified script, return a list of 2 of script commands for 
each value of Kx. (one for each the positive and negative branch 
of the plasma waves) Each command will have an omega window that is
centered corresponding the value of Kx, L & tau.
"""
import argparse
import sys
import pipes

import numpy as np

from . import calc as tc

# Globals
USAGE = """\
centered_plot.py [-h] SCRIPT [-ws OMEGA_STEPS] [-o OUTPUT] [scsr-calc args]
"""
REQUIRED_CONSTANTS = ["L", "tau"]


def get_parser():
    parser = argparse.ArgumentParser(usage=USAGE, description=__doc__)
    parser.add_argument(
        "script",
        choices=("scsr-calc", "scsr-build-chunked-jobscripts"),
        help="The script to generate commands for.",
    )
    parser.add_argument(
        "-ws",
        "--omega-steps",
        type=int,
        default=50,
        help="Number of omega values in output command. (Default: 50)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="centered/output_{w}_Kx_{Kx}.pkl",
        help="Output pkl paths pattern. Use {w} for w_plus / w_minus and {Kx} for the corresponding Kx value. (Default: 'centered/output_{w}_Kx_{Kx}.pkl')",
    )
    return parser


def main():
    given_args = sys.argv[1:]
    args, given_args = get_parser().parse_known_args(given_args)
    tc_parser = tc.get_parser()
    tc_args, remainder_args = tc_parser.parse_known_args(list(given_args))
    if args.script == "scsr-build-chunked-jobscripts":
        from .build_chunked_jobscripts import validate_args

        validate_args(tc_args)

    params, variable_params = tc.get_parameters(tc_args)

    if "w" in params or "w" in variable_params:
        raise RuntimeError("w should not be provided to this script.")
    for c in REQUIRED_CONSTANTS:
        if c not in params:
            raise RuntimeError("{} should be provided to this script.".format(c))
        if c in variable_params:
            raise RuntimeError("{} should be a constant parameter.".format(c))
    k_vals = []
    if "Kx" in params:
        k_vals = [params["Kx"]]

    if "Kx" in variable_params:
        k_vals = variable_params["Kx"]

    to_remove_indices = [i for i, p in enumerate(given_args) if p.startswith("Kx")]
    for i in reversed(to_remove_indices):
        del given_args[i]

    print()
    for Kx in k_vals:
        e_neg_kl = np.exp(-Kx * params["L"])
        w_neg = np.sqrt((1 + e_neg_kl) / 2)
        w_pos = np.sqrt((1 - e_neg_kl) / 2)
        w_neg_bounds = [w_neg - (2 / params["tau"]), w_neg + (2 / params["tau"])]
        w_pos_bounds = [w_pos - (2 / params["tau"]), w_pos + (2 / params["tau"])]

        w_neg_window = ["-v", "w={}:{}:{}".format(*w_neg_bounds, args.omega_steps)]
        w_pos_window = ["-v", "w={}:{}:{}".format(*w_pos_bounds, args.omega_steps)]

        for w_label, w_window in [("w_minus", w_neg_window), ("w_plus", w_pos_window)]:
            output_args = ["-o", args.output.format(w=w_label, Kx=Kx)]
            print(
                " ".join(
                    [args.script]
                    + [pipes.quote(s) for s in given_args]
                    + w_window
                    + output_args
                    + ["-p", "Kx={}".format(Kx)]
                )
            )
        print()


if __name__ == "__main__":
    main()
