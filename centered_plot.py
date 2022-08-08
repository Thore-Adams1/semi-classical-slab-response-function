"""
- in: Kx, L, tau
- out: command with static k,l,tau and calculated omega by p command.
"""
import argparse
import sys
import pipes

import numpy as np

import thesis_code as tc

# Globals
USAGE = """\
centered_plot.py [-h] [thesis_code.py args]
"""
REQUIRED_CONSTANTS = ["Kx", "L", "tau"]


def main():
    parser = argparse.ArgumentParser(usage=USAGE)
    parser.add_argument("-o", "--omega-steps", type=int, default=50, help="Number of omega values in output command.")
    given_args = sys.argv[1:]
    args, given_args = parser.parse_known_args(given_args)
    tc_parser = tc.get_parser()
    tc_args = tc_parser.parse_args(list(given_args))
    params, variable_params = tc.get_parameters(tc_args)
    
    if "w" in params or "w" in variable_params:
        raise RuntimeError("w should not be provided to this script.")
    for c in REQUIRED_CONSTANTS:
        if c not in params:
            raise RuntimeError("{} should be provided to this script.".format(c))
        if c in variable_params:
            raise RuntimeError("{} should be a constant parameter.".format(c))
    
    w_lower, w_upper = 0, 1
    k_window = ["-v", f"w={w_lower}:{w_upper}:{args.omega_steps}"]

    print("\n"+" ".join(
        ["thesis_code.py"] + [
        pipes.quote(s)
        for s in given_args
    ] + k_window
    ))




if __name__ == "__main__":
    main()