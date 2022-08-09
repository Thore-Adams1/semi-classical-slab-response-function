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
REQUIRED_CONSTANTS = ["L", "tau"]


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
        w_neg = np.sqrt((1-e_neg_kl) / 2)
        w_pos = np.sqrt((1+e_neg_kl) / 2)
        w_neg_bounds = [w_neg - (2 / params["tau"]), w_neg + (2 / params["tau"])]
        w_pos_bounds = [w_pos - (2 / params["tau"]), w_pos + (2 / params["tau"])]

        w_lower, w_upper = 0, 1
        w_neg_k_window = ["-v", "w={}:{}:{}".format(*w_neg_bounds, args.omega_steps)]
        w_pos_k_window = ["-v", "w={}:{}:{}".format(*w_pos_bounds, args.omega_steps)]

        for k_window in [w_neg_k_window, w_pos_k_window]:
            print(" ".join(
                ["thesis_code.py"] + [
                pipes.quote(s)
                for s in given_args
            ] + k_window
            ))
        print()




if __name__ == "__main__":
    main()