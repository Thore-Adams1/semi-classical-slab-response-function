"""plot the results from the thing bla
"""
# Standard
import argparse
import pickle
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
from re import L

# Third party
import numpy as np
from tqdm import tqdm
import matplotlib.ticker as tck
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.linalg import inv, det

# Local
from scsr.plots import generate_plots, write_plots_from_plots_pickle
"""
run:
py plot.py --help 
for information
"""


def main(args):
    if args.plot_pickle:
        print("Loading results object for {}".format(args.plot_pickle))
        with open(args.plot_pickle, "rb") as f:
            plots = pickle.load(f)
        print("Loaded plots with axes: {} by {}".format(*plots["axes"]))
    else:
        print("Generating plots with axes: {} by {}".format(*args.axes))
        plots = generate_plots(args)

    figs_dir = args.figs_dir or os.path.join(
        "figs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)

    if not args.skip_plotting:
        write_plots_from_plots_pickle(plots, plots["variable_params"], figs_dir)


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-x", "--axes", nargs=2, help="Axes to plot. Default: w Kx", default=["w", "Kx"]
    )
    parser.add_argument(
        "-xl",
        "--axis-labels",
        nargs=2,
        help="Axes labels. Default: \omega K",
        default=["\omega", "K"],
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-r",
        "--result-pickles",
        nargs="+",
        help="Pickle files with results. If multiple are given, will attempt to combine",
        metavar="path",
    )
    group.add_argument(
        "-p", "--plot-pickle", help="Pickle files with plots.", metavar="path"
    )
    parser.add_argument(
        "-o",
        "--output-pkl",
        help="Where to save pickle files with plots. Defaults to a unique path in the current directory.",
        metavar="path",
    )
    parser.add_argument(
        "-f",
        "--figs-dir",
        help="Directory in which to store figures. Defaults to a timestamped \n"
        r"'./figs/[DATE]_[TIME]' directory",
    )
    parser.add_argument(
        "-k",
        "--skip-plotting",
        action="store_true",
        help=(
            "Skip plotting the figures. Useful if you just want to save the plots\n"
            "to a plot pickle."
        ),
    )
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
