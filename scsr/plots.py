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
import matplotlib.pyplot as plt
from matplotlib import cm

# Local
from .results import load_results, PickleType
from . import maths


# Globals
labels_by_param = {
    "Kx": "K_x",
    "lc": "lc",
    "L": "L",
    "P": "P",
    "w": "\\omega",
}


def write_plot(
    axes,
    axis_labels,
    iteration_params,
    array_2d,
    func_name,
    title,
    variable_params,
    figs_dir="figs",
):

    ax_v_array = np.array(variable_params[axes[0]])
    ax_h_array = np.array(variable_params[axes[1]])
    ax_v_vals = ax_v_array.T * np.ones([len(ax_h_array), len(ax_v_array)])
    ax_h_vals = np.ones([len(ax_v_array), len(ax_h_array)]) * ax_h_array.T
    params_str = "_".join(
        "{}={:g}".format(k, v)
        for k, v in iteration_params.items()
        if k in labels_by_param
    )
    fig_name = " ".join(
        "{}={:g}".format(labels_by_param[k], v)
        for k, v in iteration_params.items()
        if k in labels_by_param
    )
    fig, ax = plt.subplots()
    c = ax.pcolor(ax_h_vals.T, ax_v_vals, array_2d, cmap=cm.inferno)  # , shading="auto"
    plt.colorbar(c, ax=ax)
    plt.xlabel(f"${axis_labels[1]}$")
    plt.ylabel(f"${axis_labels[0]}$")

    ax.set_title("{} {}".format(title, fig_name))
    fig_name = (
        "{}_{}.png".format(func_name, params_str)
        if params_str
        else "{}.png".format(func_name)
    )
    fig_path = os.path.join(figs_dir, fig_name)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return fig_path


def generate_plots(args):
    results = load_results(args.result_pickles)
    print(
        "All variable params: {}".format(
            ", ".join(f"{k} ({len(v)})" for k, v in results.variable_params.items())
        )
    )
    plots_pickle = {
        "pickle_paths": args.result_pickles,
        "variable_params": results.variable_params,
        "plots": [],
        "axes": args.axes,
        "axis_labels": args.axis_labels,
        "pickle_type": PickleType.EPSILON_PLOTS,
    }
    plot_iterator = results.iter_plots(args.axes)

    progress_bar = tqdm(
        plot_iterator,
        desc="\u2728 Calculating Epsilon \u2728",
        total=len(plot_iterator),
    )
    for i, (key_params, eps_plots) in enumerate(plot_iterator):
        progress_bar.set_postfix(
            {"plot": f"{(i+1)*4}/{plot_iterator.extra_axes_count*4}"}
        )
        index_plots = {
            "key_params": key_params,
        }
        index_plots.update(eps_plots)
        plots_pickle["plots"].append(index_plots)
    progress_bar.update()
    progress_bar.close()
    output_path = args.output_pkl or "plots.pkl"
    file_path_pattern = "{}{{}}{}".format(*os.path.splitext(output_path))
    i = 1
    while os.path.exists(output_path):
        i += 1
        output_path = file_path_pattern.format(i)
    with open(output_path, "wb") as f:
        pickle.dump(plots_pickle, f)
    print("Saved plots pickle to {}".format(output_path))
    return plots_pickle


def write_plots_from_plots_pickle(plots_pickle, variable_params, figs_dir):
    # z = 1 / (epsp_map.real**2)
    # path = write_plot({}, epsp_map.real, "result", figs_dir=args.figs_dir)
    axes = plots_pickle["axes"]
    axis_labels = plots_pickle["axis_labels"]
    for plot in plots_pickle["plots"]:
        for data, name, label in [
            (
                abs(1 / (plot["epsp"])) ** 2,
                "epsp",
                r"$\left(\frac{1}{\left|\epsilon_{S}^{+}\right|}\right)^{2}$",
            ),
            (
                abs(1 / (plot["epsm"])) ** 2,
                "epsm",
                r"$\left(\frac{1}{\left|\epsilon_{S}^{-}\right|}\right)^{2}$",
            ),
            (
                np.log(abs(plot["Hinvp"])),
                "Bulk+",
                r"$f_{-}(k, \omega)=\frac{1}{\left|\bar{\omega}^{2}-\mathscr{H}^{-}\right|}$",
            ),
            (
                np.log(abs(plot["Hinvm"])),
                "Bulk-",
                r"$f_{+}(k, \omega)=\frac{1}{\left|\bar{\omega}^{2}-\mathscr{H}^{+}\right|}$",
            ),
        ]:
            plot_path = write_plot(
                axes,
                axis_labels,
                plot["key_params"],
                data,
                name,
                label,
                variable_params,
                figs_dir=figs_dir,
            )
            print("Wrote plot: {}".format(plot_path))
