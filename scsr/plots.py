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
from .results import ResultsStorage, CombinedResults
from . import maths


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

    fig_name = "({}) = ({})".format(
        ",".join(iteration_params.keys()), ",".join(map(str, iteration_params.values()))
    )
    fig, ax = plt.subplots()
    c = ax.pcolor(ax_h_vals.T, ax_v_vals, array_2d, cmap=cm.inferno)  # , shading="auto"
    plt.colorbar(c, ax=ax)
    plt.xlabel(f"${axis_labels[1]}$")
    plt.ylabel(f"${axis_labels[0]}$")

    ax.set_title("{} {}".format(title, fig_name))
    params_str = "_".join("{}={:g}".format(k, v) for k, v in iteration_params.items())
    fig_name = (
        "{}_{}.png".format(func_name, params_str)
        if params_str
        else "{}.png".format(func_name)
    )
    fig_path = os.path.join(figs_dir, fig_name)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return fig_path


def cartesian_product(*arrays, reshaped=True):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    if reshaped:
        return arr.reshape(-1, la)
    else:
        return arr


def load_results(pickle_files):
    if len(pickle_files) == 1:
        print("Loading {}".format(pickle_files[0]))
        with open(pickle_files[0], "rb") as f:
            results = ResultsStorage.from_dict(pickle.load(f))
    else:
        all_results = []
        for pickle_file in pickle_files:
            with open(pickle_file, "rb") as f:
                print("Loading {}".format(pickle_file))
                all_results.append(ResultsStorage.from_dict(pickle.load(f)))
        results = CombinedResults(all_results)
    return results


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
    }
    additional_axes = [p for p in results.variable_params if p not in args.axes]
    src_key_params = {
        p: results.parameters[p] for p in ("L", "tau") if p in results.parameters
    }

    variable_params = results.variable_params
    param_combinations = cartesian_product(
        *[np.array(variable_params[v]) for v in args.axes],
        #  reshaped=False
    )
    combination_indices = cartesian_product(
        *[np.arange(len(variable_params[v])) for v in args.axes]
    )
    if additional_axes:
        additional_axes_count = np.product(
            [len(variable_params[p]) for p in additional_axes]
        )
        additional_axes_indices = itertools.product(
            *[range(len(variable_params[p])) for p in additional_axes]
        )
    else:
        additional_axes_indices = [()]
        additional_axes_count = 1

    progress_bar = tqdm(
        desc="\u2728 Calculating Epsilon \u2728",
        total=param_combinations.shape[0] * additional_axes_count,
    )

    def get_param(param, index):
        if param == args.axes[0]:
            return results.variable_params[param][index[0]]
        elif param == args.axes[1]:
            return results.variable_params[param][index[1]]
        else:
            return results.parameters[param]

    for i, additional_axes_index in enumerate(additional_axes_indices):
        progress_bar.set_postfix({"plot": f"{(i+1)*4}/{additional_axes_count*4}"})
        key_params = src_key_params.copy()
        for j, axis in enumerate(additional_axes):
            key_params[axis] = variable_params[axis][additional_axes_index[j]]
        epsp_map = np.empty(
            [len(variable_params[args.axes[1]]), len(variable_params[args.axes[0]])],
            dtype=np.complex128,
        )
        epsm_map = np.empty(
            [len(variable_params[args.axes[1]]), len(variable_params[args.axes[0]])],
            dtype=np.complex128,
        )
        Hinvp_map = np.empty(
            [len(variable_params[args.axes[1]]), len(variable_params[args.axes[0]])],
            dtype=np.complex128,
        )
        Hinvm_map = np.empty(
            [len(variable_params[args.axes[1]]), len(variable_params[args.axes[0]])],
            dtype=np.complex128,
        )
        for param_values, axes_index in zip(param_combinations, combination_indices):
            progress_bar.update(1)

            axes_index = list(axes_index)
            additional_axes_i = list(additional_axes_index)
            full_index = []
            for v in variable_params:
                if v in args.axes:
                    full_index.append(axes_index[args.axes.index(v)])
                else:
                    full_index.append(additional_axes_i.pop(0))
            full_index = tuple(full_index)
            values = {v: vals for v, vals in zip(args.axes, param_values)}
            index = {v: vals for v, vals in zip(results.variable_params, full_index)}
            ax_v_v, ax_h_v = values[args.axes[0]], values[args.axes[1]]
            ax_v_i, ax_h_i = index[args.axes[0]], index[args.axes[1]]

            eps = maths.get_epsilon_at_index(results, full_index)

            epsp_map[ax_h_i, ax_v_i] = eps["epsp"]
            epsm_map[ax_h_i, ax_v_i] = eps["epsm"]

            Hinvp_map[ax_h_i, ax_v_i] = eps["Hinvp"]
            Hinvm_map[ax_h_i, ax_v_i] = eps["Hinvm"]

        index_plots = {
            "epsp_map": epsp_map,
            "epsm_map": epsm_map,
            "Hinvp_map": Hinvp_map,
            "Hinvm_map": Hinvm_map,
            "key_params": key_params,
        }
        plots_pickle["plots"].append(index_plots)
    progress_bar.refresh()
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
        path_epsp = write_plot(
            axes,
            axis_labels,
            plot["key_params"],
            abs(1 / (plot["epsp_map"])) ** 2,
            "epsp",
            r"$\left(\frac{1}{\left|\epsilon_{S}^{+}\right|}\right)^{2}$",
            variable_params,
            figs_dir=figs_dir,
        )
        path_epsm = write_plot(
            axes,
            axis_labels,
            plot["key_params"],
            abs(1 / (plot["epsm_map"])) ** 2,
            "epsm",
            r"$\left(\frac{1}{\left|\epsilon_{S}^{-}\right|}\right)^{2}$",
            variable_params,
            figs_dir=figs_dir,
        )
        path_Hinvp = write_plot(
            axes,
            axis_labels,
            plot["key_params"],
            np.log(abs(plot["Hinvp_map"])),
            "Bulk+",
            r"$f_{-}(k, \omega)=\frac{1}{\left|\bar{\omega}^{2}-\mathscr{H}^{-}\right|}$",
            variable_params,
            figs_dir=figs_dir,
        )
        path_Hinvm = write_plot(
            axes,
            axis_labels,
            plot["key_params"],
            np.log(abs(plot["Hinvm_map"])),
            "Bulk-",
            r"$f_{+}(k, \omega)=\frac{1}{\left|\bar{\omega}^{2}-\mathscr{H}^{+}\right|}$",
            variable_params,
            figs_dir=figs_dir,
        )
        print("Wrote plot: {}".format(path_epsp))
        print("Wrote plot: {}".format(path_epsm))
        print("Wrote plot: {}".format(path_Hinvp))
        print("Wrote plot: {}".format(path_Hinvm))