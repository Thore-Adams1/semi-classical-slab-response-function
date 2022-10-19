"""plot the results from the thing bla
"""
import argparse
import pickle
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
from re import L

import numpy as np
from tqdm import tqdm
import matplotlib.ticker as tck
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.linalg import inv, det

"""
run:
py plot.py --help 
for information
"""


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
    from thesis_code import ResultsStorage, CombinedResults

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

            G = results.get_m_n_array_from_index("G", full_index)
            # A1 = results.get_m_n_array_from_index("A1", full_index)
            # A2 = results.get_m_n_array_from_index("A2", full_index)
            H = results.get_m_n_array_from_index("H", full_index)

            tau = get_param("tau", axes_index)
            w_bar = get_param("w", axes_index) + 1j / tau

            H_plus = np.matrix(H[0::2, 0::2]).T
            H_minus = np.matrix(H[1::2, 1::2]).T

            G_plus = np.matrix(G[::2, ::2])
            G_minus = np.matrix(G[1::2, 1::2])

            # A = A1 + A2
            # A_plus = A[0::2, 0::2]
            # A_minus = A[1::2, 1::2]

            """
            Create required arrays from output arrays
            """
            Z_plus = np.ones([np.shape(H_plus)[0]])
            Z_plus[0] = 1 / 2
            Z_plus_matrix = np.matrix(Z_plus).T
            Z_minus = np.ones([np.shape(H_plus)[0]])
            Z_minus_matrix = np.matrix(Z_minus).T

            G_vec_plus = np.matrix(G_plus[:, 0] * 2)
            G_vec_minus = np.matrix(G_minus[:, 0] / Z_minus)

            Iden = np.identity(np.shape(H_plus)[0])

            iden_w_sq = np.matrix(Iden * w_bar**2)

            Hinvp = np.linalg.inv(iden_w_sq - H_plus)
            Hinvm = np.linalg.inv(iden_w_sq - H_minus)

            """
            Calculate epsilon
            """
            epsp = (
                1 - G_vec_plus.T * Hinvp * Z_plus_matrix
            )  # The poles of this function give symmetric SPWs.
            epsm = (
                1 - G_vec_minus.T * Hinvm * Z_minus_matrix
            )  # The poles of this function give anti-symmetric SPWs

            sign_Hinvp, slog_Hinvp = np.linalg.slogdet(iden_w_sq - H_plus)
            sign_Hinvm, slog_Hinvm = np.linalg.slogdet(iden_w_sq - H_minus)

            Fp = sign_Hinvp * np.exp(slog_Hinvp)
            Fm = sign_Hinvm * np.exp(slog_Hinvm)

            epsp_map[ax_h_i, ax_v_i] = epsp[0, 0]
            epsm_map[ax_h_i, ax_v_i] = epsm[0, 0]

            Hinvp_map[ax_h_i, ax_v_i] = 1 / (Fp)
            Hinvm_map[ax_h_i, ax_v_i] = 1 / (Fm)

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


def plot(plots_pickle, variable_params, figs_dir):
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
        plot(plots, plots["variable_params"], figs_dir)


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
