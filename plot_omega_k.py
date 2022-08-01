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
py plot_omega_k.py --help 
for information
"""


def write_plot(
    iteration_params,
    array_2d,
    func_name,
    title,
    variable_params,
    figs_dir="figs",
):
    w_vals_array = np.array(variable_params["w"])
    Kx_vals_array = np.array(variable_params["Kx"])
    w_vals = w_vals_array.T * np.ones([len(Kx_vals_array), len(w_vals_array)])
    Kx_vals = np.ones([len(w_vals_array), len(Kx_vals_array)]) * Kx_vals_array.T

    fig_name = "({}) = ({})".format(
        ",".join(iteration_params.keys()), ",".join(map(str, iteration_params.values()))
    )
    fig, ax = plt.subplots()
    c = ax.pcolor(Kx_vals.T, w_vals, array_2d, cmap=cm.inferno)  # , shading="auto"
    plt.colorbar(c, ax=ax)
    plt.xlabel("$K$")
    plt.ylabel("$\omega$")

    ax.set_title("{} {}".format(title, fig_name))
    params_str = "_".join("{}={:g}".format(k, v) for k, v in iteration_params.items())
    fig_name = (
        "{}_{}.png".format(func_name, params_str)
        if params_str
        else "{}.png".format(func_name)
    )
    fig_path = os.path.join(figs_dir, fig_name)
    plt.savefig(fig_path, dpi=300)
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
    args.axes = ["m", "n"]
    args.variable_params = ["w", "Kx"]

    variable_params = results.variable_params
    param_combinations = cartesian_product(
        *[np.array(variable_params[v]) for v in args.variable_params]
    )

    combination_indices = cartesian_product(
        *[np.arange(len(variable_params[v])) for v in args.variable_params]
    )

    progress_bar = tqdm(
        zip(param_combinations, combination_indices),
        desc="\u2728 Calculating Epsilon \u2728",
        total=len(param_combinations),
    )
    epsp_map = np.empty(
        [len(variable_params["Kx"]), len(variable_params["w"])], dtype=np.complex128
    )
    epsm_map = np.empty(
        [len(variable_params["Kx"]), len(variable_params["w"])], dtype=np.complex128
    )
    Hinvp_map = np.empty(
        [len(variable_params["Kx"]), len(variable_params["w"])], dtype=np.complex128
    )
    Hinvm_map = np.empty(
        [len(variable_params["Kx"]), len(variable_params["w"])], dtype=np.complex128
    )

    for param_values, combo_indices in progress_bar:
        values = {v: vals for v, vals in zip(results.variable_params, param_values)}
        index = {v: vals for v, vals in zip(results.variable_params, combo_indices)}
        w, Kx = values["w"], values["Kx"]
        w_i, Kx_i = index["w"], index["Kx"]

        G = results.get_m_n_array_from_index("G", combo_indices)
        # A1 = results.get_m_n_array_from_index("A1", combo_indices)
        # A2 = results.get_m_n_array_from_index("A2", combo_indices)
        H = results.get_m_n_array_from_index("H", combo_indices)

        tau = 100
        w_bar = w + 1j / tau

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

        epsp_map[Kx_i, w_i] = epsp[0, 0]
        epsm_map[Kx_i, w_i] = epsm[0, 0]
        
        Hinvp_map[Kx_i, w_i] = 1/(Fp)
        Hinvm_map[Kx_i, w_i] = 1/(Fm)

    key_params = {p: results.parameters[p] for p in ("L", "tau")}
    output_path = args.output_pkl or "plots.pkl"
    file_path_pattern = "{}{{}}{}".format(*os.path.splitext(output_path))
    i = 1
    while os.path.exists(output_path):
        i += 1
        output_path = file_path_pattern.format(i)
    plots = {
        "pickle_paths": args.result_pickles,
        "epsp_map": epsp_map,
        "epsm_map": epsm_map,
        "Hinvp_map": Hinvp_map,
        "Hinvm_map": Hinvm_map,
        "key_params": key_params,
        "variable_params": results.variable_params,
    }
    with open(output_path, "wb") as f:
        pickle.dump(plots, f)
    print("Saved plots pickle to {}".format(output_path))
    return plots


def plot_omega_k(plots, variable_params, figs_dir):
    # z = 1 / (epsp_map.real**2)
    # path = write_plot({}, epsp_map.real, "result", figs_dir=args.figs_dir)
    path_epsp = write_plot(
        plots["key_params"],
        abs(1 / (plots["epsp_map"])) ** 2,
        "epsp",
        r"$\left(\frac{1}{\left|\epsilon_{S}^{+}\right|}\right)^{2}$",
        variable_params,
        figs_dir=figs_dir,
    )
    path_epsm = write_plot(
        plots["key_params"],
        abs(1 / (plots["epsm_map"])) ** 2,
        "epsm",
        r"$\left(\frac{1}{\left|\epsilon_{S}^{-}\right|}\right)^{2}$",
        variable_params,
        figs_dir=figs_dir,
    )
    path_Hinvp = write_plot(
        plots["key_params"],
        np.log(abs(plots["Hinvp_map"])),
        "Bulk+",
        r"$f_{-}(k, \omega)=\frac{1}{\left|\bar{\omega}^{2}-\mathscr{H}^{-}\right|}$",
        variable_params,
        figs_dir=figs_dir,
    )
    path_Hinvm = write_plot(
        plots["key_params"],
        np.log(abs(plots["Hinvm_map"])),
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
        with open(args.plot_pickle, "rb") as f:
            plots = pickle.load(f)
        print("Loading results object for {}".format(args.plot_pickle))
    else:
        plots = generate_plots(args)

    figs_dir = args.figs_dir or os.path.join(
        "figs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)

    plot_omega_k(plots, plots["variable_params"], figs_dir)


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
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
        action="store_true",
        help="Directory in which to store figures. Defaults to a timestamped \n"
        r"'./figs/[DATE]_[TIME]' directory",
    )
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
