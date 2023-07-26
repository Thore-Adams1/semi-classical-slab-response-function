DESC = """Compare pickled results with results from the matlab code.

Note: The following two environment variables must be set:

- `$MATLAB_EXECUTABLE`: 
    The path to the matlab executable on your machine. For example 
    ``/usr/local/bin/matlab`` or ``C:/Program Files/MATLAB/R2022a/bin/matlab.exe``.
- `$MATLAB_SCRIPTS_PATH`:
    The path to directory containing original logic for this code. This 
    directory must include `get_matelement.m` and dependent 
    files.

"""
# Standard
import pipes
import os
import argparse
import pickle
import logging
import tempfile
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third Party
import numpy as np
import scipy.io
from tqdm import tqdm
from textwrap import dedent

# Local
from scsr.results import Results, ChunkedResults


def read_array_from_csv(file_path):
    with open(file_path, "r") as f:
        numbers = []
        i = 0
        for i, line in enumerate(f, start=1):
            line = line.replace("i", "j")
            numbers.extend(complex(x) for x in line.split(","))
        return np.array(list(numbers)).reshape((-1, i))


def check_arrays_match(array_a, array_b, name="", tolerance=1e-9):
    result = True
    arrays_close = False
    assert (
        array_a.shape == array_b.shape
    ), f"{name} shape mismatch: {array_a.shape} != {array_b.shape}"
    arrays_close = np.isclose(array_a, array_b, rtol=tolerance, atol=tolerance).all()
    abs_difference = np.abs((np.abs(array_a) - np.abs(array_b)))
    tol_flat_index = abs_difference.argmax()
    tol_i = np.unravel_index(tol_flat_index, abs_difference.shape)
    tol = abs_difference[tol_i]
    if arrays_close:
        msg = "are identical byte-for-byte"
        if tol:
            msg = "are virtually identical (to tolerance {})".format(tol)
    else:
        msg = "not identical!"
        if tol:
            msg = f"diverge by a maximum of {tol:g} at index {tol_i}!"
            logging.info(
                "Values at max difference:\nA: %s\nB: %s",
                array_a[tol_i],
                array_b[tol_i],
            )
        result = False
    logging.warning("{}arrays {}".format(name + " " if name else "", msg))
    return result


def compare_matlab_csv_with_numpy_array(csv_path, numpy_array, name=""):
    matlab_array = read_array_from_csv(csv_path)
    return check_arrays_match(matlab_array, numpy_array, name=name)


def run_matlab_script(matlab_script):
    MATLAB_PATH = os.environ["MATLAB_EXECUTABLE"]
    RUN_MATLAB_FILE_COMMAND = (
        f"{pipes.quote(MATLAB_PATH)} -batch \"run('{{script_path}}')\""
    )
    cmd = RUN_MATLAB_FILE_COMMAND.format(script_path=matlab_script)
    logging.debug(f"Running {cmd}")
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE)
    stdout = p.communicate()[0]
    logging.debug("STDOUT: " + stdout.decode("utf-8"))
    if p.returncode != 0:
        raise ValueError(f"Matlab failed T_T (Return code: {p.returncode})")


def compare_results_with_matlab(results, temp_root=".validation"):
    ORIGINAL_LOGIC_PATH = os.environ["MATLAB_SCRIPTS_PATH"]
    MATLAB_CODE_HEADER = dedent(
        f"""\
    addpath {ORIGINAL_LOGIC_PATH}
    """
    )
    CREATE_M_N_ARRAY_MATLAB_CODE = dedent(
        """\
    kvec = [{Kx},0];
    % wbar = {w_bar};
    p = {P};
    L = {L};
    dtheta = {d_theta};
    dphi = {d_phi};
    vF = {Vf};
    tau = {tau};
    m_max = {m_max};
    n_max = {n_max};
    omega = {w};
    N = {N};
    wp = 1;
    A1_array = zeros(m_max+1, n_max+1);
    A2_array = A1_array;
    G_array = A1_array;
    H_array = A1_array;
    for m = 0:m_max;
        for n = 0:n_max;
            [A1,A2,G,H,~,~,~,~] = get_matelement(n,m,kvec,omega,tau,L,p,N,vF,wp);
            A1_array(m+1,n+1) = A1;
            A2_array(m+1,n+1) = A2;
            G_array(m+1,n+1) = G;
            H_array(m+1,n+1) = H;
        end
    end
    save('{matrix_file_stem}.mat', 'A1_array','A2_array','G_array','H_array')
    display('Matlab script completed successfully!')
    """
    )

    p = results.parameters
    p["N"] = p["theta_max"] / p["d_theta"]
    arrays_to_compare = {
        f: [None] * results.param_combination_count() for f in results.functions
    }
    if not os.path.exists(temp_root):  #
        os.mkdir(temp_root)
    temp_dir = tempfile.mkdtemp(prefix="results_", dir=temp_root)
    matlab_futures = {}
    param_texts = []
    with ThreadPoolExecutor() as executor:

        for i, iteration in enumerate(results.parameter_combinations()):
            iteration_index = []
            param_strs = []
            for name, (index, value) in zip(results.variable_params, iteration):
                p[name] = value
                param_strs.append(f"{name}={value:g}")
                iteration_index.append(index)
            param_text = " ".join(param_strs)
            p["w_bar"] = p["w"] + (1j / p["tau"])
            p["omega"] = p["w"]
            iteration_arrays = [
                results.get_m_n_array_from_index(f, iteration_index)
                for f in arrays_to_compare
            ]
            p["m_max"], p["n_max"] = [(n - 1) for n in iteration_arrays[0].shape]
            p["matrix_file_stem"] = os.path.abspath(
                os.path.join(temp_dir, f"matlab_results_{i}")
            )
            matlab_code = "{}{}".format(
                MATLAB_CODE_HEADER, CREATE_M_N_ARRAY_MATLAB_CODE.format(**p)
            )
            matlab_script = os.path.join(temp_dir, f"calc_results_{i}.m")
            with open(matlab_script, "w") as f:
                f.write(matlab_code)
            param_texts.append(param_text)
            matlab_futures[executor.submit(run_matlab_script, matlab_script)] = (
                i,
                iteration_index,
                iteration_arrays,
                p["matrix_file_stem"],
            )
        pbar = tqdm(
            total=results.param_combination_count(),
            disable=logging.getLogger().getEffectiveLevel() <= logging.DEBUG,
        )
        for future in as_completed(matlab_futures):
            pbar.update(1)
            i, iteration_index, iteration_arrays, matrix_file_stem = matlab_futures[
                future
            ]
            mat = scipy.io.loadmat(f"{matrix_file_stem}.mat")
            for function, python_array in zip(arrays_to_compare, iteration_arrays):
                arrays_to_compare[function][i] = (
                    python_array,
                    mat["{}_array".format(function)],
                )

        pbar.close()
    result = True
    for function in arrays_to_compare:
        for param_text, (i, (python_array, matlab_array)) in zip(
            param_texts, enumerate(arrays_to_compare[function])
        ):
            result = (
                check_arrays_match(
                    matlab_array,
                    python_array,
                    name=f"{function:<2}({i:<2}: {param_text:<40})",
                )
                and result
            )
    return result


def main():
    args = get_parser().parse_args()
    for i, log_level_name in enumerate(["ERROR", "WARNING", "INFO", "DEBUG"]):
        if i == args.verbosity:
            break
    print(f"Log level: {log_level_name}")
    log_level = getattr(logging, log_level_name)
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    if len(args.pickle_files) == 1:
        with open(args.pickle_files[0], "rb") as f:
            results = Results.from_dict(pickle.load(f))
            file_msg = args.pickle_files[0]
    else:
        all_results = []
        for pickle_file in args.pickle_files:
            with open(pickle_file, "rb") as f:
                all_results.append(Results.from_dict(pickle.load(f)))
        file_msg = f"{len(args.pickle_files)} pickles"
        results = ChunkedResults(all_results)
    if not results.functions:
        raise ValueError("No functions found in results")
    print(f"Comparing results in {file_msg} with matlab code...")
    valid = compare_results_with_matlab(results)
    if valid:
        print("\033[92mPickle file matches matlab results!\033[0m")
    else:
        print("\033[91mPickle file does NOT match matlab results!\033[0m")
        exit(1)


def get_parser():
    parser = argparse.ArgumentParser(
        description=DESC, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "pickle_files",
        help=(
            "Pickle files with results. If multiple are given, will attempt "
            "to combine them into a single results object."
        ),
        nargs="+",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Print more information",
        action="count",
        default=0,
        dest="verbosity",
    )
    return parser


if __name__ == "__main__":
    main()
