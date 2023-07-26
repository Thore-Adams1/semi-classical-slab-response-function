#! /usr/bin/env python3
DESC = """This script computes H, G, A1 and A2 electron dynamics matrices

Examples:

Run A2 & G with 200 steps:
    
    scsr-calc A2 G -p steps=200

Run A2 & G with multiprocessing, with 200 steps and lc = 4 and Kx = (0,1,2,3):
    
    scsr-calc A2 G -p lc=4 steps=200 -x -v Kx=0:3

Run A2 & G with multiprocessing, with 200 steps for 6 equally-spaced w.
w values from 0-3:
    
    scsr-calc A2 G -p steps=200 -x -v w=0:3:6

Run A2 & G with 100 steps, then write to file:
    
    scsr-calc A2 G -p steps=100 -w

Run A2 & G with 100 steps, then write to file called "A2_G_100_steps.pkl"
    
    scsr-calc A2 G -p steps=100 -w -o A2_G_100_steps.pkl

Output:
Files are written as python pickles. Pickles can be read from a new python 
session using:

    import pickle
    result = pickle.load(open('output.pkl', 'rb'))

Or, more conveniently, using the `scsr.results.load_results` function,
which returns a results object corresponding to the type of pickle. It 
expects a list of 1 (or more, if chunking is used) pickle path(s), as it
also deals with compiling results from chunked `scsr-calc` commands:

    from scsr.results import load_results
    results = load_results(['out.1.pkl', 'out.2.pkl'])
"""
# Standard
import ast
import argparse
import itertools
import pickle
import os
import datetime
import queue
import multiprocessing as mp
import math
import sys
from concurrent.futures import ThreadPoolExecutor

# Third Party
from tqdm import tqdm
import numpy as np

# Local
from scsr import maths
from scsr.results import ResultsProcessor, EpsilonResultsProcessor, readable_filesize


# Globals
FUNCTIONS = {"A2", "A1", "G", "H"}
KNOWN_PARAMS = {"P", "w", "Kx", "L", "Ln", "tau", "steps", "lc"}
PARAM_DEFAULTS = {
    #  DEFINES SIZE OF FUNCTION MATRICES
    # "m_n_size": 2,
    "mp_batch_size": 1,
    # Calculates max values
    "theta_max": 0.5 * maths.xp.pi,
    "phi_max": 2 * maths.xp.pi,
    # Place holder
    "Nf_m": 1,
    "Vf": 1,
    "Ky": 0,
    "wp": 1,
    "max_tile_size": (4, 4),
}
PARAM_DESCRIPTIONS = {
    "steps": (
        "The number of discrete steps in theta/phi axes. (The theta by phi\n"
        "grid is steps^2)."
    ),
    "theta_max": "The maximum value of theta.",
    "phi_max": "The maximum value of phi.",
    "max_tile_size": (
        "The max size of the tile of a m by n function matrix to compute \n"
        "at once. Default is (2,2), but you may get significant performance \n"
        "boosts if you increase this - especially in GPU mode."
    ),
    "mp_batch_size": (
        "The number of function arrays to process before sending them to\n"
        "the main thread (No need to adjust)."
    ),
}


def worker_calculate(
    param_queue,
    result_queue,
    progress,
    functions,
    parameters,
    dtype,
    process_id=None,
    use_gpu=False,
):
    """Worker process for multiprocessing.

    Args:
        params_pipe (mp.Pipe): Pipe to read parameters from.
        done_counter_array (mp.Array): Counter to keep track of how many jobs are done.
        process_id (int): Process ID.
        parameters (list[str]): Parameters to expect.
    """
    try:
        import psutil
    except ImportError:
        psutil = None
        print(
            "psutil not found, pip installing is recommended. It's used to check "
            "memory usage + kill processes when at risk of an out-of-memory error."
        )
    C = {}
    max_tile_size = parameters["max_tile_size"]
    while True:
        if psutil is not None:
            if psutil.virtual_memory().percent > 95:
                print("Memory usage is at 95%, killing 1 subprocess.")
                return

        param_batch = param_queue.get()
        if param_batch is None:
            return
        batch_results = []
        for iteration_params in param_batch:
            params = parameters.copy()
            params.update(iteration_params)
            mn_arrays = [
                maths.xp.full([iteration_params["mn"]] * 2, -1, dtype=dtype)
                for _ in functions
            ]
            for chunk, arrays in process_chunks(params, functions, max_tile_size, C):
                progress.value += (chunk[1] - chunk[0]) * (chunk[3] - chunk[2])
                for i, arr in enumerate(mn_arrays):
                    arr[chunk[0] : chunk[1], chunk[2] : chunk[3]] = arrays[i]
            if use_gpu:
                mn_arrays = [maths.ensure_numpy_array(arr) for arr in mn_arrays]
            batch_results.append((params["i"], mn_arrays))

        if batch_results:
            result_queue.put(batch_results)


def worker_process(
    param_queue,
    result_queue,
    progress,
    functions,
    args,
    dtype=None,
    process_id=None,
    gpu_id=None,
):
    # import cProfile
    # cProfile.runctx(
    #     "worker_calculate(param_queue, result_queue, functions, parameters, i=i)",
    #     globals(),
    #     locals(),
    #     f"debug\\prof\\prof{i+1}.prof",
    # )
    if gpu_id is not None:
        maths.set_gpu_mode(True)
        maths.cp.cuda.Device(gpu_id).use()
    if dtype is None:
        dtype = maths.xp.complex128
    parameters, _ = get_parameters(args)
    worker_calculate(
        param_queue,
        result_queue,
        progress,
        functions,
        parameters,
        dtype,
        process_id=process_id,
        use_gpu=gpu_id is not None,
    )


def process_chunks(params, functions, chunk_size, cache):
    p = params
    mn = params["mn"]
    p["w_bar"] = p["w"] + (1j / p["tau"])
    for chunk in tile_2d_arr(mn, mn, *chunk_size):
        ms, me, ns, ne = chunk
        p["mc"] = ms, me
        p["nc"] = ns, ne
        chunk_result = maths.compute_functions(functions, p, cache, result_only=True)
        arrs = [chunk_result[f]["result"] for f in functions]
        yield (ms, me, ns, ne), arrs


# @profile
def main():
    args = get_parser().parse_args()
    set_arg_defaults(args)

    start_time = datetime.datetime.now()

    gpus_to_use = []
    if args.gpu:
        maths.set_gpu_mode(True)
        import cupy as cp

        if args.gpu_ids:
            for gpu_id in args.gpu_ids:
                gpus_to_use.append(gpu_id)
        elif args.all_gpus:
            gpus_to_use = list(range(cp.cuda.runtime.getDeviceCount()))
        else:
            gpus_to_use = [cp.cuda.runtime.getDevice()]
            cp.cuda.Device(gpus_to_use[0]).use()
        if not args.use_subprocesses and len(gpus_to_use) > 1:
            print(
                f"Found >1 GPUs ({len(gpus_to_use)} found) - forcing multiprocessing mode."
            )
            args.use_subprocesses = True
        if args.use_subprocesses:
            mp.set_start_method("spawn")
        if gpus_to_use:
            print(f"Using {len(gpus_to_use)} GPU(s):")
            for gpu_id in gpus_to_use:
                gpu_info = cp.cuda.runtime.getDeviceProperties(gpu_id)
                print(
                    f"\tGPU {gpu_id}: {gpu_info['name'].decode()} "
                    f"with {gpu_info['totalGlobalMem'] / 1e9:.2f} GB of memory."
                )
        else:
            raise RuntimeError(
                "No GPUs found! Please run without --gpu or check your "
                "CUDA installation."
            )
    dtype = getattr(np, f"complex{args.dtype}")

    params, variable_params = get_parameters(args, log=True)

    if args.epsilon_only:
        results_class = EpsilonResultsProcessor
    else:
        results_class = ResultsProcessor
    result_proc = results_class(
        list(args.functions), params, variable_params, dtype=dtype
    )

    iterations = result_proc.get_tasks()
    expected_file_size = readable_filesize(result_proc.size_estimate())
    print(
        f"Expected Size: {expected_file_size} (dtype: complex{args.dtype}) "
        "Reserving memory..."
    )
    result_proc.reserve_memory()
    progress_bar = tqdm(
        desc="Computing Functions",
        total=result_proc.iteration_total,
        mininterval=args.min_update_interval,
    )
    create_postfix = lambda d: ",".join("{}={:g}".format(k, v) for k, v in d.items())
    C = {}
    if not args.use_subprocesses:
        max_tile_size = result_proc.parameters["max_tile_size"]
        for iteration_params in iterations:
            mn = iteration_params["mn"]
            progress_bar.postfix = create_postfix(iteration_params)
            p = result_proc.parameters.copy()
            p.update(iteration_params)
            mn_arrays = [
                maths.xp.full([mn] * 2, -1, dtype=maths.xp.complex128)
                for _ in args.functions
            ]
            for chunk, arrays in process_chunks(p, args.functions, max_tile_size, C):
                ms, me, ns, ne = chunk
                progress_bar.update((me - ms) * (ne - ns))
                for i, arr in enumerate(mn_arrays):
                    arr[chunk[0] : chunk[1], chunk[2] : chunk[3]] = arrays[i]
            result_proc.add_m_n_arrays(p["i"], mn_arrays)

            p.update(iteration_params)
    else:
        if args.gpu:
            args.subprocess_count = args.subprocess_count or len(gpus_to_use)
        else:
            # Limit any multiprocessing within numpy
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_MAIN_FREE"] = "1"
            os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"
            args.subprocess_count = args.subprocess_count or os.cpu_count()

        batch_size = min(
            (
                params.get("mp_batch_size", 1),
                result_proc.iteration_total // args.subprocess_count,
            )
        )
        total_arrays = 0
        for v in result_proc.m_n_arrays.values():
            total_arrays += len(v)

        progress_bar.write(
            "Using multiprocessing, max workers: {} - Batch size: {} - Arrays to compute: {}".format(
                args.subprocess_count, batch_size, total_arrays
            )
        )
        process_queue_size = args.subprocess_count * 8
        param_queue = mp.Queue(maxsize=process_queue_size)
        result_queue = mp.Queue(maxsize=process_queue_size)
        progress_values = []
        processes = []
        for i, gpu_id in zip(
            range(args.subprocess_count), itertools.cycle(gpus_to_use or [None])
        ):
            progress_value = mp.Value("i", 0, lock=False)
            progress_values.append(progress_value)
            process = mp.Process(
                target=worker_process,
                args=(
                    param_queue,
                    result_queue,
                    progress_value,
                    args.functions,
                    args,
                ),
                kwargs={"process_id": i, "gpu_id": gpu_id, "dtype": dtype},
            )
            processes.append(process)
        with ThreadPoolExecutor(max_workers=args.subprocess_count) as executor:
            for process in processes:
                executor.submit(process.start)
            executor.shutdown(wait=True)
        processing_time = datetime.datetime.now() - start_time
        progress_bar.write("--- Initialized processes: {} ---".format(processing_time))
        start_time = datetime.datetime.now()
        postfix = {"Procs": args.subprocess_count}
        progress_bar.postfix = create_postfix(postfix)

        def queue_parameters(parameter_queue, iterations, batches, batch_size):
            for _ in range(batches):
                batch = []
                for _ in range(batch_size):
                    iteration_params = next(iterations, None)
                    if iteration_params is None:
                        break
                    batch.append(iteration_params.copy())
                parameter_queue.put(batch or None)

        try:
            queue_parameters(param_queue, iterations, process_queue_size, batch_size)
            complete_arrays = 0
            while complete_arrays != total_arrays:
                try:
                    recieved = result_queue.get(timeout=1)
                except queue.Empty:
                    recieved = None
                total_progress = sum(v.value for v in progress_values)
                unreported_progress = total_progress - progress_bar.n
                if unreported_progress:
                    progress_bar.update(unreported_progress)
                if recieved is None:
                    continue
                for i, mn_arrays in recieved:
                    complete_arrays += len(mn_arrays)
                    if mn_arrays:
                        for name, array in zip(args.functions, mn_arrays):
                            result_proc.m_n_arrays[name][i] = array
                queue_parameters(param_queue, iterations, 1, batch_size)
                progress_bar.postfix = create_postfix(postfix)
            for _ in processes:
                param_queue.put(None)
            for p in processes:
                process.join()
        except Exception as exc:
            print("Killing processes")
            for p in processes:
                p.kill()
            raise exc
    progress_bar.close()
    if args.gpu:
        result_proc.numpyify()

    results_dict = result_proc.as_dict()
    results_dict["args"] = vars(args)
    processing_time = datetime.datetime.now() - start_time
    results_dict["metadata"] = {"runtime": processing_time, "cli_args": sys.argv}
    print("--- Processing time: {} ---".format(processing_time))
    if args.write:
        output_path = args.output or "results/output.pkl"
        dir_name = os.path.dirname(output_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if not args.force and os.path.exists(output_path):
            file_path_pattern = "{}{{}}{}".format(*os.path.splitext(output_path))
            i = 1
            while os.path.exists(output_path):
                i += 1
                output_path = file_path_pattern.format(i)

        print("Writing to file: {} ... ".format(os.path.realpath(output_path)), end="")
        with open(output_path, "wb+") as f:
            pickle.dump(results_dict, f)
        print("Done!")
    return results_dict


def get_parameters(args, log=False):
    """Get parameters from command line arguments.
    Args:
        args (argparse.Namespace): Command line arguments.
        log (bool): Whether to log the parameters.

    Returns:
        tuple[dict, dict]: Parameters and variable parameters.
    """
    params = PARAM_DEFAULTS.copy()
    variable_params = {}

    # Override parameters with -p arguments
    for param, value in itertools.chain.from_iterable(args.params or ()):
        if param == "max_tile_size":
            import numbers

            if isinstance(value, numbers.Number):
                value = [
                    value,
                ]
            chunk_size = list(value)
            if len(chunk_size) == 1:
                chunk_size += chunk_size
        if param in params or param in KNOWN_PARAMS:
            params[param] = value
        else:
            available_params = KNOWN_PARAMS | set(variable_params) | set(params)
            raise ValueError(
                "Unknown parameter: {}. Available params {}".format(
                    param, available_params
                )
            )
        params[param] = value
    # Override variable parameters with -v arguments
    for param, values in itertools.chain.from_iterable(args.variable_params or ()):
        if (
            param not in variable_params
            and param not in params
            and param not in KNOWN_PARAMS
        ):
            available_params = KNOWN_PARAMS | set(variable_params) | set(params)
            raise ValueError(
                "Unknown parameter: {}. Available params {}".format(
                    param, available_params
                )
            )
        variable_params[param] = values

    if log:
        print(
            "Functions: {}\nParameters: {}\nVariable: {}".format(
                args.functions, params, variable_params
            )
        )

    # --- SET UP DEPENDENT PARAMETERS ---
    # Calculates theta and phi steps
    params["d_theta"] = params["theta_max"] / (params["steps"] - 1)
    params["d_phi"] = params["phi_max"] / (params["steps"] - 1)
    # Generates arrays for theta and phi based on the values previously defined
    params["theta_array"] = maths.xp.linspace(0, params["theta_max"], params["steps"])
    params["phi_array"] = maths.xp.linspace(0, params["phi_max"], params["steps"])

    if args.chunks > 1:
        if args.chunk_parameter is None:
            args.chunk_parameter = max(
                variable_params.keys(), key=lambda p: len(variable_params[p])
            )
        if args.chunk_id is None:
            raise ValueError("No chunk id specified.")
        chunked_values = get_chunk(
            variable_params[args.chunk_parameter], args.chunks, args.chunk_id
        )
        if not chunked_values:
            raise ValueError("Chunk {} has no values.".format(args.chunk_id))
        if log:
            print(
                "Chunked values on axis {} [{}/{}]: {}".format(
                    args.chunk_parameter, args.chunk_id, args.chunks, chunked_values
                )
            )
        variable_params[args.chunk_parameter] = chunked_values

    return params, variable_params


def param_type(string):
    param, value = string.split("=")
    return param, ast.literal_eval(value)


def variable_param_type(string):
    param, values = string.split("=")
    all_values = []
    for value in values.split(","):
        steps = None
        split_steps = value.split(":")
        if len(split_steps) == 3:
            start_str, end_str, steps = split_steps
            steps = ast.literal_eval(steps)
            if steps % 1 != 0:
                raise argparse.ArgumentTypeError(
                    "Couldn't process variable param {!r}. "
                    "Steps must be an integer.".format(param)
                )
        elif len(split_steps) == 2:
            start_str, end_str = split_steps
        else:
            all_values.append(ast.literal_eval(value))
            continue
        start, end = ast.literal_eval(start_str), ast.literal_eval(end_str)
        if steps is None:
            steps = end - start
            if start % 1 != 0 or end % 1 != 0:
                raise argparse.ArgumentTypeError(
                    "Couldn't process non-integer range variable param {!r}: "
                    "{!r}. No steps specified.".format(param, value)
                )
            else:
                all_values.extend(maths.xp.linspace(start, end, 1 + end - start))
        else:
            all_values.extend(maths.xp.linspace(start, end, steps))
    return param, all_values


def get_parser():
    # indent_trailing = lambda text: f"{text}".replace("\n", f"\n\t\t")
    param_descs = PARAM_DESCRIPTIONS.copy()
    all_params = set(KNOWN_PARAMS) | set(PARAM_DEFAULTS)
    no_desc = set()
    for param in all_params:
        if param not in param_descs:
            no_desc.add(param)
    no_desc_text = ", ".join(f"`{p}`" for p in no_desc)
    param_text = "# Parameters:\n\n{}\n\n{}".format(
        no_desc_text,
        "\n".join(f"* `{param}`: {param_descs[param]}" for param in param_descs),
    )
    parser = argparse.ArgumentParser(
        description=f"{DESC}\n{param_text}",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    inputs_group = parser.add_argument_group("Inputs")
    inputs_group.add_argument(
        "functions", nargs="*", help="Functions.", choices=FUNCTIONS, default=FUNCTIONS
    )
    inputs_group.add_argument(
        "-p",
        "--params",
        type=param_type,
        nargs="+",
        help="Parameters to override. space-separated list of '[PARAM]=[VALUE]' pairs.",
        metavar="P=V",
        action="append",
    )
    inputs_group.add_argument(
        "-v",
        "--variable-params",
        nargs="+",
        type=variable_param_type,
        help=(
            "Variable parameters to override. Space-separated list of \n"
            "'[PARAM]=[VALUE1],[...],[VALUEN]' items. Values can either be a\n"
            "single value or a range of values.\n"
            "Ranges of consecutive whole numbers can be specified as 'A:B', \n"
            "where A and B are integers. Ranges of floating point numbers can\n"
            "be specified as '[VALUEA]:[VALUEB]:[STEPS]', where steps is the\n"
            "number of steps"
        ),
        action="append",
        metavar="P=V1,V2,..,VN",
    )
    inputs_group.add_argument(
        "-d",
        "--dtype",
        choices=["64", "128", "256"],
        default="128",
        help="Complex data type to use for calculations.",
    )
    mp_group = parser.add_argument_group("Processing")
    mp_group.add_argument(
        "-x", "--use-subprocesses", action="store_true", help="Use subprocesses."
    )
    mp_group.add_argument(
        "-g",
        "--gpu",
        action="store_true",
        help=("Use the GPU. Requires a CUDA-enabled GPU and CuPy to be installed."),
    )
    gpu_mode_group = mp_group.add_mutually_exclusive_group()
    gpu_mode_group.add_argument(
        "-G",
        "--gpu-ids",
        type=int,
        nargs="+",
        help=(
            "Id(s) of GPUs)s to use. Default's to the first available\n"
            "cuda-capable gpu. Implies --gpu."
        ),
    )
    gpu_mode_group.add_argument(
        "-A",
        "--all-gpus",
        action="store_const",
        const=-1,
        help="Use all available GPUs. Implies --gpu.",
    )
    mp_group.add_argument(
        "-m",
        "--subprocess-count",
        type=int,
        default=None,
        help=(
            "Use this many subprocesses. Defaults to the processor core\n"
            "count, unless --gpu is specified, in which case it defaults to 1\n"
            "per GPU in use."
        ),
    )
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "-w",
        "--write",
        action="store_true",
        help="Write a pickle file with the resultant array.",
    )
    output_group.add_argument(
        "-E",
        "--epsilon-only",
        action="store_true",
        help="Only include epsilon values in the results.",
    )
    output_group.add_argument(
        "-o",
        "--output",
        help="Output pickle file path. Defaults to 'results/output.pkl'.",
    )
    output_group.add_argument(
        "-f",
        "--force",
        action="store_true",
        help=(
            "Overwrite pickle file path it exists. Will generate a unique name\n"
            "by default."
        ),
    )
    output_group.add_argument(
        "-u",
        "--min-update-interval",
        default=0,
        type=float,
        help="Min interval (seconds) between progress bar updates.",
        metavar="seconds",
    )
    chunk_group = parser.add_argument_group("Chunking")
    chunk_group.add_argument(
        "-C", "--chunks", type=int, default=1, help="Number of chunks."
    )
    chunk_group.add_argument(
        "-P",
        "--chunk-parameter",
        type=str,
        default=None,
        help=(
            "Parameter on which to chunk. Defaults to variable parameter with\n"
            "the most values."
        ),
    )
    chunk_group.add_argument(
        "-I",
        "--chunk-id",
        type=int,
        default=None,
        help="Chunk id. (from 1 to --chunks)",
    )
    return parser


def set_arg_defaults(args):
    if args.all_gpus or args.gpu_ids:
        args.gpu = True


def get_chunk(array, chunks, chunk_id):
    """
    > for i in range(1,6): print(i, get_chunk(list(range(12)), 5, i))
    1 [0, 1, 2, 3]
    2 [2, 3, 4, 5]
    3 [4, 5, 6, 7]
    4 [6, 7, 8, 9]
    5 [10, 11]
    """
    chunk_size = len(array) / chunks
    if chunk_size % 1 != 0:
        chunk_size = chunk_size + 1
    chunk_size = int(chunk_size)
    for i in range(0, len(array), chunk_size):
        chunk_id -= 1
        if not chunk_id:
            return array[i : i + chunk_size]
    return []


def tile_2d_arr(width, height, max_width, max_height):
    p, q = max_width, max_height
    if p > width:
        p = width
    if q > height:
        q = height
    if p == 0:
        v_slices = width
    else:
        v_slices = math.ceil(width / p)
    if q == 0:
        h_slices = height
    else:
        h_slices = math.ceil(height / q)
    h_start = h_end = v_start = v_end = 0
    for h in range(h_slices - bool(height % p)):
        h_start = h * q
        h_end = h_start + p
        for v in range(v_slices - bool(width % p)):
            v_start = v * q
            v_end = v_start + q
            yield h_start, h_end, v_start, v_end
        if v_end < width:
            yield h_start, h_end, v_end, width
    if h_end < height:
        for v in range(v_slices - bool(width % p)):
            v_start = v * q
            v_end = v_start + q
            yield h_end, width, v_start, v_end
        if v_start < width:
            yield h_end, width, v_end, width


if __name__ == "__main__":
    main()
