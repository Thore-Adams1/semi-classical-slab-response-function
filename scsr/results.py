"""Module defining objects used for managing matrices of results."""
# Standard
from enum import Enum
import itertools
from bisect import bisect_left
from collections import defaultdict
import pickle

# Third Party
import numpy as np

# Local
from . import maths


class PickleType(Enum):
    # Contains n-dimensional array of indices which each point to unique m by n
    # arrays
    M_N_ARRAYS = 0
    # Contains n-dimensional plot of epsilon + associated values
    EPSILON_VALUES = 1
    # Contains 2d plot data
    EPSILON_PLOTS = 2


class ResultsBase:
    def __init__(self, parameters, variable_params):
        self.parameters = dict(parameters)
        self.variable_params = dict(variable_params)
        self.pickle_paths = []
        self.variable_param_indices = {k: i for i, k in enumerate(variable_params)}
        self.index_shape = tuple(len(v) for v in self.variable_params.values())
        self.metadata = {}

    def __repr__(self):
        p_args, v_args = self.reconstruct_readable_param_args()
        return f"<{self.__class__.__name__}: {' '.join(p_args)}, {' '.join(v_args)}>"

    def as_dict(self, **kwargs):
        dictionary = self._get_results_dict()
        dictionary.update(
            {
                "pickle_type": self.get_pickle_type(),
                "parameters": self.parameters,
                "variable_params": self.variable_params,
            }
        )
        dictionary.update(kwargs)
        return dictionary

    def _get_results_dict(self):
        return {}

    def get_pickle_type(self):
        raise NotImplementedError("Must be implemented by subclass")

    def parameter_combinations(self):
        # --- LOOP OVER CARTESIAN PRODUCT OF VARIABLE PARAMETERS ---
        enumerated_values = (
            enumerate(values) for values in self.variable_params.values()
        )
        return itertools.product(*enumerated_values)

    def param_combination_count(self):
        param_combinations = 1
        for v in self.variable_params.values():
            param_combinations *= len(v)
        return param_combinations

    # def get_m_n_array_from_index(self, function, index):
    #     raise NotImplementedError("Must be implemented in subclass")

    def get_param_at_index(self, param_name, index):
        param_value = self.parameters.get(param_name)
        if param_value is not None:
            return param_value
        i = index[self.variable_param_indices[param_name]]
        return self.variable_params[param_name][i]

    def update_metadata(self, result_dict):
        meta = result_dict.get("metadata")
        if meta is None:
            meta = result_dict.get("details", {})
        args = result_dict.get("args")
        if args is not None:
            meta["parsed_args"] = args
        self.metadata = meta

    def reconstruct_readable_param_args(self, all_chunks=False):
        parsed_args = self.metadata.get("parsed_args")
        if parsed_args is None:
            return [], []
        p_args = []
        if all_chunks:
            params = list(itertools.chain.from_iterable(parsed_args["params"]))
        else:
            arg_params = {
                k for k, v in itertools.chain.from_iterable(parsed_args["params"])
            }
            params = [(k, v) for k, v in self.parameters.items() if k in arg_params]
        if params:
            param_fmt = lambda v: (
                ",".join(str(i) for i in v) if hasattr(v, "__getitem__") else f"{v:g}"
            )
            for k, v in params:
                if k != "max_tile_size":
                    p_args.append(f"{k}={param_fmt(v)}")
        if all_chunks:
            variable_params = list(
                itertools.chain.from_iterable(parsed_args["variable_params"])
            )
        else:
            variable_params = self.variable_params.items()
        v_args = []
        for k, v in variable_params:
            vparam_fmt = lambda v: (
                ",".join(str(i) for i in v)
                if len(v) < 3
                else f"{v[0]}:{v[-1]}:{len(v)}"
            )
            if len(v) > 3:
                v_args.append(f"{k}={vparam_fmt(v)}")
        return p_args, v_args

    def get_param_args(self, all_chunks=False):
        p_args, v_args = self.reconstruct_readable_param_args(all_chunks=all_chunks)
        args = []
        if p_args:
            args.extend(["-p"] + p_args)
        if v_args:
            args.extend(["-v"] + v_args)
        return args


class Results(ResultsBase):
    def __init__(self, functions, parameters, variable_params):
        self.functions = functions
        self.m_n_arrays = []
        super().__init__(parameters, variable_params)

    def _get_results_dict(self, **kwargs):
        return {
            "functions": self.functions,
            "m_n_arrays": self.m_n_arrays,
        }

    def get_pickle_type(self):
        return PickleType.M_N_ARRAYS

    def get_dtype(self):
        dtype_bits = self.metadata.get("parsed_args", {}).get("dtype", 128)
        return getattr(np, f"complex{dtype_bits}")

    def get_m_n_array_from_values(self, function, iteration_params):
        index = tuple(
            self.variable_params[k].index(v) for k, v in iteration_params.items()
        )
        return self.get_m_n_array_from_index(function, index)

    def get_m_n_array_from_index(self, function, index):
        return self.m_n_arrays[function][np.ravel_multi_index(index, self.index_shape)]

    @classmethod
    def from_dict(cls, dictionary):
        instance = cls(
            dictionary["functions"],
            dictionary["parameters"],
            dictionary["variable_params"],
        )
        instance.m_n_arrays = dictionary["m_n_arrays"]
        instance.pickle_paths = dictionary.get("pickle_paths", ())
        instance.update_metadata(dictionary)
        return instance

    def iter_plots(self, axes=("w", "Kx")):
        return PlotIterator(self, axes)

    def get_epsilon_at_index(self, index):
        return maths.get_epsilon_at_index(self, index)


class ChunkedResults(Results):
    """Combine the results of a set of runs.

    Args:
        results (list[Results]): list of results.
    """

    def __init__(self, results):
        results = list(results)
        (
            self.chunked_param,
            self.variable_params,
            self.ordered_chunk_ends,
            self.results,
        ) = self._reconstruct_chunks(results)
        self.parameters = results[0].parameters
        self.functions = results[0].functions
        super().__init__(self.functions, self.parameters, self.variable_params)

    def __repr__(self):
        p_args, v_args = self.reconstruct_readable_param_args()
        return "<{}: {}, {}, {}/{}>".format(
            self.__class__.__name__,
            " ".join(p_args),
            " ".join(v_args),
            self.chunked_param,
            len(self.results),
        )

    def _reconstruct_chunks(self, results):
        if len(results) < 2:
            raise ValueError(
                "No or only one result provided. Just use Results "
                "directly - cba to test this case"
            )
        variable_params = defaultdict(list)
        result_chunk_maxes = []
        chunked_param = None
        valid_variable_params = set(results[0].variable_params)
        for i, result in enumerate(results):
            if i == 0:
                for param_name, values in result.variable_params.items():
                    variable_params[param_name].extend(values)
                continue
            else:
                if set(result.variable_params) != valid_variable_params:
                    raise ValueError("Results must have the same variable parameters.")
            for param_name, values in result.variable_params.items():
                if values != results[0].variable_params[param_name]:
                    if chunked_param is None:
                        chunked_param = param_name
                    else:
                        if chunked_param != param_name:
                            raise ValueError(
                                "Cannot parse chunks on multiple axes. "
                                f"result {i+1}/{len(results)} has unique {param_name} "
                                f"whereas chunks were expected on {chunked_param}"
                            )
            if chunked_param is None:
                raise ValueError("No chunked parameters found.")
        variable_params[chunked_param] = []
        for result in results:
            values = result.variable_params[chunked_param]
            result_chunk_maxes.append((max(values), result))
            variable_params[chunked_param].extend(values)

        for k, v in variable_params.items():
            variable_params[k] = sorted(v)
        ordered_chunk_ends = []
        results = []
        for chunk_max, result in sorted(result_chunk_maxes, key=lambda x: x[0]):
            ordered_chunk_ends.append(chunk_max)
            results.append(result)
        return (chunked_param, variable_params, ordered_chunk_ends, results)

    def _get_chunked_index(self, index):
        value = self.variable_params[self.chunked_param][index]
        chunk_end_index = bisect_left(self.ordered_chunk_ends, value)
        mapped_result = self.results[chunk_end_index]
        if mapped_result is None:
            raise ValueError("No result found for index {}".format(index))
        return mapped_result, mapped_result.variable_params[self.chunked_param].index(
            value
        )

    def get_m_n_array_from_index(self, function, index):
        index = list(index)
        chunked_param_index = self.variable_param_indices[self.chunked_param]
        (
            mapped_result,
            index[chunked_param_index],
        ) = self._get_chunked_index(index[chunked_param_index])
        return mapped_result.get_m_n_array_from_index(function, tuple(index))

    def reconstruct_readable_param_args(self, all_chunks=None):
        return self.results[0].reconstruct_readable_param_args(all_chunks=True)


def calculate_m_n_sizes(results):
    # --- SET UP INTEGRAL ARRAYS (m*n) ---
    iteration_total = 0
    m_n_array_total = 1
    for v in results.variable_params.values():
        m_n_array_total *= len(v)
    m_n_array_sizes = [None] * m_n_array_total
    given_lc = results.parameters.get("lc")
    max_m_n_size = 1
    for i, values in enumerate(results.parameter_combinations()):
        iteration_params = {k: v for k, (_, v) in zip(results.variable_params, values)}
        p = results.parameters.copy()
        p.update(iteration_params)
        if given_lc is None:
            lc = 10 * p["Kx"] * p["L"] / (2 * maths.xp.pi)
            m_n_size = 2 * int(maths.xp.ceil(lc))
        else:
            m_n_size = 2 * int(maths.xp.ceil(given_lc))
        max_m_n_size = max(max_m_n_size, m_n_size)
        m_n_array_sizes[i] = m_n_size
        iteration_total += m_n_size**2
    return m_n_array_sizes, max_m_n_size, iteration_total


class ProcessorBase(Results):
    def __init__(self, functions, parameters, variable_params, dtype=np.complex128):
        super().__init__(functions, parameters, variable_params)
        self.dtype = dtype
        if not variable_params:
            raise ValueError("No variable parameters provided.")
        self.m_n_array_total = 1
        for v in variable_params.values():
            self.m_n_array_total *= len(v)
        self.m_n_arrays = {f: [None] * self.m_n_array_total for f in functions}
        self.m_n_array_sizes = [None] * self.m_n_array_total
        (
            self.m_n_array_sizes,
            self.parameters["max_m_n_size"],
            self.iteration_total,
        ) = calculate_m_n_sizes(self)

    def get_tasks(self):
        f = next(iter(self.functions), None)
        for i, values in enumerate(self.parameter_combinations()):
            # A unique combination of variable parameters.
            iteration_params = {k: v for k, (_, v) in zip(self.variable_params, values)}
            iteration_params["i"] = i
            m_max = n_max = self.m_n_array_sizes[i]
            iteration_params["mn"] = m_max  # , n_max
            yield iteration_params


class ResultsProcessor(ProcessorBase):
    def size_estimate(self):
        return (self.iteration_total * len(self.functions)) * maths.xp.dtype(
            self.dtype
        ).itemsize

    def reserve_memory(self):
        """reserve memory for the arrays.

        Returns:
            int: filesize estimate in bytes.
        """
        # reserve array mem - so that memory errors are raised early
        for function in self.functions:
            for i, m_n_array in enumerate(self.m_n_arrays[function]):
                if m_n_array is None:
                    m_n_size = self.m_n_array_sizes[i]
                    self.m_n_arrays[function][i] = arr = np.empty(
                        (m_n_size, m_n_size), dtype=self.dtype
                    )
                    arr.fill(0)

    def numpyify(self):
        for f, arrs in self.m_n_arrays.items():
            for i, arr in enumerate(arrs):
                if arr is not None:
                    self.m_n_arrays[f][i] = maths.ensure_numpy_array(arr)
        for k, v in self.parameters.items():
            self.parameters[k] = maths.ensure_numpy_array(v)

    def add_m_n_array(self, function, i, array):
        self.m_n_arrays[function][i] = maths.ensure_numpy_array(array)

    def add_m_n_arrays(self, i, arrays):
        for f, arr in zip(self.functions, arrays):
            self.m_n_arrays[f][i] = maths.ensure_numpy_array(arr)


class EpsilonResults(Results):
    def get_pickle_type(self):
        return PickleType.EPSILON_VALUES

    def _get_results_dict(self):
        return {
            "functions": self.functions,
            "epsilon_functions": self.epsilon_functions,
            "epsilon_values": self.epsilon_values,
        }

    @classmethod
    def from_dict(cls, dictionary):
        instance = cls(
            dictionary["functions"],
            dictionary["parameters"],
            dictionary["variable_params"],
        )
        instance.epsilon_values = dictionary["epsilon_values"]
        instance.epsilon_functions = dictionary["epsilon_functions"]
        instance.pickle_paths = dictionary.get("pickle_paths", ())
        instance.update_metadata(dictionary)
        return instance

    def get_epsilon_at_index(self, index):
        return self.epsilon_values[:, np.ravel_multi_index(index, self.index_shape)]


class ChunkedEpsilonResults(ChunkedResults, EpsilonResults):
    def get_epsilon_at_index(self, index):
        index = list(index)
        chunked_param_index = self.variable_param_indices[self.chunked_param]
        (
            mapped_result,
            index[chunked_param_index],
        ) = self._get_chunked_index(index[chunked_param_index])
        return mapped_result.get_epsilon_at_index(tuple(index))


class EpsilonResultsProcessor(ProcessorBase, EpsilonResults):
    def __init__(self, functions, parameters, variable_params, dtype=np.complex128):
        super().__init__(functions, parameters, variable_params, dtype=dtype)
        self.epsilon_functions = maths.EPSILON_FUNCTIONS
        self.epsilon_values = np.zeros(
            (len(self.epsilon_functions), self.m_n_array_total), dtype=self.dtype
        )

    def size_estimate(self):
        return (self.epsilon_values.size) * maths.xp.dtype(self.dtype).itemsize

    def reserve_memory(self):
        """reserve memory for the arrays.

        Returns:
            int: filesize estimate in bytes.
        """
        # reserve array mem - so that memory errors are raised early
        self.epsilon_values.fill(0)

    def numpyify(self):
        for f, arrs in self.m_n_arrays.items():
            for i, arr in enumerate(arrs):
                if arr is not None:
                    self.m_n_arrays[f][i] = maths.ensure_numpy_array(arr)
        for k, v in self.parameters.items():
            self.parameters[k] = maths.ensure_numpy_array(v)

    def add_m_n_arrays(self, i, arrays):
        for f, arr in zip(self.functions, arrays):
            self.m_n_arrays[f][i] = maths.ensure_numpy_array(arr)
        index = np.unravel_index(i, self.index_shape)
        eps = maths.get_epsilon_at_index(self, index)
        self.epsilon_values[:, i] = eps
        for f in self.functions:
            self.m_n_arrays[f][i] = None


class PlotIterator:
    def __init__(self, results, axes):
        self.results = results
        variable_params = results.variable_params
        extra_axes = [p for p in variable_params if p not in axes]
        self.combination_indices = maths.cartesian_product(
            *[np.arange(len(variable_params[v])) for v in axes]
        )
        if extra_axes:
            self.extra_axes_count = np.product(
                [len(variable_params[p]) for p in extra_axes]
            )
            self.extra_axes_indices = itertools.product(
                *[range(len(variable_params[p])) for p in extra_axes]
            )
        else:
            self.extra_axes_indices = [()]
            self.extra_axes_count = 1

        self.axes = axes
        self.extra_axes = extra_axes
        self.length = self.extra_axes_count

    def __len__(self):
        return self.length

    def __iter__(self):
        return self._iter()

    def _iter(self):
        results, axes = self.results, self.axes
        src_key_params = {p: results.parameters[p] for p in results.parameters}
        axes_i = (
            results.variable_param_indices[axes[0]],
            results.variable_param_indices[axes[1]],
        )
        for i, extra_axes_index in enumerate(self.extra_axes_indices):
            key_params = src_key_params.copy()
            for j, axis in enumerate(self.extra_axes):
                key_params[axis] = results.variable_params[axis][extra_axes_index[j]]
            eps_plots = {}
            for eps_f in maths.EPSILON_FUNCTIONS:
                eps_plots[eps_f] = np.empty(
                    [
                        len(results.variable_params[axes[1]]),
                        len(results.variable_params[axes[0]]),
                    ],
                    dtype=np.complex128,
                )
            for axes_index in self.combination_indices:

                axes_index = list(axes_index)
                extra_axes_i = list(extra_axes_index)
                full_index = []
                for v in results.variable_params:
                    if v in axes:
                        axis_i = axes_index[axes.index(v)]
                        axis_v = results.variable_params[v][axis_i]
                        key_params[v] = axis_v
                        full_index.append(axis_i)
                    else:
                        full_index.append(extra_axes_i.pop(0))
                full_index = tuple(full_index)

                eps = results.get_epsilon_at_index(full_index)
                plot_coord = (full_index[axes_i[1]], full_index[axes_i[0]])
                for j, (eps_f, plot) in enumerate(eps_plots.items()):
                    plot[plot_coord] = eps[j]

            yield key_params, eps_plots


def load_results(pickle_files):
    if not pickle_files:
        raise ValueError("No pickle files specified")
    with open(pickle_files[0], "rb") as f:
        results_dict = pickle.load(f)
    # Type defaults to PickleType.M_N_ARRAYS for backwards compatibility.
    pickle_type = results_dict.get("pickle_type", PickleType.M_N_ARRAYS)
    if pickle_type == PickleType.M_N_ARRAYS:
        cls = Results
        chunked_cls = ChunkedResults
    elif pickle_type == PickleType.EPSILON_VALUES:
        cls = EpsilonResults
        chunked_cls = ChunkedEpsilonResults
    else:
        raise ValueError("Unknown pickle type: %s" % pickle_type)
    if len(pickle_files) == 1:
        print("Loading {}".format(pickle_files[0]))
        results = cls.from_dict(results_dict)
    else:
        all_results = []
        for pickle_file in pickle_files:
            with open(pickle_file, "rb") as f:
                print("Loading {}".format(pickle_file))
                all_results.append(cls.from_dict(pickle.load(f)))
        results = chunked_cls(all_results)
    return results


def readable_filesize(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Pi{suffix}"
