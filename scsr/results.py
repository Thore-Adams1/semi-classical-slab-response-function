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
    EPSILON_PLOTS = 1


class ResultsBase:
    def __init__(self, parameters, variable_params):
        self.parameters = dict(parameters)
        self.variable_params = dict(variable_params)
        self._variable_param_indices = {k: i for i,k in enumerate(variable_params)}
        self.index_shape = tuple(len(v) for v in self.variable_params.values())

    def as_dict(self, **kwargs):
        kwargs.update(
            {
                "parameters": self.parameters,
                "variable_params": self.variable_params,
            }
        )
        return kwargs

    def parameter_combinations(self):
        # --- LOOP OVER CARTESIAN PRODUCT OF VARIABLE PARAMETERS ---
        enumerated_values = (
            enumerate(values) for values in self.variable_params.values()
        )
        return itertools.product(*enumerated_values)

    def param_combination_count(self):
        m_n_array_total = 1
        for v in self.variable_params.values():
            m_n_array_total *= len(v)
        return m_n_array_total

    def get_m_n_array_from_index(self, function, index):
        raise NotImplementedError("Must be implemented in subclass")

    def get_param_at_index(self, param_name, index):
        param_value = self.parameters.get(param_name)
        if param_value is not None:
            return param_value
        i = index[self._variable_param_indices[param_name]]
        return self.variable_params[param_name][i]

    @classmethod
    def from_dict(cls, dictionary):
        instance = cls(
            dictionary["parameters"],
            dictionary["variable_params"],
        )
        instance.processing_time = dictionary.get("processing_time", 0)
        return instance


class ResultsStorage(ResultsBase):
    def __init__(self, functions, parameters, variable_params):
        self.functions = functions
        self.m_n_arrays = []
        self.processing_time = 0
        super().__init__(parameters, variable_params)

    def as_dict(self, **kwargs):
        kwargs.update(super().as_dict())
        kwargs.setdefault("pickle_type", PickleType.M_N_ARRAYS)
        kwargs.update(
            {
                "functions": self.functions,
                "m_n_arrays": self.m_n_arrays,
            }
        )
        return kwargs

    def get_m_n_array_from_values(self, function, iteration_params):
        index = []
        for k, v in iteration_params.items():
            index.append(self.variable_params[k].index(v))
        index = tuple(index)
        return self.m_n_arrays[function][np.ravel_multi_index(index, self.index_shape)]

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
        instance.processing_time = dictionary.get("processing_time", 0)
        return instance


class CombinedResults(ResultsBase):
    """Combine the results of a set of runs.

    Args:
        results (list[ResultsStorage]): list of results.
    """

    def __init__(self, results):
        results = list(results)
        self.variable_params = defaultdict(list)
        self.chunked_param = None
        if len(results) < 2:
            raise ValueError(
                "No or only one result provided. Just use ResultsStorage "
                "directly - cba to test this case"
            )
        self.parameters = results[0].parameters
        self.functions = results[0].functions
        valid_variable_params = set(results[0].variable_params)
        self.chunk_ends_to_result = {}
        result_chunk_maxes = []
        for i, result in enumerate(results):
            if i == 0:
                for param_name, values in result.variable_params.items():
                    self.variable_params[param_name].extend(values)
                continue
            else:
                if set(result.variable_params) != valid_variable_params:
                    raise ValueError("Results must have the same variable parameters.")
            for param_name, values in result.variable_params.items():
                if values != results[0].variable_params[param_name]:
                    if self.chunked_param is None:
                        self.chunked_param = param_name
                    else:
                        if self.chunked_param != param_name:
                            raise ValueError("Cannot parse chunks on multiple axes.")
            if self.chunked_param is None:
                raise ValueError("No chunked parameters found.")

        self.variable_params[self.chunked_param] = []
        for result in results:
            values = result.variable_params[self.chunked_param]
            result_chunk_maxes.append((max(values), result))
            self.variable_params[self.chunked_param].extend(values)

        for k, v in self.variable_params.items():
            self.variable_params[k] = sorted(v)
        self.ordered_chunk_ends = []
        self.results = []
        for chunk_max, result in sorted(result_chunk_maxes, key=lambda x: x[0]):
            self.ordered_chunk_ends.append(chunk_max)
            self.results.append(result)
        super().__init__(self.parameters, self.variable_params)

    def _get_chunked_index(self, index):
        value = self.variable_params[self.chunked_param][index]
        chunk_end_index = bisect_left(self.ordered_chunk_ends, value)
        mapped_result = self.results[chunk_end_index]
        return mapped_result, mapped_result.variable_params[self.chunked_param].index(
            value
        )

    def get_m_n_array_from_values(self, function, values):
        mapped_index = []
        mapped_result = None
        for v, value in zip(self.variable_params, values):
            i = self.variable_params[v].index(value)
            if v == self.chunked_param:
                mapped_result, mapped_partial_index = self._get_chunked_index(i)
                mapped_index.append(mapped_partial_index)
            else:
                mapped_index.append(i)
        if mapped_result is None:
            raise ValueError("No result found for values {}".format(values))
        return mapped_result.get_m_n_array_from_index(function, tuple(mapped_index))

    def get_m_n_array_from_index(self, function, index):
        mapped_index = []
        mapped_result = None
        for v, i in zip(self.variable_params, index):
            if v == self.chunked_param:
                mapped_result, mapped_partial_index = self._get_chunked_index(i)
                mapped_index.append(mapped_partial_index)
            else:
                mapped_index.append(i)
        if mapped_result is None:
            raise ValueError("No result found for index {}".format(index))
        return mapped_result.get_m_n_array_from_index(function, tuple(mapped_index))


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


class ProcessorMixin():
    def get_tasks(self):
        f = next(iter(self.functions), None)
        for i, values in enumerate(self.parameter_combinations()):
            # A unique combination of variable parameters.
            iteration_params = {k: v for k, (_, v) in zip(self.variable_params, values)}
            iteration_params["i"] = i
            m_max = n_max = self.m_n_array_sizes[i]
            iteration_params["mn"] = m_max  # , n_max
            yield iteration_params


class ResultsProcessor(ResultsStorage, ProcessorMixin):
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
        self.m_n_array_sizes, self.parameters["max_m_n_size"], self.iteration_total = calculate_m_n_sizes(self)

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
        return (
            (self.iteration_total * len(self.functions)) * maths.xp.dtype(self.dtype).itemsize
        )


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


class EpsilonResultsStorage(ResultsStorage):
    def as_dict(self, **kwargs):
        kwargs.update(super().as_dict(pickle_type=PickleType.EPSILON_VALUES))
        kwargs["epsilon_functions"] = self.epsilon_functions
        kwargs["epsilon_values"] = self.epsilon_values
        del kwargs["m_n_arrays"]
        return kwargs

    @classmethod
    def from_dict(cls, dictionary):
        instance = cls(
            dictionary["functions"],
            dictionary["parameters"],
            dictionary["variable_params"],
        )
        instance.processing_time = dictionary.get("processing_time", 0)
        instance.epsilon_values = dictionary["epsilon_values"]
        instance.epsilon_functions = dictionary["epsilon_functions"]
        return instance


class EpsilonResultsProcessor(EpsilonResultsStorage, ProcessorMixin):
    EPSILON_FUNCTIONS = ["epsp", "epsm", "Hinvp", "Hinvm"]
    def __init__(self, functions, parameters, variable_params, dtype=np.complex128):
        super().__init__(functions, parameters, variable_params)
        self.functions = functions
        self.dtype = dtype
        if not variable_params:
            raise ValueError("No variable parameters provided.")
        self.m_n_array_total = 1
        for v in variable_params.values():
            self.m_n_array_total *= len(v)
        self.m_n_arrays = {f: [None] * self.m_n_array_total for f in functions}
        self.m_n_array_sizes = [None] * self.m_n_array_total
        self.m_n_array_sizes, self.parameters["max_m_n_size"], self.iteration_total = calculate_m_n_sizes(self)
        self.epsilon_functions = EpsilonResultsProcessor.EPSILON_FUNCTIONS
        self.epsilon_values = np.zeros((len(EpsilonResultsProcessor.EPSILON_FUNCTIONS), self.m_n_array_total), dtype=self.dtype)

    def as_dict(self, **kwargs):
        dictionary = super().as_dict(pickle_type=PickleType.EPSILON_VALUES)
        dictionary["epsilon_functions"] = self.epsilon_functions
        dictionary["epsilon_values"] = self.epsilon_values
        return dictionary

    def reserve_memory(self):
        """reserve memory for the arrays.

        Returns:
            int: filesize estimate in bytes.
        """
        # reserve array mem - so that memory errors are raised early
        self.epsilon_values.fill(0)
        return (self.epsilon_values.size) * maths.xp.dtype(self.dtype).itemsize

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
        self.epsilon_values[:, i] = [eps[k] for k in self.epsilon_functions]
        for f in self.functions:
            self.m_n_arrays[f][i] = None


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


def readable_filesize(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Pi{suffix}"
