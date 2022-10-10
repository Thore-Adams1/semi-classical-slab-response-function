#! /usr/bin/env python3
"""This script computes H, G, A1 and A2 electron dynamics matrices

Usage:

    -p [parameters] -v [variables]

    Run A2 & G with 200 steps:
      python phys_code.py A2 G -p steps=200

    Run A2 & G with multiprocessing, with 200 steps and lc = 4 and Kx = (0,1,2,3):
      python phys_code.py A2 G -p lc=4 steps=200 -x -v Kx=0:3

    Run A2 & G with multiprocessing, with 200 steps for 6 equally-spaced w.
    w values from 0-3:
      python phys_code.py A2 G -p steps=200 -x -v w=0:3:6

    Run A2 & G with 100 steps, plot the tild arrays as a func of (phi, theta) (and save them), 
    then write to file:
      python phys_code.py A2 G -p steps=100 --save-figs -w

    Run A2 & G with 100 steps, then write to file called "A2_G_100_steps.pkl"
      python phys_code.py A2 G -p steps=100 -w -o A2_G_100_steps.pkl

Output:
    Files are written as python pickles. Pickles can be read from a new python 
    session using:

        import pickle
        result = pickle.load(open('output.pkl', 'rb'))
"""
# Standard
import ast
import argparse
from bisect import bisect_left
import itertools
import pickle
import os
import datetime
import queue
from collections import defaultdict
import operator
import multiprocessing as mp
import math
from concurrent.futures import ThreadPoolExecutor
from subprocess import getoutput

# Third Party
from tqdm import tqdm
import numpy as np
import numpy as xp

try:
    import cupy as cp
except ImportError:
    cp = None

# Globals
FUNCTIONS = {"A2", "A1", "G", "H"}
KNOWN_PARAMS = {"P", "w", "Kx", "L", "Ln", "tau", "steps", "lc"}
PARAM_DEFAULTS = {
    #  DEFINES SIZE OF FUNCTION MATRICES
    # "m_n_size": 2,
    "mp_batch_size": 1,
    # Calculates max values
    "theta_max": 0.5 * xp.pi,
    "phi_max": 2 * xp.pi,
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
        "The max size of the tile of a m by n function matrix to compute \n" "at once."
    ),
    "mp_batch_size": (
        "The number of function arrays to process before sending them to\n"
        "the main thread (No need to adjust)."
    ),
}


def values_differ(d1, d2, keys):
    """Check if any of the values for the given keys differ between d1 & d2.


        d1 (dict): Dictionary.

        keys (iterable[str]): Keys to check.

    Returns:
        bool: True if values differ, False otherwise.
    """
    for key in keys:
        if d1.get(key) != d2.get(key):
            return True


def op_across_axes(arr_a, arr_b, axes, op):
    """Multiply two arrays along the given axes."""
    return (op(arr_a, arr_b.transpose(axes))).transpose(np.argsort(axes))


def stack_broadcast(arr_a, arr_b):
    """
    Given arr_a (with n dims) and arr_b (with m dims), return a new array with
    n+m dims, with arr_b broadcast across on every new leading axis.
    """
    return xp.broadcast_to(arr_b, (*arr_a.shape, *arr_b.shape))


def stack_op(arr_a, arr_b, op):
    """
    Given arr_a (with n dims) and arr_b (with m dims), return a new array with
    n+m dims, where each axis is a combination of an arr_a value and an arr_b
    value.
    """
    stacked_bcast = stack_broadcast(arr_a, arr_b)
    l = list(range(stacked_bcast.ndim))
    axes = l[arr_a.ndim :] + l[: arr_a.ndim]
    return op_across_axes(arr_a, stacked_bcast, axes, op)


def smul(arr_a, arr_b):
    return stack_op(arr_a, arr_b, operator.mul)


def sadd(arr_a, arr_b):
    return stack_op(arr_a, arr_b, operator.add)


def mul_axes(arr_a, arr_b, axes):
    return op_across_axes(arr_a, arr_b, axes, operator.mul)


def mn_mul(arr_a, arr_b):
    l = list(range(arr_b.ndim))
    axes = l[arr_a.ndim :] + l[: arr_a.ndim]
    return op_across_axes(arr_a, arr_b, axes, operator.mul)


def update_arrays(p, cache):
    C = cache
    last_p = C.get("last_p")
    if not C:
        C["vel"] = calc_velocity(p)
        C["vel_z"] = C["vel"][..., 2]
        C["vel_z_sq"] = C["vel_z"] ** 2
        C["dth_x_dphi_x_Nf_m"] = p["d_theta"] * p["d_phi"] * p["Nf_m"]
        C["theta_sins"] = xp.sin(p["theta_array"]).reshape([-1, 1])

    m_changed = last_p is None or values_differ(p, last_p, ["mc"])
    n_changed = last_p is None or values_differ(p, last_p, ["nc"])
    L_changed = last_p is None or values_differ(p, last_p, ("L",))
    ms, me = p["mc"]
    ns, ne = p["nc"]
    if m_changed:
        C["m"] = xp.arange(ms, me).reshape(-1, 1)
    if n_changed:
        C["n"] = xp.arange(ns, ne).reshape(1, -1)
    if m_changed or n_changed:
        if n_changed or L_changed:
            C["qn"] = C["n"] * xp.pi / p["L"]
            C["Ln"] = p["L"] / (2 - (C["n"] == 0))
            C["qn_sq_vel_z_sq"] = smul(C["qn"] ** 2, C["vel_z_sq"])
        if m_changed or L_changed:
            # x = xp.random.rand(1000,1000,1000)
            # z = x.T
            # z =
            # z.T * xp.array([0,1])
            C["qm"] = C["m"] * xp.pi / p["L"]
            C["qm_vel_z"] = stack_op(C["qm"], C["vel_z"], operator.mul)
            C["neg_1_m"] = (-1) ** C["m"]
            C["1_m_neg_1_m"] = 1 - (-1) ** C["m"]

    # m_changed = last_p is None or p["m"] != last_p["m"]
    k_changed = last_p is None or values_differ(p, last_p, ("Kx", "Ky"))
    w_til_changed = k_changed or values_differ(p, last_p, ("w", "tau"))
    e_exp_changed = w_til_changed or values_differ(p, last_p, ("L",))
    d_changed = e_exp_changed or values_differ(p, last_p, ("P",))
    if k_changed:
        C["K"] = xp.array([p["Kx"], p["Ky"]])
        C["k_dot_v"] = xp.dot(C["vel"][..., :2], C["K"])
        #
        Kx_vel_z = p["Kx"] * C["vel_z"]
        C["Kx_vel_z_1j"] = 1j * Kx_vel_z
        C["k_dot_v_m_Kx_vel_z_1j"] = C["k_dot_v"] - C["Kx_vel_z_1j"]
        C["k_dot_v_p_Kx_vel_z_1j"] = C["k_dot_v"] + C["Kx_vel_z_1j"]
    if k_changed or m_changed:
        C["k_dot_v_p_qm_vel_z"] = C["k_dot_v"] + C["qm_vel_z"]
        C["k_dot_v_m_qm_vel_z"] = C["k_dot_v"] - C["qm_vel_z"]
        # C["qm_sq_vel_z_sq"] = C["qm"] ** 2 * C["vel_z_sq"]
        C["qm_sq_vel_z_sq"] = stack_op(C["qm"] ** 2, C["vel_z_sq"], operator.mul)
    if w_til_changed:
        # calculate omega tild
        C["w_til"] = (
            p["w"]
            + (1j / p["tau"])
            - (p["Kx"] * C["vel"][..., 0] + p["Ky"] * C["vel"][..., 1])
        )
        C["w_til_sq"] = C["w_til"] ** 2
        C["w_til_vel_z"] = C["w_til"] * C["vel_z"]
        C["w_bar_w_til_vel_z"] = p["w_bar"] * C["w_til_vel_z"]
    if e_exp_changed:
        C["e_kx_L"] = xp.exp(-p["Kx"] * p["L"])
        C["e_exp"] = (1j * C["w_til"] * p["L"]) / (C["vel"][..., 2])
        C["e_iwl"] = xp.exp(C["e_exp"])
        C["e_iwl_sq"] = C["e_iwl"] ** 2
        C["e_iwl_cu"] = C["e_iwl_sq"] * C["e_iwl"]
        C["e_iwl_m_e_kx_L"] = C["e_iwl"] - C["e_kx_L"]
        C["e_kx_L_m_e_iwl"] = -C["e_iwl_m_e_kx_L"]

        C["e_iwl_x_e_kx_L"] = C["e_iwl"] * C["e_kx_L"]
        C["e_iwl_sq_x_e_kx_L"] = C["e_iwl_sq"] * C["e_kx_L"]
        C["e_iwl_cu_x_e_kx_L"] = C["e_iwl_cu"] * C["e_kx_L"]
        C["e_iwl_cu_p_e_iwl"] = C["e_iwl_cu"] + C["e_iwl"]

        C["Gts_arr_1"] = C["e_iwl_m_e_kx_L"] * C["e_iwl"]
        C["Gts_arr_3"] = C["e_iwl_sq"] * C["e_kx_L"] - C["e_iwl"]
        C["Gts_arr_4"] = 1 - C["e_iwl"] * C["e_kx_L"]
        C["e_kx_L_m_e_iwl"] = C["e_kx_L"] - C["e_iwl"]
    if w_til_changed or m_changed:
        C["w_til_m_qm_vel_z"] = C["w_til"] - C["qm_vel_z"]
        C["w_til_p_qm_vel_z"] = C["w_til"] + C["qm_vel_z"]
        C["w_til_sq_m_qm_sq_vel_z_sq"] = C["w_til_sq"] - C["qm_sq_vel_z_sq"]
    if w_til_changed or k_changed:
        C["w_til_m_k_dot_v"] = C["w_til"] * C["k_dot_v"]
        C["w_til_m_Kx_vel_z_1j"] = C["w_til"] - C["Kx_vel_z_1j"]
        C["w_til_p_Kx_vel_z_1j"] = C["w_til"] + C["Kx_vel_z_1j"]
        C["G_A"] = C["k_dot_v_p_Kx_vel_z_1j"] / C["w_til_m_Kx_vel_z_1j"]
        C["G_B"] = C["k_dot_v_m_Kx_vel_z_1j"] / C["w_til_p_Kx_vel_z_1j"]
        C["G_D"] = C["G_B"] - C["G_A"]

        C["H_a"] = C["k_dot_v_p_Kx_vel_z_1j"] ** 2 / (C["w_til_m_Kx_vel_z_1j"])
        C["H_b"] = ((C["k_dot_v_m_Kx_vel_z_1j"]) ** 2) / (C["w_til_p_Kx_vel_z_1j"])
        C["H_c"] = C["H_a"] + C["H_b"]
    if w_til_changed or k_changed or m_changed:
        C["A1_a"] = C["w_til_m_k_dot_v"] + C["qm_sq_vel_z_sq"]
        C["A1_fac1"] = (C["k_dot_v_p_qm_vel_z"]) ** 2 / (C["w_til_m_qm_vel_z"]) + (
            C["k_dot_v_m_qm_vel_z"]
        ) ** 2 / (C["w_til_p_qm_vel_z"])
        C["A2_fac1"] = (
            C["k_dot_v_p_qm_vel_z"] / C["w_til_m_qm_vel_z"]
            + C["k_dot_v_m_qm_vel_z"] / C["w_til_p_qm_vel_z"]
        )
        C["G_C"] = C["G_A"] + smul(C["neg_1_m"] * C["e_kx_L"], C["G_B"])

    if e_exp_changed or m_changed:
        C["G_fac_4"] = 2 * C["e_iwl"] - smul(C["neg_1_m"], (C["e_iwl_sq"] + 1))
        C["neg_m_e_iwl"] = 1 - smul(C["neg_1_m"], C["e_iwl"])
        C["At2s_fac1"] = 2 * C["e_iwl"] - smul(C["neg_1_m"], (C["e_iwl_sq"] + 1))
        neg_1_m_e_iwl_cu_p_e_ewl = smul(C["neg_1_m"], C["e_iwl_cu_p_e_iwl"])
        C["A1_fac2"] = (
            2
            * C["w_bar_w_til_vel_z"]
            * C["A1_a"]
            / (
                (C["w_til_sq"] - C["qn_sq_vel_z_sq"])
                * (C["w_til_sq"] - C["qm_sq_vel_z_sq"])
            )
        )

        C["Hts1_a"] = 2 * C["e_iwl_sq"] - neg_1_m_e_iwl_cu_p_e_ewl
        C["Hts1_b"] = (C["e_iwl_x_e_kx_L"] - C["e_iwl_sq"]) + smul(
            C["neg_1_m"], C["e_iwl_m_e_kx_L"] * C["e_iwl_sq"]
        )
        C["Hts1_c"] = (C["e_iwl_cu_x_e_kx_L"] - C["e_iwl_sq"]) + smul(
            C["neg_1_m"], (C["e_iwl"] - C["e_iwl_sq_x_e_kx_L"])
        )

    if e_exp_changed or m_changed or k_changed:
        C["Phim0"] = 1 - C["neg_1_m"] * C["e_kx_L"]
    if e_exp_changed or m_changed or k_changed or w_til_changed:
        C["H_d"] = C["neg_m_e_iwl"] * (
            C["A1_fac1"]
            - C["H_a"]
            - smul(C["neg_1_m"] * C["e_kx_L"], C["H_b"])
            + smul(C["Phim0"], C["k_dot_v_m_Kx_vel_z_1j"])
        )
    if d_changed:
        # calcuate d
        C["d"] = 1 / (1 - (p["P"] ** 2 * C["e_iwl_sq"]))
        C["d_P"] = p["P"] * C["d"]
        C["P_e_iwl"] = p["P"] * C["e_iwl"]
    C["last_p"] = p.copy()


# @profile
def compute_functions(functions, p, cache, result_only=False):
    """Compute a function. MORE DETAIL COULD HELP - MAYBE A BETTER NAME?
    - Calculates each function as a function of Vf, theta and phi
    - Perfoms the integral by multiplying by sin(theta)
    Args:
        functions (list[str]): Functions to compute. Can include ("H", "A1", "A2", "G").
        p (dict[str:object]): parameters used for calculations.
        C (dict[str:xp.ndarray]): Arrays used for calculations.
    Returns:
        dict[str:dict[str:xp.ndarray]]: Arrays of the functions in the format:
            {
                FUNCTION_NAME: {
                    "array": xp.ndarray[theta*phi],
                    "integral": xp.ndarray[theta*phi],
                },
                ...
            }
    """
    # return {f: {"result":0} for f in functions}
    w_bar = p["w_bar"]
    update_arrays(p, cache)
    C = cache
    P, L, Ln, Kx, wp, Vf = (
        p["P"],
        p["L"],
        C["Ln"],
        p["Kx"],
        p["wp"],
        p["Vf"],
    )

    all_arrays = {Kx: {} for Kx in functions}
    m, qm, n, qn = C["m"], C["qm"], C["n"], C["qn"]
    e_iwl = C["e_iwl"]
    k_dot_v = C["k_dot_v"]
    vel_z = C["vel_z"]
    neg_1_n = (-1) ** n

    if "G" in functions or "A2" in functions or "A1" in functions or "H" in functions:
        Ln = L / (2 - (n == 0))
        symmetry = (1 + (-1) ** (m + n)) / 2
        symmetry_vel_z = smul(symmetry, vel_z)
        A2_fac2 = symmetry_vel_z * C["A2_fac1"]
        At2b = C["neg_m_e_iwl"] * A2_fac2
        At2s = C["At2s_fac1"] * A2_fac2
        Atilde2 = At2b + At2s * ((sadd(neg_1_n, C["P_e_iwl"]) * (C["d_P"])))
        all_arrays["A2"] = {}
        all_arrays["A2"]["array"] = Atilde2

    if "G" in functions:
        fac2 = symmetry_vel_z * C["G_C"]
        fac3 = symmetry_vel_z * C["G_D"]

        Gtb = (A2_fac2 - fac2) * C["neg_m_e_iwl"] - mn_mul(
            (1 - C["neg_1_m"] * C["e_kx_L"]), fac3
        )

        Gts = (
            A2_fac2 * C["G_fac_4"]
            + (
                C["G_A"] * (C["e_kx_L_m_e_iwl"] + smul(neg_1_n, C["Gts_arr_1"]))
                + C["G_B"] * (C["Gts_arr_3"] + smul(neg_1_n, C["Gts_arr_4"]))
            )
            * symmetry_vel_z
        )

        Gtilde = Gtb + Gts * ((C["d_P"]) * sadd(neg_1_n, C["P_e_iwl"]))
        all_arrays["G"]["array"] = Gtilde

    if "A1" in functions or "H" in functions:
        w_til_sq_m_qn_sq_vel_z_sq = C["w_til_sq"] - C["qn_sq_vel_z_sq"]
        A1_divisor1 = w_til_sq_m_qn_sq_vel_z_sq * C["w_til_sq_m_qm_sq_vel_z_sq"]
        A1_b = C["w_til_vel_z"] / w_til_sq_m_qn_sq_vel_z_sq
        At1b = mn_mul((n == m) * (Ln / 1j), C["A1_fac1"]) - mn_mul(
            symmetry, C["neg_m_e_iwl"]
        ) * A1_b * (C["A1_fac1"] + k_dot_v)

        At1s1 = mn_mul(-symmetry, C["A1_fac2"]) * (
            2 * C["e_iwl_sq"] - smul(C["neg_1_m"], C["e_iwl_cu_p_e_iwl"])
        )
        e_iwl_n = 1 - smul(neg_1_n, e_iwl)
        e_iwl_n_fac5 = e_iwl_n * C["neg_m_e_iwl"]
        At1s2 = mn_mul(symmetry, e_iwl_n_fac5)

        Atilde1 = At1b + (C["d_P"]) * (P * At1s1 + At1s2)

        all_arrays["A1"] = {}
        all_arrays["A1"]["array"] = Atilde1

    if "H" in functions:
        H_c = C["H_c"]
        H_d = C["H_d"]

        Htb = (
            mn_mul((n == m) * (Ln / 1j), C["A1_fac1"])
            - mn_mul(symmetry, A1_b) * H_d
            + smul(symmetry * (1j * Kx * C["Phim0"] / (Kx**2 + qn**2)), H_c)
        )

        fac1 = 2 * C["A1_a"] / A1_divisor1

        fac2 = C["k_dot_v_p_Kx_vel_z_1j"] / (
            w_til_sq_m_qn_sq_vel_z_sq * C["w_til_m_Kx_vel_z_1j"]
        )

        fac3 = (C["k_dot_v_m_Kx_vel_z_1j"]) / (
            w_til_sq_m_qn_sq_vel_z_sq * C["w_til_p_Kx_vel_z_1j"]
        )
        symmetry_w_bar_w_til_vel_z = smul(symmetry, C["w_bar_w_til_vel_z"])
        Hts1 = -symmetry_w_bar_w_til_vel_z * (
            fac1 * C["Hts1_a"] + fac2 * C["Hts1_b"] + fac3 * C["Hts1_c"]
        )

        neg_1_m_m_e_iwl_n = mn_mul(C["neg_1_m"], e_iwl_n)
        Hts2 = symmetry_w_bar_w_til_vel_z * (
            fac1 * e_iwl_n_fac5
            - neg_1_m_m_e_iwl_n * fac2 * C["e_kx_L_m_e_iwl"]
            - fac3 * e_iwl_n * (1 - C["e_iwl_x_e_kx_L"])
        )

        Htilde = Htb + C["d_P"] * (P * Hts1 + Hts2)

        all_arrays["H"] = {}
        all_arrays["H"]["array"] = Htilde

    # calculate the phi*theta integrals for each of the functions
    fA = (1j * w_bar / Ln) * (wp / Vf) ** 2 * (3 / (4 * xp.pi) ** 2)
    fGH = fA * 4 * xp.pi / (Kx * Kx + qm**2)
    for func_name, func_arrays in all_arrays.items():
        func_arrays["integral"] = func_arrays["array"] * C["theta_sins"]
        func_arrays["result"] = (
            xp.sum(func_arrays["integral"], axis=(2, 3)) * C["dth_x_dphi_x_Nf_m"]
        )
        if func_name in ("G", "H"):
            func_arrays["result"] *= fGH
        else:
            func_arrays["result"] *= fA
        # func_arrays["result"] = complex(func_arrays["result"])
        if result_only:
            # Clear some memory if these aren't needed
            del func_arrays["array"]
            del func_arrays["integral"]
    return all_arrays


def calc_velocity(p):
    """Get velocity in x, y and z.

    Args:
        params (float): parameters.

    Returns:
        xp.ndarray[theta*phi*3]: x, y and z velocity.
    """
    sin_theta_array = xp.sin(p["theta_array"])[:, xp.newaxis]
    velocity = xp.full((len(p["theta_array"]), len(p["phi_array"]), 3), float(p["Vf"]))
    velocity[..., 0] *= xp.cos(p["phi_array"]) * sin_theta_array
    velocity[..., 1] *= xp.sin(p["phi_array"]) * sin_theta_array
    velocity[..., 2] *= xp.cos(p["theta_array"])[:, xp.newaxis]
    return velocity


class ResultsBase:
    def __init__(self):
        self.parameters = {}
        self.variable_params = {}

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


class ResultsStorage(ResultsBase):
    def __init__(self, functions, parameters, variable_params):
        super().__init__()
        self.functions = functions
        self.parameters = parameters
        self.variable_params = variable_params
        self.m_n_arrays = []
        self.index_arrays = {}
        self.processing_time = 0

    def as_dict(self, **kwargs):
        kwargs.update(
            {
                "functions": self.functions,
                "parameters": self.parameters,
                "variable_params": self.variable_params,
                "m_n_arrays": self.m_n_arrays,
                "index_arrays": self.index_arrays,
            }
        )
        return kwargs

    def get_m_n_array_from_values(self, function, iteration_params):
        index = []
        for k, v in iteration_params.items():
            index.append(self.variable_params[k].index(v))
        index = tuple(index)
        return self.m_n_arrays[function][self.index_arrays[function][index]]

    def get_m_n_array_from_index(self, function, index):
        return self.m_n_arrays[function][self.index_arrays[function][tuple(index)]]

    @classmethod
    def from_dict(cls, dictionary):
        instance = cls(
            dictionary["functions"],
            dictionary["parameters"],
            dictionary["variable_params"],
        )
        instance.m_n_arrays = dictionary["m_n_arrays"]
        instance.index_arrays = dictionary["index_arrays"]
        return instance


class CombinedResults(ResultsBase):
    """Combine the results of a set of runs.

    Args:
        results (list[ResultsStorage]): list of results.
    """

    def __init__(self, results):
        super().__init__()
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


class ResultsProcessor(ResultsStorage):
    def __init__(self, functions, parameters, variable_params, dtype=np.complex128):
        super().__init__(functions, parameters, variable_params)
        self.dtype = dtype
        if not variable_params:
            raise ValueError("No variable parameters provided.")
        m_n_array_total = 1
        for v in variable_params.values():
            m_n_array_total *= len(v)
        self.m_n_arrays = {f: [None] * m_n_array_total for f in functions}
        self.m_n_array_sizes = [None] * m_n_array_total
        self.index_arrays = {k: {} for k in functions}
        assert variable_params, "No variable parameters specified!"
        param_axes = [len(v) for v in variable_params.values()]
        for array_name in functions:
            self.index_arrays[array_name] = index_array = xp.empty(
                param_axes, dtype=int
            )
            index_array[:] = -1

        # --- SET UP INTEGRAL ARRAYS (m*n) ---
        self.iteration_total = 0
        given_lc = self.parameters.get("lc")
        max_m_n_size = 1
        for i, values in enumerate(self.parameter_combinations()):
            iteration_params = {k: v for k, (_, v) in zip(self.variable_params, values)}
            p = self.parameters.copy()
            p.update(iteration_params)
            if given_lc is None:
                lc = 10 * p["Kx"] * p["L"] / (2 * xp.pi)
                m_n_size = 2 * int(xp.ceil(lc))
            else:
                m_n_size = 2 * int(xp.ceil(given_lc))
            max_m_n_size = max(max_m_n_size, m_n_size)
            index = []
            for j, _ in values:
                index.append(j)
            self.m_n_array_sizes[i] = m_n_size
            for function in functions:
                self.index_arrays[function][tuple(index)] = i

            self.iteration_total += m_n_size**2
        self.parameters["max_m_n_size"] = max_m_n_size

    def reserve_memory(self):
        # reserve array mem - so that memory errors are raised early
        for function in self.functions:
            for i, m_n_array in enumerate(self.m_n_arrays[function]):
                if m_n_array is None:
                    m_n_size = self.m_n_array_sizes[i]
                    self.m_n_arrays[function][i] = arr = np.empty(
                        (m_n_size, m_n_size), dtype=self.dtype
                    )
                    arr.fill(0)

    def get_tasks(self):
        f = next(iter(self.functions), None)
        for i, values in enumerate(self.parameter_combinations()):
            # A unique combination of variable parameters.
            iteration_params = {k: v for k, (_, v) in zip(self.variable_params, values)}
            iteration_params["i"] = i
            m_max = n_max = self.m_n_array_sizes[i]
            iteration_params["mn"] = m_max  # , n_max
            yield iteration_params

    def numpyify(self):
        for f, arrs in self.m_n_arrays.items():
            for i, arr in enumerate(arrs):
                if arr is not None:
                    self.m_n_arrays[f][i] = ensure_numpy_array(arr)
        for f, arr in self.index_arrays.items():
            self.index_arrays[f] = ensure_numpy_array(arr)

    def add_m_n_array(self, function, i, array):
        self.m_n_arrays[function][i] = ensure_numpy_array(array)

    def add_m_n_arrays(self, i, arrays):
        for f, arr in zip(self.functions, arrays):
            self.m_n_arrays[f][i] = ensure_numpy_array(arr)


def ensure_numpy_array(arr):
    if cp is not None and isinstance(arr, cp.ndarray):
        return arr.get()
    return arr


def worker_calculate(
    param_queue, result_queue, progress, functions, parameters, dtype, process_id=None
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
                xp.full([iteration_params["mn"]] * 2, -1, dtype=dtype)
                for _ in functions
            ]
            for chunk, arrays in process_chunks(params, functions, max_tile_size, C):
                progress.value += (chunk[1] - chunk[0]) * (chunk[3] - chunk[2])
                for i, arr in enumerate(mn_arrays):
                    arr[chunk[0] : chunk[1], chunk[2] : chunk[3]] = arrays[i]
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
        import cupy as cp

        globals()["xp"] = cp
        cp.cuda.Device(gpu_id).use()
    if dtype is None:
        dtype = xp.complex128
    parameters, _ = get_parameters(args)
    worker_calculate(
        param_queue,
        result_queue,
        progress,
        functions,
        parameters,
        dtype,
        process_id=process_id,
    )


def process_chunks(params, functions, chunk_size, cache):
    p = params
    mn = params["mn"]
    p["w_bar"] = p["w"] + (1j / p["tau"])
    for chunk in tile_2d_arr(mn, mn, *chunk_size):
        ms, me, ns, ne = chunk
        p["mc"] = ms, me
        p["nc"] = ns, ne
        chunk_result = compute_functions(functions, p, cache, result_only=True)
        arrs = [chunk_result[f]["result"] for f in functions]
        yield (ms, me, ns, ne), arrs


# @profile
def main(args):
    """Main function."""
    start_time = datetime.datetime.now()

    gpus_to_use = []
    if args.gpu:
        if cp is None:
            raise RuntimeError(
                "Couldn't import CuPy. Required for GPU.\n"
                "See: https://docs.cupy.dev/en/stable/install.html"
            )
        else:
            globals()["xp"] = cp
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

    result_proc = ResultsProcessor(
        list(args.functions), params, variable_params, dtype=dtype
    )

    iterations = result_proc.get_tasks()
    expected_file_size = (
        result_proc.iteration_total
        * len(args.functions)
        * xp.dtype(dtype).itemsize
        / 1024
        / 1024
        / 1024
    )
    print(
        f"Expected File Size: {expected_file_size:.1f} GB (Data Type: complex{args.dtype})"
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
                xp.full([mn] * 2, -1, dtype=xp.complex128)
                for _ in result_proc.functions
            ]
            for chunk, arrays in process_chunks(
                p, result_proc.functions, max_tile_size, C
            ):
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
                    result_proc.functions,
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
                        for name, array in zip(result_proc.functions, mn_arrays):
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
    params["theta_array"] = xp.linspace(0, params["theta_max"], params["steps"])
    params["phi_array"] = xp.linspace(0, params["phi_max"], params["steps"])

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
        if chunked_values is None:
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
                all_values.extend(xp.linspace(start, end, 1 + end - start))
        else:
            all_values.extend(xp.linspace(start, end, steps))
    return param, all_values


def get_parser():
    indent_trailing = lambda text: f"{text}".replace("\n", f"\n\t\t")
    param_descs = PARAM_DESCRIPTIONS.copy()
    all_params = set(KNOWN_PARAMS) | set(PARAM_DEFAULTS)
    no_desc = set()
    for param in all_params:
        if param not in param_descs:
            no_desc.add(param)
    no_desc_text = f"{', '.join(no_desc)}\n\t"
    param_text = "Parameters:\n\t{}{}".format(
        no_desc_text,
        "\n\t".join(
            f"{param}: {indent_trailing(param_descs[param])}" for param in param_descs
        ),
    )
    parser = argparse.ArgumentParser(
        description=f"{__doc__}\n{param_text}",
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
    arguments = get_parser().parse_args()
    set_arg_defaults(arguments)
    main(arguments)
