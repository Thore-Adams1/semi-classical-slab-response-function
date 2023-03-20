# Standard
import operator

# Third Party
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

# Globals
xp = np
FUNCTIONS = {"A2", "A1", "G", "H"}
KNOWN_PARAMS = {"P", "w", "Kx", "L", "Ln", "tau", "steps", "lc"}
EPSILON_FUNCTIONS = ("epsp", "epsm", "Hinvp", "Hinvm", "Ptilde")
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


def set_gpu_mode(enabled):
    """Set whether to use GPU or not.

    Args:
        enabled (bool): True to use GPU, False to use CPU.
    """
    if enabled:
        if cp is None:
            raise RuntimeError(
                "Couldn't import CuPy. Required for GPU.\n"
                "See: https://docs.cupy.dev/en/stable/install.html"
            )
        else:
            globals()["xp"] = cp
    else:
        globals()["xp"] = np


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

        C["Hts1_a"] = 2 * C["e_iwl_sq"] - neg_1_m_e_iwl_cu_p_e_ewl
        C["Hts1_b"] = (C["e_iwl_x_e_kx_L"] - C["e_iwl_sq"]) + smul(
            C["neg_1_m"], C["e_iwl_m_e_kx_L"] * C["e_iwl_sq"]
        )
        C["Hts1_c"] = (C["e_iwl_cu_x_e_kx_L"] - C["e_iwl_sq"]) + smul(
            C["neg_1_m"], (C["e_iwl"] - C["e_iwl_sq_x_e_kx_L"])
        )
    if e_exp_changed or m_changed or n_changed:
        C["A1_fac2"] = (
            2
            * C["w_bar_w_til_vel_z"]
            * C["A1_a"]
            / (
                (C["w_til_sq"] - C["qn_sq_vel_z_sq"])
                * (C["w_til_sq"] - C["qm_sq_vel_z_sq"])
            )
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


def ensure_numpy_array(obj):
    if cp is not None and isinstance(obj, cp.ndarray):
        # get numpy array from cupy array
        return obj.get()
    return obj


def get_epsilon_at_index(results, index):
    G = results.get_m_n_array_from_index("G", index)
    H = results.get_m_n_array_from_index("H", index)

    tau = results.get_param_at_index("tau", index)
    w_bar = results.get_param_at_index("w", index) + 1j / tau

    H_plus = np.matrix(H[0::2, 0::2]).T
    H_minus = np.matrix(H[1::2, 1::2]).T

    G_plus = np.matrix(G[::2, ::2])
    G_minus = np.matrix(G[1::2, 1::2])

    # Create required arrays from output arrays
    Z_plus = np.ones([H_plus.shape[0]])
    Z_plus[0] = 1 / 2
    Z_plus_matrix = np.matrix(Z_plus).T
    Z_minus = np.ones([H_plus.shape[0]])
    Z_minus_matrix = np.matrix(Z_minus).T

    G_vec_plus = np.matrix(G_plus[:, 0] * 2)
    G_vec_minus = np.matrix(G_minus[:, 0] / Z_minus)

    Iden = np.identity(H_plus.shape[0])

    iden_w_sq = np.matrix(Iden * w_bar**2)

    Hinvp = np.linalg.inv(iden_w_sq - H_plus)
    Hinvm = np.linalg.inv(iden_w_sq - H_minus)

    # Calculate epsilon
    epsp = (
        1 - G_vec_plus.T * Hinvp * Z_plus_matrix
    )  # The poles of this function give symmetric SPWs.
    epsm = (
        1 - G_vec_minus.T * Hinvm * Z_minus_matrix
    )  # The poles of this function give anti-symmetric SPWs

    sign_p, slog_p = np.linalg.slogdet(Hinvp)
    sign_m, slog_m = np.linalg.slogdet(Hinvm)

    # VPW
    Fp = sign_p * np.exp(slog_p)
    Fm = sign_m * np.exp(slog_m)

    # Calculating P tilde:
    A1 = results.get_m_n_array_from_index("A1", index)
    A2 = results.get_m_n_array_from_index("A2", index)
    A = A1 - A2

    L = results.get_param_at_index("L", index)

    # A_plus = np.matrix(A[::2, ::2]).T
    A_minus = np.matrix(A[1::2, 1::2]).T

    chi_plus = np.array(np.linalg.inv(iden_w_sq - H_plus - G_vec_plus) * A_minus)
    # chi_minus = np.array(np.linalg.inv(iden_w_sq - H_minus - G_vec_minus) * A_minus)

    # Defining constants
    k = 2

    # Inputting the dimensions of chi matrix
    m_max, n_max = chi_plus.shape

    # Calculating q_n and q_m
    m = np.arange(m_max).reshape(-1, 1)
    n = np.arange(n_max).reshape(1, -1)
    q_m = np.pi * m / L
    q_n = np.pi * n / L

    # Calculating L_m
    L_m = L / (2 - (m == 0).astype(int))

    # Calculating P_n
    exp_k_L = np.exp(-k * L)
    P_n = np.zeros((1, n_max), dtype=results.get_dtype())
    P_n[0, :] = (
        chi_plus.T * ((1 - (-1) ** m * exp_k_L) / (L_m * (k**2 + q_m**2)))
    ).sum(axis=0)

    P_tilde = np.sum(
        2 * np.pi * (k * P_n) / (k**2 + q_n**2) * (1 - (-1) ** n * exp_k_L)
    )

    return (
        # It's critical that the order of the terms here matches the order
        # of EPSILON_FUNCTIONS.
        epsp[0, 0],
        epsm[0, 0],
        Fp,
        Fm,
        P_tilde,
    )
