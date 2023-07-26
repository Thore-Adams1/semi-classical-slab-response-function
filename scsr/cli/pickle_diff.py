import numpy as np


def check_arrays_match(array_a, array_b, name="", location=""):
    result = True
    arrays_close = False
    for tol in [0, 1e-18, 1e-15, 1e-12, 1e-10, 1e-6]:
        arrays_close = np.isclose(array_a, array_b, rtol=tol, atol=tol).all()
        if arrays_close:
            break
    if arrays_close:
        if tol:
            msg = "virtually identical (to tolerance {})".format(tol)
        else:
            msg = "identical byte-for-byte"
    else:
        msg = "not identical!"
        result = False
    print("{}arrays {}are {}".format(name + " " if name else "", location, msg))
    return result


ARRAY_KEYS = {"m_n_arrays", "index_array"}


def ensure_pickle_arrays_virtually_identical(
    pickle_path, other_pickle_path, arrays_only=True
):
    import pickle

    with open(pickle_path, "rb") as fa:
        with open(other_pickle_path, "rb") as fb:
            if fa.read() == fb.read():
                print(
                    "{} and {} are byte-for-byte identical!".format(
                        pickle_path, other_pickle_path
                    )
                )
                return True
    with open(pickle_path, "rb") as fa:
        pickle_a = pickle.load(fa)
    with open(other_pickle_path, "rb") as fb:
        pickle_b = pickle.load(fb)
    result = True
    if not arrays_only:
        assert (
            pickle_a.keys() == pickle_b.keys()
        ), "Keys are not identical: {} != {}".format(pickle_a.keys(), pickle_b.keys())
        for key in pickle_a.keys():
            if key in ARRAY_KEYS:
                continue
            elif key == "parameters":
                params_a = pickle_a["parameters"]
                params_b = pickle_b["parameters"]
                assert (
                    params_a.keys() == params_b.keys()
                ), "Param keys are not identical: {} != {}".format(
                    params_a.keys(), params_b.keys()
                )
                for param_key in params_a.keys():
                    value_a, value_b = params_a[param_key], params_b[param_key]
                    if isinstance(value_a, np.ndarray):
                        result = result and check_arrays_match(
                            value_a, value_b, name="parameter: {}".format(param_key)
                        )
                    else:
                        assert value_a == value_b, "parameter: {}: {} != {}".format(
                            param_key, value_a, value_b
                        )
                if result:
                    print("[parameters]: identical")
            else:
                assert pickle_a[key] == pickle_b[key], f"{key} is not identical"
                print(f"[{key}]: identical")
    for (k, o_arrays), (j, n_arrays) in zip(
        pickle_a["m_n_arrays"].items(), pickle_b["m_n_arrays"].items()
    ):
        assert len(o_arrays) == len(
            n_arrays
        ), "pickle a array count ({}) != pickle b array count ({})".format(
            len(o_arrays), len(n_arrays)
        )
        for i, (oa, na) in enumerate(zip(o_arrays, n_arrays)):
            result = (
                check_arrays_match(
                    oa, na, name="[{}]".format(k), location="at index {} ".format(i)
                )
                and result
            )
        for key in ARRAY_KEYS:
            assert k == j, "Keys are not identical. pickle a: {}, pickle b: {}".format(
                pickle_a[key].keys(), pickle_b[key].keys()
            )
    for (k, o_array), (j, n_array) in zip(
        pickle_a["index_arrays"].items(), pickle_a["index_arrays"].items()
    ):
        result = check_arrays_match(oa, na, "[{} INDEX ARRAY]".format(k)) and result
    if result:
        print("\npickles are virtually identical!")
    else:
        print("\npickles differ!")
    return result


def get_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Check whether two pickles from scsr-calc match."
    )
    parser.add_argument("pickle_path", type=str, help="path to pickle file")
    parser.add_argument(
        "other_pickle_path", type=str, help="path to another pickle file"
    )
    parser.add_argument(
        "-A",
        "--arrays-only",
        action="store_true",
        help="only check arrays, not parameters",
    )
    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()
    ensure_pickle_arrays_virtually_identical(args.pickle_path, args.other_pickle_path)
