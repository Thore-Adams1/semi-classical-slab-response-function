import time
import os
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start} seconds")
        return result
    return wrapper

def load():
    # Run a cpu-intensive load
    for i in range(1000000):
        for j in range(1000000):
            i * j

@timed
def numpy_load():
    import psutil
    print(psutil.Process().cpu_affinity())
    import numpy as np
    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # os.environ["OPENBLAS_MAIN_FREE"]="1"
    # os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    # os.environ["NUMEXPR_NUM_THREADS"] = "1"
    n = 5000
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    c = np.dot(a, b)

def numpy_mp_load(cpus=4):
    print("\nStarting processes...")
    import multiprocessing as mp
    processes = []
    t1 = time.time()
    for i in range(cpus):
        p = mp.Process(target=numpy_load)
        processes.append(p)
    with ThreadPoolExecutor(max_workers=cpus) as executor:
        for process in processes:
            executor.submit(process.start)
        executor.shutdown(wait=True)    
    t2 = time.time()
    for i in processes:
        i.join()
    t3 = time.time()
    print (f"Multiprocessing took {t3 - t2} seconds (setup: {t2 - t1} seconds)\n")

THREADING_VARS = ("OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OPENBLAS_MAIN_FREE",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS")

# def set_threading_vars():
#     for var in ("OMP_NUM_THREADS",
#     "MKL_NUM_THREADS",
#     "OPENBLAS_NUM_THREADS",
#     "OPENBLAS_MAIN_FREE",
#     "VECLIB_MAXIMUM_THREADS",
#     "NUMEXPR_NUM_THREADS"):
#         os.environ[var] = "1"

# def get_threading_vars()

if __name__ == "__main__":
    # numpy_load()
    # numpy_mp_load(2)
    # Limit any multiprocessing within numpy
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_MAIN_FREE"]="1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    import sys
    from importlib import reload
    for k,v in list(sys.modules.items()):
        if "numpy" in k:
            try:
                reload(v)
            except Exception as e:
                print(v)
                print(e)

            
    numpy_mp_load(6)
    numpy_load()
    