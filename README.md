# Semi-classical Slab Response Function

> Add description of repo.

## Requirements

Install the required dependencies:

```bash
$ pip install -r requirements.txt
```

If intending to run on the gpu, [`CuPy`](https://cupy.dev/) and it's required dependencies must also be installed.

## Development

All the python code is formatted using [`black`](https://pypi.org/project/black/). 
Automatic formatting can be set up as a git pre-commit hook by running:

```bash
$ git-scripts/install-git-hooks
```

Alternatively, `black` can be run manually using:

```bash
black scsr/ *.py
```

### Running tests

To run some tests and compare against the original matlab logic, first set the following environment variables:

- `$MATLAB_EXECUTABLE`: The path to the matlab executable on your machine.
  Examples: `'/usr/local/bin/matlab'` or `'C:/Program Files/MATLAB/R2022a/bin/matlab.exe'`
- `$MATLAB_SCRIPTS_PATH`: The path to directory containing the original logic for
  this code. This directory must include `get_matelement.m` and dependent functions.
  
Then, to run the tests:

```sh
$ test/test_thesis_code_against_matlab.sh
```

## Running

```
$ python thesis_code.py --help
usage: thesis_code.py [-h] [-p P=V [P=V ...]] [-v P=V1,V2,..,VN [P=V1,V2,..,VN ...]] [-d {64,128,256}] [-x] [-g] [-G GPU_IDS [GPU_IDS ...] | -A] [-m SUBPROCESS_COUNT] [-w] [-o OUTPUT] [-f] [-u seconds]
                      [-C CHUNKS] [-P CHUNK_PARAMETER] [-I CHUNK_ID]
                      [{A1,G,A2,H} ...]

This script computes H, G, A1 and A2 electron dynamics matrices

Usage:
    -p [parameters] -v [variables]

    Run A2 & G with 200 steps:
      python thesis_code.py A2 G -p steps=200

    Run A2 & G with multiprocessing, with 200 steps and lc = 4 and Kx = (0,1,2,3):
      python thesis_code.py A2 G -p lc=4 steps=200 -x -v Kx=0:3

    Run A2 & G with multiprocessing, with 200 steps for 6 equally-spaced w.
    w values from 0-3:
      python thesis_code.py A2 G -p steps=200 -x -v w=0:3:6

    Run A2 & G with 100 steps, then write to file:
      python thesis_code.py A2 G -p steps=100 -w

    Run A2 & G with 100 steps, then write to file called "A2_G_100_steps.pkl"
      python thesis_code.py A2 G -p steps=100 -w -o A2_G_100_steps.pkl

Output:
    Files are written as python pickles. Pickles can be read from a new python
    session using:

        import pickle
        result = pickle.load(open('output.pkl', 'rb'))

    Or, more conveniently, using the `scsr.results.load_results` function,
    which returns a results object corresponding to the type of pickle. It
    expects a list of 1 (or more, if chunking is used) pickle path(s):

        from scsr.results import load_results
        results = load_results(['out.1.pkl', 'out.2.pkl'])

Parameters:
        Ky, Ln, L, Kx, Vf, w, lc, Nf_m, tau, wp, P
        steps: The number of discrete steps in theta/phi axes. (The theta by phi
                grid is steps^2).
        theta_max: The maximum value of theta.
        phi_max: The maximum value of phi.
        max_tile_size: The max size of the tile of a m by n function matrix to compute
                at once.
        mp_batch_size: The number of function arrays to process before sending them to
                the main thread (No need to adjust).

options:
  -h, --help            show this help message and exit

Inputs:
  {A1,G,A2,H}           Functions.
  -p P=V [P=V ...], --params P=V [P=V ...]
                        Parameters to override. space-separated list of '[PARAM]=[VALUE]' pairs.
  -v P=V1,V2,..,VN [P=V1,V2,..,VN ...], --variable-params P=V1,V2,..,VN [P=V1,V2,..,VN ...]
                        Variable parameters to override. Space-separated list of
                        '[PARAM]=[VALUE1],[...],[VALUEN]' items. Values can either be a
                        single value or a range of values.
                        Ranges of consecutive whole numbers can be specified as 'A:B',
                        where A and B are integers. Ranges of floating point numbers can
                        be specified as '[VALUEA]:[VALUEB]:[STEPS]', where steps is the
                        number of steps
  -d {64,128,256}, --dtype {64,128,256}
                        Complex data type to use for calculations.

Processing:
  -x, --use-subprocesses
                        Use subprocesses.
  -g, --gpu             Use the GPU. Requires a CUDA-enabled GPU and CuPy to be installed.
  -G GPU_IDS [GPU_IDS ...], --gpu-ids GPU_IDS [GPU_IDS ...]
                        Id(s) of GPUs)s to use. Default's to the first available
                        cuda-capable gpu. Implies --gpu.
  -A, --all-gpus        Use all available GPUs. Implies --gpu.
  -m SUBPROCESS_COUNT, --subprocess-count SUBPROCESS_COUNT
                        Use this many subprocesses. Defaults to the processor core
                        count, unless --gpu is specified, in which case it defaults to 1
                        per GPU in use.

Output:
  -w, --write           Write a pickle file with the resultant array.
  -o OUTPUT, --output OUTPUT
                        Output pickle file path. Defaults to 'results/output.pkl'.
  -f, --force           Overwrite pickle file path it exists. Will generate a unique name
                        by default.
  -u seconds, --min-update-interval seconds
                        Min interval (seconds) between progress bar updates.

Chunking:
  -C CHUNKS, --chunks CHUNKS
                        Number of chunks.
  -P CHUNK_PARAMETER, --chunk-parameter CHUNK_PARAMETER
                        Parameter on which to chunk. Defaults to variable parameter with
                        the most values.
  -I CHUNK_ID, --chunk-id CHUNK_ID
                        Chunk id. (from 1 to --chunks)
```

## Performance Testing

Testing was performed with the below command, suffixed with the additional arguments in the below table: 

```bash
$ python thesis_code.py G H -v w=0:1.2:3 Kx=0.001:0.4:10 -p steps=150 L=50 tau=10 P=0 -w
```
Testing with:
- **System A**: i5-10600K / 32GB RAM / RTX 3080 (10GB VRAM)
- **System B**: i7-12700KF / 32GB RAM (DDR5) / RTX 3080 (10GB VRAM)

|Variant|Additional Args|System A (H:M:S.ms)|System B|
|--------|---------|----------|---------|
|No Multiprocessing\*|| `0:06:38.946743`|`0:05:18.815062`|
|Multiprocessing (process per virtual cpu)|`-x`|`0:02:09.609565`|`0:01:07.827395`|
|GPU|`--gpu`|`0:00:14.559752`|`0:00:07.670285`|
|GPU (Larger tile size)|`--gpu -p max_tile_size=12,12`|`0:00:05.814009`|`0:00:05.757186`|

>\*With some Numpy builds, threading is enabled by default. Here we are seeing some of the benefits of that included in the time. We disable this threading when using multiprocessing mode.
