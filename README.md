# Semi-classical Slab Response Function

Add description of repo.

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If intending to run on the gpu, [`CuPy`](https://cupy.dev/) and it's required dependencies must also be installed.

## Development

This code is all formatted using [`black`](https://pypi.org/project/black/). Automatic formatting can be set up as a git hook by running:

```bash
git-scripts/install-git-hooks
```

## Running

```
$ python thesis_code.py --help
usage: thesis_code.py [-h] [-p P=V [P=V ...]] [-v P=V1,V2,..,VN [P=V1,V2,..,VN ...]] [-x] [-g] [--gpu-id]
                      [-m MAX_SUBPROCESSES] [-w] [-o OUTPUT] [-f] [-u seconds] [-C CHUNKS] [-P CHUNK_PARAMETER]
                      [-I CHUNK_ID]
                      [{A1,G,H,A2} ...]

This script computes H, G, A1 and A2 electron dynamics matrices

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

Parameters:
        Vf, wp, Ln, lc, Kx, tau, w, P, Nf_m, Ky, L
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
  {A1,G,H,A2}           Functions.
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

Processing:
  -x, --use-subprocesses
                        Use subprocesses.
  -g, --gpu             Use the GPU. Requires a CUDA-enabled GPU and CuPy to be installed.
  --gpu-id              GPU id. Default's to the first available gpu.
  -m MAX_SUBPROCESSES, --max-subprocesses MAX_SUBPROCESSES
                        Use this many subprocesses. Defaults to the processor core count.

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
                        Parameter on which to chunk. Defaults to first variable parameter
  -I CHUNK_ID, --chunk-id CHUNK_ID
                        Chunk id. (from 1 to --chunks)
```

## Performance Testing

Testing was performed with the below command, suffixed with the additional arguments in the below table: 

```bash
python thesis_code.py G H -v w=0:1.2:3 Kx=0.001:0.4:10 -p steps=150 L=50 tau=10 P=0 -w
```
Testing with an i5-10600K / 32GB RAM / RTX 3080 (10GB VRAM):
|Variant|Additional Args|Time (H:M:S.ms)|
|--------|---------|----------|
|No Multiprocessing\*|| `0:06:38.946743`|
|Multiprocessing (process per virtual cpu)|`-x`|`0:02:09.609565`|
|GPU|`--gpu`|`0:00:14.559752`|
|GPU (Larger tile size)|`--gpu -p max_tile_size=12,12`|`0:00:05.814009`|

>\*With some Numpy builds, threading is enabled by default. Here we are seeing some of the benefits of that included in the time. We disable this threading when using multiprocessing mode.
