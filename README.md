# Semi-classical Slab Response Function

> **TODO:** Add description of repo.

## Requirements

Install the required dependencies:

```bash
$ pip install -r requirements.txt
```

If intending to run on the gpu, [`CuPy`](https://cupy.dev/) and it's required dependencies must also be installed.

## Usage

Visit the [documentation](http://SPraill.github.io/semi-classical-slab-response-function)
to get started.

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

All changes should be done in a local fork of the repo then you can 
[create a pull request](https://github.com/Spraill/semi-classical-slab-response-function/compare)
to [`main`](https://github.com/Spraill/semi-classical-slab-response-function/tree/main)

## Performance Testing

Testing was performed with the below command, suffixed with the additional arguments in the below table: 

```bash
$ scsr-calc G H -v w=0:1.2:3 Kx=0.001:0.4:10 -p steps=150 L=50 tau=10 P=0 -w
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

>**\*** With some Numpy builds, threading is enabled by default. Here we are seeing some of the benefits of that included in the time. We disable this threading when using multiprocessing mode.
