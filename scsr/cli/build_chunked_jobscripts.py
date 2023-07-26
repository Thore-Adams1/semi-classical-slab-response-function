#! /usr/bin/env python
import os
import sys
import pipes
import argparse
import itertools
from textwrap import dedent

import scsr.cli.calc as tc

DESC = """\
This script accepts a scsr-calc command and generates a command
to run each chunk. Slurm jobscripts can also be created for each chunk
using `-J`.
"""
USAGE = """\
scsr-build-chunked-jobscripts [-h] [-J] [args for scsr-calc]
"""
jobscript = """\
#!/bin/bash --login
###
#job name
#SBATCH --job-name='sam_praill_job_{name}'
#job stdout file
#SBATCH --output={log_dir}/log.stdout.%J
#job stderr file
#SBATCH --error={log_dir}/log.stderr.%J
#maximum job time in D-HH:MM
#SBATCH --time={max_days}-00:00
#number of parallel processes (tasks) you are requesting - maps to MPI processes
#SBATCH --ntasks=1
#memory per process in MB
#tasks to run per node (change for hybrid OpenMP/MPI)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
{gpu_sbatch}{chunks_sbatch}###
echo "Loading Python"
module load python/3.7.0
{gpu_commands}
echo "RUNNING CODE:"
{command}"""
sys.path.append(os.path.dirname(__file__))


def get_unique_dir(pattern, start=1):
    path = pattern.format(start)
    while os.path.exists(path):
        start += 1
        path = pattern.format(start)
    return path


def get_commands(args, tc_args, given_args, pickles_dir):
    commands = ["cd " + pipes.quote(os.path.dirname(tc.__file__))]
    if args.create_job_script:
        chunk_range = ["$SLURM_ARRAY_TASK_ID"]
    else:
        chunk_range = range(1, tc_args.chunks + 1)
    for chunk in chunk_range:
        sanitised_args = [
            pipes.quote(s)
            for s in given_args
            + [
                "--force",
                "-u",
                "60",
                "--output",
            ]
        ]
        sanitised_args.extend(
            [
                os.path.join(pickles_dir, "out.{}.pkl".format(chunk)).replace(
                    " ", r"\ "
                ),
                "-I",
                str(chunk),
            ]
        )
        command_parts = [
            "python3",
            tc.__file__,
        ] + sanitised_args
        commands.append(" ".join(command_parts))
    return commands


def validate_args(tc_args):
    if tc_args.chunk_id is not None:
        raise ValueError(
            "Chunk ID shouldn't be provided to this script. It's determined internally for each jobscript."
        )
    if tc_args.chunks is None:
        raise ValueError("No chunks specified.")


def get_parser():
    parser = argparse.ArgumentParser(description=DESC, usage=USAGE)
    parser.add_argument("-J", "--create-job-script", action="store_true")
    return parser


def main():
    given_args = sys.argv[1:]
    args, given_args = get_parser().parse_known_args(given_args)
    tc_parser = tc.get_parser()
    tc_args = tc_parser.parse_args(list(given_args))
    tc.set_arg_defaults(tc_args)
    validate_args(tc_args)

    var_str = "no_params"
    if tc_args.params is not None:
        ignore_vars = {"max_tile_size"}
        var_str = "_".join(
            "{}_{}".format(k, v)
            for k, v in itertools.chain.from_iterable(tc_args.params)
            if k not in ignore_vars
        )

    if tc_args.output is not None:
        dirname = os.path.dirname(tc_args.output)
        var_str += "_" + os.path.splitext(os.path.basename(tc_args.output))[0]
        results_dir = "results.{}_{{}}".format(var_str)
        if dirname:
            results_dir = os.path.join(dirname, results_dir)
    else:
        results_dir = "results.{}_{{}}".format(var_str)

    job_root = get_unique_dir(results_dir)
    pickles_dir = os.path.join(job_root, "pickles")
    log_dir = os.path.join(job_root, "log")
    commands = get_commands(args, tc_args, given_args, pickles_dir)
    print("Job root dir: {}\nCommands:\n\n{}\n".format(job_root, "\n".join(commands)))
    if args.create_job_script:
        if tc_args.gpu:
            gpu_sbatch = (
                "#SBATCH --gres=gpu:2\n" "#SBATCH -p gpu_v100  # to request V100 GPUs\n"
            )
            gpu_commands = dedent(
                """\
                echo "GPU INFO:"
                nvidia-smi
                nvidia-smi -L
                echo "AVAIL CUDA VERSIONS:"
                module avail CUDA
                echo "Loading CUDA:"
                module load CUDA/11.5
                echo "checking cupy"
                python3 -c "import sys; print('Python Path: {}'.format(sys.path))"
                python3 -m pip install cupy-cuda115 --user
                echo "CuPy Packages: " `python3 -m pip freeze | grep cupy`
                python3 -c "import cupy; print(cupy.show_config())"
            """
            )
        else:
            gpu_sbatch, gpu_commands = "", ""
        for dir_path in (job_root, pickles_dir, log_dir):
            os.makedirs(dir_path)
        job_script_path = os.path.join(job_root, "job.sh")
        chunks_config = "#SBATCH --array=1{}\n".format(
            "-{}".format(tc_args.chunks) if tc_args.chunks > 1 else ""
        )
        with open(job_script_path, "w") as f:
            f.write(
                jobscript.format(
                    name=var_str,
                    command="\n".join(commands),
                    gpu_sbatch=gpu_sbatch,
                    gpu_commands=gpu_commands,
                    chunks_sbatch=chunks_config,
                    max_days=1 if tc_args.gpu else 3,
                    log_dir=log_dir,
                )
            )
        print(
            "To run the jobscripts, run the following command:\n\n{}".format(
                "sbatch --account=[ACCOUNT] {}".format(
                    job_script_path.replace("\\", "\\\\")
                )
            )
        )


if __name__ == "__main__":
    main()
