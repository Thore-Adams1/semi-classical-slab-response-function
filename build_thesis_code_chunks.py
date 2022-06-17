import os
import sys
import pipes
import argparse
import itertools

jobscript = """\
#!/bin/bash --login
###
#job name
#SBATCH --job-name='sam_praill_job_{name}'
#job stdout file
#SBATCH --output=bench.out.%J
#job stderr file
#SBATCH --error=bench.err.%J
#maximum job time in D-HH:MM
#SBATCH --time=3-00:00
#number of parallel processes (tasks) you are requesting - maps to MPI processes
#SBATCH --ntasks=1
#memory per process in MB
#tasks to run per node (change for hybrid OpenMP/MPI)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
###
#now run normal batch commands
module load python/3.7.0
{command}"""
import thesis_code as tc


def get_unique_dir(pattern, start=1):
    path = pattern.format(start)
    while os.path.exists(path):
        start += 1
        path = pattern.format(start)
    return path


def get_commands(tc_args, given_args, pickles_dir):
    commands = []
    for chunk in range(1, tc_args.chunks + 1):
        sanitised_args = [
            pipes.quote(s)
            for s in given_args
            + [
                "--force",
                "--output",
                os.path.join(pickles_dir, "out.{}.pkl".format(chunk)),
                "-I",
                str(chunk),
                "-u",
                "60",
            ]
        ]
        command_parts = [
            "python3",
            os.path.join("~/sam-phys-code/", os.path.basename(tc.__file__)),
        ] + sanitised_args
        commands.append(" ".join(command_parts))
    return commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-J", "--create-job-scripts", action="store_true")
    given_args = sys.argv[1:]
    args, given_args = parser.parse_known_args(given_args)
    tc_parser = tc.get_parser()
    tc_args = tc_parser.parse_args(list(given_args))
    if tc_args.chunk_id is not None:
        raise ValueError(
            "Chunk ID shouldn't be provided to this script. It's determined internally for each jobscript."
        )
    if tc_args.chunks is None:
        raise ValueError("Chunks should be provided to this script.")

    if tc_args.output is not None:
        raise ValueError("Output should NOT be provided to this script.")

    var_str = "no_params"
    if tc_args.params is not None:
        var_str = "_".join(
            "{}_{}".format(k, v)
            for k, v in itertools.chain.from_iterable(tc_args.params)
        )
    job_root = get_unique_dir("results.{}.{{}}".format(var_str))
    pickles_dir = os.path.join(job_root, "pickles")
    jobscript_dir = os.path.join(job_root, "jobscripts")
    commands = get_commands(tc_args, given_args, pickles_dir)
    print("Job root dir: {}\nCommands:\n\n{}\n".format(job_root, "\n".join(commands)))
    if args.create_job_scripts:
        for dir_path in (job_root, pickles_dir, jobscript_dir):
            os.makedirs(dir_path)
        job_scripts = []
        job_script_path_template = os.path.join(jobscript_dir, "job.{}.sh")
        for i, command in enumerate(commands, start=1):
            job_script_path = job_script_path_template.format(i)
            with open(job_script_path, "w") as f:
                f.write(jobscript.format(name=var_str, command=command))
            job_scripts.append(job_script_path)
        print(
            "To run the jobscripts, run the following commands:\n\n{}".format(
                "\n".join(
                    "sbatch --account=scw1772 {}".format(job_script)
                    for job_script in job_scripts
                ),
            )
        )


if __name__ == "__main__":
    main()
