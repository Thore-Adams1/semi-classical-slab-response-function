import argparse
import os
import shutil
import humanize
from collections import namedtuple
import terminaltables
import glob

from matplotlib import interactive

Dir = namedtuple("Dir", ["path", "file_count", "size"])
File = namedtuple("File", ["path", "size"])


def get_dir(dir_path):
    recursive_contents = glob.glob(os.path.join(dir_path, "**"), recursive=True)
    return Dir(
        path=dir_path,
        file_count=len(recursive_contents),
        size=sum(os.path.getsize(f) for f in recursive_contents),
    )


def get_file(file_path):
    return File(path=file_path, size=os.path.getsize(file_path))


def deletion_confirmed(obj: Dir or File):
    if isinstance(obj, Dir):
        print(
            "{}\n({} files, {} bytes)".format(
                obj.path,
                humanize.number.intcomma(obj.file_count),
                humanize.naturalsize(obj.size),
            ),
            end="",
        )
        response = input(" delete? [y/n]> ")
    elif isinstance(obj, File):
        print("{}\n({} bytes)".format(obj.path, humanize.naturalsize(obj.size)), end="")
        response = input(" delete? [y/n]> ")
    return response.lower().strip() == "y"


def queue_deletion_by_pattern(pattern, files=True, dirs=True, interactive=True):
    for match in glob.glob(pattern):
        result = None
        if dirs and os.path.isdir(match):
            result = get_dir(match)
        elif files and os.path.isfile(match):
            result = get_file(match)
        if result is not None and (not interactive or deletion_confirmed(result)):
            yield result


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Cleanup all results (figs, jobs, pickles).",
    )
    parser.add_argument("-f", "--figs", action="store_true", help="Cleanup figs.")
    parser.add_argument("-j", "--jobs", action="store_true", help="Cleanup jobs.")
    parser.add_argument("-p", "--pickles", action="store_true", help="Cleanup pickles.")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Delete results 1-by-1 after confirmation.",
    )
    parser.add_argument(
        "-F",
        "--force",
        action="store_true",
        help="Skip final confirmation before deletion.",
    )
    parser.add_argument(
        "-d",
        "--dir",
        default=os.getcwd(),
        help="Dir to cleanup (default is current directory).",
    )
    return parser


def cleanup_results():
    args = get_parser().parse_args()
    args.figs = args.all or args.figs
    args.jobs = args.all or args.jobs
    args.pickles = args.all or args.pickles
    if not any([args.figs, args.jobs, args.pickles]):
        print("No cleanup options selected. Exiting.")
    to_delete = []
    deletion_patterns = []
    if args.figs:
        deletion_patterns.append(([os.path.join(args.dir, "*figs*")], {"files": False}))
    if args.jobs:
        deletion_patterns.append(
            ([os.path.join(args.dir, "*results*")], {"files": False})
        )
    if args.pickles:
        deletion_patterns.append(([os.path.join(args.dir, "*.pkl")], {"dirs": False}))
    for fn_args, fn_kwargs in deletion_patterns:
        to_delete.extend(
            queue_deletion_by_pattern(
                *fn_args, **fn_kwargs, interactive=args.interactive
            )
        )
    print(manifest(to_delete))
    if not to_delete:
        print("No files to delete. Exiting.")
        exit(0)
    if not args.force and input("Ok to Delete? [y/n]> ").lower().strip() != "y":
        print("Exiting. (Not deleting.)")
        exit(0)

    for obj in to_delete:
        print("Deleting {}".format(obj.path))
        if isinstance(obj, Dir):
            shutil.rmtree(obj.path)
        elif isinstance(obj, File):
            os.remove(obj.path)


def manifest(to_delete):

    rows = [["Path", "Files", "Size"]]
    total_files = 0
    total_size = 0
    for obj in sorted(to_delete, key=lambda x: x.path):
        if isinstance(obj, Dir):
            rows.append(
                [
                    obj.path,
                    humanize.number.intcomma(obj.file_count),
                    humanize.naturalsize(obj.size),
                ]
            )
            total_files += obj.file_count
        else:
            rows.append([obj.path, "", humanize.naturalsize(obj.size)])
        total_files += 1
        total_size += obj.size
    summary_rows = [
        ["Total Items", "Total Files", "Total Size"],
        [
            len(to_delete),
            humanize.number.intcomma(total_files),
            humanize.naturalsize(total_size),
        ],
    ]
    files_table = terminaltables.AsciiTable(rows)
    summary_table = terminaltables.AsciiTable(summary_rows)
    return "{}\n{}".format(files_table.table, summary_table.table)


if __name__ == "__main__":
    cleanup_results()
