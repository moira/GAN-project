import os
import re
import sys
import glob
import shutil
import zipfile
import argparse
import subprocess

import cs236781.answers as answers
import cs236781.jupyter_utils as jupyter_utils

SUBMISSION_NAME_PATTERN = re.compile(r"hw\d-(\d+_?)+")
SUBMISSION_ZIPF_PATTERN = re.compile(SUBMISSION_NAME_PATTERN.pattern + r"\.zip")


def parse_cli():
    def is_dir(dirname):
        if not os.path.isdir(dirname):
            raise argparse.ArgumentTypeError(f"{dirname} is not a directory")
        else:
            return dirname

    def is_file(filename):
        if not os.path.isfile(filename):
            raise argparse.ArgumentTypeError(f"{filename} is not a file")
        else:
            return filename

    p = argparse.ArgumentParser(description="CS236781 Course Homework Tools")
    sp = p.add_subparsers(help="Sub-command help")

    # Prepare distribution
    sp_dist = sp.add_parser(
        "prepare-dist", help="Prepare homework for distribution to " "students"
    )
    sp_dist.set_defaults(subcmd_fn=prepare_dist)
    sp_dist.add_argument("--hw-dir", "-i", type=is_dir, help="hw folder", required=True)
    sp_dist.add_argument(
        "--out-dir",
        "-o",
        type=str,
        help="Output folder",
        default="./out",
        required=False,
    )

    # Prepare submission
    sp_subm = sp.add_parser(
        "prepare-submission", help="Prepare homework for submission"
    )
    sp_subm.set_defaults(subcmd_fn=prepare_submission)
    sp_subm.add_argument(
        "--hw-dir", type=is_dir, help="hw folder", default=".", required=False
    )
    sp_subm.add_argument(
        "--out-dir",
        "-o",
        type=is_dir,
        help="Output folder",
        default=".",
        required=False,
    )
    sp_subm.add_argument(
        "--id",
        "-i",
        type=int,
        help="Submitter id",
        action="append",
        metavar="ID",
        required=True,
        dest="submitter_ids",
    )
    sp_subm.add_argument(
        "--skip-run",
        "-R",
        action="store_true",
        help="Skip running notebooks",
        required=False,
    )
    sp_subm.add_argument(
        "--allow-errors",
        "-E",
        action="store_true",
        help="Allow errors when running notebooks",
        required=False,
    )

    # Clear notebook outputs
    sp_clear = sp.add_parser("clear-nb", help="clear outputs from notebooks")
    sp_clear.set_defaults(subcmd_fn=clear_notebooks)
    sp_clear.add_argument(
        "nb_paths", type=is_file, help="notebooks to run", metavar="NB_PATH", nargs="+"
    )

    # Run notebooks
    sp_run = sp.add_parser("run-nb", help="run jupyter notebooks")
    sp_run.set_defaults(subcmd_fn=run_notebooks)
    sp_run.add_argument(
        "nb_paths", type=is_file, help="notebooks to run", metavar="NB_PATH", nargs="+"
    )
    sp_run.add_argument(
        "--allow-errors",
        "-E",
        action="store_true",
        help="Allow errors when running notebooks",
        required=False,
    )

    parsed = p.parse_args()

    if "subcmd_fn" not in parsed:
        p.print_help()
        sys.exit()

    return parsed


def zipdir(path, archive_name=None):
    """
    Zips an entire directory tree.
    The archive will be placed beside the directory. For example, given a
    path of `foo/bar/baz/`, this function will create `foo/bar/baz.zip`.

    :param path: Path of the directory to zip.
    :param archive_name: The name of the archive to create. By default it's
        None, which means use the name of the zipped folder.
    :return: Path of the created archive.
    """
    path_dirname = os.path.dirname(path)
    path_basename = os.path.basename(path)

    if archive_name is None:
        archive_name = path_basename

    archive_path = f"{os.path.join(path_dirname, archive_name)}.zip"

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as ziph:
        for root, dirs, files in os.walk(path):
            arch_root = os.path.relpath(root, path_dirname)

            for file in files:
                file_path = os.path.join(root, file)
                arch_path = os.path.join(arch_root, file)

                ziph.write(file_path, arch_path)

    return archive_path


def create_submission_name(hw_dir, submitter_ids):
    hw_module_dir = glob.glob(os.path.join(hw_dir, "hw?"))[0]
    hw_name = os.path.basename(hw_module_dir)

    submitter_ids = [str(submitter_id) for submitter_id in submitter_ids]
    submission_name = f'{hw_name}-{str.join("_", submitter_ids)}'

    assert re.match(SUBMISSION_NAME_PATTERN, submission_name)
    return submission_name


def copytree_ignore_fn(src, names, is_distribution=True):
    def ignore_predicate(name: str) -> bool:
        return (
            name.startswith(".")
            or name == "__pycache__"
            or name == "data"
            or re.match(SUBMISSION_ZIPF_PATTERN, name)
        )

    # Completely drop results folders when creating a distribution,
    # but keep these folders when creating a submission
    if is_distribution and os.path.basename(src) == "results":
        return names

    # Completely drop checkpoints folders
    if os.path.basename(src) == "checkpoints":
        return names

    # Go over names and select the ones to drop
    return [name for name in names if ignore_predicate(name)]


def clear_notebooks(nb_paths, **kwargs):
    print(f">> Clearing {len(nb_paths)} notebooks...")
    nb_paths.sort()
    for nb_path in nb_paths:
        jupyter_utils.nbconvert(nb_path, clear_output=True)


def run_notebooks(nb_paths, allow_errors=False, **kwargs):
    print(f">> Running {len(nb_paths)} notebooks...")
    nb_paths.sort()
    for nb_path in nb_paths:
        try:
            jupyter_utils.nbconvert(
                nb_path, execute=True, inplace=True, allow_errors=allow_errors
            )
        except subprocess.CalledProcessError:
            error_msg = (
                f"Got errors while executing notebook {nb_path}. "
                f"Make sure you've implemented everything and that all "
                f"tests pass."
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)


def prepare_dist(hw_dir, out_dir, **kwargs):
    """
    Prepares a homework assignment for distribution to students.
    Clears solutions from the code and wraps everything in a zip file.
    :param hw_dir: root directory of assignment.
    :param out_dir: output directory for the zip file.
    """
    hw_dir_basename = os.path.basename(hw_dir)

    dest_dir = os.path.join(out_dir, hw_dir_basename)

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    try:
        # Copy assignment directory to a temporary folder
        shutil.copytree(hw_dir, dest_dir, symlinks=False, ignore=copytree_ignore_fn)

        # Copy extra files
        extra_files = (
            "main.py",
            "py-sbatch.sh",
            "jupyter-lab.sh",
            "environment.yml",
            "environment.lock.yml",
            ".gitignore",
        )
        for filename in extra_files:
            shutil.copy(filename, dest_dir)

        # Clear notebook outputs
        nb_paths = glob.glob(os.path.join(dest_dir, "*.ipynb"))
        clear_notebooks(nb_paths)

        # Remove solutions from the code
        for root, dirs, files in os.walk(dest_dir):
            py_files = [f for f in files if f.lower().endswith(".py")]
            for py_file in py_files:
                py_file = os.path.join(root, py_file)

                with open(py_file, "r") as file_handle:
                    content = file_handle.read()
                    new_content, n_subs_code, n_subs_answers = answers.clear_solutions(
                        content
                    )

                if new_content is None:
                    continue

                with open(py_file, "w") as file_handle:
                    file_handle.write(new_content)

                if n_subs_code > 0:
                    print(f">> {py_file}: {n_subs_code} code-blocks replaced")
                if n_subs_answers > 0:
                    print(f">> {py_file}: {n_subs_answers} answers replaced")

        # Create an archive for distribution
        dist_zip = zipdir(dest_dir)
        print(">> Created ", dist_zip)

    finally:
        shutil.rmtree(dest_dir)


def prepare_submission(hw_dir, out_dir, submitter_ids, skip_run, **kwargs):
    """
    Creates a submission zip file students can submit.
    This function will run all the notebooks in the assignment, merge them into
    a single html file for submission and wrap everything in a zip file.
    :param hw_dir: Root directory of the assignment.
    :param out_dir: Where to write the zip file to.
    :param submitter_ids: list of ID numbers of submitters.
    :param skip_run: Whether to skip running the notebooks.
    """
    submission_name = create_submission_name(hw_dir, submitter_ids)

    nb_paths = glob.glob(os.path.join(hw_dir, "*.ipynb"))
    nb_paths.sort()

    # 1. Run all notebooks and save outputs within them
    if not skip_run:
        run_notebooks(nb_paths, **kwargs)

    # 2. Run nbnmerge to merge them
    nb_merged = os.path.join(hw_dir, f"{submission_name}.ipynb")
    jupyter_utils.nbmerge(nb_paths, nb_merged)

    # 3. Run nbconvert to convert the merged notebook to html
    jupyter_utils.nbconvert(nb_merged)
    os.remove(nb_merged)

    # 4. Zip the hw folder, excluding 'data' folder and temp files.
    dest_dir = os.path.join(out_dir, submission_name)
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    try:

        def ignore_fn(src, names):
            return copytree_ignore_fn(src, names, is_distribution=False)

        # Copy assignment directory to a temporary folder
        shutil.copytree(hw_dir, dest_dir, symlinks=False, ignore=ignore_fn)

        # Create an archive for submission
        sub_zip = zipdir(dest_dir)
        print(f">> Created submission {sub_zip}. Good luck!")

    finally:
        shutil.rmtree(dest_dir)


if __name__ == "__main__":
    parsed_args = parse_cli()
    parsed_args.subcmd_fn(**vars(parsed_args))
