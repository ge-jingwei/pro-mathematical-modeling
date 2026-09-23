"""Deploy the second-generation model to the lab server and run it on GPU."""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, "tools")

from remote_tool import execute, file_upload, run_background  # noqa: E402

LOCAL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE = "/home/gejw/mm/mm_v2"

FILES = [
    "neuro",
    "Ques1/events.py",
    "Ques1/__init__.py",
    "tools/io_mat.py",
    "tools/__init__.py",
    "Ques2/run_q2_v2.py",
    "configs/cti.yaml",
]


def stage() -> str:
    staging = tempfile.mkdtemp(prefix="mm_v2_")
    for rel in FILES:
        source = os.path.join(LOCAL_ROOT, rel)
        target = os.path.join(staging, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.isdir(source):
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(source, target)
    return staging


def main() -> None:
    staging = stage()
    try:
        print("uploading", staging)
        remote = file_upload(staging, remote_dir="/home/gejw/mm", remote_name="mm_v2")
        print("uploaded to", remote)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    execute("mkdir -p /home/gejw/mm/mm_v2/C题 && ln -sfn /home/gejw/mm/C题/dataset /home/gejw/mm/mm_v2/C题/dataset", workdir=None, conda_env=None)
    pid, log = run_background("python -m Ques2.run_q2_v2 --permutations 2000", workdir=REMOTE, conda_env="gejw", log="logs/q2_v2.log")
    print("started pid=%s log=%s" % (pid, log))


if __name__ == "__main__":
    main()
