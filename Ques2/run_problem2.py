"""Run three gated stages locally on the server or dispatch from the workstation."""
from pathlib import Path
import argparse
import datetime
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", action="store_true")
    p.add_argument("--data", type=Path, default=ROOT / "C题" / "dataset")
    p.add_argument("--cache", type=Path, default=ROOT / "Ques1" / "results")
    p.add_argument("--out", type=Path)
    p.add_argument("--font", default="/home/gejw/mm/q1_20260924/assets/msyh.ttc")
    p.add_argument("--stage", type=int, choices=[1, 2, 3], default=3)
    args = p.parse_args()
    if not args.server:
        dispatch(args)
        return
    from utils import stage1, choose_device, setup, dump
    import numpy as np
    np.random.seed(20260924)
    out = args.out or ROOT / "Ques2" / "outputs" / datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    for folder in ["figures", "tables", "model"]:
        (out / folder).mkdir(parents=True, exist_ok=True)
    started = time.time()
    device, hardware = choose_device()
    setup(args.font)
    dump(out / "model" / "run_config.json", dict(seed=20260924, hardware=hardware,
         data=str(args.data), cache=str(args.cache), command=sys.argv,
         stage=args.stage, preencoding_commit="61eda6e"))
    try:
        clean, X, meta = stage1(args.data, args.cache, out, device)
        if args.stage >= 2:
            from model import stage2
            stage2(X, meta, out, device)
        if args.stage >= 3:
            from model import stage3
            stage3(X, meta, out, device)
            from utils import verify_results, write_report
            verify_results(X, meta, out, device)
            write_report(out)
        dump(out / "model" / "completion.json", dict(status="complete", completed_stage=args.stage, elapsed_seconds=time.time()-started))
        print(f"Completed stage {args.stage}: {out}", flush=True)
    except Exception as exc:
        (out / "blocked.md").write_text(f"# 运行暂停\n\n已停止后续阶段，未补造结果。\n\n{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise


def dispatch(args):
    import shlex
    sys.path.insert(0, str(ROOT / "tools"))
    import remote_tool as remote
    stamp = datetime.datetime.now().strftime("q2_%Y%m%d_%H%M%S")
    base = remote.ROOT_PATH.rstrip("/") + "/" + stamp
    py = "/home/gejw/.conda/envs/kenny_ok/bin/python"
    local = args.out or ROOT / "Ques2" / "outputs" / stamp
    if local.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {local}")
    for path in (ROOT / "Ques2").glob("*.py"):
        remote.file_upload(path, base + "/Ques2")
    remote.file_upload(ROOT / "Ques1" / "src", base + "/Ques1", "src")
    remote.file_upload(ROOT / "Ques1" / "qa", base + "/Ques1", "qa")
    remote.file_upload(args.data, base, "dataset")
    for pattern in ["*_epochs.npz", "events.csv", "trial_quality.csv", "recordings.csv"]:
        for path in args.cache.glob(pattern):
            remote.file_upload(path, base + "/Ques1/results")
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        remote.file_upload(font, base + "/assets")
        args.font = base + "/assets/msyh.ttc"
    command = f"{py} -B Ques2/run_problem2.py --server --data dataset --cache Ques1/results --out outputs --stage {args.stage} --font {shlex.quote(args.font)}"
    code = remote.execute(command, conda_env="", workdir=base)
    local.mkdir(parents=True)
    remote.download(base + "/outputs", str(local))
    if code:
        raise RuntimeError(f"Remote run failed; inspect downloaded diagnostics: {local}")


if __name__ == "__main__":
    main()


