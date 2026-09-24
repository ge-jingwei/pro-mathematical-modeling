"""Run the compact Question 3 pipeline on the designated CUDA server."""
from pathlib import Path
import argparse
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--data", type=Path, default=ROOT / "C题/dataset")
    parser.add_argument("--q1", type=Path, default=ROOT / "Ques1/results")
    parser.add_argument("--q2", type=Path, default=ROOT / "Ques2/outputs/final")
    parser.add_argument("--out", type=Path, default=ROOT / "Ques3/outputs/final")
    parser.add_argument("--remote-root", default="/home/gejw/mm/q3_20260924_mvp")
    parser.add_argument("--font", default="/home/gejw/mm/q1_20260924/assets/msyh.ttc")
    args = parser.parse_args()
    if not args.server:
        dispatch(args)
        return
    sys.path.insert(0, str(ROOT))
    from Ques3.utils import stage1, dump, q2_utils
    if args.out.exists() and not args.resume:
        raise FileExistsError(f"Use a new output directory or --resume: {args.out}")
    for folder in ["tables", "model", "figures"]:
        (args.out / folder).mkdir(parents=True, exist_ok=True)
    device, hardware = q2_utils.choose_device()
    q2_utils.setup(args.font)
    started = time.time()
    dump(args.out / "model/run_config.json", dict(seed=20260924, hardware=hardware, stage=args.stage,
         command=sys.argv, preencoding_commit="488e576", data=str(args.data), q1=str(args.q1), q2=str(args.q2)))
    dump(args.out / "model/completion.json", dict(status="running", stage=args.stage))
    try:
        trials = stage1(args.data, args.q1, args.q2, args.out)
        if args.stage >= 2:
            from Ques3.cognitive_model import run_models
            run_models(trials, args, device)
        if args.stage >= 3:
            from Ques3.verify import verify
            verify(args.out, args.q2, device)
        dump(args.out / "model/completion.json", dict(status="complete", stage=args.stage, elapsed_s=time.time() - started))
        (args.out / "blocked.md").unlink(missing_ok=True)
    except Exception as exc:
        dump(args.out / "model/completion.json", dict(status="failed", stage=args.stage, error=str(exc)))
        (args.out / "blocked.md").write_text(f"# 运行暂停\n\n{type(exc).__name__}: {exc}\n\n未生成后续阶段结果。\n", encoding="utf-8")
        raise


def dispatch(args):
    sys.path.insert(0, str(ROOT / "tools"))
    import remote_tool as remote
    base = args.remote_root.rstrip("/")
    if args.out.exists() and not args.resume:
        raise FileExistsError(f"Use --resume or a new destination: {args.out}")
    for path in (ROOT / "Ques3").glob("*.py"):
        remote.file_upload(path, base + "/Ques3")
    for name in ["model.py", "utils.py"]:
        remote.file_upload(ROOT / "Ques2" / name, base + "/Ques2")
    if not args.resume:
        remote.file_upload(ROOT / "Ques1/src", base + "/Ques1", "src")
        remote.file_upload(ROOT / "Ques1/qa", base + "/Ques1", "qa")
        for path in args.data.glob("VisualCog*_Task-*.mat"):
            remote.file_upload(path, base + "/dataset")
        for pattern in ["*_epochs.npz", "events.csv", "recordings.csv"]:
            for path in args.q1.glob(pattern):
                remote.file_upload(path, base + "/Ques1/results")
        for pattern in ["*_to_*_final.json", "validation_epochs.npz"]:
            for path in (args.q2 / "model").glob(pattern):
                remote.file_upload(path, base + "/Ques2/outputs/final/model")
        remote.file_upload(args.q2 / "tables/trials.csv", base + "/Ques2/outputs/final/tables")
    py = "/home/gejw/.conda/envs/kenny_ok/bin/python"
    out_name = args.out.name
    command = f"{py} -B Ques3/ques3.py --server --data dataset --q1 Ques1/results --q2 Ques2/outputs/final --out {shlex.quote(out_name)} --stage {args.stage} --font {shlex.quote(args.font)}"
    if args.resume:
        command += " --resume"
    code = remote.execute(command, conda_env="", workdir=base)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    remote.download(base + "/" + out_name, str(args.out.parent))
    if code:
        raise RuntimeError(f"Remote execution failed; diagnostics downloaded to {args.out}")
    (args.out / "blocked.md").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
