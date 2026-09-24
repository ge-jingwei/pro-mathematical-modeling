"""Run the cognitive model using the original Question 2 mechanism folds."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import ROOT, DATA, output_run, dump


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--q1", type=Path, default=ROOT / "Ques1/output")
    parser.add_argument("--q2", type=Path, default=ROOT / "Ques2/output")
    parser.add_argument("--out", type=Path, default=ROOT / "Ques3/output")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--font")
    args = parser.parse_args(argv)
    from Ques3.utils import stage1, q2_utils
    device, hardware = q2_utils.choose_device(args.device)
    q2_utils.setup(args.font)
    with output_run(args.out, "Ques3", args.overwrite, args.stage) as out:
        args.out = out
        dump(out / "run_config.json", dict(seed=20260924, hardware=hardware, stage=args.stage,
             data=str(args.data.resolve()), q1=str(args.q1.resolve()), q2=str(args.q2.resolve())))
        trials = stage1(args.data, args.q1, args.q2, out)
        if args.stage >= 2:
            from Ques3.cognitive_model import run_models
            run_models(trials, args, device)


if __name__ == "__main__":
    main()
