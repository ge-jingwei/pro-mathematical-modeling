"""Run the mechanism model and optional nested decoding into one flat output."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import ROOT, DATA, output_run, dump


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--cache", type=Path, default=ROOT / "Ques1/output")
    parser.add_argument("--out", type=Path, default=ROOT / "Ques2/output")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--skip-nested", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--font")
    args = parser.parse_args(argv)
    from Ques2.utils import stage1, choose_device, setup
    from Ques2.model import stage2, stage3
    device, hardware = choose_device(args.device)
    setup(args.font)
    with output_run(args.out, "Ques2", args.overwrite, args.stage) as out:
        dump(out / "run_config.json", dict(seed=20260924, hardware=hardware, stage=args.stage,
             data=str(args.data.resolve()), cache=str(args.cache.resolve()), nested=not args.skip_nested))
        _, X, meta = stage1(args.data, args.cache, out, device)
        if args.stage >= 2:
            stage2(X, meta, out, device)
        if args.stage >= 3:
            stage3(X, meta, out, device)
            if not args.skip_nested:
                from Ques2.nested import run
                run(out, args.data, device, hardware)


if __name__ == "__main__":
    main()
