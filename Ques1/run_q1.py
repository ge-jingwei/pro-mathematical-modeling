import argparse
from pathlib import Path

import yaml

from Ques1.events import write_trials
from Ques1.quality import write_quality_outputs
from Ques1.reliability import write_reliability
from Ques1.validate import write_nested_validation


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["events", "quality", "reliability", "baselines"],
        default="events",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "cti.yaml")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    data_dir = ROOT / config["paths"]["data_dir"]
    output_dir = ROOT / config["paths"]["q1_output_dir"]
    if args.stage == "events":
        output_path = output_dir / "trials.csv"
        trials = write_trials(data_dir, output_path)
        print(f"Validated {len(trials)} trials across {trials['file'].nunique()} records")
        print(output_path)
    elif args.stage == "quality":
        trials, report = write_quality_outputs(data_dir, output_dir, config)
        print(f"Measured {len(trials)} trials and {len(report)} channel records")
        print(output_dir / "quality_report.csv")
    elif args.stage == "reliability":
        output_path = output_dir / "reliability.npz"
        write_reliability(data_dir, output_path, config)
        print(output_path)
    else:
        summary = write_nested_validation(data_dir, output_dir, config)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
