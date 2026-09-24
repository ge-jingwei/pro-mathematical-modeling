"""Shared paths, flat-output lifecycle, and reproducibility metadata."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "C题" / "dataset"


def source_hashes(question):
    paths = [ROOT / "pipeline.py"]
    for number in range(1, int(question.removeprefix("Ques")) + 1):
        folder = ROOT / f"Ques{number}"
        paths.extend(sorted(folder.rglob("*.py")))
        paths.append(folder / "requirements.txt")
    return {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


@contextmanager
def output_run(out, question, overwrite=False, stage=3):
    out = Path(out).resolve()
    protected = [ROOT, DATA.resolve(), *(ROOT / f"Ques{i}" for i in (1, 2, 3))]
    if any(out == p or out in p.parents for p in protected):
        raise ValueError(f"Unsafe output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if any(p.is_dir() for p in out.iterdir()):
        raise ValueError(f"Output must be flat: {out}")
    files = list(out.iterdir())
    if files and not overwrite:
        raise FileExistsError(f"Use --overwrite to replace generated results: {out}")
    if files and overwrite and not (out / "completion.json").is_file():
        raise ValueError("Refusing to clear an unrecognized output directory")
    if overwrite:
        for path in files:
            path.unlink()
    started = time.time()
    status = dict(status="running", stage=stage, source_sha256=source_hashes(question))
    dump(out / "completion.json", status)
    try:
        yield out
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    else:
        status["status"] = "complete"
    finally:
        status["elapsed_seconds"] = time.time() - started
        dump(out / "completion.json", status)
