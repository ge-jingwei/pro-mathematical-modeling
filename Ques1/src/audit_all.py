"""Audit every exported PDF after all analysis jobs have completed."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

p=argparse.ArgumentParser(); p.add_argument("--out",required=True); a=p.parse_args()
out=Path(a.out); auditor=Path(__file__).resolve().parents[1]/"qa"/"audit_figure_collisions.py"
rows=[]
for path in sorted((out/"figures").glob("*.pdf")):
    result=subprocess.run([sys.executable,str(auditor),str(path),"--json-out",str(path.with_suffix(".collision.json"))],capture_output=True,text=True)
    rows.append(dict(file=path.name,code=result.returncode,summary=result.stdout[-3000:]))
    print(path.name,result.returncode,result.stdout[-140:])
(out/"figure_audit.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
assert len(rows)>=11
assert all(row["code"]==0 for row in rows),"One or more figures require repair."
