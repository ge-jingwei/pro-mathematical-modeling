"""Read raw channels, event transitions, and recording-level metadata."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks

CHANNELS = ["F3", "Fz", "F4"]


def transitions(x, levels=None):
    selected = x != 0 if levels is None else np.isin(x, levels)
    return np.flatnonzero(selected & np.r_[True, x[1:] != x[:-1]])


def load_record(path):
    m = loadmat(path, squeeze_me=True)
    data = np.asarray(m["data"], dtype=float)
    labels = [str(np.asarray(x).item()) for x in m["DataLabel"]]
    fs = float(m["SampleRate"])
    assert data.shape[0] == 10 and fs == 256
    assert labels[:3] == ["Fz", "F3", "F4"]
    assert np.isfinite(data).all()
    assert np.allclose(np.diff(data[9]), 1 / fs, atol=1e-10)
    cue = transitions(data[7], [-1, 1])
    target = transitions(data[8], [-1, 1])
    clicks = transitions(data[8], [-2, 2])
    rows = []
    for k, start in enumerate(cue):
        stop = cue[k + 1] if k + 1 < len(cue) else data.shape[1]
        ti = target[(target > start) & (target < stop)]
        ci = clicks[(clicks > start) & (clicks < stop)]
        end = start + 1
        while end < len(data[7]) and data[7, end] == data[7, start]:
            end += 1
        rows.append(dict(record=path.stem, trial=k, sample=int(start),
                         cue=int(data[7, start]), cue_duration_s=(end-start)/fs,
                         target_sign=int(np.sign(data[8, ti[0]])) if len(ti) else np.nan,
                         target_delay_s=(ti[0]-start)/fs if len(ti) else np.nan,
                         click_sign=int(np.sign(data[8, ci[0]])) if len(ci) else np.nan,
                         click_delay_s=(ci[0]-start)/fs if len(ci) else np.nan,
                         target_count=len(ti), click_count=len(ci)))
    events = pd.DataFrame(rows)
    agreement = float(np.mean(events.cue == events.target_sign))
    inferred = 1 if agreement > .95 and len(clicks) == 0 else 2 if .25 < agreement < .75 and len(clicks) > 0 else 0
    assert inferred != 0, "Ambiguous project mapping requires review."
    assert inferred == int(path.stem.rsplit("-", 1)[1])
    events["project"] = inferred
    eeg = data[[1, 0, 2]].copy()
    ecg = data[6]
    peaks, _ = find_peaks(np.abs(ecg-np.median(ecg)), distance=int(.35*fs),
                          prominence=3*np.median(np.abs(ecg-np.median(ecg))))
    summary = dict(record=path.stem, project=inferred, group_letter=path.stem[9],
                   shape=list(data.shape), labels=labels, fs=fs,
                   duration_s=data.shape[1]/fs, timestamp_fs=1/np.median(np.diff(data[9])),
                   n_cues=len(cue), n_left=int(sum(events.cue==-1)),
                   n_right=int(sum(events.cue==1)), target_events=len(target),
                   click_events=len(clicks), cue_target_agreement=agreement,
                   cue_duration_median_s=float(events.cue_duration_s.median()),
                   target_delay_median_s=float(events.target_delay_s.median()),
                   click_delay_median_s=float(events.click_delay_s.median()) if len(clicks) else None,
                   action_values=np.unique(data[8]).tolist(),
                   nonfinite_count=int((~np.isfinite(data)).sum()),
                   ecg_std=float(ecg.std()), ecg_nonflat=bool(np.ptp(ecg)>0),
                   ecg_candidate_peak_rate_per_min=len(peaks)/len(ecg)*fs*60,
                   raw_saturation_fractions=np.mean(np.abs(eeg)>=999.9,axis=1).tolist(),
                   eeg_units="recorded amplitude unit; physical calibration unspecified",
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return dict(name=path.stem, project=inferred, fs=fs, eeg=eeg, ecg=ecg,
                events=events, summary=summary)


def explore(data_dir, out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    records=[load_record(p) for p in sorted(Path(data_dir).glob("VisualCog*_Task-*.mat"))]
    assert len(records)==4
    (out/"exploration.json").write_text(json.dumps([r["summary"] for r in records],ensure_ascii=False,indent=2),encoding="utf-8")
    pd.concat([r["events"] for r in records]).to_csv(out/"events.csv",index=False)
    pd.DataFrame([{k:v for k,v in r["summary"].items() if not isinstance(v,list)} for r in records]).to_csv(out/"recordings.csv",index=False)
    print(json.dumps([r["summary"] for r in records],ensure_ascii=False,indent=2),flush=True)
    return records


if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--out",required=True)
    a=p.parse_args(); explore(a.data,a.out)
