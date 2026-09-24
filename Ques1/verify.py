"""Run reproducibility and scientific-invariant checks on the server."""
from pathlib import Path
import tempfile
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat,savemat
from Ques1.src.dataio import load_record,transitions
from Ques1.src.preprocess import TIME,BASE,P300
from Ques1.src.classification import gpu_predictions


def verify(data, out, device):
    out=Path(out); checks={}
    assert np.array_equal(transitions(np.array([0,-1,-1,0,1,1,0,-1])),[1,4,7])
    checks["event_onsets_not_nonzero_samples"]=True
    path=next(Path(data).glob("VisualCogA_Task-1.mat")); before=load_record(path)
    with tempfile.TemporaryDirectory(prefix="q1_integrity_") as tmp:
        changed=loadmat(path); changed["data"][3:6]=123456789.
        dest=Path(tmp)/path.name; savemat(dest,{k:v for k,v in changed.items() if not k.startswith("_")})
        after=load_record(dest)
        assert np.array_equal(before["eeg"],after["eeg"])
        assert before["events"].equals(after["events"])
    checks["machine_filtered_channels_do_not_affect_input"]=True
    assert np.all(TIME[P300]>=.25) and np.all(TIME[P300]<=.5) and np.all(TIME[BASE]<0)
    checks["epoch_windows"]=True
    assert len(list(out.glob("*_epochs.npz")))==4
    for p in out.glob("*_epochs.npz"):
        z=np.load(p)
        assert np.allclose(z["clean"][...,BASE].mean(-1),0,atol=1e-9)
        assert z["clean"].shape==z["raw"].shape==(100,3,len(TIME))
        assert z["keep"].sum()>=20
    checks["saved_epochs_shape_and_baseline"]=True
    cls=pd.read_csv(out/"classification.csv")
    assert cls.auc.between(0,1).all() and cls.accuracy.between(0,1).all()
    assert set(cls.scheme)=={"leave_record_out","purged_block5"}
    checks["classification_metrics_and_validation_schemes"]=True
    recovery=pd.read_csv(out/"synthetic_recovery.csv")
    assert recovery.peak_retention.between(.97,1.03).all()
    assert (recovery.latency_error_ms.abs()<=1000/256).all()
    checks["injected_pulse_recovery"]=True
    rng=np.random.default_rng(7)
    x=rng.normal(size=(20,3)); y=np.tile([-1,1],10)[:,None]
    splits=[(np.arange(10),np.arange(10,20))]
    a=gpu_predictions(x,y,splits,device); y2=y.copy(); y2[10:]*=-1
    b=gpu_predictions(x,y2,splits,device)
    assert np.allclose(a[10:],b[10:])
    checks["heldout_labels_never_enter_classifier_fit"]=True
    (out/"verification.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    return checks

