"""Run Question 1 analyses with reproducible flat outputs."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import platform
import time
import numpy as np
import pandas as pd
from Ques1.src.dataio import explore,CHANNELS
from Ques1.src.preprocess import prepare,injection_check,TIME,BASE,P300,filter_record,epochs,baseline
from Ques1.src.inference import choose_device,group_erp,bootstrap,waveform_metrics,cohen_d,fit_erp,weighted_mean
from Ques1.src.classification import evaluate
from Ques1.src.plotting import figures
from Ques1.src.supplement import supplement


def write_csv(rows,out,name):
    pd.DataFrame(rows).to_csv(out/name,index=False)


def main(argv=None):
    parser=argparse.ArgumentParser()
    from pipeline import DATA, ROOT, output_run
    parser.add_argument("--data",type=Path,default=DATA); parser.add_argument("--out",type=Path,default=ROOT/"Ques1/output")
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--device",default="auto")
    parser.add_argument("--font"); parser.add_argument("--bootstrap",type=int,default=1000)
    parser.add_argument("--permutations",type=int,default=500)
    args=parser.parse_args(argv)
    if args.bootstrap < 2 or args.permutations < 1:
        parser.error("bootstrap must be >= 2 and permutations >= 1")
    with output_run(args.out,"Ques1",args.overwrite) as out:
        run(args,out)


def run(args,out):
    start=time.time(); device,hardware=choose_device(args.device)
    records=explore(args.data,out)
    for r in records:
        prepare(r); print("QC",r["name"],int(r["keep"].sum()),"/",len(r["keep"]),flush=True)
        assert r["keep"].sum()>=20
    pd.concat([r["qc"] for r in records]).to_csv(out/"trial_quality.csv",index=False)
    write_csv([row for r in records for row in r["artifact_metrics"]],out,"artifact_metrics.csv")
    write_csv([dict(record=r["name"],**dict(zip(["baseline_rms","peak_to_peak","max_jump","hf_rms"],r["thresholds"]))) for r in records],out,"quality_thresholds.csv")
    curves=[]; metrics=[]; fits=[]; fitcurves=[]; effects=[]; recordmetrics=[]; artifact_extra=[]
    rng=np.random.default_rng(20260924)
    for project in [1,2]:
        rr=[r for r in records if r["project"]==project]; means={}; boot={}
        for cond in [-1,1]:
            means[cond],groups=group_erp(rr,cond)
            boot[cond]=bootstrap(groups,device,args.bootstrap,seed=20260924+project*10+cond)
            for stage in ["raw_all","raw","filtered","clean"]:
                mean,_=group_erp(rr,cond,"raw" if stage=="raw_all" else stage,matched=stage!="raw_all")
                for c,ch in enumerate(CHANNELS):
                    b=boot[cond][:,c] if stage=="clean" else None
                    row=dict(project=project,condition=cond,channel=ch,stage=stage,**waveform_metrics(mean[c],b))
                    row["n_left_or_right"]=int(sum(np.sum((r["events"].cue.to_numpy()==cond)&(r["keep"] if stage!="raw_all" else True)) for r in rr))
                    metrics.append(row)
                    low,high=np.quantile(b,[.025,.975],axis=0) if b is not None else (np.full(len(TIME),np.nan),np.full(len(TIME),np.nan))
                    curves.extend(dict(project=project,condition=cond,channel=ch,stage=stage,time_ms=t*1000,amplitude=a,ci_low=l,ci_high=h) for t,a,l,h in zip(TIME,mean[c],low,high))
            for c,ch in enumerate(CHANNELS):
                rows,fit,winner=fit_erp(means[cond][c])
                fits.extend(dict(project=project,condition=cond,channel=ch,**r) for r in rows)
                fitcurves.extend(dict(project=project,condition=cond,channel=ch,time_ms=t*1000,fitted=y,model=winner["model"]) for t,y in zip(TIME,fit) if .1<=t<=.65)
        diff=means[1]-means[-1]; db=boot[1]-boot[-1]
        for c,ch in enumerate(CHANNELS):
            lo,hi=np.quantile(db[:,c],[.025,.975],axis=0)
            curves.extend(dict(project=project,condition=0,channel=ch,stage="difference",time_ms=t*1000,amplitude=y,ci_low=l,ci_high=h) for t,y,l,h in zip(TIME,diff[c],lo,hi))
            for stage in ["raw","clean"]:
                for scope in ["pooled",*[r["name"] for r in rr]]:
                    selected=rr if scope=="pooled" else [r for r in rr if r["name"]==scope]
                    vals={cond:np.concatenate([r[f"{stage}_epochs"][(r["events"].cue.to_numpy()==cond)&r["keep"]][:,c,P300].mean(1) for r in selected]) for cond in [-1,1]}
                    ds=[cohen_d(rng.choice(vals[-1],len(vals[-1]),replace=True),rng.choice(vals[1],len(vals[1]),replace=True)) for _ in range(1000)]
                    dl,dh=np.quantile(ds,[.025,.975])
                    effects.append(dict(project=project,scope=scope,stage=stage,channel=ch,cohen_d=cohen_d(vals[-1],vals[1]),ci_low=dl,ci_high=dh,
                                        n_left=len(vals[-1]),n_right=len(vals[1]),right_minus_left=vals[1].mean()-vals[-1].mean()))
        print("ERP completed",project,flush=True)
    for r in records:
        for cond in [-1,1]:
            mask=(r["events"].cue.to_numpy()==cond)&r["keep"]
            for stage in ["raw","clean"]:
                mean=weighted_mean(r[f"{stage}_epochs"][mask],r["weights"][mask])
                for c,ch in enumerate(CHANNELS):
                    recordmetrics.append(dict(record=r["name"],project=r["project"],condition=cond,channel=ch,stage=stage,n=int(mask.sum()),**waveform_metrics(mean[c])))
        for c,ch in enumerate(CHANNELS):
            for cond in [-1,1]:
                mask=(r["events"].cue.to_numpy()==cond)&r["keep"]
                raw=weighted_mean(r["raw_epochs"][mask],r["weights"][mask])[c]
                clean=weighted_mean(r["clean_epochs"][mask],r["weights"][mask])[c]
                artifact_extra.append(dict(record=r["name"],project=r["project"],channel=ch,condition=cond,
                                           p300_waveform_correlation=float(np.corrcoef(raw[P300],clean[P300])[0,1]),
                                           matched_peak_ratio=float(clean[P300].max()/raw[P300].max()) if abs(raw[P300].max())>1e-9 else np.nan))
    for rows,name in [(curves,"erp_curves.csv"),(metrics,"p300_metrics.csv"),(fits,"fit_parameters.csv"),(fitcurves,"fit_curves.csv"),
                      (effects,"effect_sizes.csv"),(recordmetrics,"record_p300_metrics.csv"),(artifact_extra,"preservation_metrics.csv")]: write_csv(rows,out,name)
    print("Classification started",flush=True)
    cls,pred,roc=evaluate(records,device,args.permutations)
    cls.to_csv(out/"classification.csv",index=False); pred.to_csv(out/"predictions.csv",index=False); roc.to_csv(out/"roc_curves.csv",index=False)
    print("Injection and sensitivity checks",flush=True)
    write_csv([row for r in records for row in injection_check(r)],out,"synthetic_recovery.csv")
    sens=[]; sens_metrics=[]
    for hp in [.05,.1,.3]:
        for r in records:
            x=r["continuous_clean"] if hp==.1 else filter_record(r["eeg"],hp=hp)
            r["sensitivity_epochs"]=baseline(epochs(x,r["events"]["sample"].to_numpy()))
        for project in [1,2]:
            rr=[r for r in records if r["project"]==project]
            mean=np.mean([group_erp(rr,c,"sensitivity")[0] for c in [-1,1]],axis=0)
            for c,ch in enumerate(CHANNELS):
                sens.extend(dict(project=project,channel=ch,highpass=hp,time_ms=t*1000,amplitude=y) for t,y in zip(TIME,mean[c]))
                sens_metrics.append(dict(project=project,channel=ch,highpass=hp,**waveform_metrics(mean[c])))
    write_csv(sens,out,"sensitivity_curves.csv"); write_csv(sens_metrics,out,"sensitivity_metrics.csv")
    environment=dict(hardware=hardware,python=platform.python_version(),platform=platform.platform(),
                     versions={n:importlib.metadata.version(n) for n in ["numpy","scipy","pandas","matplotlib","scikit-learn","PyWavelets","torch"]},
                     bootstrap=args.bootstrap,permutations=args.permutations,seed=20260924,
                     channels_used_for_analysis=[1,0,2],channel_index_base=0,
                     excluded_channels_zero_based=[3,4,5],pooling="equal record weights within each cue condition",
                     confidence_intervals="conditional within-record trial bootstrap; not population intervals",
                     elapsed_seconds=time.time()-start)
    (out/"run_manifest.json").write_text(json.dumps(environment,indent=2),encoding="utf-8")
    figures(out,args.font)
    supplement(records,out)
    print("Completed",json.dumps(environment),flush=True)


if __name__=="__main__": main()



