"""Supplementary sensitivity, morphology, fit validation, and exported trials."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from dataio import CHANNELS
from preprocess import TIME,P300,mad
from inference import weighted_mean,waveform_metrics,group_erp
from plotting import axes_grid,save,COLORS
import matplotlib.pyplot as plt


def flexible(t,c,b,a1,m1,s1,a2,m2,s2):
    return c+b*t+a1*np.exp(-.5*((t-m1)/s1)**2)+a2*np.exp(-.5*((t-m2)/s2)**2)


def fit_flexible(y):
    mask=(TIME>=0)&(TIME<=.8); t=TIME[mask]; z=y[mask]
    scale=max(np.ptp(z),np.max(np.abs(z)),1.)
    lo=[-3*scale,-6*scale,-5*scale,.12,.025,-5*scale,.45,.025]
    hi=[3*scale,6*scale,5*scale,.45,.2,5*scale,.8,.3]
    best=None
    for sign in [-1,1]:
        for m2 in [.6,.7]:
            p0=[z[0],0,sign*scale/2,.33,.07,scale,.65,.10]
            p0[-2]=m2
            try:
                p,cov=curve_fit(flexible,t,z,p0=p0,bounds=(lo,hi),maxfev=20000)
                mse=np.mean((z-flexible(t,*p))**2)
                if best is None or mse<best[0]: best=(mse,p,cov)
            except (ValueError,RuntimeError): pass
    mse,p,cov=best
    pred=flexible(TIME,*p)
    distance=np.minimum((p-lo)/(np.array(hi)-lo),(np.array(hi)-p)/(np.array(hi)-lo))
    return dict(r2=float(1-mse/np.var(z)),rmse=float(np.sqrt(mse)),offset=p[0],slope=p[1],
                a1=p[2],mu1_ms=p[3]*1000,sigma1_ms=p[4]*1000,a2=p[5],mu2_ms=p[6]*1000,sigma2_ms=p[7]*1000,
                boundary_hit=bool(np.any(distance<.005)),parameter_se=json.dumps(np.sqrt(np.diag(cov)).tolist())),pred


def supplement(records,out):
    out=Path(out); rows=[]; candidates=[]; features=[]; rec_curves=[]; fits=[]; fitcurves=[]; cross=[]
    for r in records:
        np.savez_compressed(out/(r["name"]+"_epochs.npz"),time_s=TIME,raw=r["raw_epochs"],clean=r["clean_epochs"],
                            labels=r["events"].cue.to_numpy(),samples=r["events"]["sample"].to_numpy(),keep=r["keep"],weights=r["weights"],channels=CHANNELS)
        for i in np.flatnonzero(r["keep"]):
            for c,ch in enumerate(CHANNELS):
                y=r["clean_epochs"][i,c]
                features.append(dict(record=r["name"],project=r["project"],trial=i,condition=r["events"].cue.iloc[i],channel=ch,
                                     weight=r["weights"][i],mean_amplitude=y[P300].mean(),peak=y[P300].max(),signed_area=np.trapezoid(y[P300],TIME[P300])))
        for cond in [-1,1]:
            mask=(r["events"].cue.to_numpy()==cond)&r["keep"]
            mean=weighted_mean(r["clean_epochs"][mask],r["weights"][mask])
            for c,ch in enumerate(CHANNELS):
                rec_curves.extend(dict(record=r["name"],project=r["project"],condition=cond,channel=ch,time_ms=t*1000,amplitude=y) for t,y in zip(TIME,mean[c]))
    for project in [1,2]:
        rr=[r for r in records if r["project"]==project]
        for cond in [-1,1]:
            mean,_=group_erp(rr,cond)
            for c,ch in enumerate(CHANNELS):
                ids,props=find_peaks(mean[c],prominence=0)
                valid=[(i,p) for i,p in zip(ids,props["prominences"]) if .25<=TIME[i]<=.5 and mean[c,i]>0]
                if valid:
                    i,pr=max(valid,key=lambda item:item[1])
                    candidates.append(dict(project=project,condition=cond,channel=ch,peak=mean[c,i],latency_ms=TIME[i]*1000,prominence=pr))
                else: candidates.append(dict(project=project,condition=cond,channel=ch,peak=np.nan,latency_ms=np.nan,prominence=np.nan))
                fit,pred=fit_flexible(mean[c]); fits.append(dict(project=project,condition=cond,channel=ch,**fit))
                fitcurves.extend(dict(project=project,condition=cond,channel=ch,time_ms=t*1000,fitted=y) for t,y in zip(TIME,pred) if 0<=t<=.8)
                for train,test in [(rr[0],rr[1]),(rr[1],rr[0])]:
                    tm,_=group_erp([train],cond); vm,_=group_erp([test],cond)
                    _,prediction=fit_flexible(tm[c]); use=(TIME>=0)&(TIME<=.8)
                    cross.append(dict(project=project,condition=cond,channel=ch,train=train["name"],test=test["name"],
                                      r2=1-np.sum((vm[c,use]-prediction[use])**2)/max(np.sum((vm[c,use]-vm[c,use].mean())**2),1e-20),
                                      rmse=np.sqrt(np.mean((vm[c,use]-prediction[use])**2))))
            for mode in ["primary","equal_weights","strict4mad"]:
                means=[]; n=0
                for r in rr:
                    keep=r["keep"].copy()
                    if mode=="strict4mad":
                        q=r["qc"][["baseline_rms","peak_to_peak","max_jump","hf_rms"]].to_numpy()
                        keep &= (q<=np.median(q,axis=0)+4*np.maximum(mad(q,axis=0),1e-8)).all(1)
                    mask=keep&(r["events"].cue.to_numpy()==cond)
                    w=np.ones(mask.sum()) if mode=="equal_weights" else r["weights"][mask]
                    means.append(weighted_mean(r["clean_epochs"][mask],w)); n+=mask.sum()
                y=np.mean(means,axis=0)
                for c,ch in enumerate(CHANNELS): rows.append(dict(project=project,condition=cond,channel=ch,mode=mode,n=n,**waveform_metrics(y[c])))
    for data,name in [(rows,"aggregation_sensitivity.csv"),(candidates,"local_peak_candidates.csv"),(features,"trial_features.csv"),
                      (rec_curves,"record_erp_curves.csv"),(fits,"descriptive_fit_parameters.csv"),(fitcurves,"descriptive_fit_curves.csv"),(cross,"fit_cross_record.csv")]:
        pd.DataFrame(data).to_csv(out/name,index=False)
    wave=pd.read_csv(out/"erp_curves.csv"); fitted=pd.DataFrame(fitcurves)
    for project in [1,2]:
        fig,axs=axes_grid()
        for row,(cond,label) in enumerate([(-1,"左靶"),(1,"右靶")]):
            for c,ch in enumerate(CHANNELS):
                ax=axs[row,c]
                w=wave[(wave.project==project)&(wave.condition==cond)&(wave.channel==ch)&(wave.stage=="clean")&(wave.time_ms>=0)&(wave.time_ms<=800)]
                f=fitted[(fitted.project==project)&(fitted.condition==cond)&(fitted.channel==ch)]
                ax.plot(w.time_ms,w.amplitude,color=COLORS["clean"],label="去噪响应")
                ax.plot(f.time_ms,f.fitted,color=COLORS["right"],ls="--",label="双高斯与线性项")
                ax.axvspan(250,500,color=".9",zorder=0); ax.axhline(0,color=".7",lw=.7)
                ax.set(xlim=(0,800),xticks=[0,250,500,800],xlabel="刺激后时间（毫秒）",ylabel="幅值（原记录单位）",title=f"{ch}  {label}")
        fig.legend(*axs[0,0].get_legend_handles_labels(),loc="upper center",ncol=2,bbox_to_anchor=(.5,1.0))
        save(fig,out/"figures"/f"project{project}_descriptive_fits")
