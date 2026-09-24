"""Conservative raw-signal preprocessing and label-blind quality control."""
import numpy as np
import pandas as pd
import pywt
from scipy import signal

PRE, POST = 51, 205
TIME = np.arange(-PRE, POST + 1) / 256
BASE = TIME < 0
P300 = (TIME >= .25) & (TIME <= .5)


def mad(x, axis=None):
    m=np.median(x,axis=axis,keepdims=True)
    return 1.4826*np.median(np.abs(x-m),axis=axis)


def wavelet_high_only(x):
    coeff=pywt.wavedec(x,"sym4",level=4,axis=-1,mode="symmetric")
    sigma=np.median(np.abs(coeff[-1]),axis=-1,keepdims=True)/.67448975
    for k in range(1,4):
        threshold=sigma*np.sqrt(2*np.log(coeff[-k].shape[-1]))
        coeff[-k]=pywt.threshold(coeff[-k],threshold,mode="soft")
    return pywt.waverec(coeff,"sym4",axis=-1,mode="symmetric")[...,:x.shape[-1]]


def filter_record(eeg,fs=256,hp=.1,wavelet=True):
    x=eeg.copy()
    for c in range(3):
        good=np.abs(x[c])<999.9
        x[c,~good]=np.interp(np.flatnonzero(~good),np.flatnonzero(good),x[c,good])
    b,a=signal.iirnotch(50,30,fs)
    x=signal.filtfilt(b,a,x,axis=-1)
    sos=signal.butter(4,[hp,30],btype="bandpass",fs=fs,output="sos")
    x=signal.sosfiltfilt(sos,x,axis=-1,padlen=int(fs*15))
    return wavelet_high_only(x) if wavelet else x


def epochs(x,samples):
    return np.stack([x[:,i-PRE:i+POST+1] for i in samples])


def baseline(x):
    return x-x[...,BASE].mean(-1,keepdims=True)


def quality(raw,clean,samples):
    re=epochs(raw,samples); ce=epochs(clean,samples)
    base_rms=np.sqrt(np.mean(baseline(ce)[...,BASE]**2,axis=-1)).max(1)
    ptp=np.ptp(ce,axis=-1).max(1)
    jump=np.max(np.abs(np.diff(re,axis=-1)),axis=(1,2))
    hf=signal.sosfiltfilt(signal.butter(3,[35,100],fs=256,btype="bandpass",output="sos"),raw,axis=-1)
    hf_rms=np.sqrt(np.mean(epochs(hf,samples)**2,axis=(1,2)))
    features=np.column_stack([base_rms,ptp,jump,hf_rms])
    thresholds=np.median(features,axis=0)+8*np.maximum(mad(features,axis=0),1e-8)
    flag=features>thresholds
    saturation=np.array([np.any(np.abs(raw[:,max(0,s-512):min(raw.shape[1],s+512)])>=999.9) for s in samples])
    flat=np.any(np.std(re,axis=-1)<1e-8,axis=1)
    keep=~(saturation|flat|flag.any(1))
    scale=max(np.median(base_rms[keep]),1e-6)
    weights=np.clip(scale/np.maximum(base_rms,scale),.2,1.)
    reasons=[";".join(n for n,b in zip(["saturation_within_2s","flat","baseline_outlier","amplitude_outlier","gradient_outlier","high_frequency_outlier"],[saturation[i],flat[i],*flag[i]]) if b) for i in range(len(samples))]
    table=pd.DataFrame(dict(keep=keep,weight=weights,reason=reasons,baseline_rms=base_rms,peak_to_peak=ptp,max_jump=jump,hf_rms=hf_rms))
    return keep,weights,table,thresholds


def band_energy(x,lo,hi):
    f,p=signal.welch(x,fs=256,nperseg=8192,axis=-1)
    use=(f>=lo)&(f<=hi)
    if lo>=30:
        use &= ~(((f>=48)&(f<=52))|((f>=98)&(f<=102)))
    return p[:,use].sum(1)*(f[1]-f[0])


def prepare(record):
    raw=record["eeg"]; samples=record["events"]["sample"].to_numpy()
    filtered=filter_record(raw,wavelet=False)
    clean=wavelet_high_only(filtered)
    keep,weights,qc,thresholds=quality(raw,clean,samples)
    events=record["events"].reset_index(drop=True)
    record.update(raw_epochs=baseline(epochs(raw,samples)),filtered_epochs=baseline(epochs(filtered,samples)),
                  clean_epochs=baseline(epochs(clean,samples)),keep=keep,weights=weights,
                  qc=pd.concat([events,qc],axis=1),thresholds=thresholds)
    rows=[]
    for name,x in [("raw",raw),("filtered",filtered),("clean",clean)]:
        drift=band_energy(x,.03125,.5); high=band_energy(x,30,100)
        e=baseline(epochs(x,samples))[keep]
        slope=np.polyfit(TIME[BASE],epochs(x,samples)[keep][...,BASE].reshape(-1,BASE.sum()).T,1)[0].reshape(-1,3)
        for c,ch in enumerate(["F3","Fz","F4"]):
            rows.append(dict(record=record["name"],project=record["project"],channel=ch,stage=name,
                             drift_energy=drift[c],high_frequency_energy=high[c],
                             baseline_slope_energy=float(np.mean(slope[:,c]**2)*np.var(TIME[BASE])),
                             trial_variance=float(np.var(e[:,c,P300],axis=0,ddof=1).mean()),
                             kept=int(keep.sum()),total=len(keep)))
    record["artifact_metrics"]=rows
    record["continuous_clean"]=clean
    return record


def injection_check(record):
    n=record["eeg"].shape[1]
    pulse=np.zeros((3,n)); centers=record["events"]["sample"].to_numpy()[record["keep"]]
    grid=np.arange(-256,257)/256
    template=5*np.exp(-.5*((grid-.35)/.07)**2)
    for s in centers:
        pulse[:,s-256:s+257]+=template
    reference=baseline(epochs(pulse,centers)).mean(0)
    recovered=baseline(epochs(filter_record(record["eeg"]+pulse)-record["continuous_clean"],centers)).mean(0)
    rows=[]
    for c,ch in enumerate(["F3","Fz","F4"]):
        a,b=reference[c],recovered[c]
        rows.append(dict(record=record["name"],channel=ch,synthetic_amplitude=5,
                         peak_retention=float(b[P300].max()/a[P300].max()),
                         latency_error_ms=float((TIME[P300][np.argmax(b[P300])]-TIME[P300][np.argmax(a[P300])])*1000),
                         area_retention=float(np.trapezoid(b[P300],TIME[P300])/np.trapezoid(a[P300],TIME[P300])),
                         correlation=float(np.corrcoef(a,b)[0,1])))
    return rows

