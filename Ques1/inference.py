"""GPU resampling, effect sizes, and constrained descriptive curve models."""
import json
import numpy as np
import torch
from scipy.optimize import curve_fit
from Ques1.src.preprocess import TIME,BASE,P300


def choose_device(requested="auto"):
    torch.set_num_threads(2)
    if requested == "cpu" or (requested == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu"), dict(device="cpu", torch_version=torch.__version__)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "auto":
        index = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])
        requested = f"cuda:{index}"
    device = torch.device(requested)
    if device.type != "cuda":
        raise ValueError("Expected auto, cpu, cuda or cuda:N")
    torch.cuda.set_device(device)
    return device, dict(device=str(device), name=torch.cuda.get_device_name(device),
                        torch_version=torch.__version__, cuda=torch.version.cuda)


def weighted_mean(x,w):
    return np.einsum("n,nct->ct",w/w.sum(),x)


def group_erp(records,condition,stage="clean",matched=True):
    groups=[]
    for r in records:
        mask=r["events"].cue.to_numpy()==condition
        if matched: mask &= r["keep"]
        x=r[f"{stage}_epochs"][mask]
        w=r["weights"][mask] if matched else np.ones(mask.sum())
        groups.append((x,w))
    return np.mean([weighted_mean(x,w) for x,w in groups],axis=0),groups


def bootstrap(groups,device,repeats=1000,seed=20260924):
    generator=torch.Generator(device=device).manual_seed(seed)
    result=None
    for x,w in groups:
        xt=torch.as_tensor(x,dtype=torch.float32,device=device)
        wt=torch.as_tensor(w,dtype=torch.float32,device=device)
        ix=torch.randint(len(x),(repeats,len(x)),device=device,generator=generator)
        counts=torch.zeros((repeats,len(x)),device=device)
        counts.scatter_add_(1,ix,torch.ones_like(ix,dtype=torch.float32))
        weights=counts*wt; weights/=weights.sum(1,keepdim=True)
        estimates=(weights@xt.reshape(len(x),-1)).reshape(repeats,3,x.shape[-1])
        result=estimates if result is None else result+estimates
    torch.cuda.synchronize(device)
    return (result/len(groups)).cpu().numpy()


def waveform_metrics(y,boot=None):
    win=y[P300]; times=TIME[P300]
    i=np.argmax(win)
    out=dict(peak=float(win[i]),latency_ms=float(times[i]*1000),mean_amplitude=float(win.mean()),
             signed_area=float(np.trapezoid(win,times)),positive_area=float(np.trapezoid(np.maximum(win,0),times)),
             snr_db=float(10*np.log10(max(np.mean(win**2),1e-20)/max(np.mean(y[BASE]**2),1e-20))),
             peak_at_window_edge=bool(i==0 or i==len(win)-1),positive_peak=bool(win[i]>0))
    if boot is not None:
        for name,value in [("peak",boot[:,P300].max(1)),("latency_ms",times[np.argmax(boot[:,P300],axis=1)]*1000),
                           ("mean_amplitude",boot[:,P300].mean(1)),("signed_area",np.trapezoid(boot[:,P300],times,axis=1))]:
            out[name+"_ci_low"],out[name+"_ci_high"]=map(float,np.quantile(value,[.025,.975]))
        noise=np.var(boot[:,P300],axis=0,ddof=1).mean()
        out["bootstrap_snr_db"]=float(10*np.log10(max(np.mean(win**2),1e-20)/max(noise,1e-20)))
        out["positive_mean_supported"]=bool(out["mean_amplitude_ci_low"]>0)
    return out


def cohen_d(left,right):
    sd=np.sqrt(((len(left)-1)*np.var(left,ddof=1)+(len(right)-1)*np.var(right,ddof=1))/(len(left)+len(right)-2))
    return float((np.mean(right)-np.mean(left))/sd) if sd>0 else np.nan


def gaussian(t,c,a,mu,sigma):
    return c+a*np.exp(-.5*((t-mu)/sigma)**2)


def double_gaussian(t,c,a0,mu0,s0,a1,mu1,s1):
    return gaussian(t,c,a0,mu0,s0)+a1*np.exp(-.5*((t-mu1)/s1)**2)


def fit_erp(y):
    mask=(TIME>=.1)&(TIME<=.65); t=TIME[mask]; z=y[mask]
    scale=max(np.ptp(z),abs(z).max(),1.)
    specs=[("gaussian",gaussian,[-4*scale,0,.25,.02],[4*scale,8*scale,.5,.18]),
           ("double_gaussian",double_gaussian,[-4*scale,-8*scale,.08,.02,0,.25,.02],
            [4*scale,8*scale,.25,.12,8*scale,.5,.18])]
    rows=[]; curves={}
    for name,func,low,high in specs:
        best=None
        for mu in [.30,.38,.46]:
            p0=[float(np.median(z)),max(np.ptp(z)/2,.01),mu,.07]
            if name=="double_gaussian": p0=[float(np.median(z)),-scale/3,.18,.05,scale/2,mu,.07]
            try:
                p,cov=curve_fit(func,t,z,p0=p0,bounds=(low,high),maxfev=10000)
                pred=func(t,*p); sse=float(np.sum((z-pred)**2))
                if best is None or sse<best[0]: best=(sse,p,cov)
            except (RuntimeError,ValueError): pass
        if best is None: continue
        sse,p,cov=best; n=len(t); k=len(p)
        full=func(TIME,*p); curves[name]=full
        r2=1-sse/max(np.sum((z-z.mean())**2),1e-20)
        wp=y[P300]; fp=full[P300]
        distance=np.minimum((p-low)/(np.array(high)-low),(np.array(high)-p)/(np.array(high)-low))
        row=dict(model=name,r2=float(r2),rmse=float(np.sqrt(sse/n)),
                 p300_r2=float(1-np.sum((wp-fp)**2)/max(np.sum((wp-wp.mean())**2),1e-20)),
                 p300_rmse=float(np.sqrt(np.mean((wp-fp)**2))),
                 aicc=float(n*np.log(max(sse/n,1e-20))+2*k+2*k*(k+1)/(n-k-1)),
                 parameters=json.dumps(p.tolist()),parameter_se=json.dumps(np.sqrt(np.diag(cov)).tolist()),
                 boundary_hit=bool(np.any(distance<.005)),offset=p[0],
                 p300_amplitude=p[-3],p300_center_ms=1000*p[-2],p300_sigma_ms=1000*p[-1],
                 early_amplitude=p[1] if k==7 else np.nan,early_center_ms=1000*p[2] if k==7 else np.nan,
                 early_sigma_ms=1000*p[3] if k==7 else np.nan)
        rows.append(row)
    winner=min(rows,key=lambda r:r["aicc"])
    if len(rows)==2 and abs(rows[0]["aicc"]-rows[1]["aicc"])<2: winner=rows[0]
    for row in rows: row["selected"]=(row is winner)
    return rows,curves[winner["model"]],winner

