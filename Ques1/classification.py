"""Leakage-controlled GPU ridge classification with fixed features and penalty."""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,accuracy_score,balanced_accuracy_score,roc_curve
from Ques1.src.preprocess import TIME


def features(epochs,window="p300",channel=None):
    edges=np.linspace(.25,.5,6) if window=="p300" else np.linspace(-.199,0,6)
    x=epochs if channel is None else epochs[:,[channel]]
    return np.concatenate([x[..., (TIME>=a)&(TIME<b)].mean(-1) for a,b in zip(edges[:-1],edges[1:])],axis=1)


def gpu_predictions(x,ys,splits,device):
    xt=torch.tensor(x,dtype=torch.float64,device=device)
    yt=torch.tensor(ys,dtype=torch.float64,device=device)
    pred=torch.zeros_like(yt)
    for train,test in splits:
        tr=torch.as_tensor(train,device=device); te=torch.as_tensor(test,device=device)
        mu=xt[tr].mean(0); sd=xt[tr].std(0).clamp_min(1e-8)
        a=(xt[tr]-mu)/sd; b=(xt[te]-mu)/sd
        a=torch.cat([torch.ones((len(train),1),device=device,dtype=torch.float64),a],1)
        b=torch.cat([torch.ones((len(test),1),device=device,dtype=torch.float64),b],1)
        penalty=torch.eye(a.shape[1],device=device,dtype=torch.float64)*10
        penalty[0,0]=0
        coef=torch.linalg.solve(a.T@a+penalty,a.T@yt[tr])
        pred[te]=b@coef
    return pred.cpu().numpy()


def evaluate(records,device,repeats=500):
    rows=[]; predictions=[]; rocrows=[]
    rng=np.random.default_rng(20260924)
    for project in [1,2]:
        rr=[r for r in records if r["project"]==project]
        y=np.concatenate([r["events"].cue.to_numpy()[r["keep"]] for r in rr])
        groups=np.concatenate([np.full(sum(r["keep"]),i) for i,r in enumerate(rr)])
        trials=np.concatenate([r["events"].trial.to_numpy()[r["keep"]] for r in rr])
        blocks=trials//20
        ys=np.repeat(y[:,None],repeats+1,axis=1)
        for i in range(repeats):
            for g in [0,1]:
                for block in range(5):
                    ids=np.flatnonzero((groups==g)&(blocks==block))
                    ys[ids,i+1]=rng.permutation(y[ids])
        splits_record=[(np.flatnonzero(groups!=g),np.flatnonzero(groups==g)) for g in [0,1]]
        splits_block=[]
        for b in range(5):
            test=np.flatnonzero(blocks==b)
            train=np.flatnonzero((trials < b*20-2)|(trials >= (b+1)*20+2))
            splits_block.append((train,test))
        for stage in ["raw","clean"]:
            e=np.concatenate([r[f"{stage}_epochs"][r["keep"]] for r in rr])
            settings=[("p300","all",None),("baseline","all",None)]+[("p300",ch,c) for c,ch in enumerate(["F3","Fz","F4"])]
            for window,ch,c in settings:
                x=features(e,window,c)
                for scheme,splits in [("leave_record_out",splits_record),("purged_block5",splits_block)]:
                    if ch!="all" and scheme!="leave_record_out": continue
                    pred=gpu_predictions(x,ys if ch=="all" else ys[:,:1],splits,device)
                    score=pred[:,0]; labels=(score>=0)*2-1
                    auc=roc_auc_score(y,score); acc=accuracy_score(y,labels)
                    null=np.array([roc_auc_score(ys[:,i],pred[:,i]) for i in range(1,pred.shape[1])])
                    cis=[]
                    for _ in range(500):
                        ids=np.concatenate([rng.choice(np.flatnonzero(groups==g),sum(groups==g),replace=True) for g in [0,1]])
                        if len(np.unique(y[ids]))==2: cis.append([roc_auc_score(y[ids],score[ids]),accuracy_score(y[ids],labels[ids])])
                    ci=np.quantile(cis,[.025,.975],axis=0)
                    row=dict(project=project,stage=stage,window=window,channel=ch,scheme=scheme,n=len(y),
                             auc=auc,auc_ci_low=ci[0,0],auc_ci_high=ci[1,0],accuracy=acc,
                             accuracy_ci_low=ci[0,1],accuracy_ci_high=ci[1,1],
                             balanced_accuracy=balanced_accuracy_score(y,labels),majority_accuracy=max(np.mean(y==1),np.mean(y==-1)),
                             permutation_p=(1+sum(null>=auc))/(1+len(null)) if len(null) else np.nan,
                             permutations=len(null),ridge_penalty=10,features=x.shape[1])
                    rows.append(row)
                    if ch=="all":
                        for i in range(len(y)):
                            predictions.append(dict(project=project,stage=stage,window=window,scheme=scheme,
                                                    record=rr[groups[i]]["name"],trial=trials[i],label=y[i],score=score[i],predicted=labels[i]))
                        fpr,tpr,_=roc_curve(y,score)
                        rocrows.extend(dict(project=project,stage=stage,window=window,scheme=scheme,fpr=a,tpr=b) for a,b in zip(fpr,tpr))
    return pd.DataFrame(rows),pd.DataFrame(predictions),pd.DataFrame(rocrows)
