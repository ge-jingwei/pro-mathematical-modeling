"""Generate traceable scientific figures and audit panel geometry."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from Ques1.qa.audit_panel_alignment import require_matplotlib_panel_alignment

COLORS={"left":"#0072B2","right":"#D55E00","raw":"#949494","clean":"#0072B2"}


def setup(font=None):
    if not font:
        font=next((str(p) for p in [Path("C:/Windows/Fonts/msyh.ttc"),Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")] if p.exists()),None)
    if font:
        font_manager.fontManager.addfont(font)
        name=font_manager.FontProperties(fname=font).get_name()
    else: name="DejaVu Sans"
    plt.rcParams.update({"font.family":name,"font.size":9,"axes.titlesize":10,"axes.labelsize":9,
                         "axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False,
                         "legend.frameon":False,"pdf.fonttype":42,"svg.fonttype":"none","lines.linewidth":1.4})


def save(fig,path):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(path)+".alignment.json",strict=True,
                                      tolerance_pt=1.5,gutter_tolerance_pt=1.5)
    for ext in ["png","pdf","svg"]: fig.savefig(str(path)+"."+ext,dpi=300)
    plt.close(fig)


def axes_grid(rows=2,cols=3):
    fig,axs=plt.subplots(rows,cols,figsize=(10,5.8) if rows==2 else (10,3.2),squeeze=False)
    fig.subplots_adjust(left=.09,right=.975,bottom=.17 if rows==1 else .13,top=.74 if rows==1 else .90,hspace=.48,wspace=.32)
    return fig,axs


def decorate(ax,title):
    ax.set_title(title,pad=7)
    ax.axvline(0,color=".65",lw=.7,ls=":")
    ax.axhline(0,color=".75",lw=.7)
    ax.axvspan(250,500,color="#E5E5E5",alpha=.45,zorder=0)
    ax.set_xlim(-200,800); ax.set_xticks([-200,0,250,500,800])
    ax.set_xlabel("刺激后时间（毫秒）")
    ax.set_ylabel("幅值（原记录单位）")


def figures(out,font=None):
    out=Path(out); dest=out
    setup(font)
    wave=pd.read_csv(out/"erp_curves.csv"); fits=pd.read_csv(out/"fit_curves.csv")
    for project in [1,2]:
        w=wave[wave.project==project]
        fig,axs=axes_grid()
        for row,(cond,label) in enumerate([(-1,"左靶"),(1,"右靶")]):
            for c,ch in enumerate(["F3","Fz","F4"]):
                ax=axs[row,c]
                for stage,display,color,ls in [("raw_all","原始全部","#BBBBBB",":"),("raw","原始同试次","#777777","--"),("clean","去噪后",COLORS["clean"],"-")]:
                    z=w[(w.condition==cond)&(w.channel==ch)&(w.stage==stage)]
                    ax.plot(z.time_ms,z.amplitude,color=color,ls=ls,label=display)
                decorate(ax,f"{ch}  {label}")
        fig.legend(*axs[0,0].get_legend_handles_labels(),loc="upper center",ncol=3,bbox_to_anchor=(.5,1.0))
        save(fig,dest/f"project{project}_raw_clean")
        fig,axs=axes_grid()
        for c,ch in enumerate(["F3","Fz","F4"]):
            for cond,label,color in [(-1,"左靶",COLORS["left"]),(1,"右靶",COLORS["right"])]:
                z=w[(w.condition==cond)&(w.channel==ch)&(w.stage=="clean")]
                axs[0,c].plot(z.time_ms,z.amplitude,color=color,label=label)
                axs[0,c].fill_between(z.time_ms,z.ci_low,z.ci_high,color=color,alpha=.15,lw=0)
            z=w[(w.condition==0)&(w.channel==ch)&(w.stage=="difference")]
            axs[1,c].plot(z.time_ms,z.amplitude,color="#6C4D90")
            axs[1,c].fill_between(z.time_ms,z.ci_low,z.ci_high,color="#6C4D90",alpha=.18,lw=0)
            decorate(axs[0,c],f"{ch}  左右响应"); decorate(axs[1,c],f"{ch}  右减左差异波")
        fig.legend(*axs[0,0].get_legend_handles_labels(),loc="upper center",ncol=2,bbox_to_anchor=(.5,1.0))
        save(fig,dest/f"project{project}_left_right_difference")
        fig,axs=axes_grid()
        for row,(cond,label) in enumerate([(-1,"左靶"),(1,"右靶")]):
            for c,ch in enumerate(["F3","Fz","F4"]):
                ax=axs[row,c]; z=w[(w.condition==cond)&(w.channel==ch)&(w.stage=="clean")]
                f=fits[(fits.project==project)&(fits.condition==cond)&(fits.channel==ch)]
                z=z[(z.time_ms>=100)&(z.time_ms<=650)]
                ax.plot(z.time_ms,z.amplitude,color=COLORS["clean"],label="去噪响应")
                ax.plot(f.time_ms,f.fitted,color=COLORS["right"],ls="--",label="拟合曲线")
                ax.set_title(f"{ch}  {label}",pad=7); ax.axhline(0,color=".75",lw=.7)
                ax.axvspan(250,500,color="#E5E5E5",alpha=.45,zorder=0)
                ax.set(xlim=(80,670),xticks=[100,250,400,550,650],xlabel="刺激后时间（毫秒）",ylabel="幅值（原记录单位）")
        fig.legend(*axs[0,0].get_legend_handles_labels(),loc="upper center",ncol=2,bbox_to_anchor=(.5,1.0))
        save(fig,dest/f"project{project}_fits")
    roc=pd.read_csv(out/"roc_curves.csv")
    fig,axs=plt.subplots(1,2,figsize=(8,3.8)); fig.subplots_adjust(left=.09,right=.97,bottom=.16,top=.78,wspace=.35)
    for project,ax in zip([1,2],axs):
        for stage,window,label,color,ls in [("raw","p300","原始", "#888888","--"),("clean","p300","去噪",COLORS["clean"],"-"),("clean","baseline","刺激前对照",COLORS["right"],":")]:
            z=roc[(roc.project==project)&(roc.stage==stage)&(roc.window==window)&(roc.scheme=="leave_record_out")]
            ax.plot(z.fpr,z.tpr,color=color,ls=ls,label=label)
        ax.plot([0,1],[0,1],color=".7",lw=.7,ls="--"); ax.set(xlim=(0,1),ylim=(0,1),xlabel="假阳性率",ylabel="真阳性率",title=f"项目{project}  跨记录验证")
    fig.legend(*axs[0].get_legend_handles_labels(),loc="upper center",ncol=3,bbox_to_anchor=(.5,.99))
    save(fig,dest/"classification_roc")
    artifact=pd.read_csv(out/"artifact_metrics.csv")
    fig,axs=axes_grid(1,3)
    for c,(metric,label) in enumerate([("drift_energy","低频漂移能量"),("high_frequency_energy","高频肌电代理能量"),("trial_variance","试次间方差")]):
        a=artifact.pivot(index=["record","channel"],columns="stage",values=metric)
        for record,g in a.groupby(level=0):
            axs[0,c].plot([0,1],np.maximum([g.raw.mean(),g.clean.mean()],1e-10),marker="o",label=record.replace("VisualCog",""))
        axs[0,c].set_xticks([0,1],["原始","去噪"]); axs[0,c].set_yscale("log"); axs[0,c].set_title(label); axs[0,c].set_ylabel("能量或方差（对数刻度）")
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc="upper center",ncol=4,bbox_to_anchor=(.5,1.0))
    save(fig,dest/"artifact_suppression")
    sensitivity=pd.read_csv(out/"sensitivity_curves.csv")
    fig,axs=axes_grid()
    for project in [1,2]:
        for c,ch in enumerate(["F3","Fz","F4"]):
            for hp,color in [(.05,"#009E73"),(.1,"#0072B2"),(.3,"#D55E00")]:
                z=sensitivity[(sensitivity.project==project)&(sensitivity.channel==ch)&np.isclose(sensitivity.highpass,hp)]
                axs[project-1,c].plot(z.time_ms,z.amplitude,color=color,label=f"高通 {hp:g} 赫兹")
            decorate(axs[project-1,c],f"项目{project}  {ch}")
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc="upper center",ncol=3,bbox_to_anchor=(.5,1.0))
    save(fig,dest/"highpass_sensitivity")


