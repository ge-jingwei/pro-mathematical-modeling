"""Build the Chinese report directly from saved, verified numerical results."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def table(frame,digits=3):
    rows=[]
    for row in frame.itertuples(index=False,name=None):
        rows.append(["—" if pd.isna(v) else f"{v:.{digits}f}" if isinstance(v,(float,np.floating)) else str(v) for v in row])
    return "| "+" | ".join(map(str,frame.columns))+" |\n|"+"|".join(["---"]*len(frame.columns))+"|\n"+"\n".join("| "+" | ".join(r)+" |" for r in rows)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True); args=p.parse_args(); out=Path(args.out)
    load=lambda n:pd.read_csv(out/n)
    rec=load("recordings.csv"); qc=load("trial_quality.csv"); m=load("p300_metrics.csv"); cls=load("classification.csv")
    art=load("artifact_metrics.csv"); desc=load("descriptive_fit_parameters.csv"); simple=load("fit_parameters.csv")
    peaks=load("local_peak_candidates.csv"); effects=load("effect_sizes.csv"); recover=load("synthetic_recovery.csv")
    sens=load("aggregation_sensitivity.csv"); cross=load("fit_cross_record.csv")
    aliases={"VisualCogA_Task-1":"甲组任务一","VisualCogA_Task-2":"甲组任务二","VisualCogB_Task-1":"乙组任务一","VisualCogB_Task-2":"乙组任务二"}
    counts=qc.groupby(["record","cue"]).keep.agg(["sum","count"]).reset_index()
    audit=[]
    for _,r in rec.iterrows():
        g=counts[counts.record==r.record]
        audit.append({"记录":aliases[r.record],"项目":r.project,"数据形状":f"10×{round(r.duration_s*256)}","时长/秒":r.duration_s,
                      "左/右提示":f"{r.n_left}/{r.n_right}","保留左/右":f"{g[g.cue==-1]['sum'].iloc[0]}/{g[g.cue==1]['sum'].iloc[0]}",
                      "提示与后续方向一致率":r.cue_target_agreement})
    reduction=[]
    for (project,ch),g in art.groupby(["project","channel"]):
        row={"项目":project,"通道":ch}
        for metric,title in [("drift_energy","低频能量降低/%"),("high_frequency_energy","高频能量降低/%"),("trial_variance","同试次方差降低/%"),("baseline_slope_energy","基线斜率能量降低/%")]:
            raw=g[g.stage=="raw"][metric].mean(); clean=g[g.stage=="clean"][metric].mean()
            row[title]=100*(1-clean/raw)
        reduction.append(row)
    pd.DataFrame(reduction).to_csv(out/"artifact_reduction_summary.csv",index=False)
    clean=m[m.stage=="clean"].copy(); clean["靶向"]=clean.condition.map({-1:"左",1:"右"})
    primary=cls[(cls.stage=="clean")&(cls.window=="p300")&(cls.channel=="all")].copy()
    primary["验证方式"]=primary.scheme.map({"leave_record_out":"跨记录留出","purged_block5":"连续五块留出"})
    effect=effects[(effects.stage=="clean")&(effects.scope=="pooled")]
    parameter=desc.copy(); parameter["靶向"]=parameter.condition.map({-1:"左",1:"右"})
    parameter["触边"]=parameter.boundary_hit.map({True:"是",False:"否"})
    local=peaks.copy(); local["靶向"]=local.condition.map({-1:"左",1:"右"})
    fz=clean[clean.channel=="Fz"]
    reasons=qc.loc[~qc.keep,"reason"].str.split(";").explode().value_counts()
    text=["# 问题一 原始额区脑电的视觉响应提取与拟合\n",
"本报告只处理问题一。全部数据计算已在指定服务器实际执行，最终信号仅来自原始前三通道。核心结论是：项目一存在较明确的额中线三百毫秒附近局部正峰；项目二包含更强的晚期慢正波，不能把五百毫秒窗口边界最大值直接当作可靠的三百毫秒成分潜伏期。当前严格跨记录验证没有证实可靠的左右区分能力。\n",
"## 一 数据探索与项目核对\n",
"**映射修正：任务编号一对应实验项目一，任务编号二对应实验项目二；文件字母不是项目编号。** 下表的甲、乙分别指文件名中的字母组，不代表已知受试者身份。题目没有提供个体标识，不能将其直接当作两个独立受试者。",
 table(pd.DataFrame(audit)),
"每份数据均为十通道、每秒二百五十六点；时间戳相邻差严格为一除以二百五十六秒，未发现缺失或非有限值。原始前三通道依次为额中线、左额、右额；输出统一为左额、额中线、右额。第四至六通道未参与处理、调参、拟合或评价。心电通道仅用于记录质量检查，四份均非平坦，候选心搏频率约每分钟七十至七十二次；此处为粗略系统连通性证据，不是心律诊断。\n",
"**事件编码与题面简写并不完全相同。** 视觉提示均为正负一的持续段，使用持续段起点而不是每一个非零采样点；提示中位持续时长为零点二〇三一秒。任务一第九通道只有正负一，缺少明确的正负二点击脉冲，不能声称获得其真实点击反应时。任务二同时存在约提示后二点二一五秒出现的正负一方向标记以及稍后的正负二单点点击。任务一提示方向与后续方向标记一致率为百分之一百和百分之九十八，而任务二约为百分之四十七和百分之四十八；结合题目中预知位置与只知形状的区别，支持按任务编号的映射。所有正负一标记暂称后续方向标记，不把推定的目标含义当成设备官方编码定义。\n",
"任务二点击的中位时间为提示后三点六〇九秒和三点五三七秒，远晚于本分析的零点八秒截取终点；这排除了本窗口内的直接点击，但不能排除眼动和准备活动。输入没有物理幅值标定信息，因此所有幅值均报告为原记录单位，不擅自写成微伏。\n",
"## 二 针对弱晚期响应的保守预处理\n",
"一、保留原参考，不做三通道平均重参考或盲目独立成分删除。只有三个额区通道且缺少眼电参考，无法可靠鉴别共同额区成分究竟来自眨眼还是真实响应。\n\n二、先标记绝对幅值不小于九百九十九点九的饱和点。仅为避免滤波器振铃传播，在连续处理副本中插值这些点；距离提示正负两秒内出现饱和的整段试次剔除，插值不被当成可恢复的真实脑电。\n\n三、连续信号使用五十赫兹陷波、四阶零点一至三十赫兹零相位带通；双向处理的幅频响应相当于单程响应平方。随后做四层对称小波分解，仅对约十六至一百二十八赫兹的三层细节软阈值收缩，保留低频近似和八至十六赫兹细节，以降低对宽慢正波的直接损伤。高频收缩并不能保证移除低频眼动。\n\n四、取提示前一百九十九点二一九毫秒至提示后八百点七八一毫秒，共二百五十七点；用负时间的五十一点均值校正基线。预先规定分析窗为二百五十至五百毫秒。\n\n五、在每份记录内用刺激前均方根、峰峰值、最大相邻跳变和三十五至一百赫兹均方根作与类别无关的质量检查；超过中位数加八倍稳健标准差的试次剔除。稳健标准差为一点四八二六乘中位绝对偏差。按刺激前噪声给予零点二至一的有界逆噪声权重，左右标签不参与质量规则。报告同时保留无权平均及四倍稳健标准差严格规则作为敏感性分析。\n\n六、每个方向先分别计算各记录的加权响应，再对两份记录等权平均，避免记录试次数不同改变项目权重。原始同试次曲线使用完全相同的保留集合及权重；全部原始曲线作为额外对照。\n",
"保留试次共三百零八个，占四百个的百分之七十七。以下为剔除原因出现次数，同一试次可有多个原因：\n"+table(pd.DataFrame({"原因":reasons.index,"次数":reasons.values}).replace({"saturation_within_2s":"正负两秒内饱和","baseline_outlier":"基线异常","amplitude_outlier":"幅度异常","gradient_outlier":"相邻跳变异常","high_frequency_outlier":"高频异常","flat":"平坦信号"})),
"## 三 去噪效果与信号保留\n",
"低频漂移能量取连续谱零点〇三一二五至零点五赫兹；高频肌电代理能量取三十至一百赫兹并排除工频附近。二者分别是频带积分，不是经验证的纯伪影能量。试次方差使用相同保留试次的目标时间窗，基线斜率能量使用刺激前拟合斜率平方乘基线时间方差。连续能量变化包含饱和插值影响，不能全部解释成神经噪声清除率。\n",
 table(pd.DataFrame(reduction)),
"负的降低百分比表示对应指标上升，应如实保留。低频脑电与伪影频谱重叠，单靠能量下降无法证明视觉特征得到保留。\n",
 f"额外向各保留事件注入峰幅五、中心三百五十毫秒、宽度七十毫秒的已知高斯信号，通过完整连续预处理后作差验证：峰值保留率为{100*recover.peak_retention.min():.3f}%—{100*recover.peak_retention.max():.3f}%，面积保留率为{100*recover.area_retention.min():.3f}%—{100*recover.area_retention.max():.3f}%，采样网格上的潜伏期误差均为零。这只能证明该预设形状在既定保留集合上的处理失真很小，不能证明真实未知响应或低频眼动已被正确分离。\n",
"## 四 指定时间窗的响应指标\n",
"下表峰值是指定窗口的代数最大值，允许为负；负值不叫正向三百毫秒成分。面积为带符号积分，单位为原记录幅值乘秒。信噪比定义为窗口响应均方除以刺激前响应均方，再取十分贝对数；这是描述性代理指标，基线很小时可偏高，因此另存了基于重采样不确定性的信噪比。\n",
 table(clean[["project","靶向","channel","peak","latency_ms","signed_area","snr_db","bootstrap_snr_db"]].rename(columns={"project":"项目","channel":"通道","peak":"窗口最大值","latency_ms":"最大值时间/毫秒","signed_area":"带符号面积","snr_db":"基线信噪比/分贝","bootstrap_snr_db":"重采样信噪比/分贝"})),
"**项目二的十二项时间解释必须谨慎：六条方向与通道组合的窗口最大值都在五百毫秒边界，并非窗内定位成功。** 为区分晚期慢波与局部正峰，另在原曲线上寻找二百五十至五百毫秒的正向局部最大值，以突出度最大者为候选；没有正局部峰则保留空值，不强造峰。\n".replace("十二项","各项"),
 table(local[["project","靶向","channel","peak","latency_ms","prominence"]].rename(columns={"project":"项目","channel":"通道","peak":"局部正峰","latency_ms":"局部峰时间/毫秒","prominence":"突出度"})),
"额中线的窗口平均幅值及百分之九十五区间如下。区间由每份记录内部有放回重采样一千次得到，条件于现有两份记录、现有质量筛选与权重；没有考虑全体受试者抽样，也没有做时间点多重比较校正。图中的区间都是逐点区间，不能当作整段同时置信带。\n",
 table(fz[["project","靶向","mean_amplitude","mean_amplitude_ci_low","mean_amplitude_ci_high"]].rename(columns={"project":"项目","mean_amplitude":"窗均值","mean_amplitude_ci_low":"区间下界","mean_amplitude_ci_high":"区间上界"})),
"## 五 数学曲线拟合\n",
"先在一百至六百五十毫秒拟合带常数项的单高斯及双高斯，以小样本修正信息准则选择；正成分中心限制在二百五十至五百毫秒。完整候选参数、拟合优度、均方根误差和边界触发标记均已输出。该限制模型在部分通道明显不足：它不能用一个受限早期正峰解释强烈晚期慢波，低拟合优度不是应被隐藏的问题。\n",
 table(simple[simple.selected][["project","condition","channel","model","r2","rmse","boundary_hit"]].replace({"condition":{-1:"左",1:"右"},"model":{"gaussian":"单高斯","double_gaussian":"双高斯"},"boundary_hit":{True:"是",False:"否"}}).rename(columns={"project":"项目","condition":"靶向","channel":"通道","model":"模型","r2":"决定系数","rmse":"均方根误差","boundary_hit":"参数触边"})),
"因此增加明确标为描述性的零至八百毫秒模型，分离线性缓慢趋势和两个宽成分：\n\n$$y(t)=c+bt+A_1\\exp[-(t-\\mu_1)^2/(2\\sigma_1^2)]+A_2\\exp[-(t-\\mu_2)^2/(2\\sigma_2^2)].$$\n\n时间变量用秒；第一成分中心限制在一百二十至四百五十毫秒，第二成分中心限制在四百五十至八百毫秒，振幅允许正负。两个成分只是数学分量，不能未经验证分别命名为特定神经源。下表给出全部十二条曲线的参数；中心和宽度以毫秒显示。\n",
 table(parameter[["project","靶向","channel","offset","slope","a1","mu1_ms","sigma1_ms","a2","mu2_ms","sigma2_ms","r2","rmse","触边"]].rename(columns={"project":"项目","channel":"通道","offset":"常数","slope":"斜率","a1":"幅度一","mu1_ms":"中心一","sigma1_ms":"宽度一","a2":"幅度二","mu2_ms":"中心二","sigma2_ms":"宽度二","r2":"决定系数","rmse":"均方根误差"})),
"宽窗拟合的样本内决定系数虽高，但这是平滑平均曲线上的描述能力。跨记录训练—验证结果如下，项目一甚至大多为负，说明数学曲线不能直接宣称具备跨记录预测能力。参数协方差给出的标准误仅为独立残差假设下的优化器近似，不作为生理参数置信区间。\n",
 table(cross.groupby("project").agg(决定系数最小值=("r2","min"),决定系数中位数=("r2","median"),决定系数最大值=("r2","max"),平均均方根误差=("rmse","mean")).reset_index().rename(columns={"project":"项目"})),
"## 六 左右区分能力与无泄漏验证\n",
"每试次提取三个通道在目标窗五个等宽子窗的平均值，共十五维。分类使用固定惩罚强度十的岭回归判别，训练折内标准化；不按测试表现挑选模型或改变分数方向。主要验证为整份记录留出，补充验证为每份记录按原始试次顺序分五块、测试块两侧再排除两试次。没有使用伪平均样本进行训练。\n\n质量阈值和去噪参数不依赖类别，但由整份记录的无标签数据确定，因此本结果属于离线、记录自适应预处理，不等价于完全训练集校准的在线泛化测试。连续零相位滤波同样使用事件前后数据。\n",
 table(primary[["project","验证方式","auc","auc_ci_low","auc_ci_high","accuracy","balanced_accuracy","majority_accuracy","permutation_p"]].rename(columns={"project":"项目","auc":"曲线下面积","auc_ci_low":"区间下界","auc_ci_high":"区间上界","accuracy":"准确率","balanced_accuracy":"平衡准确率","majority_accuracy":"多数类基准","permutation_p":"置换概率"})),
"区间为固定折外预测的记录内重采样区间，没有重新训练每个自助样本，可能低估训练不确定性。置换检验在记录与连续二十试次块内打乱标签五百次，并对每次置换重新训练分类器。当前没有显著的高于随机解码证据。部分连续分块结果低于二分之一，提示时间不稳定或偶然偏差，不能在看到测试结果后倒转标签使其变成较高正确率。刺激前负对照也已输出，未出现可信的先验方向解码。\n",
"效应量定义为右靶减左靶的单试次窗口均值除以合并标准差；下表为未加权汇总试次，因此不等同于前文记录等权、质量加权曲线的差值。所有区间为探索性、未作多重比较校正。\n",
 table(effect[["project","channel","cohen_d","ci_low","ci_high"]].rename(columns={"project":"项目","channel":"通道","cohen_d":"标准化均值差","ci_low":"区间下界","ci_high":"区间上界"})),
"项目二左额通道的汇总单试次效应量约零点三七，但这没有转化为稳定跨记录解码；不得将单个未经校正的效应量区间解读成已证实的纯视物形状编码。文件内效应量和单通道分类均另表保存。\n",
"## 七 生理解释和项目差异\n",
"一、项目一额中线左右局部正峰分别约三百五十二和三百二十四毫秒，时间上符合目标窗口；左额在该窗口仍为负，不能把其代数最大值叫作正峰。\n\n二、项目二额中线局部正峰约三百四十四和三百五十九毫秒，但叠加了显著更大的五百至八百毫秒慢正波。其较大幅值可能涉及更强持续加工、期待、眼动或基线趋势；现有数据不能区分这些解释，更不能直接量化认知负荷。\n\n三、三个测点均在额区。额中线不能写成中央或顶区电极；缺少中央、顶区和枕区测点，所以不能验证题目所述中央—顶区最大分布，也不能做可信全头皮拓扑或源定位。三百毫秒左右的额区响应与视觉认知相关，但不构成经典顶区成分的充分证据。\n\n四、左右三角形同时改变形状方向、空间注意和可能的眼动，缺少反平衡对照及眼电，不能声称只保留了纯形状信息。此处最可靠的是事件锁定额区波形的描述及处理失真控制，而不是神经源或机制归因。\n\n五、不同项目的记录可能来自不同受试者或不同会话；缺少身份资料，项目差异仅为本数据集描述，不进行虚构的配对受试者显著性检验。\n",
"## 八 稳健性和交付核验\n",
"高通截止频率分别取零点〇五、零点一和零点三赫兹，项目一额中线候选时间仍在约三百二十八至三百五十五毫秒；项目二窗口最大值仍在五百毫秒边界，边界解释对三种设置一致，但其幅值对高通设置敏感。无权聚合与严格四倍稳健标准差筛选结果已完整保存，不能挑选最有利的一组代替预设主结果。\n",
 table(sens[sens.channel=="Fz"][["project","condition","mode","n","peak","latency_ms"]].replace({"primary":"主方案","equal_weights":"无权平均","strict4mad":"严格筛选"}).rename(columns={"project":"项目","condition":"靶向","mode":"设置","n":"保留数","peak":"窗口最大值","latency_ms":"最大值时间/毫秒"})),
"自动测试检查事件段起点、输入禁用通道不影响原始分析、截取时间、已存试次形状与基线、测试标签不进入分类训练、合成脉冲保留。最终图形经过面板对齐和文字碰撞核验。所有数表均由代码直接生成，没有手工改写数值。\n",
"## 九 图形索引\n",
"![项目一左右响应与差异波](figures/project1_left_right_difference.png)\n\n![项目二左右响应与差异波](figures/project2_left_right_difference.png)\n\n![项目一原始与去噪](figures/project1_raw_clean.png)\n\n![项目二原始与去噪](figures/project2_raw_clean.png)\n\n![项目一描述性拟合](figures/project1_descriptive_fits.png)\n\n![项目二描述性拟合](figures/project2_descriptive_fits.png)\n\n![分类曲线](figures/classification_roc.png)\n\n![伪影指标](figures/artifact_suppression.png)\n\n![滤波敏感性](figures/highpass_sensitivity.png)\n",
"图中灰带表示预设二百五十至五百毫秒窗口，阴影表示条件于现有记录的百分之九十五逐点重采样区间。原始比较图的点线为全部原始试次，虚线为同一保留集合，实线为去噪结果。另有两幅受限拟合图用于展示限制模型的不足。全部十一幅图同时保存矢量格式、便携文档格式和每英寸三百点位图。\n",
"## 十 结论\n",
"已完成原始数据读取、项目映射核验、事件对齐、保守降噪、三个观测点左右响应提取、两层曲线拟合、伪影指标、信号保留验证、左右分类及图表导出。项目一的额中线局部正峰最符合预设时间特征；项目二须区分局部早峰与强晚期慢波。当前证据不足以确认纯形状解码或经典中央顶区成分，去噪也没有带来稳定的左右分类提升。这个负面结果是分析结论的一部分，不应通过选模、改标签或移除不利试次掩盖。\n"]
    text.append("## 十一 方法学依据\n\n一、波利奇关于三百毫秒成分及其额区与颞顶区子成分的综述，二〇〇七年，数字对象标识符：10.1016/j.clinph.2007.04.019。用于限定空间分布与成分归属的解释，不作为本数据集结果证据。\n\n二、坦纳等关于高通滤波引入伪成分的实证研究，二〇一五年，数字对象标识符：10.1111/psyp.12437。用于支持保守高通、信号注入及截止频率敏感性检查；不意味着某个固定截止频率对所有实验最优。\n\n三、科学计算库前向反向滤波官方文档，函数标识为 scipy.signal.filtfilt（前向反向滤波）与 scipy.signal.sosfiltfilt（二阶节前向反向滤波）。实际软件版本以运行清单为准。\n")
    (out/"问题一分析报告.md").write_text("\n\n".join(text),encoding="utf-8")
    print("Report saved",out/"问题一分析报告.md")


if __name__=="__main__": main()


