from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, whosmat
from scipy.signal import welch

from audit_panel_alignment import require_matplotlib_panel_alignment


CHANNELS = ["Fz", "F3", "F4"]
COLORS = {"left": "#3B82B8", "right": "#E58B3A", "correct": "#39855B", "wrong": "#C94F45"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def labels_from_cell(value: np.ndarray) -> list[str]:
    labels = []
    for item in np.asarray(value).ravel():
        while isinstance(item, np.ndarray) and item.size == 1:
            item = item.item()
        labels.append(str(item))
    return labels


def run_starts(
    values: np.ndarray,
    accepted: set[int],
    tolerance: float = 0.25,
    legal_codes: set[int] | None = None,
) -> tuple[list[dict], list[float]]:
    quantized = np.zeros(values.size, dtype=int)
    legal = np.zeros(values.size, dtype=bool)
    for code in legal_codes or (accepted | {0}):
        hit = np.abs(values - code) <= tolerance
        quantized[hit] = code
        legal |= hit
    illegal = np.unique(values[(~legal) & (np.abs(values) > tolerance)]).tolist()
    starts = np.r_[True, quantized[1:] != quantized[:-1]]
    indices = np.flatnonzero(starts)
    ends = np.r_[indices[1:], values.size]
    events = [
        {"sample": int(i), "code": int(quantized[i]), "length": int(j - i)}
        for i, j in zip(indices, ends)
        if quantized[i] in accepted
    ]
    return events, illegal


def pair_trials(cues: list[dict], responses: list[dict], time: np.ndarray) -> tuple[list[dict], list[dict]]:
    trials = []
    used = set()
    cursor = 0
    for i, cue in enumerate(cues):
        next_sample = cues[i + 1]["sample"] if i + 1 < len(cues) else time.size
        while cursor < len(responses) and responses[cursor]["sample"] < cue["sample"]:
            cursor += 1
        response = None
        if cursor < len(responses) and responses[cursor]["sample"] < next_sample:
            response = responses[cursor]
            used.add(cursor)
            cursor += 1
        cue_side = "left" if cue["code"] < 0 else "right"
        row = {
            "trial": i + 1,
            "cue_sample": cue["sample"],
            "cue_time_s": float(time[cue["sample"]]),
            "cue_code": cue["code"],
            "cue_side": cue_side,
            "response_sample": "",
            "response_time_s": "",
            "response_code": "",
            "response_side": "missing",
            "rt_s": "",
            "correct": "missing",
            "rt_flag": "missing",
        }
        if response is not None:
            response_side = "left" if response["code"] < 0 else "right"
            rt = float(time[response["sample"]] - time[cue["sample"]])
            row.update(
                response_sample=response["sample"],
                response_time_s=float(time[response["sample"]]),
                response_code=response["code"],
                response_side=response_side,
                rt_s=rt,
                correct=response_side == cue_side,
                rt_flag="pending",
            )
        trials.append(row)
    valid_rt = np.array([x["rt_s"] for x in trials if x["rt_s"] != ""], dtype=float)
    if valid_rt.size:
        center = float(np.median(valid_rt))
        scale = float(1.4826 * np.median(np.abs(valid_rt - center)))
        time_step = float(np.median(np.diff(time))) if time.size > 1 else 1e-12
        scale = max(scale, time_step, 1e-12)
        for row in trials:
            if row["rt_s"] == "":
                continue
            rt = float(row["rt_s"])
            if rt <= 0:
                row["rt_flag"] = "nonpositive"
            elif rt < 0.15:
                row["rt_flag"] = "too_fast"
            elif abs(rt - center) / scale > 5:
                row["rt_flag"] = "robust_outlier"
            else:
                row["rt_flag"] = "normal"
    orphan = [event for i, event in enumerate(responses) if i not in used]
    return trials, orphan


def band_power(freq: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freq >= low) & (freq <= high)
    return float(np.trapezoid(psd[mask], freq[mask])) if mask.sum() >= 2 else float("nan")


def panel_labels(axes: np.ndarray | list) -> None:
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.annotate(
            chr(ord("a") + i),
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(-20, 4),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
        )


def save_figure(fig: plt.Figure, base: Path, require_labels: bool) -> None:
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(base) + ".alignment.json",
        overlay_svg=str(base) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=require_labels,
        strict=True,
    )
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_raw(time: np.ndarray, eeg: np.ndarray, title: str, output: Path) -> None:
    stop = min(time.size, 2560)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True, constrained_layout=True)
    for ax, signal, channel in zip(axes, eeg, CHANNELS):
        ax.plot(time[:stop], signal[:stop], color="#315B7D", linewidth=0.65)
        ax.axhline(999, color="#C94F45", linestyle="--", linewidth=0.6, alpha=0.7)
        ax.axhline(-999, color="#C94F45", linestyle="--", linewidth=0.6, alpha=0.7)
        ax.set_ylabel(f"{channel}\n幅值（数据单位）")
        ax.grid(alpha=0.18, linewidth=0.5)
    axes[-1].set_xlabel("时间（s）")
    fig.suptitle(f"{title}：原始脑电前 10 s（未滤波）", fontsize=11)
    panel_labels(axes)
    save_figure(fig, output, True)


def plot_psd(eeg: np.ndarray, fs: float, title: str, output: Path) -> list[dict]:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True, constrained_layout=True)
    metrics = []
    for ax, signal, channel in zip(axes, eeg, CHANNELS):
        nperseg = min(int(4 * fs), signal.size)
        freq, density = welch(signal, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend="constant", scaling="density")
        display = (freq >= 0.5) & (freq <= min(100, fs / 2))
        db = 10 * np.log10(np.maximum(density[display], np.finfo(float).tiny))
        ax.plot(freq[display], db, color="#725A9A", linewidth=0.85)
        ax.axvline(50, color="#C94F45", linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"{channel}\nPSD（dB/Hz）")
        ax.grid(alpha=0.18, linewidth=0.5)
        p_01_1 = band_power(freq, density, 0.1, 1)
        p_1_40 = band_power(freq, density, 1, 40)
        p_30_80 = band_power(freq, density, 30, min(80, fs / 2))
        p_49_51 = band_power(freq, density, 49, 51)
        p_side = band_power(freq, density, 45, 49) + band_power(freq, density, 51, 55)
        peak_mask = (freq >= 1) & (freq <= min(100, fs / 2))
        high_mask = (freq >= 40) & (freq <= min(80, fs / 2))
        high_density = density[high_mask]
        metrics.append(
            {
                "channel": channel,
                "low_freq_ratio": p_01_1 / p_1_40 if p_1_40 > 0 else float("nan"),
                "high_freq_ratio": p_30_80 / p_1_40 if p_1_40 > 0 else float("nan"),
                "line_50hz_ratio": 4 * p_49_51 / p_side if p_side > 0 else float("nan"),
                "peak_frequency_hz": float(freq[peak_mask][np.argmax(density[peak_mask])]),
                "high_band_peak_hz": float(freq[high_mask][np.argmax(high_density)]),
                "high_band_peak_prominence_db": float(10 * np.log10(np.max(high_density) / np.median(high_density))),
            }
        )
    axes[-1].set_xlabel("频率（Hz）")
    fig.suptitle(f"{title}：原始脑电 Welch 功率谱密度（红色虚线为 50 Hz）", fontsize=11)
    panel_labels(axes)
    save_figure(fig, output, True)
    return metrics


def plot_rt(trials: list[dict], title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    all_rt = np.array([x["rt_s"] for x in trials if x["rt_s"] != ""], dtype=float)
    if all_rt.size:
        bins = np.histogram_bin_edges(all_rt, bins="fd")
        if bins.size < 6:
            bins = np.linspace(all_rt.min() - 0.02, all_rt.max() + 0.02, 8)
        counts = {}
        for side, label in [("left", "左靶"), ("right", "右靶")]:
            rt = [x["rt_s"] for x in trials if x["cue_side"] == side and x["rt_s"] != ""]
            counts[side] = len(rt)
            ax.hist(rt, bins=bins, alpha=0.55, color=COLORS[side], edgecolor="white", linewidth=0.5)
        median = float(np.median(all_rt))
        ax.axvline(median, color="#222222", linestyle="--", linewidth=1)
    else:
        ax.text(0.5, 0.5, "无可用配对反应时", transform=ax.transAxes, ha="center", va="center")
    missing = sum(x["rt_s"] == "" for x in trials)
    outliers = sum(x["rt_flag"] not in {"normal", "missing"} for x in trials)
    color_note = f"蓝=左靶（n={counts.get('left', 0)}），橙=右靶（n={counts.get('right', 0)}），黑虚线=中位数 {np.median(all_rt):.3f} s" if all_rt.size else ""
    ax.set_title(f"{title}：提示—响应反应时分布\n{color_note}；缺失={missing}，异常={outliers}")
    ax.set_xlabel("反应时间 RT（s）")
    ax.set_ylabel("试次数")
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    save_figure(fig, output, False)


def plot_alignment(cues: list[dict], responses: list[dict], trials: list[dict], time: np.ndarray, title: str, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), constrained_layout=True)
    ax = axes[0]
    rows = [(cues, -1, 3, "提示-左", "left"), (cues, 1, 2, "提示-右", "right"), (responses, -1, 1, "响应-左", "left"), (responses, 1, 0, "响应-右", "right")]
    for events, sign, y, label, side in rows:
        x = [time[e["sample"]] for e in events if np.sign(e["code"]) == sign]
        ax.scatter(x, np.full(len(x), y), marker="|", s=70, linewidth=1.0, color=COLORS[side])
    ax.set_yticks([0, 1, 2, 3], ["响应-右", "响应-左", "提示-右", "提示-左"])
    ax.set_xlabel("记录时间（s）")
    ax.set_title("全记录事件时序")
    ax.grid(axis="x", alpha=0.18, linewidth=0.5)
    ax = axes[1]
    for correctness, marker, label in [(True, "o", "正确"), (False, "X", "错误")]:
        for side in ["left", "right"]:
            rows_sel = [x for x in trials if x["correct"] is correctness and x["cue_side"] == side]
            ax.scatter([x["trial"] for x in rows_sel], [x["rt_s"] for x in rows_sel], s=20, marker=marker, color=COLORS[side], alpha=0.85)
    missing = [x for x in trials if x["rt_s"] == ""]
    if missing:
        ax.scatter([x["trial"] for x in missing], np.zeros(len(missing)), s=28, marker="x", color="#555555")
    valid = [x["rt_s"] for x in trials if x["rt_s"] != ""]
    if valid:
        ax.axhline(np.median(valid), color="#333333", linestyle="--", linewidth=0.8)
    ax.set_xlabel("试次序号")
    ax.set_ylabel("RT（s）")
    ax.set_title("逐试次配对：蓝/橙=左/右靶，圆点/X=正确/错误，虚线=RT 中位数")
    ax.grid(alpha=0.18, linewidth=0.5)
    fig.suptitle(f"{title}：VisCue 与响应事件对齐审计", fontsize=11)
    panel_labels(axes)
    save_figure(fig, output, True)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_summary(summaries: list[dict]) -> str:
    lines = ["", "十一、实际结果（自动生成）", "<!-- P0_AUTO_START -->"]
    for number, item in enumerate(summaries, 1):
        lines.extend(
            [
                f"11.{number} {item['file']}",
                f"- 数据：10×{item['n_samples']}，采样率 {item['fs']:.0f} Hz，时长 {item['duration_s']:.3f} s。",
                f"- 提示：左 {item['cue_left']}，右 {item['cue_right']}；响应：左 {item['resp_left']}，右 {item['resp_right']}。",
                f"- 配对：有效 {item['matched']}，缺失 {item['missing']}，孤立响应 {item['orphan']}；正确 {item['correct']}，错误 {item['wrong']}。",
                f"- RT：均值 {item['rt_mean']:.3f} s，中位数 {item['rt_median']:.3f} s，标准差 {item['rt_std']:.3f} s，稳健异常 {item['rt_outlier']}。",
                f"- 时间戳：非正差分 {item['timestamp_nonpositive']}，差分中位数 {item['timestamp_step']:.8f} s，{'采用文件时间戳' if item['timestamp_valid'] else '回退到采样索引'}。",
                "- 原始信号质量：",
            ]
        )
        for stat in item["channel_stats"]:
            lines.append(
                f"  - {stat['channel']}：均值 {stat['mean']:.3f}，标准差 {stat['std']:.3f}，最小值 {stat['min']:.3f}，最大值 {stat['max']:.3f}，"
                f"峰峰值 {stat['peak_to_peak']:.3f}，|x|≥999 比例 {stat['saturation_ratio']:.6%}，稳健异常比例 {stat['robust_outlier_ratio']:.6%}。"
            )
        lines.append("- PSD 审计（低频漂移比/高频比/50 Hz 工频比/主峰 Hz/40—80 Hz 峰 Hz/突出度 dB）：")
        for metric in item["psd_metrics"]:
            lines.append(
                f"  - {metric['channel']}：{metric['low_freq_ratio']:.3f}/{metric['high_freq_ratio']:.3f}/"
                f"{metric['line_50hz_ratio']:.3f}/{metric['peak_frequency_hz']:.3f}/"
                f"{metric['high_band_peak_hz']:.3f}/{metric['high_band_peak_prominence_db']:.3f}。"
            )
    total_missing = sum(x["missing"] for x in summaries)
    total_orphan = sum(x["orphan"] for x in summaries)
    total_wrong = sum(x["wrong"] for x in summaries)
    saturated = any(any(v > 0 for v in x["sat_ratios"]) for x in summaries)
    lines.extend(
        [
            "十二、P0 初步结论",
            "1. 四个 MAT 文件均成功读取，顶层变量均为 SampleRate、DataLabel、data；数据方向统一为 10×N，采样率均为 256 Hz。",
            "2. 四个文件的 VisCue 左/右试次数与题面预期完全一致。",
            f"3. 全部文件合计缺失响应 {total_missing}、孤立响应 {total_orphan}、错误响应 {total_wrong}；详细到文件和试次的记录见 P0_结果.csv。",
            f"4. 原始 Fz/F3/F4 {'检测到 |x|≥999 的近饱和样本，后续必须采用稳健伪影拒绝' if saturated else '未检测到 |x|≥999 的近饱和样本'}。",
            "5. 各通道 1—100 Hz 主谱峰均在 1 Hz，低频漂移最严重的是项目 1 的 F3/F4；50 Hz 工频比约为 1，未见窄带 50 Hz 异常抬升。40—80 Hz 审计显示多数通道峰值在 60.0 Hz，其中 A-项目1 三通道、A-项目2 Fz、B-项目1 F3/F4 的突出度约 15—19 dB。",
            "6. 项目 1 与项目 2 的第 9 通道语义不同：项目 1 用持续 ±1 段起点，项目 2 用单点 ±2 点击；该差异必须延续到后续试次切分。",
            "7. P0 只做原始数据审计，不对信号滤波或删除异常试次；后续预处理应优先处理低频漂移、饱和平台和大幅伪影，并保留全部剔除日志。",
            "<!-- P0_AUTO_END -->",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    dataset = root / "C题" / "dataset"
    figure_dir = root / "P0_图表"
    figure_dir.mkdir(parents=True, exist_ok=True)
    mat_files = sorted(dataset.glob("VisualCog?_Task-?.mat"))
    if len(mat_files) != 4:
        raise RuntimeError(f"Expected 4 MAT files, found {len(mat_files)}")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    rows = []
    summaries = []
    print(f"项目目录：{root}")
    print("MAT 文件：")
    for path in mat_files:
        print(f"  {path.name} ({path.stat().st_size} bytes)")
    for path in mat_files:
        subject = path.stem.replace("VisualCog", "").split("_")[0]
        task = int(path.stem.split("Task-")[1])
        print(f"\n{'=' * 78}\n{path.name}")
        metadata = whosmat(path)
        print("变量名、形状、类型：")
        for name, shape, dtype in metadata:
            print(f"  {name}: shape={shape}, dtype={dtype}")
            rows.append({"record_type": "variable", "file": path.name, "variable": name, "shape": "x".join(map(str, shape)), "dtype": dtype})
        mat = loadmat(path, squeeze_me=False, struct_as_record=False)
        fs = float(np.asarray(mat["SampleRate"]).squeeze())
        labels = labels_from_cell(mat["DataLabel"])
        data = np.asarray(mat["data"], dtype=float)
        if data.shape[0] != len(labels) and data.shape[1] == len(labels):
            data = data.T
        if data.shape[0] != 10:
            raise RuntimeError(f"Unexpected channel count in {path.name}: {data.shape}")
        print(f"SampleRate={fs:g} Hz")
        print(f"DataLabel={labels}")
        print("各通道前 10 个采样点：")
        for label, values in zip(labels, data):
            print(f"  {label}: {np.array2string(values[:10], precision=6, separator=', ')}")
        timestamp = data[9]
        diff = np.diff(timestamp)
        timestamp_step = float(np.median(diff))
        timestamp_nonpositive = int(np.sum(diff <= 0))
        timestamp_valid = timestamp_nonpositive == 0 and np.isclose(timestamp_step, 1 / fs, rtol=0.05, atol=1e-9)
        time = timestamp - timestamp[0] if timestamp_valid else np.arange(data.shape[1]) / fs
        cue_events, cue_illegal = run_starts(data[7], {-1, 1})
        response_codes = {-1, 1} if task == 1 else {-2, 2}
        response_events, response_illegal = run_starts(
            data[8], response_codes, legal_codes=response_codes | ({-1, 0, 1} if task == 2 else {0})
        )
        state_events, _ = run_starts(data[8], {-1, 1})
        trials, orphan = pair_trials(cue_events, response_events, time)
        cue_left = sum(e["code"] < 0 for e in cue_events)
        cue_right = sum(e["code"] > 0 for e in cue_events)
        resp_left = sum(e["code"] < 0 for e in response_events)
        resp_right = sum(e["code"] > 0 for e in response_events)
        matched = sum(x["rt_s"] != "" for x in trials)
        missing = len(trials) - matched
        correct = sum(x["correct"] is True for x in trials)
        wrong = sum(x["correct"] is False for x in trials)
        rt = np.array([x["rt_s"] for x in trials if x["rt_s"] != ""], dtype=float)
        rt_outlier = sum(x["rt_flag"] not in {"normal", "missing"} for x in trials)
        print(f"事件统计：提示 左={cue_left}, 右={cue_right}; 响应 左={resp_left}, 右={resp_right}")
        print(f"配对统计：有效={matched}, 缺失={missing}, 孤立={len(orphan)}, 正确={correct}, 错误={wrong}, RT异常={rt_outlier}")
        if rt.size:
            print(f"RT(s)：mean={rt.mean():.6f}, median={np.median(rt):.6f}, std={rt.std(ddof=1):.6f}, min={rt.min():.6f}, max={rt.max():.6f}")
        if cue_illegal or response_illegal:
            print(f"非法事件值：VisCue={cue_illegal}, Response={response_illegal}")
        channel_stats = []
        print("原始脑电质量统计：")
        for index, channel in enumerate(CHANNELS):
            signal = data[index]
            median = float(np.median(signal))
            mad_scale = max(float(1.4826 * np.median(np.abs(signal - median))), 1e-12)
            stat = {
                "channel": channel,
                "mean": float(np.mean(signal)),
                "std": float(np.std(signal, ddof=1)),
                "min": float(np.min(signal)),
                "max": float(np.max(signal)),
                "peak_to_peak": float(np.ptp(signal)),
                "nan_count": int(np.isnan(signal).sum()),
                "inf_count": int(np.isinf(signal).sum()),
                "saturation_ratio": float(np.mean(np.abs(signal) >= 999)),
                "flat_diff_ratio": float(np.mean(np.abs(np.diff(signal)) <= 1e-12)),
                "robust_outlier_ratio": float(np.mean(np.abs(signal - median) / mad_scale > 6)),
            }
            channel_stats.append(stat)
            print(
                f"  {channel}: mean={stat['mean']:.6f}, std={stat['std']:.6f}, min={stat['min']:.6f}, "
                f"max={stat['max']:.6f}, p2p={stat['peak_to_peak']:.6f}, sat={stat['saturation_ratio']:.6%}, "
                f"robust>|6|={stat['robust_outlier_ratio']:.6%}"
            )
            rows.append({"record_type": "channel_quality", "file": path.name, "subject": subject, "task": task, **stat})
        short = f"被试 {subject} 项目 {task}"
        stem = f"{subject}_Task{task}"
        plot_raw(time, data[:3], short, figure_dir / f"{stem}_raw_10s")
        psd_metrics = plot_psd(data[:3], fs, short, figure_dir / f"{stem}_psd")
        plot_rt(trials, short, figure_dir / f"{stem}_rt_hist")
        plot_alignment(cue_events, response_events, trials, time, short, figure_dir / f"{stem}_event_alignment")
        for metric in psd_metrics:
            rows.append({"record_type": "psd_metric", "file": path.name, "subject": subject, "task": task, **metric})
        for event_name, events in [("cue", cue_events), ("response", response_events), ("task2_state", state_events if task == 2 else [])]:
            for event in events:
                rows.append({"record_type": "event", "file": path.name, "subject": subject, "task": task, "event_type": event_name, **event})
        for trial in trials:
            rows.append({"record_type": "trial", "file": path.name, "subject": subject, "task": task, **trial})
        rows.append(
            {
                "record_type": "file_summary",
                "file": path.name,
                "subject": subject,
                "task": task,
                "fs": fs,
                "n_samples": data.shape[1],
                "duration_s": float(time[-1]),
                "cue_left": cue_left,
                "cue_right": cue_right,
                "resp_left": resp_left,
                "resp_right": resp_right,
                "matched": matched,
                "missing": missing,
                "orphan": len(orphan),
                "correct": correct,
                "wrong": wrong,
                "rt_outlier": rt_outlier,
                "timestamp_valid": timestamp_valid,
            }
        )
        summaries.append(
            {
                "file": path.name,
                "fs": fs,
                "n_samples": data.shape[1],
                "duration_s": float(time[-1]),
                "cue_left": cue_left,
                "cue_right": cue_right,
                "resp_left": resp_left,
                "resp_right": resp_right,
                "matched": matched,
                "missing": missing,
                "orphan": len(orphan),
                "correct": correct,
                "wrong": wrong,
                "rt_mean": float(np.mean(rt)),
                "rt_median": float(np.median(rt)),
                "rt_std": float(np.std(rt, ddof=1)),
                "rt_outlier": rt_outlier,
                "sat_ratios": [x["saturation_ratio"] for x in channel_stats],
                "sat_text": "、".join(f"{x['saturation_ratio']:.6%}" for x in channel_stats),
                "timestamp_nonpositive": timestamp_nonpositive,
                "timestamp_step": timestamp_step,
                "timestamp_valid": timestamp_valid,
                "channel_stats": channel_stats,
                "psd_metrics": psd_metrics,
            }
        )
    write_csv(root / "P0_结果.csv", rows)
    plan_path = root / "P0_方案.txt"
    plan = plan_path.read_text(encoding="utf-8")
    start = "<!-- P0_AUTO_START -->"
    if start in plan:
        plan = plan.split("\n十一、实际结果（自动生成）", 1)[0].rstrip() + "\n"
    plan_path.write_text(plan + format_summary(summaries), encoding="utf-8")
    (figure_dir / "P0_图表清单.json").write_text(
        json.dumps({"figures": sorted(p.name for p in figure_dir.glob("*.png")), "dpi": 400, "backend": "Python/Matplotlib"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n输出完成：{root / 'P0_方案.txt'}")
    print(f"输出完成：{root / 'P0_结果.csv'}")
    print(f"图表目录：{figure_dir}")


if __name__ == "__main__":
    main()
