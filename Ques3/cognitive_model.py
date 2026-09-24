"""Small reset-state model, train-only fitting, and honest held-out evaluation."""
import numpy as np
import pandas as pd
import torch
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from Ques3.utils import FS, SEED, CHANNELS, dump, table, load_early, fold_sources, observation_windows

NAMES = ["tau_m", "tau_z", "w_e", "w_m", "bias", "decision_threshold"]
BOUNDS = np.array([[.08, 3.], [.08, 4.], [-2., 2.], [-2., 2.], [-2., 2.], [.08, 3.]])


def canonical(params):
    p = np.asarray(params, dtype=float).copy()
    norm = max(np.linalg.norm(p[2:5]), 1e-8)
    p[2:6] /= norm
    return p


@njit(cache=False)
def simulate(evidence, params, dt):
    n, length = evidence.shape
    memory, decision = np.zeros((n, length)), np.zeros((n, length))
    am, az = 1 - np.exp(-dt / params[0]), 1 - np.exp(-dt / params[1])
    for j in range(1, length):
        memory[:, j] = memory[:, j - 1] + am * (-memory[:, j - 1] + evidence[:, j - 1])
        drive = params[2] * evidence[:, j - 1] + params[3] * memory[:, j - 1] + params[4]
        decision[:, j] = decision[:, j - 1] + az * (-decision[:, j - 1] + drive)
    return memory, decision


def gpu_candidates(evidence, candidates, device):
    e = torch.as_tensor(evidence, device=device, dtype=torch.float64)
    p = torch.as_tensor(candidates, device=device, dtype=torch.float64)
    z = torch.zeros((len(p), len(e), e.shape[-1]), device=device, dtype=torch.float64)
    m = torch.zeros((len(p), len(e)), device=device, dtype=torch.float64)
    am, az = (1 - torch.exp(-1 / FS / p[:, 0, None])), (1 - torch.exp(-1 / FS / p[:, 1, None]))
    for j in range(1, e.shape[-1]):
        drive = p[:, 2, None] * e[None, :, j - 1] + p[:, 3, None] * m + p[:, 4, None]
        z[:, :, j] = z[:, :, j - 1] + az * (-z[:, :, j - 1] + drive)
        m = m + am * (-m + e[None, :, j - 1])
    return z.cpu().numpy()


def crossings(z, threshold, times, horizon):
    allowed = times <= horizon
    hit = (np.abs(z) >= threshold) & allowed[None]
    exists = hit.any(1)
    first = hit.argmax(1)
    predicted = np.where(exists, np.sign(z[np.arange(len(z)), first]), 0).astype(int)
    latency = np.where(exists, times[first], np.nan)
    return predicted, latency


def loss(z, p, target, latency, times, horizon):
    pred, cross = crossings(z, p[-1], times, horizon)
    last = np.searchsorted(times, horizon, side="right") - 1
    response = target != 0
    probability = expit(2 * z[:, last] / max(p[-1], .05))
    if response.any():
        binary = (target[response] + 1) / 2
        bce = -np.mean(binary * np.log(probability[response] + 1e-8) + (1 - binary) * np.log(1 - probability[response] + 1e-8))
        time_error = np.mean(np.abs(np.where(np.isfinite(cross[response]), cross[response], horizon + 1) - latency[response]))
        threshold_fit = np.mean((np.abs(z[response, np.minimum(np.rint(latency[response] * FS).astype(int), z.shape[1] - 1)]) / p[-1] - 1) ** 2)
    else:
        bce = time_error = threshold_fit = 0.
    return float(np.mean(pred != target) + .3 * bce + .55 * time_error + .03 * min(threshold_fit, 100) + .002 * np.sum(np.square(p[2:5])))


def fit_model(evidence, response, latency, times, horizon, device):
    rng = np.random.default_rng(SEED)
    candidates = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], (160, 6))
    candidates[:, :2] = np.exp(rng.uniform(np.log(BOUNDS[:2, 0]), np.log(BOUNDS[:2, 1]), (160, 2)))
    candidates[0] = [1.5, 1.5, .05, .05, 1., .9]
    z = gpu_candidates(evidence, candidates, device)
    scores = np.array([loss(curve, p, response, latency, times, horizon) for curve, p in zip(z, candidates)])
    best = int(np.argmin(scores))
    trace = pd.DataFrame(candidates, columns=NAMES)
    trace["loss"], trace["candidate"] = scores, np.arange(len(scores))
    evaluated = [(float(scores[best]), candidates[best].copy())]
    def objective(p):
        value = loss(simulate(evidence, p, 1 / FS)[1], p, response, latency, times, horizon)
        evaluated.append((value, np.asarray(p).copy()))
        return value
    local = minimize(objective, candidates[best], method="Powell", bounds=BOUNDS, options=dict(maxiter=18, maxfev=450, xtol=.015, ftol=.001))
    selected_loss, selected = min(evaluated, key=lambda pair: pair[0])
    canonical_p = canonical(selected)
    return canonical_p, trace, dict(random_candidates=160, local_evaluations=int(local.nfev),
        local_success=bool(local.success), local_message=str(local.message), local_loss=float(local.fun),
        selected_loss=float(selected_loss), alpha=1., beta=0., task2_prior=0., scale_constraint="unit norm of w_e,w_m,bias")


def fit_mapping(y, baseline, memory, decision, mask, training, device):
    select = mask & training[:, None]
    design = np.stack([memory, decision], axis=-1)[select]
    residual = np.moveaxis(y - baseline, 1, 2)[select]
    if len(design) < 100:
        raise ValueError("Too few valid training EEG samples")
    x = torch.as_tensor(design, device=device, dtype=torch.float64)
    v = torch.as_tensor(residual, device=device, dtype=torch.float64)
    gram = x.T @ x
    ridge = .1 * torch.trace(gram) / 2 + 1e-8
    weights = torch.linalg.solve(gram + ridge * torch.eye(2, device=device, dtype=torch.float64), x.T @ v).cpu().numpy()
    return weights, float(ridge.cpu()), len(design)


def safe_corr(a, b, rank=False):
    if len(a) < 3 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return np.nan
    return float((spearmanr(a, b) if rank else pearsonr(a, b)).statistic)


def eeg_metrics(y, prediction, mask, fold, model):
    rows = []
    for c, channel in enumerate(CHANNELS):
        actual, fitted = y[:, c][mask], prediction[:, c][mask]
        sse = np.sum((actual - fitted) ** 2)
        sst = np.sum((actual - actual.mean()) ** 2)
        rows.append(dict(fold=fold, model=model, channel=channel, samples=len(actual),
            nrmse=np.sqrt(np.mean((actual - fitted) ** 2)) / max(np.std(actual), 1e-10),
            r2=1 - sse / sst, pearson=safe_corr(actual, fitted)))
    return rows


def behavior_metrics(frame):
    rows, time_rows, cms = [], [], []
    groups = list(frame.groupby("fold", sort=False)) + [("pooled", frame)]
    for name, x in groups:
        responded = x.response_direction != 0
        y = x.loc[responded, "response_direction"].to_numpy()
        pred = x.loc[responded, "predicted_response"].to_numpy()
        binary = x.loc[responded, "predicted_binary_response"].to_numpy()
        recall = recall_score(y, binary, labels=[-1, 1], average=None, zero_division=0)
        rows.append(dict(fold=name, n=len(y), accuracy=accuracy_score(y, binary), balanced_accuracy=balanced_accuracy_score(y, binary),
            macro_f1=f1_score(y, binary, labels=[-1, 1], average="macro", zero_division=0), recall_left=recall[0], recall_right=recall[1],
            threshold_accuracy=accuracy_score(y, pred),
            threshold_balanced_accuracy=float(np.mean([np.mean(pred[y == side] == side) for side in [-1, 1]])),
            threshold_macro_f1=f1_score(y, pred, labels=[-1, 1], average="macro", zero_division=0),
            threshold_recall_left=float(np.mean(pred[y == -1] == -1)), threshold_recall_right=float(np.mean(pred[y == 1] == 1)),
            threshold_response_coverage=float(np.mean(pred != 0)),
            actual_no_response=int(sum(~responded))))
        cm = confusion_matrix(y, binary, labels=[-1, 1])
        for i, actual in enumerate([-1, 1]):
            for j, predicted in enumerate([-1, 1]):
                cms.append(dict(fold=name, actual=actual, predicted=predicted, count=int(cm[i, j]), classifier="binary_readout"))
        for actual in [-1, 1]:
            for predicted in [-1, 0, 1]:
                cms.append(dict(fold=name, actual=actual, predicted=predicted, count=int(np.sum((y == actual) & (pred == predicted))), classifier="threshold"))
        observed = x.loc[responded, "cue_to_response_latency"].to_numpy()
        predicted_time = x.loc[responded, "predicted_crossing_time"].to_numpy()
        for method, estimate in [("threshold_crossed_only", predicted_time), ("training_median_baseline", x.loc[responded, "baseline_time"].to_numpy())]:
            valid = np.isfinite(estimate)
            error = estimate[valid] - observed[valid]
            time_rows.append(dict(fold=name, model=method, actual_responses=len(observed), evaluated=int(sum(valid)),
                coverage=float(np.mean(valid)), mae=float(np.mean(np.abs(error))) if valid.any() else np.nan,
                rmse=float(np.sqrt(np.mean(error ** 2))) if valid.any() else np.nan,
                spearman=safe_corr(observed[valid], estimate[valid], rank=True)))
    return pd.DataFrame(rows), pd.DataFrame(time_rows), pd.DataFrame(cms)


def run_models(trials, args, device):
    out = args.out
    x = load_early(args.q2, trials)
    behavior = (trials.task_id == 2).to_numpy()
    # The global storage grid is fixed, not estimated from held-out outcomes.
    times = np.arange(0, 12 + 1 / FS, 1 / FS)
    y, observed = observation_windows(args.data, trials, times)
    outputs, eeg_rows, parameter_rows, trajectory, checks = [], [], [], {}, []
    for train_group, test_group in [("A", "B"), ("B", "A")]:
        fold = f"{train_group}_to_{test_group}"
        print(f"Stage 2: fit {fold} on training responses only", flush=True)
        training = behavior & (trials.record_group == train_group).to_numpy()
        testing = behavior & (trials.record_group == test_group).to_numpy()
        latency = trials.cue_to_response_latency.to_numpy(float)
        response = trials.response_direction.to_numpy(int)
        horizon = float(np.ceil((np.nanmax(latency[training]) + .75) * FS) / FS)
        if horizon > times[-1]:
            raise ValueError("Training observation horizon exceeds fixed safety grid")
        q, evidence, baseline, provenance = fold_sources(x, trials, args.q2, train_group, times, device)
        fit_times = times[times <= horizon]
        params, search, fitted = fit_model(evidence[training, :len(fit_times)], response[training], latency[training], fit_times, horizon, device)
        memory, decision = simulate(evidence, params, 1 / FS)
        memory[~behavior], decision[~behavior] = np.nan, np.nan
        pred, cross = crossings(decision[behavior], params[-1], times, horizon)
        full_pred = np.zeros(len(trials), dtype=int)
        full_cross = np.full(len(trials), np.nan)
        full_pred[behavior], full_cross[behavior] = pred, cross
        final_index = len(fit_times) - 1
        # A binary readout is separate from the genuine threshold response.
        binary = np.where(decision[:, final_index] >= 0, 1, -1)
        binary[full_pred != 0] = full_pred[full_pred != 0]
        weight, ridge, fit_samples = fit_mapping(y, baseline, memory, decision, observed, training, device)
        extended = baseline + np.einsum("nts,sc->nct", np.stack([memory, decision], axis=-1), weight)
        test_mask = observed & testing[:, None]
        ee = eeg_metrics(y, baseline, test_mask, fold, "q2_only") + eeg_metrics(y, extended, test_mask, fold, "q2_plus_states")
        for row in ee:
            row["scope"] = "full_pre_response"
        eeg_rows.extend(ee)
        late = test_mask & (times[None] > provenance["evidence_available_s"])
        late_rows = eeg_metrics(y, baseline, late, fold, "q2_only") + eeg_metrics(y, extended, late, fold, "q2_plus_states")
        for row in late_rows:
            row["scope"] = "after_evidence"
        eeg_rows.extend(late_rows)
        search.to_csv(out / f"tables/{fold}_parameter_search.csv", index=False)
        fit_rows = trials[behavior].copy()
        fit_rows["fold"] = fold
        fit_rows["split"] = np.where(training[behavior], "train", "test")
        fit_rows["predicted_response"] = full_pred[behavior]
        fit_rows["predicted_binary_response"] = binary[behavior]
        fit_rows["predicted_crossing_time"] = full_cross[behavior]
        fit_rows["max_decision_state"] = decision[behavior, :len(fit_times)].max(1)
        fit_rows["min_decision_state"] = decision[behavior, :len(fit_times)].min(1)
        fit_rows["final_memory_state"] = memory[behavior, final_index]
        fit_rows["final_decision_state"] = decision[behavior, final_index]
        fit_rows["decision_threshold"] = params[-1]
        fit_rows["prediction_horizon"] = horizon
        fit_rows["baseline_time"] = np.nanmedian(latency[training])
        fit_rows["eeg_eligible"] = observed[behavior].any(1)
        fit_rows.to_csv(out / f"tables/{fold}_trial_predictions.csv", index=False)
        outputs.append(fit_rows[fit_rows.split == "test"])
        np.savez_compressed(out / f"model/{fold}_states.npz", row_id=trials.row_id.to_numpy(), time_s=times,
            q_left=q[:, 0], q_right=q[:, 1], q_common=q[:, 2], source_evidence=q[:, 1] - q[:, 0],
            evidence=evidence, memory=memory, decision=decision, observed_eeg=y, q2_eeg=baseline, extended_eeg=extended,
            eeg_mask=observed, behavioral_model_available=behavior, training=training, testing=testing,
            threshold=params[-1], horizon=horizon, parameters=params, mapping=weight)
        provenance.update(fitted, horizon=horizon, parameters=dict(zip(NAMES, map(float, params))), mapping=weight.tolist(),
            mapping_ridge=ridge, mapping_training_samples=fit_samples, behavioral_training_ids=trials.row_id[training].tolist(),
            test_ids=trials.row_id[testing].tolist(), eeg_training_trials=int(sum(observed[training].any(1))),
            eeg_test_trials=int(sum(observed[testing].any(1))))
        dump(out / f"model/{fold}_parameters.json", provenance)
        parameter_rows.extend([dict(fold=fold, parameter=name, value=float(value)) for name, value in zip(NAMES, params)])
        checks.append(dict(fold=fold, train=int(sum(training)), test=int(sum(testing)), disjoint=not bool(np.any(training & testing)),
            initial_memory_zero=bool(np.all(memory[behavior, 0] == 0)), initial_decision_zero=bool(np.all(decision[behavior, 0] == 0)),
            finite=bool(np.isfinite(memory[behavior]).all() and np.isfinite(decision[behavior]).all()),
            eeg_fit_samples=fit_samples, eeg_test_trials=int(sum(observed[testing].any(1))), threshold_predictions_no_response=int(sum(full_pred[testing] == 0))))
        trajectory[fold] = dict(time=times, evidence=evidence, memory=memory, decision=decision, params=params,
                                predictions=fit_rows[fit_rows.split == "test"], observed=observed, y=y, baseline=baseline, extended=extended)
        print(f"Stage 2 saved {fold}: loss={fitted['selected_loss']:.4f}; test n={sum(testing)}; EEG n={sum(observed[testing].any(1))}", flush=True)
    predictions = pd.concat(outputs, ignore_index=True).sort_values("row_id")
    predictions.to_csv(out / "tables/out_of_fold_predictions.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(out / "tables/parameters.csv", index=False)
    pd.DataFrame(eeg_rows).to_csv(out / "tables/eeg_metrics.csv", index=False)
    dump(out / "model/numerical_checks.json", checks)
    if not all(c["disjoint"] and c["initial_memory_zero"] and c["initial_decision_zero"] and c["finite"] for c in checks):
        raise ValueError("Numerical or split verification failed")
    (out / "stage2_check.md").write_text("# 阶段二检查\n\n两折均通过训练测试隔离、零初值及有限数值检查。每折一百六十个有界候选和一次局部精调；完整参数、搜索损失、训练编号及所有试次的源轨迹均已保存。任务一的记忆、决策状态留空，不把缺失点击包装成遗漏机制。\n\n源反演直接调用问题二模型，不输入测试提示或点击标签。每折使用已有的训练记录专属问题二模型，并重新用训练记录校准证据尺度。源差为右源减左源；先收集固定零点八秒证据再开始读入模型，禁止早于取证结束使用整窗估计。后续源波形是从早期证据和问题二动力学外推，不是实测晚期源定位。\n\n记忆状态表示海马记忆匹配功能的潜在状态，不声称前额三电极直接测得海马活动。记忆输入系数固定为一，任务二先验为零，不拟合无法识别的任务一行为先验。决策驱动和阈值统一归一化，消除共同尺度任意性。状态在每试次开始清零；首越阈才产生实际阈值应答。二分类额外提供截止时刻符号读出，未越阈时不会伪造越阈时间。\n\n认知优化优先拟合行为，不联合优化脑电；映射仅在训练记录用岭回归估计。小型局部优化由处理器执行；源反演、批量候选状态递推与脑电映射使用所选显卡。脑电观测沿用问题一全记录零相位去噪，跨记录不混合，但它是离线重建评价，不是在线因果预测；整个应答前窗均报告，另存硬质控覆盖率。\n", encoding="utf-8")
    if args.stage >= 3:
        print("Stage 3: evaluate untouched recording groups", flush=True)
        choice, timing, confusion = behavior_metrics(predictions)
        choice.to_csv(out / "tables/choice_metrics.csv", index=False)
        timing.to_csv(out / "tables/time_metrics.csv", index=False)
        confusion.to_csv(out / "tables/confusion_matrix.csv", index=False)
        from Ques3.figures import make_figures, write_report
        make_figures(trials, predictions, trajectory, choice, timing, confusion, pd.DataFrame(eeg_rows), out)
        write_report(trials, predictions, choice, timing, pd.DataFrame(eeg_rows), checks, out)
        print(choice.to_string(index=False), flush=True)
        print(timing.to_string(index=False), flush=True)

