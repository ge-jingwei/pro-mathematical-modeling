"""Train-only task centering, sparse ranking, and bounded linear model search."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix

SEED = 20260924
CLASSIFIERS = [("logistic", c) for c in [.01, .1, 1., 10.]] + [("svm", c) for c in [.01, .1, 1., 10.]] + [("lda", s) for s in [.1, .5, .9]]
SIZES = {"erp": [8, 15], "source": [6, 12], "combined": [8, 12, 15]}


def estimator(name, value):
    if name == "logistic":
        return LogisticRegression(C=value, penalty="l2", class_weight="balanced", solver="liblinear", max_iter=3000, random_state=SEED)
    if name == "svm":
        return LinearSVC(C=value, penalty="l2", class_weight="balanced", dual="auto", max_iter=20000, random_state=SEED)
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=value, priors=[.5, .5])


def fit_transformer(F, y, tasks, scheme):
    task_means = {int(t):F[tasks == t].mean(0) for t in np.unique(tasks)}
    centered = F-np.stack([task_means[int(t)] for t in tasks])
    scaler = StandardScaler().fit(centered)
    weights = np.ones(F.shape[1])
    if scheme == "combined":
        weights[10:] = .5
    z = scaler.transform(centered)*weights
    selector = LogisticRegression(C=1., penalty="l1", solver="liblinear", class_weight="balanced", max_iter=3000, random_state=SEED).fit(z, y)
    rank = np.argsort(-np.abs(selector.coef_[0]), kind="stable")
    return dict(task_means=task_means, scaler=scaler, weights=weights, rank=rank, l1_coefficients=selector.coef_[0]), z


def transform(F, tasks, state, size):
    centered = F-np.stack([state["task_means"][int(t)] for t in tasks])
    return (state["scaler"].transform(centered)*state["weights"])[:, state["rank"][:size]]


def fit_classifier(F, y, tasks, scheme, size, family, parameter):
    transformer, z = fit_transformer(F, y, tasks, scheme)
    classifier = estimator(family, parameter).fit(z[:, transformer["rank"][:size]], y)
    return dict(transformer=transformer, classifier=classifier, size=size, scheme=scheme)


def evaluate_classifier(bundle, F, tasks):
    z = transform(F, tasks, bundle["transformer"], bundle["size"])
    return bundle["classifier"].predict(z), bundle["classifier"].decision_function(z)


def metrics(y, pred, score):
    cm = confusion_matrix(y, pred, labels=[-1, 1])
    if set(y) != {-1, 1}:
        raise ValueError("Both classes are required in every evaluation fold")
    return dict(n=len(y), accuracy=float(accuracy_score(y, pred)), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, labels=[-1, 1], average="macro", zero_division=0)),
                auc=float(roc_auc_score(y, score)), left_recall=float(cm[0, 0]/cm[0].sum()), right_recall=float(cm[1, 1]/cm[1].sum()),
                left_left=int(cm[0, 0]), left_right=int(cm[0, 1]), right_left=int(cm[1, 0]), right_right=int(cm[1, 1]))

