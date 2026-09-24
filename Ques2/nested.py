"""Nested train-only source decoding and reproducible fold artifacts."""
from pathlib import Path
import hashlib
import pickle
import time
import numpy as np
import pandas as pd
from Ques2.source_features import ALPHAS, fit_state, feature_sets, feature_names
from Ques2.classifier import CLASSIFIERS, SIZES, fit_transformer, transform, estimator, fit_classifier, evaluate_classifier, metrics
from Ques2.iter_utils import load_data, outer_splits, purge, save_state, mechanism_rows, finalize, dump


def run(out, data, device, hardware):
    np.random.seed(20260924)
    started = time.time()
    dump(out / "nested_completion.json", dict(status="running"))
    protocol = dict(seed=20260924, inversion_alphas=ALPHAS, classifier_grid=CLASSIFIERS, selected_dimensions=SIZES,
                    schemes=["erp", "source", "combined"], main_metric="mean inner-fold balanced accuracy",
                    inner_validation="all available chronological blocks, train-only one-trial purge",
                    outer_validation="two cross-recording directions plus five leave-block-out folds per recording group",
                    model="refit the original fifteen-parameter simple model in every inner and outer training split",
                    feature_counts=dict(erp=15, source=12, combined=15), combined_scalp_weight=.5,
                    no_test_rejection=True, known_test_set_from_prior_iteration=True,
                    exploratory_evaluation="not a new prospectively untouched confirmation set", hardware=hardware,
                    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")})
    dump(out / "nested_protocol.json", protocol)
    try:
        X, meta = load_data(out, data, out)
        labels, tasks = meta.y.to_numpy(), meta.task.to_numpy()
        predictions, metric_rows, selections, inner_rows, mechanisms, manifests = [], [], [], [], [], []
        state_cache = {}
        def state_for(ids, name):
            key = tuple(ids)
            if key not in state_cache:
                print(f"  Model fit {name}: {len(ids)} input trials", flush=True)
                state_cache[key] = fit_state(X, meta, ids, device)
            return state_cache[key]
        splits = list(outer_splits(meta))
        for outer_number, split in enumerate(splits, 1):
            tag, mode, group, train, test = split
            print(f"Outer {outer_number}/{len(splits)} {tag}: train={len(train)}, test={len(test)}", flush=True)
            blocks = sorted(meta.iloc[train].block.unique())
            fold_candidates = []
            inner_manifests = []
            for block in blocks:
                valid = train[meta.block.to_numpy()[train] == block]
                inner = purge(train[meta.block.to_numpy()[train] != block], valid, meta)
                state = state_for(inner, f"{tag}_inner{block}")
                kept = state["training_indices"]
                name = f"{tag}_inner{block}"
                save_state(out / f"nested_{name}.json", state)
                inner_manifests.append(dict(block=int(block), training=inner.tolist(), fitting=kept.tolist(), validation=valid.tolist(), state=name))
                for ai, alpha in enumerate(ALPHAS):
                    fi, _, _, _ = feature_sets(X[kept], tasks[kept], state, alpha, device)
                    fv, _, _, _ = feature_sets(X[valid], tasks[valid], state, alpha, device)
                    for scheme in ["erp", "source", "combined"]:
                        if scheme == "erp" and ai:
                            continue
                        transform_state, zi = fit_transformer(fi[scheme], labels[kept], tasks[kept], scheme)
                        for size in SIZES[scheme]:
                            zj = transform(fv[scheme], tasks[valid], transform_state, size)
                            for family, parameter in CLASSIFIERS:
                                clf = estimator(family, parameter).fit(zi[:, transform_state["rank"][:size]], labels[kept])
                                pred, score = clf.predict(zj), clf.decision_function(zj)
                                fold_candidates.append(dict(outer=tag, inner_block=int(block), scheme=scheme, alpha=float(alpha if scheme != "erp" else 0.),
                                                            size=size, family=family, parameter=parameter, **metrics(labels[valid], pred, score)))
            candidate_frame = pd.DataFrame(fold_candidates)
            inner_rows.extend(fold_candidates)
            keys = ["scheme", "alpha", "size", "family", "parameter"]
            candidate_summary = candidate_frame.groupby(keys, sort=False)[["balanced_accuracy", "accuracy", "auc"]].mean().reset_index()
            candidate_summary["score_std"] = candidate_frame.groupby(keys, sort=False).balanced_accuracy.std().to_numpy()
            candidate_summary.to_csv(out / f"nested_{tag}_inner_search.csv", index=False)
            winners = {}
            for scheme in ["erp", "source", "combined"]:
                candidates = candidate_summary[candidate_summary.scheme == scheme]
                ranked = candidates.sort_values(["balanced_accuracy", "size"], ascending=[False, True], kind="stable")
                winners[scheme] = ranked.iloc[0].to_dict()
            selected_scheme = max(["erp", "source", "combined"], key=lambda s:winners[s]["balanced_accuracy"])
            state = state_for(train, tag+"_final")
            kept = state["training_indices"]
            save_state(out / f"nested_{tag}_final.json", state)
            outer_manifest = dict(tag=tag, mode=mode, group=group, train=train.tolist(), fit=kept.tolist(), test=test.tolist(),
                                  inner=inner_manifests, selected_scheme=selected_scheme, winners=winners)
            manifests.append(outer_manifest)
            for scheme in ["erp", "source", "combined"]:
                winner = winners[scheme]
                alpha = float(winner["alpha"]) if scheme != "erp" else ALPHAS[0]
                ft, qt, operator, strength = feature_sets(X[kept], tasks[kept], state, alpha, device)
                fe, qe, _, _ = feature_sets(X[test], tasks[test], state, alpha, device)
                bundle = fit_classifier(ft[scheme], labels[kept], tasks[kept], scheme, int(winner["size"]), winner["family"], float(winner["parameter"]))
                pred, score = evaluate_classifier(bundle, fe[scheme], tasks[test])
                row = dict(mode=mode, fold=tag, group=group, scheme=scheme, chosen=scheme == selected_scheme,
                           inner_balanced_accuracy=winner["balanced_accuracy"], alpha=alpha, size=int(winner["size"]), family=winner["family"], parameter=winner["parameter"],
                           **metrics(labels[test], pred, score))
                metric_rows.append(row)
                selections.append(dict(fold=tag, scheme=scheme, selected_indices=bundle["transformer"]["rank"][:bundle["size"]].tolist(),
                                       selected_features=[feature_names(scheme)[i] for i in bundle["transformer"]["rank"][:bundle["size"]]],
                                       inner_balanced_accuracy=float(winner["balanced_accuracy"])))
                records = [dict(mode=mode, fold=tag, group=group, scheme=scheme, chosen=scheme == selected_scheme, index=int(idx),
                                task=int(tasks[idx]), block=int(meta.block.iloc[idx]), y=int(y), prediction=int(p), score=float(s))
                           for idx, y, p, s in zip(test, labels[test], pred, score)]
                predictions.extend(records)
                artifact = dict(bundle=bundle, state=state, alpha=alpha, training_indices=kept, test_indices=test,
                                operator=operator, regularization=strength, features=feature_names(scheme))
                with (out / f"nested_{tag}_{scheme}.pkl").open("wb") as stream:
                    pickle.dump(artifact, stream)
                np.savez_compressed(out / f"nested_{tag}_{scheme}_features.npz", train=ft[scheme], test=fe[scheme], train_indices=kept, test_indices=test)
                if scheme == "source":
                    mechanisms += mechanism_rows(X, qt, qe, state, kept, test, meta, mode, tag, alpha, operator)
                print(f"  {scheme}: inner BA={winner['balanced_accuracy']:.3f}, test ACC={row['accuracy']:.3f}, BA={row['balanced_accuracy']:.3f}, AUC={row['auc']:.3f}", flush=True)
            pd.DataFrame(metric_rows).to_csv(out / "nested_fold_metrics.csv", index=False)
            pd.DataFrame(predictions).to_csv(out / "nested_predictions.csv", index=False)
            dump(out / "nested_split_manifest.json", manifests)
        pd.DataFrame(inner_rows).to_csv(out / "nested_inner_fold_scores.csv", index=False)
        pd.DataFrame(mechanisms).to_csv(out / "nested_separability.csv", index=False)
        dump(out / "nested_feature_selection.json", selections)
        finalize(out, X, meta, device)
        dump(out / "nested_completion.json", dict(status="complete", outer_folds=len(splits), unique_fits=len(state_cache), elapsed_seconds=time.time()-started))
        print(f"Completed: {out}; elapsed={time.time()-started:.1f}s", flush=True)
    except Exception as exc:
        dump(out / "nested_completion.json", dict(status="failed", error=str(exc)))
        raise

