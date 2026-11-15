from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FAILURES, make_features

SEED = 42


@dataclass(frozen=True)
class EvaluationConfig:
    holdout_start: str = "2020-07-08"
    beta: float = 2.0


def build_models() -> dict[str, Pipeline]:
    return {
        "Logistic regression": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED))]),
        "Random forest": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))]),
        "Histogram gradient boosting": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(max_iter=220, learning_rate=0.06, max_leaf_nodes=20, l2_regularization=1.0, class_weight="balanced", random_state=SEED))]),
    }


def _metrics(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction).ravel()
    return {"roc_auc": roc_auc_score(y_true, probability), "pr_auc": average_precision_score(y_true, probability),
            "precision": precision_score(y_true, prediction, zero_division=0), "recall": recall_score(y_true, prediction, zero_division=0),
            "f1": f1_score(y_true, prediction, zero_division=0), "f2": fbeta_score(y_true, prediction, beta=2, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, prediction), "brier_score": brier_score_loss(y_true, probability),
            "threshold": threshold, "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)}


def _event_folds(index: pd.DatetimeIndex) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Validate around events 2 and 3, training strictly on earlier history."""
    folds = []
    for event_number in (2, 3):
        start = pd.Timestamp(FAILURES[event_number - 1][0])
        train_mask = index < start - pd.Timedelta("48h")
        valid_mask = (index >= start - pd.Timedelta("7d")) & (index < start)
        folds.append((np.flatnonzero(train_mask), np.flatnonzero(valid_mask), f"event_{event_number}"))
    return folds


def select_threshold(y_true: np.ndarray, probability: np.ndarray, beta: float) -> tuple[float, float]:
    candidates = np.linspace(0.02, 0.80, 395)
    scores = [fbeta_score(y_true, probability >= value, beta=beta, zero_division=0) for value in candidates]
    best = int(np.argmax(scores))
    return float(candidates[best]), float(scores[best])


def _save_confusion(y_true: pd.Series, probability: np.ndarray, threshold: float, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.4, 4.5))
    sns.heatmap(confusion_matrix(y_true, probability >= threshold), annot=True, fmt=",d", cmap="YlGn", cbar=False, ax=axis)
    axis.set(xlabel="Predicted class", ylabel="Actual class", title="Chronological holdout confusion matrix")
    fig.tight_layout(); fig.savefig(output, dpi=180, bbox_inches="tight"); plt.close(fig)


def run_evaluation(data: pd.DataFrame, output_dir: Path, config: EvaluationConfig = EvaluationConfig()) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y, events = make_features(data)
    holdout_start = pd.Timestamp(config.holdout_start)
    development = X.index < holdout_start
    X_dev, y_dev = X.loc[development], y.loc[development]
    X_test, y_test = X.loc[~development], y.loc[~development]
    models = build_models(); rows = []; pooled: dict[str, tuple[list[int], list[float]]] = {}
    for name, model in models.items():
        fold_rows = []; pooled_y: list[int] = []; pooled_p: list[float] = []
        for train_idx, valid_idx, fold_name in _event_folds(X_dev.index):
            fitted = clone(model).fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])
            probability = fitted.predict_proba(X_dev.iloc[valid_idx])[:, 1]
            truth = y_dev.iloc[valid_idx]
            pooled_y.extend(truth.tolist()); pooled_p.extend(probability.tolist())
            fold_rows.append({"fold": fold_name, "pr_auc": average_precision_score(truth, probability), "roc_auc": roc_auc_score(truth, probability)})
        rows.append({"model": name, "pr_auc_mean": float(np.mean([r["pr_auc"] for r in fold_rows])), "pr_auc_std": float(np.std([r["pr_auc"] for r in fold_rows], ddof=1)),
                     "roc_auc_mean": float(np.mean([r["roc_auc"] for r in fold_rows])), "roc_auc_std": float(np.std([r["roc_auc"] for r in fold_rows], ddof=1))})
        pooled[name] = (pooled_y, pooled_p)
    cv_results = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)
    best_name = str(cv_results.iloc[0]["model"]); pooled_y, pooled_p = pooled[best_name]
    threshold, validation_f2 = select_threshold(np.asarray(pooled_y), np.asarray(pooled_p), config.beta)
    best_model = models[best_name].fit(X_dev, y_dev)
    holdout_probability = best_model.predict_proba(X_test)[:, 1]
    holdout = _metrics(y_test, holdout_probability, threshold)
    _save_confusion(y_test, holdout_probability, threshold, output_dir / "confusion_matrix.png")
    joblib.dump(best_model, output_dir / "best_model.joblib")
    alert_times = X_test.index[holdout_probability >= threshold]
    failure_start = pd.Timestamp(FAILURES[3][0])
    pre_failure_alerts = alert_times[alert_times < failure_start]
    first_alert = pre_failure_alerts.min() if len(pre_failure_alerts) else None
    lead_hours = (failure_start - first_alert).total_seconds() / 3600 if first_alert is not None else None
    summary = {"dataset_rows": int(len(data)), "window_rows": int(len(X)), "failure_events": 4, "prediction_horizon_hours": 24,
               "development_end": config.holdout_start, "holdout_event": 4, "best_model": best_name,
               "threshold_selection": {"method": "maximum pooled event-validation F2", "selected_threshold": threshold, "validation_f2": validation_f2},
               "holdout": holdout, "event_detection": {"detected": first_alert is not None, "first_alert": first_alert.isoformat() if first_alert is not None else None, "lead_hours": lead_hours},
               "features": X.columns.tolist(), "evaluation_note": "Events 2-3 used for forward event validation; July event 4 is the untouched chronological holdout."}
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
