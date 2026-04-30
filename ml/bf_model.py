#!/usr/bin/env python3
"""
Train a GradientBoostingClassifier to predict Bloom filter benefit.

Feature set — build-side only (deployable at plan time):
  build_cardinality, distinct_estimate, filter_size_bytes,
  build_time_ms, distinctness_ratio (engineered)

Excluded probe features and rationale:
  rows_in, probe_batches — post-hoc probe observations, not available at plan
    time in the Rust optimizer.  (Earlier "Option B" model retained these and
    got F1=1.0 — that's the oracle upper bound, not a deployable model.)
  rows_eliminated, rows_out, probe_time_ms — same reason plus direct label
    leakage (rows_eliminated is in the label formula).

Run from repo root:  python ml/bf_model.py
Outputs: ml/bf_model.json, ml/feature_importances.png, ml/WRITEUP.md
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

REPO_ROOT  = Path(__file__).resolve().parent.parent
ML_DIR     = Path(__file__).resolve().parent
CSV_PATH   = REPO_ROOT / "benchmarks" / "results" / "bf_metrics.csv"
MODEL_JSON = ML_DIR / "bf_model.json"
PNG_PATH   = ML_DIR / "feature_importances.png"
WRITEUP    = ML_DIR / "WRITEUP.md"

FEATURE_COLS = [
    "build_cardinality",
    "distinct_estimate",
    "filter_size_bytes",
    "build_time_ms",
    "distinctness_ratio",
]
LABEL = "was_beneficial"

# Metrics from the previous oracle model (probe features included, not deployable).
# Kept here so _write_writeup can show the gap.
_ORACLE_METRICS = {
    "f1": 1.0, "precision": 1.0, "recall": 1.0, "roc_auc": 1.0,
    "top_features": "rows_in (0.8884), probe_batches (0.1013)",
}


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["distinctness_ratio"] = df["distinct_estimate"] / np.maximum(df["build_cardinality"], 1)
    return out[FEATURE_COLS]


def sample_weights(y: pd.Series) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    w = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
    return np.array([w[yi] for yi in y])


# ── JSON export ───────────────────────────────────────────────────────────────

def _tree_to_nodes(tree) -> list:
    nodes = []
    for i in range(tree.node_count):
        feat = int(tree.feature[i])
        if feat == -2:  # TREE_UNDEFINED → leaf
            nodes.append({
                "feature": -1, "threshold": None,
                "left": -1, "right": -1,
                "value": float(tree.value[i][0][0]),
            })
        else:
            nodes.append({
                "feature": feat,
                "threshold": float(tree.threshold[i]),
                "left": int(tree.children_left[i]),
                "right": int(tree.children_right[i]),
                "value": None,
            })
    return nodes


def export_model(clf: GradientBoostingClassifier, metrics: dict) -> dict:
    # init_score: log-odds of the positive class prior from the fitted DummyClassifier.
    # DummyClassifier is fitted with sample_weight so class_prior_ is the weighted prior.
    prior = np.clip(float(clf.init_.class_prior_[1]), 1e-7, 1 - 1e-7)
    init_score = float(np.log(prior / (1.0 - prior)))

    return {
        "model_type": "gradient_boosted_trees",
        "feature_names": FEATURE_COLS,
        "n_estimators": int(clf.n_estimators_),
        "learning_rate": float(clf.learning_rate),
        "init_score": init_score,
        "trees": [{"nodes": _tree_to_nodes(stage[0].tree_)} for stage in clf.estimators_],
        "metrics": metrics,
    }


# ── Inference from JSON ───────────────────────────────────────────────────────

def predict_from_json(features_dict: dict, model: dict) -> float:
    """Score one example using the exported JSON model. Returns P(beneficial=1).

    Replicates sklearn GBT binary predict_proba without importing sklearn:
      raw = init_score + lr * sum(leaf_value per tree)
      P  = sigmoid(raw)
    """
    x     = [features_dict[f] for f in model["feature_names"]]
    score = model["init_score"]
    lr    = model["learning_rate"]
    for tree in model["trees"]:
        nodes = tree["nodes"]
        idx = 0
        while nodes[idx]["feature"] != -1:
            n = nodes[idx]
            idx = n["left"] if x[n["feature"]] <= n["threshold"] else n["right"]
        score += lr * nodes[idx]["value"]
    return 1.0 / (1.0 + np.exp(-score))


# ── Unit test ─────────────────────────────────────────────────────────────────

def _unit_test_json(clf, model, X_test):
    """Confirm predict_from_json matches clf.predict_proba on 5 random rows."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=5, replace=False)
    X_sub = X_test.iloc[idx]
    sk_probs = clf.predict_proba(X_sub)[:, 1]

    all_pass = True
    for i, sp in enumerate(sk_probs):
        row = X_sub.iloc[i]
        feat = {k: float(row[k]) for k in model["feature_names"]}
        jp   = predict_from_json(feat, model)
        diff = abs(jp - sp)
        ok   = diff < 1e-6
        all_pass = all_pass and ok
        print(f"  [{i}] json={jp:.7f}  sklearn={sp:.7f}  Δ={diff:.2e}  {'✓' if ok else '✗ FAIL'}")

    if all_pass:
        print("predict_from_json unit test: PASS (5/5 rows within 1e-6)")
    else:
        raise AssertionError("predict_from_json unit test FAILED — JSON export is incorrect")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows.  Positive rate: {df[LABEL].mean():.3%}")
    print(df.groupby("query_id")[LABEL].agg(["sum", "count", "mean"])
            .sort_values("mean", ascending=False).to_string())

    X = engineer_features(df)
    y = df[LABEL]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    sw = sample_weights(y_tr)
    print(f"\nTrain: {len(X_tr)}  Test: {len(X_te)}")

    clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    clf.fit(X_tr, y_tr, sample_weight=sw)

    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]

    f1   = f1_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec  = recall_score(y_te, y_pred)
    auc  = roc_auc_score(y_te, y_prob)
    cm   = confusion_matrix(y_te, y_pred)

    metrics = {
        "f1":        round(float(f1),   4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec),  4),
        "roc_auc":   round(float(auc),  4),
    }

    print("\n=== Test-set metrics (build-side-only, deployable) ===")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v}")
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):\n{cm}")

    if metrics["f1"] < 0.3:
        print("\n⚠  F1 < 0.3 — investigate before proceeding.")

    # Feature importances plot
    imp = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot.barh(ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Feature Importances — Build-side-only GradientBoostingClassifier")
    ax.set_xlabel("Mean decrease in impurity")
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)
    print(f"\nSaved {PNG_PATH}")

    model = export_model(clf, metrics)
    MODEL_JSON.write_text(json.dumps(model, indent=2))
    print(f"Saved {MODEL_JSON}  ({len(model['trees'])} trees)")

    print("\n=== predict_from_json unit test ===")
    _unit_test_json(clf, model, X_te)

    imp_desc = imp.sort_values(ascending=False)
    _write_writeup(metrics, imp_desc, cm)
    print(f"\nWrote {WRITEUP}")

    top3 = list(imp_desc.items())[:3]
    print("\n── Headline ──────────────────────────────")
    print(f"  Top 3 features: {', '.join(f'{f} ({v:.4f})' for f, v in top3)}")
    print(f"  Test-set F1   : {metrics['f1']}")
    print(f"  (Sweet-spot budget printed by evaluation.py --sweep)")


def _write_writeup(metrics, imp_desc, cm):
    oracle = _ORACLE_METRICS
    lines = [
        "# ML Bloom Filter Budget Policy — Results\n",
        "## 1. Oracle baseline (probe features included — NOT deployable)\n",
        "> Trained with `rows_in` + `probe_batches` as features.  These are only",
        "> available after the query runs, so this model cannot be used in the",
        "> Rust planner.  Kept here as an upper-bound reference.\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| f1 | {oracle['f1']} |",
        f"| precision | {oracle['precision']} |",
        f"| recall | {oracle['recall']} |",
        f"| roc_auc | {oracle['roc_auc']} |",
        "",
        f"Top features: {oracle['top_features']}",
        "",
        "## 2. Build-side-only model (deployable)\n",
        "Features available at plan time: `build_cardinality`, `distinct_estimate`,",
        "`filter_size_bytes`, `build_time_ms`, `distinctness_ratio`.\n",
        "> **Note on `build_time_ms`:** this is measured during filter construction,",
        "> so it is technically post-build. The Rust side can estimate it from",
        "> cardinality (linear in build_cardinality), so it is borderline-acceptable.",
        "> Flagged here; remove it if the Rust estimator is unavailable.\n",
        "| Metric | Value |",
        "|--------|-------|",
        *[f"| {k} | {v} |" for k, v in metrics.items()],
        "",
        "### Confusion Matrix (rows = actual, cols = predicted)",
        "```",
        "              Pred 0   Pred 1",
        f"  Actual 0  {cm[0,0]:7d}  {cm[0,1]:7d}",
        f"  Actual 1  {cm[1,0]:7d}  {cm[1,1]:7d}",
        "```",
        "",
        "### Feature Importances (mean decrease in impurity, descending)\n",
        "| Feature | Importance |",
        "|---------|------------|",
        *[f"| `{f}` | {v:.4f} |" for f, v in imp_desc.items()],
        "",
        "## 3. Evaluation results\n",
        "_Run `python ml/evaluation.py --budget <bytes>` to populate._",
        "",
        "## 4. Budget sweep\n",
        "_Run `python ml/evaluation.py --sweep` to populate._",
        "",
        "## 5. Knapsack tests\n",
        "_Run `python ml/knapsack.py` for PASS/FAIL output._",
        "",
    ]
    WRITEUP.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
