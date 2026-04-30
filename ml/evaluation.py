#!/usr/bin/env python3
"""
4-way Bloom filter policy comparison using bf_metrics.csv.

Policies:
  no_bf        — build nothing; net benefit = 0 everywhere
  single_hop   — first filter alphabetically by col_name per query
                 (assumption: col_name sort order approximates primary join edge)
  always_build — build every candidate; sum net_benefit_ms
  ml_guided    — model-predicted P(beneficial) → global 0/1 knapsack under memory budget

Duplicates: the CSV has multiple runs per (query_id, col_name).  We average
metrics across runs before evaluation to avoid triple-counting.  (Training in
bf_model.py keeps all rows for more signal.)

Usage (from repo root):
  # Single budget
  python ml/evaluation.py [--model ml/bf_model.json] \
                          [--metrics benchmarks/results/bf_metrics.csv] \
                          [--budget BYTES]

  # Budget sweep
  python ml/evaluation.py --sweep [--model ml/bf_model.json] \
                                  [--metrics benchmarks/results/bf_metrics.csv]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ML_DIR    = Path(__file__).resolve().parent
REPO_ROOT = ML_DIR.parent
WRITEUP   = ML_DIR / "WRITEUP.md"

DEFAULT_MODEL   = ML_DIR / "bf_model.json"
DEFAULT_METRICS = REPO_ROOT / "benchmarks" / "results" / "bf_metrics.csv"

SWEEP_BUDGETS = [10_000, 25_000, 50_000, 100_000, 200_000, 500_000]

sys.path.insert(0, str(ML_DIR))
from bf_model import predict_from_json, FEATURE_COLS  # noqa: E402
from knapsack import select_filters_exact              # noqa: E402


# ── Data loading ──────────────────────────────────────────────────────────────

def load_and_aggregate(path: Path) -> pd.DataFrame:
    """Average numeric columns across runs for each (query_id, col_name) pair."""
    df = pd.read_csv(path)
    numeric = [
        "build_cardinality", "distinct_estimate", "filter_size_bytes",
        "build_time_ms", "probe_batches", "rows_in", "rows_out",
        "rows_eliminated", "probe_time_ms", "net_benefit_ms", "was_beneficial",
    ]
    agg = df.groupby(["query_id", "col_name"])[numeric].mean().reset_index()
    agg["distinctness_ratio"] = agg["distinct_estimate"] / np.maximum(agg["build_cardinality"], 1)
    return agg


# ── Policies ──────────────────────────────────────────────────────────────────

def policy_no_bf(_group: pd.DataFrame) -> float:
    return 0.0


def policy_single_hop(group: pd.DataFrame) -> float:
    # Assumption: alphabetically first col_name approximates the primary join edge.
    first = group.sort_values("col_name").iloc[0]
    return float(first["net_benefit_ms"])


def policy_always_build(group: pd.DataFrame) -> float:
    return float(group["net_benefit_ms"].sum())


def _ml_detail(df: pd.DataFrame, model: dict, budget_bytes: int) -> list:
    """Run the global 0/1 knapsack and return full per-candidate detail.

    Each entry is a dict with:
      key, query_id, col_name — identity
      predicted_p             — model P(beneficial)
      weight                  — filter_size_bytes
      net_benefit             — actual net_benefit_ms (averaged across runs)
      was_beneficial          — label average (0-1; use >= 0.5 as bool)
      selected                — True if knapsack chose this filter

    Candidate key: "<query_id>:<col_name>" because col_name alone is not
    unique across queries (e.g. l_orderkey appears in q3, q5, q7, q8, q9).
    """
    candidates = []
    for _, row in df.iterrows():
        key  = f"{row['query_id']}:{row['col_name']}"
        feat = {c: float(row[c]) for c in model["feature_names"]}
        candidates.append({
            "key":           key,
            "query_id":      row["query_id"],
            "col_name":      row["col_name"],
            "predicted_p":   predict_from_json(feat, model),
            "weight":        int(row["filter_size_bytes"]),
            "net_benefit":   float(row["net_benefit_ms"]),
            "was_beneficial": float(row["was_beneficial"]),
        })

    knapsack_in = [{"col_name": c["key"], "value": c["predicted_p"], "weight": c["weight"]}
                   for c in candidates]
    selected_keys = set(select_filters_exact(knapsack_in, budget_bytes))
    for c in candidates:
        c["selected"] = c["key"] in selected_keys
    return candidates


def ml_guided_global(df: pd.DataFrame, model: dict, budget_bytes: int) -> dict:
    """Returns {query_id: net_benefit_ms_sum} for filters chosen by the knapsack."""
    per_query: dict = {}
    for c in _ml_detail(df, model, budget_bytes):
        if c["selected"]:
            per_query[c["query_id"]] = per_query.get(c["query_id"], 0.0) + c["net_benefit"]
    return per_query


# ── Per-budget totals (shared by single-run and sweep) ───────────────────────

def _policy_totals(df: pd.DataFrame, model: dict, budget_bytes: int) -> dict:
    """Return {policy: total_net_benefit_ms} summed across all queries."""
    ml_per_query = ml_guided_global(df, model, budget_bytes)
    totals = {"no_bf": 0.0, "single_hop": 0.0, "always_build": 0.0, "ml_guided": 0.0}
    for _qid, group in df.groupby("query_id"):
        totals["no_bf"]        += policy_no_bf(group)
        totals["single_hop"]   += policy_single_hop(group)
        totals["always_build"] += policy_always_build(group)
        totals["ml_guided"]    += ml_per_query.get(_qid, 0.0)
    return totals


def _per_query_rows(df: pd.DataFrame, model: dict, budget_bytes: int) -> pd.DataFrame:
    """Return per-query breakdown DataFrame (used by single-budget mode)."""
    ml_per_query = ml_guided_global(df, model, budget_bytes)
    rows = []
    for qid, group in df.groupby("query_id"):
        rows.append({
            "query":        qid,
            "no_bf":        policy_no_bf(group),
            "single_hop":   policy_single_hop(group),
            "always_build": policy_always_build(group),
            "ml_guided":    ml_per_query.get(qid, 0.0),
        })
    result = pd.DataFrame(rows).set_index("query")
    totals = result.sum().rename("TOTAL")
    return pd.concat([result, totals.to_frame().T])


# ── Single-budget mode ────────────────────────────────────────────────────────

def run_evaluation(model_path: Path, metrics_path: Path, budget_bytes: int) -> pd.DataFrame:
    with open(model_path) as fh:
        model = json.load(fh)

    df = load_and_aggregate(metrics_path)

    total_memory = int(df["filter_size_bytes"].sum())
    if budget_bytes < 0:
        budget_bytes = total_memory // 2

    print(f"Candidates : {len(df)} distinct (query_id, col_name) pairs")
    print(f"Budget     : {budget_bytes:,} bytes")
    print(f"Total mem  : {total_memory:,} bytes  ({budget_bytes/total_memory:.0%} utilisation)\n")

    result = _per_query_rows(df, model, budget_bytes)
    _print_table(result)
    _append_eval_to_writeup(result, budget_bytes, total_memory)
    print(f"\nAppended results to {WRITEUP}")
    return result


def _print_table(result: pd.DataFrame) -> None:
    width = [10, 12, 12, 14, 12]
    header = (f"{'query':<{width[0]}} {'no_bf':>{width[1]}} {'single_hop':>{width[2]}}"
              f" {'always_build':>{width[3]}} {'ml_guided':>{width[4]}}")
    print(header)
    print("-" * len(header))
    for idx, row in result.iterrows():
        marker = " ←" if str(idx) == "TOTAL" else ""
        print(
            f"{str(idx):<{width[0]}} "
            f"{row['no_bf']:>{width[1]}.2f} "
            f"{row['single_hop']:>{width[2]}.2f} "
            f"{row['always_build']:>{width[3]}.2f} "
            f"{row['ml_guided']:>{width[4]}.2f}"
            f"{marker}"
        )


def _append_eval_to_writeup(result: pd.DataFrame, budget: int, total_memory: int) -> None:
    ml_total = result.loc["TOTAL", "ml_guided"]
    ab_total = result.loc["TOTAL", "always_build"]

    lines = [
        "\n## 3. Evaluation results\n",
        f"Budget: **{budget:,} bytes** / Total filter memory: **{total_memory:,} bytes**\n",
        "| query | no_bf | single_hop | always_build | ml_guided |",
        "|-------|------:|-----------:|-------------:|----------:|",
    ]
    for idx, row in result.iterrows():
        lines.append(
            f"| **{idx}** | {row['no_bf']:.2f} | {row['single_hop']:.2f} |"
            f" {row['always_build']:.2f} | {row['ml_guided']:.2f} |"
        )
    lines += [
        "",
        f"ml_guided {'beats' if ml_total > ab_total else 'does NOT beat'} always_build "
        f"({ml_total:.2f} vs {ab_total:.2f}).",
        "",
    ]

    _patch_writeup("## 3. Evaluation results", "## 4. Budget sweep", "\n".join(lines))


# ── Budget sweep mode ─────────────────────────────────────────────────────────

def run_sweep(model_path: Path, metrics_path: Path) -> pd.DataFrame:
    with open(model_path) as fh:
        model = json.load(fh)
    df = load_and_aggregate(metrics_path)

    total_memory = int(df["filter_size_bytes"].sum())
    print(f"Candidates : {len(df)} distinct (query_id, col_name) pairs")
    print(f"Total mem  : {total_memory:,} bytes\n")

    sweep_rows = []
    for budget in SWEEP_BUDGETS:
        t = _policy_totals(df, model, budget)
        sweep_rows.append({"budget_bytes": budget, **t})

    sweep = pd.DataFrame(sweep_rows)
    csv_path = ML_DIR / "budget_sweep.csv"
    sweep.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # Print table
    print(f"\n{'budget_bytes':>14} {'no_bf':>10} {'single_hop':>12} {'always_build':>14} {'ml_guided':>12}")
    print("-" * 66)
    for _, row in sweep.iterrows():
        print(f"{row['budget_bytes']:>14,.0f} {row['no_bf']:>10.2f} {row['single_hop']:>12.2f}"
              f" {row['always_build']:>14.2f} {row['ml_guided']:>12.2f}")

    # Sweet spot: budget where ml_guided is highest (least negative / most positive)
    best_idx    = sweep["ml_guided"].idxmax()
    sweet_budget = int(sweep.loc[best_idx, "budget_bytes"])
    sweet_val    = sweep.loc[best_idx, "ml_guided"]
    print(f"\nSweet-spot budget: {sweet_budget:,} bytes  →  ml_guided total = {sweet_val:.2f} ms")

    _save_sweep_plot(sweep)
    _append_sweep_to_writeup(sweep, sweet_budget, sweet_val, total_memory)
    print(f"Appended budget sweep to {WRITEUP}")

    # Diagnostic for the sweet-spot budget
    detail = _ml_detail(df, model, sweet_budget)
    diag   = _sweet_spot_diagnostic(detail, sweet_budget)
    _append_diagnostic_to_writeup(diag, sweet_budget)
    print(f"Appended sweet-spot diagnostic to {WRITEUP}")

    return sweep


def _sweet_spot_diagnostic(candidates: list, budget_bytes: int) -> dict:
    """Print and return diagnostic stats for the sweet-spot budget."""
    selected = sorted([c for c in candidates if     c["selected"]],
                      key=lambda c: c["net_benefit"], reverse=True)
    skipped  =        [c for c in candidates if not c["selected"]]

    # "Actually beneficial" = avg net_benefit_ms > 0 (expected-value semantics).
    # Using net_benefit > 0 rather than was_beneficial >= 0.5 because after averaging
    # 3 runs a filter can have positive expected net_benefit but was_beneficial < 0.5
    # (e.g. beneficial in 1/3 runs with a large payoff, non-beneficial in 2/3 with
    # small losses).  The knapsack optimises expected net_benefit, so this is the
    # right ground truth for precision/recall in the diagnostic.
    sel_beneficial  = [c for c in selected if c["net_benefit"] > 0]
    skip_beneficial = [c for c in skipped  if c["net_benefit"] > 0]
    skip_beneficial_top5 = sorted(skip_beneficial,
                                  key=lambda c: c["net_benefit"], reverse=True)[:5]

    n_sel   = len(selected)
    n_skip  = len(skipped)
    n_sel_b = len(sel_beneficial)
    n_skip_b = len(skip_beneficial)
    total_beneficial = n_sel_b + n_skip_b

    prec_pct   = n_sel_b   / max(n_sel, 1)          * 100
    recall_pct = n_sel_b   / max(total_beneficial, 1) * 100

    total_net = sum(c["net_benefit"] for c in selected)
    top3_net  = sum(c["net_benefit"] for c in selected[:3])
    # Express top-3 contribution relative to |total| so sign flips don't mislead
    top3_pct  = (top3_net / total_net * 100) if total_net != 0 else float("nan")

    print(f"\n=== Sweet-spot diagnostic: budget={budget_bytes:,} bytes ===\n")
    print("Note: precision/recall use net_benefit>0 as ground truth (expected-value),")
    print("not was_beneficial>=0.5 (majority-vote), because a filter beneficial in 1/3")
    print("runs with large payoff has positive expected value despite was_beneficial<0.5.\n")
    print("Selected filters (col_name, predicted_prob, actual_net_benefit_ms, was_beneficial):")
    for c in selected:
        sign = "+" if c["net_benefit"] >= 0 else ""
        print(f"  {c['query_id']}.{c['col_name']:<22}"
              f"  p={c['predicted_p']:.2f}"
              f"  actual={sign}{c['net_benefit']:.1f}"
              f"  beneficial={int(round(c['was_beneficial']))}")

    print("\nSkipped filters that would have been beneficial (false negatives, up to 5):")
    if skip_beneficial_top5:
        for c in skip_beneficial_top5:
            sign = "+" if c["net_benefit"] >= 0 else ""
            print(f"  {c['query_id']}.{c['col_name']:<22}"
                  f"  p={c['predicted_p']:.2f}"
                  f"  actual={sign}{c['net_benefit']:.1f}"
                  f"  beneficial={int(round(c['was_beneficial']))}")
    else:
        print("  (none)")

    top3_sign = "+" if top3_net >= 0 else ""
    tot_sign  = "+" if total_net >= 0 else ""
    print(f"\nSummary at budget={budget_bytes:,}:")
    print(f"  Selected:  {n_sel:3d} filters, {n_sel_b:3d} actually beneficial"
          f"  →  precision-at-budget = {prec_pct:.0f}%")
    print(f"  Skipped:   {n_skip:3d} filters, {n_skip_b:3d} actually beneficial"
          f"  →  recall-at-budget    = {recall_pct:.0f}%")
    print(f"  Selected total actual net_benefit: {tot_sign}{total_net:.2f} ms")
    if not (top3_pct != top3_pct):  # NaN check
        print(f"  Of that, top 3 picks contributed: {top3_sign}{top3_net:.2f} ms"
              f"  (= {top3_pct:.0f}% of total)")

    # Verdict
    if top3_pct > 80:
        verdict = (f"Sweet-spot win is fragile: top 3 picks contribute {top3_pct:.0f}%"
                   f" of total benefit. Result may not generalize.")
    elif prec_pct < 50:
        verdict = (f"Sweet-spot precision-at-budget is {prec_pct:.0f}%"
                   f" — model is over-picking but knapsack ranking compensates.")
    else:
        verdict = (f"Sweet-spot win is robust: precision-at-budget = {prec_pct:.0f}%,"
                   f" top 3 picks contribute {top3_pct:.0f}% of total benefit.")
    print(f"\n{verdict}")

    return {
        "budget":        budget_bytes,
        "n_sel":         n_sel,
        "n_sel_b":       n_sel_b,
        "n_skip":        n_skip,
        "n_skip_b":      n_skip_b,
        "prec_pct":      prec_pct,
        "recall_pct":    recall_pct,
        "total_net":     total_net,
        "top3_net":      top3_net,
        "top3_pct":      top3_pct,
        "verdict":       verdict,
        "selected":      selected,
        "fn_top5":       skip_beneficial_top5,
    }


def _append_diagnostic_to_writeup(diag: dict, budget_bytes: int) -> None:
    d = diag
    tot_sign  = "+" if d["total_net"] >= 0 else ""
    top3_sign = "+" if d["top3_net"]  >= 0 else ""
    top3_pct_str = f"{d['top3_pct']:.0f}%" if d["top3_pct"] == d["top3_pct"] else "n/a"

    sel_rows = []
    for c in d["selected"]:
        sign = "+" if c["net_benefit"] >= 0 else ""
        sel_rows.append(
            f"| {c['query_id']}.{c['col_name']} | {c['predicted_p']:.2f}"
            f" | {sign}{c['net_benefit']:.1f} | {int(round(c['was_beneficial']))} |"
        )

    fn_rows = []
    for c in d["fn_top5"]:
        sign = "+" if c["net_benefit"] >= 0 else ""
        fn_rows.append(
            f"| {c['query_id']}.{c['col_name']} | {c['predicted_p']:.2f}"
            f" | {sign}{c['net_benefit']:.1f} | {int(round(c['was_beneficial']))} |"
        )

    lines = [
        "\n## 6. Sweet-spot diagnostic\n",
        f"Budget: **{budget_bytes:,} bytes**\n",
        "### Selected filters\n",
        "| filter | pred_p | actual_net_ms | was_beneficial |",
        "|--------|-------:|--------------:|:--------------:|",
        *sel_rows,
        "",
        "### False negatives (skipped but actually beneficial, up to 5)\n",
        "| filter | pred_p | actual_net_ms | was_beneficial |",
        "|--------|-------:|--------------:|:--------------:|",
        *(fn_rows if fn_rows else ["| — | — | — | — |"]),
        "",
        "### Summary\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Selected | {d['n_sel']} filters, {d['n_sel_b']} actually beneficial |",
        f"| Precision-at-budget | {d['prec_pct']:.0f}% |",
        f"| Skipped | {d['n_skip']} filters, {d['n_skip_b']} actually beneficial |",
        f"| Recall-at-budget | {d['recall_pct']:.0f}% |",
        f"| Total net benefit (selected) | {tot_sign}{d['total_net']:.2f} ms |",
        f"| Top-3 contribution | {top3_sign}{d['top3_net']:.2f} ms ({top3_pct_str} of total) |",
        "",
        f"**{d['verdict']}**",
        "",
    ]
    _patch_writeup("## 6. Sweet-spot diagnostic", "", "\n".join(lines))


def _save_sweep_plot(sweep: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"no_bf": "grey", "single_hop": "orange", "always_build": "red", "ml_guided": "steelblue"}
    labels = {"no_bf": "no_bf", "single_hop": "single_hop",
              "always_build": "always_build", "ml_guided": "ml_guided (knapsack)"}
    for policy, color in colors.items():
        ls = "--" if policy in ("no_bf", "always_build") else "-"
        ax.plot(sweep["budget_bytes"], sweep[policy], label=labels[policy],
                color=color, linestyle=ls, linewidth=2, marker="o", markersize=5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("Budget (bytes, log scale)")
    ax.set_ylabel("Total net benefit (ms)")
    ax.set_title("Policy comparison across memory budgets")
    ax.legend(loc="lower right")
    fig.tight_layout()
    png_path = ML_DIR / "budget_sweep.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved {png_path}")


def _append_sweep_to_writeup(sweep: pd.DataFrame, sweet_budget: int,
                              sweet_val: float, total_memory: int) -> None:
    ab_total = float(sweep["always_build"].iloc[0])
    lines = [
        "\n## 4. Budget sweep\n",
        f"Total filter memory: **{total_memory:,} bytes**.",
        f"Sweet-spot budget: **{sweet_budget:,} bytes** → ml_guided total = **{sweet_val:.2f} ms**",
        f"(always_build baseline = {ab_total:.2f} ms at all budgets)\n",
        "| budget_bytes | no_bf | single_hop | always_build | ml_guided |",
        "|-------------:|------:|-----------:|-------------:|----------:|",
    ]
    for _, row in sweep.iterrows():
        marker = " ◀ sweet spot" if int(row["budget_bytes"]) == sweet_budget else ""
        lines.append(
            f"| {int(row['budget_bytes']):,} | {row['no_bf']:.2f} | {row['single_hop']:.2f} |"
            f" {row['always_build']:.2f} | {row['ml_guided']:.2f} |{marker}"
        )
    lines += [
        "",
        "See `ml/budget_sweep.png` for the trend plot.",
        "",
    ]
    _patch_writeup("## 4. Budget sweep", "## 5. Knapsack tests", "\n".join(lines))


# ── WRITEUP patch helper ──────────────────────────────────────────────────────

def _patch_writeup(start_marker: str, end_marker: str, new_content: str) -> None:
    """Replace the section between start_marker and end_marker (exclusive) in WRITEUP."""
    if not WRITEUP.exists():
        WRITEUP.write_text(new_content + "\n")
        return
    text = WRITEUP.read_text()
    suffix = ""
    if end_marker and end_marker in text:  # empty end_marker → replace to end of file
        suffix = "\n" + text[text.index(end_marker):]
    if start_marker in text:
        text = text[: text.index(start_marker)]
    WRITEUP.write_text(text + new_content + suffix + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Bloom filter policy evaluation")
    p.add_argument("--model",   default=str(DEFAULT_MODEL),   help="Path to bf_model.json")
    p.add_argument("--metrics", default=str(DEFAULT_METRICS), help="Path to bf_metrics.csv")
    p.add_argument("--budget",  type=int, default=-1,
                   help="Memory budget in bytes for single-budget mode (default: half of total)")
    p.add_argument("--sweep",   action="store_true",
                   help="Run budget sweep instead of single-budget evaluation")
    args = p.parse_args()

    if args.sweep:
        run_sweep(Path(args.model), Path(args.metrics))
    else:
        run_evaluation(Path(args.model), Path(args.metrics), args.budget)


if __name__ == "__main__":
    main()
