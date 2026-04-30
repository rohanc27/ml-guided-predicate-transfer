#!/usr/bin/env python3
"""
0/1 knapsack budget policy for Bloom filter selection.

Weight scaling: bytes → KB (÷1024, floor, min 1).
This caps DP memory at O(n × budget_kb) instead of O(n × budget_bytes).
For our workload (n ≈ 50, budgets ≤ a few hundred MB) KB granularity is
more than adequate — filter sizes that differ by < 1 KB are treated identically.
"""


def _kb(bytes_val: int) -> int:
    return max(1, bytes_val // 1024)


def select_filters_exact(candidates: list, budget_bytes: int) -> list:
    """0/1 knapsack via DP.

    candidates: list of {"col_name": str, "value": float, "weight": int (bytes)}
    Returns: list of selected col_names.
    """
    if not candidates or budget_bytes <= 0:
        return []

    n = len(candidates)
    W = _kb(budget_bytes)
    ws = [_kb(c["weight"]) for c in candidates]
    vs = [c["value"] for c in candidates]

    # 1-D DP rolling array; iterate capacity backwards to enforce 0/1 constraint.
    dp = [0.0] * (W + 1)
    # chosen[i][w] = True when item i is included in the optimal solution at capacity w.
    chosen = [[False] * (W + 1) for _ in range(n)]

    for i in range(n):
        for w in range(W, ws[i] - 1, -1):
            candidate_val = dp[w - ws[i]] + vs[i]
            if candidate_val > dp[w]:
                dp[w] = candidate_val
                chosen[i][w] = True

    # Traceback
    selected = []
    w = W
    for i in range(n - 1, -1, -1):
        if chosen[i][w]:
            selected.append(candidates[i]["col_name"])
            w -= ws[i]
    return selected


def select_filters_greedy(candidates: list, budget_bytes: int) -> list:
    """Greedy 0/1 knapsack: sort by value/weight ratio descending, take while it fits.

    candidates: list of {"col_name": str, "value": float, "weight": int (bytes)}
    Returns: list of selected col_names.
    """
    if not candidates or budget_bytes <= 0:
        return []

    ranked = sorted(
        candidates,
        key=lambda c: c["value"] / max(c["weight"], 1),
        reverse=True,
    )
    selected = []
    remaining = budget_bytes
    for c in ranked:
        if c["weight"] <= remaining:
            selected.append(c["col_name"])
            remaining -= c["weight"]
    return selected


# ── Unit tests ────────────────────────────────────────────────────────────────

def _run_tests() -> bool:
    results = []

    # ── Test 1: Trivial — all 3 filters fit within generous budget ────────────
    cands1 = [
        {"col_name": "a", "value": 0.9, "weight": 1000},
        {"col_name": "b", "value": 0.7, "weight": 1500},
        {"col_name": "c", "value": 0.5, "weight": 2000},
    ]
    budget1 = 10_000
    e1 = set(select_filters_exact(cands1, budget1))
    g1 = set(select_filters_greedy(cands1, budget1))
    ok1 = e1 == {"a", "b", "c"} and g1 == {"a", "b", "c"}
    print(f"Test 1 (trivial — all fit):")
    print(f"  exact  = {sorted(e1)}")
    print(f"  greedy = {sorted(g1)}")
    print(f"  Expected: all 3   →  {'PASS' if ok1 else 'FAIL'}")
    results.append(ok1)

    # ── Test 2: Tight budget — 5 filters, only top-2 by value fit ────────────
    # best1+best2 = 5500 bytes ≤ 5600; any combo including medium/low1/low2 overflows.
    cands2 = [
        {"col_name": "best1",  "value": 0.9, "weight": 3000},
        {"col_name": "best2",  "value": 0.8, "weight": 2500},
        {"col_name": "medium", "value": 0.5, "weight": 4000},
        {"col_name": "low1",   "value": 0.3, "weight": 5000},
        {"col_name": "low2",   "value": 0.1, "weight": 6000},
    ]
    budget2 = 5_600
    e2 = set(select_filters_exact(cands2, budget2))
    g2 = set(select_filters_greedy(cands2, budget2))
    ok2 = e2 == {"best1", "best2"} and g2 == {"best1", "best2"}
    print(f"\nTest 2 (tight budget — 5 items, only 2 fit):")
    print(f"  exact  = {sorted(e2)}")
    print(f"  greedy = {sorted(g2)}")
    print(f"  Expected: {{best1, best2}}   →  {'PASS' if ok2 else 'FAIL'}")
    results.append(ok2)

    # ── Test 3: Greedy-vs-exact gap ───────────────────────────────────────────
    # 'big' has the highest value/weight ratio (0.90/5120 ≈ 1.758e-4), so greedy
    # picks it first (5 KB used, 2 KB left).  Neither med1 (4 KB) nor med2 (3 KB)
    # fits in the remaining 2 KB, so greedy keeps only {big}=0.90.
    # Exact finds med1+med2 (4+3=7 KB = budget) → value 1.20 > 0.90.
    cands3 = [
        {"col_name": "big",  "value": 0.90, "weight": 5120},  # ratio ≈ 1.758e-4 (highest)
        {"col_name": "med1", "value": 0.70, "weight": 4096},  # ratio ≈ 1.709e-4
        {"col_name": "med2", "value": 0.50, "weight": 3072},  # ratio ≈ 1.628e-4
    ]
    budget3 = 7_168  # 7 KB
    e3 = set(select_filters_exact(cands3, budget3))
    g3 = set(select_filters_greedy(cands3, budget3))
    ev3 = sum(c["value"] for c in cands3 if c["col_name"] in e3)
    gv3 = sum(c["value"] for c in cands3 if c["col_name"] in g3)
    ok3 = e3 == {"med1", "med2"} and g3 == {"big"} and ev3 > gv3
    print(f"\nTest 3 (greedy-vs-exact gap):")
    print(f"  greedy selects {sorted(g3)!s:<20}  total value = {gv3:.2f}")
    print(f"  exact  selects {sorted(e3)!s:<20}  total value = {ev3:.2f}")
    print(f"  Exact beats greedy by {ev3 - gv3:.2f}   →  {'PASS' if ok3 else 'FAIL'}")
    results.append(ok3)

    print(f"\n{'All 3 tests PASS ✓' if all(results) else 'SOME TESTS FAILED ✗'}")
    return all(results)


def _append_to_writeup(output: str) -> None:
    from pathlib import Path
    writeup = Path(__file__).resolve().parent / "WRITEUP.md"
    marker  = "## 5. Knapsack tests"
    section = f"\n{marker}\n\n```\n" + output + "```\n"
    if writeup.exists():
        text = writeup.read_text()
        if marker in text:
            text = text[: text.index(marker)]
        writeup.write_text(text + section)
    else:
        writeup.write_text(section)


if __name__ == "__main__":
    import io
    import sys

    # Capture output so we can both print it and write it to WRITEUP.md
    buf = io.StringIO()

    class _Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)
        def flush(self):
            sys.__stdout__.flush()

    sys.stdout = _Tee()
    ok = _run_tests()
    sys.stdout = sys.__stdout__

    _append_to_writeup(buf.getvalue())
    sys.exit(0 if ok else 1)
