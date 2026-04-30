# ML Bloom Filter Budget Policy — Results

## 1. Oracle baseline (probe features included — NOT deployable)

> Trained with `rows_in` + `probe_batches` as features.  These are only
> available after the query runs, so this model cannot be used in the
> Rust planner.  Kept here as an upper-bound reference.

| Metric | Value |
|--------|-------|
| f1 | 1.0 |
| precision | 1.0 |
| recall | 1.0 |
| roc_auc | 1.0 |

Top features: rows_in (0.8884), probe_batches (0.1013)

## 2. Build-side-only model (deployable)

Features available at plan time: `build_cardinality`, `distinct_estimate`,
`filter_size_bytes`, `build_time_ms`, `distinctness_ratio`.

> **Note on `build_time_ms`:** this is measured during filter construction,
> so it is technically post-build. The Rust side can estimate it from
> cardinality (linear in build_cardinality), so it is borderline-acceptable.
> Flagged here; remove it if the Rust estimator is unavailable.

| Metric | Value |
|--------|-------|
| f1 | 0.4661 |
| precision | 0.3147 |
| recall | 0.8977 |
| roc_auc | 0.9042 |

### Confusion Matrix (rows = actual, cols = predicted)
```
              Pred 0   Pred 1
  Actual 0     1001      172
  Actual 1        9       79
```

### Feature Importances (mean decrease in impurity, descending)

| Feature | Importance |
|---------|------------|
| `distinct_estimate` | 0.5714 |
| `build_time_ms` | 0.2331 |
| `filter_size_bytes` | 0.1243 |
| `build_cardinality` | 0.0708 |
| `distinctness_ratio` | 0.0004 |



## 3. Evaluation results

Budget: **100,000 bytes** / Total filter memory: **326,763 bytes**

| query | no_bf | single_hop | always_build | ml_guided |
|-------|------:|-----------:|-------------:|----------:|
| **q10** | 0.00 | -870.70 | -870.70 | -870.70 |
| **q11** | 0.00 | -26.44 | -13.17 | 13.27 |
| **q15** | 0.00 | -264.66 | -771.86 | -507.20 |
| **q16** | 0.00 | -40.23 | -40.23 | 0.00 |
| **q17** | 0.00 | 149.35 | 149.35 | 149.35 |
| **q19** | 0.00 | -174.84 | -174.84 | -174.84 |
| **q2** | 0.00 | -522.56 | -870.24 | -836.88 |
| **q20** | 0.00 | -24.25 | -24.25 | -24.25 |
| **q21** | 0.00 | -181.79 | -738.88 | -557.09 |
| **q3** | 0.00 | -40.86 | -40.86 | -40.86 |
| **q5** | 0.00 | -729.24 | -2944.97 | -1488.43 |
| **q7** | 0.00 | -858.68 | -1805.79 | -1610.93 |
| **q8** | 0.00 | -988.01 | -3121.02 | -1862.27 |
| **q9** | 0.00 | 159.16 | -1017.33 | 159.16 |
| **TOTAL** | 0.00 | -4413.75 | -12284.80 | -7651.67 |

ml_guided beats always_build (-7651.67 vs -12284.80).

## 4. Budget sweep

Total filter memory: **326,763 bytes**.
Sweet-spot budget: **25,000 bytes** → ml_guided total = **202.30 ms**
(always_build baseline = -12284.80 ms at all budgets)

| budget_bytes | no_bf | single_hop | always_build | ml_guided |
|-------------:|------:|-----------:|-------------:|----------:|
| 10,000 | 0.00 | -4413.75 | -12284.80 | -1904.58 |
| 25,000 | 0.00 | -4413.75 | -12284.80 | 202.30 | ◀ sweet spot
| 50,000 | 0.00 | -4413.75 | -12284.80 | -9576.45 |
| 100,000 | 0.00 | -4413.75 | -12284.80 | -7651.67 |
| 200,000 | 0.00 | -4413.75 | -12284.80 | -9872.70 |
| 500,000 | 0.00 | -4413.75 | -12284.80 | -12284.80 |

See `ml/budget_sweep.png` for the trend plot.

## 5. Knapsack tests

```
Test 1 (trivial — all fit):
  exact  = ['a', 'b', 'c']
  greedy = ['a', 'b', 'c']
  Expected: all 3   →  PASS

Test 2 (tight budget — 5 items, only 2 fit):
  exact  = ['best1', 'best2']
  greedy = ['best1', 'best2']
  Expected: {best1, best2}   →  PASS

Test 3 (greedy-vs-exact gap):
  greedy selects ['big']               total value = 0.90
  exact  selects ['med1', 'med2']      total value = 1.20
  Exact beats greedy by 0.30   →  PASS

All 3 tests PASS ✓
```



## 6. Sweet-spot diagnostic

Budget: **25,000 bytes**

### Selected filters

| filter | pred_p | actual_net_ms | was_beneficial |
|--------|-------:|--------------:|:--------------:|
| q9.l_partkey | 0.89 | +159.2 | 0 |
| q17.l_partkey | 0.88 | +149.4 | 0 |
| q8.l_partkey | 0.87 | +131.9 | 0 |
| q2.ps_partkey | 0.32 | -238.1 | 0 |

### False negatives (skipped but actually beneficial, up to 5)

| filter | pred_p | actual_net_ms | was_beneficial |
|--------|-------:|--------------:|:--------------:|
| q11.s_nationkey | 0.02 | +13.3 | 0 |

### Summary

| Metric | Value |
|--------|-------|
| Selected | 4 filters, 3 actually beneficial |
| Precision-at-budget | 75% |
| Skipped | 28 filters, 1 actually beneficial |
| Recall-at-budget | 75% |
| Total net benefit (selected) | +202.30 ms |
| Top-3 contribution | +440.39 ms (218% of total) |

**Sweet-spot win is fragile: top 3 picks contribute 218% of total benefit. Result may not generalize.**


