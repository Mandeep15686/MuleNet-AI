# MuleNet AI — Testing and Validation Report

**Test environment:** Python 3.12, PyTorch 2.15, pandas 3.0, scikit-learn 1.3+  
**Test date:** November 2026  
**Dataset used for smoke tests:** Synthetic (IBM AML dataset not available in test environment)

---

## Test Results Summary

| Test | Status | Notes |
|---|---|---|
| Syntax check — all 6 Python files | ✓ PASS | No syntax errors |
| Import test — all 6 modules | ✓ PASS | All imports resolve cleanly |
| `resolve_columns` with IBM AML headers | ✓ PASS | Maps all 11 required columns |
| `build_events` directional split | ✓ PASS | 4,000 events from 2,000 transactions |
| `build_account_features` | ✓ PASS | 26 features from 160 accounts |
| `build_rolling_snapshots` | ✓ PASS | 6 windows, correct account counts |
| `FeatureOnlyGNN` forward pass | ✓ PASS | Input (200, 26) → logits (200,) |
| `fit_feature_only_gnn` (seeds=1, epochs=3) | ✓ PASS | Parquet written, correct schema |
| `fit_random_forest_candidate` | ✓ PASS | Validation AP 0.339 (synthetic) |
| `validation_gate_random_forest` | ✓ PASS | Correctly excludes RF when gain < threshold |
| `make_reason_thresholds` | ✓ PASS | 12 threshold entries |
| `account_reason_codes` | ✓ PASS | Returns 2 codes on test account |
| `json_safe` (nan, numpy types) | ✓ PASS | NaN → None, int64 → int |
| `:: separator` in explain.py | ✓ PASS | 100% account ID overlap |
| `:: separator` in rings.py | ✓ PASS | 0 remaps needed |
| `remap_account_ids` | ✓ PASS | Returns {} (no mismatch) |
| `build_edge_list` | ✓ PASS | 118 directed edges |
| `extract_subgraph` | ✓ PASS | Subgraph extracted correctly |
| `louvain_communities` | ✓ PASS | 17 communities from 54 nodes |
| `load_preds` (bootstrap) | ✓ PASS | Loads and renames score columns |
| `bootstrap_ap_diff` | ✓ PASS | 100 iterations, correct diff shape |
| Dashboard `_TEMPLATE` loading | ✓ PASS | 39,632 chars, all placeholders |
| **Full end-to-end smoke test** | **✓ PASS** | All 17 output files written |

---

## End-to-End Smoke Test Details

**Configuration:** 4,000 synthetic transactions, 40 timestamps, 120 accounts, 2 seeds, 5 epochs

**Files written by `mulenet/train.py`:**

```
account_explanations_v292.csv       (122,667 bytes)
explanation_thresholds_v292.json    (    401 bytes)
feature_imputer_v292.joblib         (  1,407 bytes)
feature_names_v292.joblib           (    563 bytes)
feature_only_gnn_predictions_v292.npz (13,072 bytes)
feature_only_gnn_seed_42_v292.pt    ( 10,077 bytes)
feature_only_gnn_seed_52_v292.pt    ( 10,077 bytes)
feature_scaler_v292.joblib          (  1,207 bytes)
investigator_review_queue_v292.csv  ( 48,142 bytes)
metrics_v292.json                   (  8,235 bytes)
model_comparison_v292.csv           (  2,453 bytes)
model_selection_v292.json           (  2,701 bytes)
random_forest_candidate_seed_42_v292.joblib (114,244 bytes)
random_forest_candidate_seed_52_v292.joblib (117,444 bytes)
seed_stability_v292.csv             (  3,411 bytes)
temporal_robustness_v292.csv        (  2,002 bytes)
test_predictions_by_seed.parquet    ( 28,574 bytes)
```

**Synthetic AP scores** (not comparable to IBM AML results):

| Window | AP |
|---|---|
| test_1 | 0.1179 |
| test_2 | 0.2497 |
| test_3 | 0.1444 |

**Parquet schema verified:** `seed`, `snapshot`, `account_id`, `risk_score`, `label`  
**Account ID format verified:** `BANK::ACCOUNT` (e.g., `101::26`)

---

## What Was NOT Verified

The following cannot be verified without the IBM AML HI-Small dataset:

1. **Reported AP scores** (0.0287 / 0.0328 / 0.0267) — require the full 5M-row IBM dataset
2. **Full explain pipeline** (`mulenet/explain.py` end-to-end with 1,000 flagged accounts)
3. **Full rings pipeline** (`mulenet/rings.py` with real transaction graph)
4. **Dashboard rendering** — HTML structure reviewed, browser rendering untested
5. **Bootstrap values** — require two separate training runs on IBM AML
6. **Training runtime** — expected 20–60 min on CPU with full dataset
7. **GPU compatibility** — CUDA device path not tested (no GPU in test environment)

---

## IBM AML Dataset Verification Checklist

When the dataset is available, verify these before production use:

- [ ] `load_transactions` loads all 5,073,939 rows without error
- [ ] `resolve_columns` maps all IBM AML headers correctly
- [ ] `build_rolling_snapshots` produces valid windows (verify cutoff dates ~Sep 2022)
- [ ] Class distribution: approximately 1:1,000 positive ratio
- [ ] `temporal_robustness_v292.csv` shows AP ≈ 0.027–0.033 on test windows
- [ ] No NaN values in final `test_predictions_by_seed.parquet`
- [ ] `account_explanations_v292.csv` has ≥90% rows with specific reason codes
- [ ] Ring detection finds ≥50 rings with ≥2 flagged members
- [ ] Dashboard HTML file is <5 MB and opens in browser

---

## Known Limitations Confirmed by Testing

1. **Class imbalance** is severe even on synthetic data — AP scores are low relative to accuracy.
2. **RF blending** correctly excludes the RF when gain < 0.0005 on synthetic data.
3. **Louvain fallback chain** works correctly (networkx Louvain used by default).
4. **Separator consistency** was a real bug — fixed and verified by overlap test (100%).
