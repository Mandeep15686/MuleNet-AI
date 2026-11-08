# MuleNet AI — Architecture & Implementation Notes

## Overview

MuleNet AI is a money-mule detection pipeline built on the IBM AML HI-Small
synthetic transaction dataset. The system is designed around two non-negotiable
constraints:

1. **No data leakage** — features, thresholds, and model weights are determined
   exclusively from transactions that occurred *before* the prediction cutoff
   for each evaluation window.
2. **Explainability** — every flagged account receives human-readable reason
   codes derived from historical behavioural features, not from opaque gradient
   attributions.

---

## Temporal Evaluation Design

```
Timeline ──────────────────────────────────────────────────────────────────▶

Transactions:  [──────────────────────────────────────────────────────────]
                      │        │        │        │        │        │
Quantile cut:        0.40     0.50     0.60     0.70     0.80     0.90
                      │        │        │        │        │        │
Windows:        TRAIN  │  VAL1  │  VAL2  │ TEST1  │ TEST2  │ TEST3  │
                       │        │        │        │        │        │

Each window:
  history  = all transactions BEFORE the left-side cutoff
  target   = transactions BETWEEN left and right cutoff
  features = computed from history only
  labels   = accounts appearing in a laundering transaction inside target
```

**Why six windows (not three)?** The two validation windows are used for:
- Early stopping (validation AP governs when training stops)
- RF blending weight selection (requires both windows to improve simultaneously)
- Threshold calibration at the target FPR (from validation negatives only)

The three test windows are held out entirely until after all model and
threshold decisions are locked.

---

## Feature Engineering

All features are account-level aggregations over the history window.

| Feature Group | Features | Signal |
|---|---|---|
| **Volume** | `in_count`, `out_count` | Activity level |
| **Amount distribution** | `{in,out}_log_amount_{mean,std,max}` | Unusual amounts |
| **Counterparty diversity** | `fan_in_degree`, `fan_out_degree` | Hub patterns |
| **Currency / format mix** | `{in,out}_{currency,format}_count` | Multi-currency layering |
| **Throughput (6h buckets)** | `ft_active_6h_windows`, `ft_log_sum_mean_currency`, `ft_log_max` | Burst throughput |
| **Rapid forwarding** | `rapid_out_1h_count`, `rapid_forward_ratio_mean`, `median_in_to_out_minutes`, `rapid_out_counterparties` | Rapid pass-through |
| **Activity balance** | `out_in_count_ratio` | One-directional flow |

**Rapid forwarding definition:** An outgoing event in the same currency as a
strictly-earlier incoming event, within a 1-hour window. This is a
*behavioural association*, not a funds-tracing claim.

**6-hour throughput:** `min(incoming, outgoing)` within each 6-hour bucket per
(account, currency). This captures "mule throughput" — funds received and
immediately re-sent — without counting passive storage or single-direction flow.

---

## Models

### Primary: Feature-Only GNN (`FeatureOnlyGNN`)

A 2-layer MLP with identity adjacency (no cross-account message passing).

```
input: x ∈ ℝ^d  (account features, scaled)

layer 1: Linear(2d → hidden) → ReLU → Dropout(0.30)
         [x, x] concatenation simulates "self-message" only
layer 2: Linear(2*hidden → hidden) → ReLU → Dropout(0.30)
output:  Linear(hidden → 1) → BCEWithLogitsLoss
```

This is intentionally named "GNN" because it was originally developed as a
comparison baseline against GraphSAGE. The finding — that it *beats* GraphSAGE
on this dataset — is itself the research result.

**Training:** 100 epochs max, Adam, patience=12, pos_weight capped at 100.

### Secondary Candidate: Random Forest

300 trees, `balanced_subsample` class weighting, retained only if validation
AP improves on **both** validation windows by at least 0.0005.

### Opt-in Experiments (not in default scorer)

- **GraphSAGE** (`--experiment-graphsage`): 2-hop mean aggregation over
  historical edges. Same training loop as the primary model.
- **LightGBM** (`--experiment-lightgbm`): requires `pip install lightgbm>=4.0`.

---

## Explanation Layer (`mulenet/explain.py`)

Thresholds are fitted from training-history accounts only (90th percentile by
default). Reason codes are assigned if a feature exceeds its threshold.

| Code | Trigger |
|---|---|
| `HISTORICAL_RAPID_FORWARDING_PATTERN` | rapid_out_1h_count or rapid_forward_ratio_mean above threshold |
| `HIGH_HISTORICAL_THROUGHPUT` | ft_active_6h_windows or ft_log_sum_mean_currency above threshold |
| `HIGH_HISTORICAL_FAN_OUT` | fan_out_degree above threshold |
| `HIGH_HISTORICAL_FAN_IN` | fan_in_degree above threshold |
| `HISTORICAL_OUT_IN_ACTIVITY_IMBALANCE` | out_in_count_ratio above threshold |
| `HIGH_HISTORICAL_TRANSACTION_ACTIVITY` | in_count or out_count above threshold |
| `SHORT_HISTORICAL_IN_TO_OUT_GAP` | median_in_to_out_minutes below 5th-percentile threshold |
| `MODEL_RANKING_CONTEXT` | Fallback: no single feature crossed a threshold |
| `INSUFFICIENT_HISTORY` | Account has fewer than a minimum number of events |

---

## Ring Detection (`mulenet/rings.py`)

```
1. Build weighted directed edge list (all accounts × all transactions)
2. Extract subgraph: edges touching ≥1 flagged account + 1-hop neighbours
3. Project to undirected: merge A→B and B→A into single edge, sum weights
4. Run Louvain community detection (networkx.community.louvain_communities)
5. Keep communities with ≥2 flagged members
6. Score: ring_score = mean GNN risk score of flagged members
7. Assign typology:
     LAYERING_RING          high flag_density (>0.5), low n_flagged
     FAN_OUT_HUB            high fan_out in flagged members
     RAPID_TRANSFER_CHAIN   high rapid_event_count in dossiers
     MIXED                  all others
```

**Louvain fallback chain:** networkx Louvain → python-louvain → label propagation.
The pure-Python label-propagation fallback requires no extra dependencies.

---

## Account ID Format

All modules use the same `BANK::ACCOUNT` format for account IDs:

```
"2::" + "12345678"  →  "2::12345678"
```

This scopes account numbers to their bank, preventing collisions between
accounts with identical numbers at different institutions.

---

## Output Parquet Schema (`test_predictions_by_seed.parquet`)

This file bridges the training module and all post-processing modules.

| Column | Type | Description |
|---|---|---|
| `seed` | int | Training seed (-1 for ensemble average) |
| `snapshot` | str | Window name: test_1 / test_2 / test_3 |
| `account_id` | str | BANK::ACCOUNT format |
| `risk_score` | float | Raw GNN sigmoid score (before ECDF ranking) |
| `label` | int | Proxy label (1 = involved in a laundering txn in target window) |

Post-processing modules (`explain.py`, `rings.py`) use the ensemble row
(seed == -1) for the final ranked list. The `bootstrap.py` module uses per-seed
rows to run the paired bootstrap.

---

## Anti-Leakage Guarantees

| Potential Leakage Vector | Mitigation |
|---|---|
| Future transactions in features | History window is strictly `timestamp < cutoff` |
| Future labels leaking into thresholds | Reason thresholds fitted on training accounts only |
| Test performance affecting threshold | Threshold calibrated on validation negatives only |
| RF blend selected on test performance | Blend weight determined from validation windows only |
| Random splits mixing time periods | Only chronological splits used (from v2.2 onwards) |
| Preprocessing fitted on all data | Imputer and scaler fitted on training snapshot accounts only |

---

## Reproducibility

The pipeline is seeded at multiple levels:

- `random.seed(seed)` — Python random
- `np.random.seed(seed)` — NumPy
- `torch.manual_seed(seed)` — PyTorch CPU
- `torch.cuda.manual_seed_all(seed)` — PyTorch CUDA (if available)
- `RandomForestClassifier(random_state=seed)` — sklearn RF
- `louvain_communities(G, seed=seed)` — networkx Louvain

Default seeds: `[42, 52, 62]`. Final scores are the mean across all seeds.
Per-seed scores are archived in `feature_only_gnn_predictions_v292.npz`.
