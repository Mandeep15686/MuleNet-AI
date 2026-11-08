# MuleNet AI — Version History

## v2.9.2 (current — `mulenet/train.py`)

Built from `mulenet_v26.py` with:
- VERSION string set to "2.9.2" to match README and downstream tool references
- Artifact filenames use `_v292` suffix throughout
- Added `test_predictions_by_seed.parquet` export (bridging train → post-process)
- Changed `--device` default from `cuda` to `cpu` for broader compatibility
- Extended `fit_feature_only_gnn()` return to expose `seeded` dict

## v2.6 (`mulenet_v26.py`)

Key additions over v2.5:
- Vectorised 6-h throughput buckets (replaced O(n²) lambda aggregation)
- Grouped `observed=True` groupby fix (patch_v26_groupby.py applied)
- Performance optimisation (patch_v26_performance.py applied)
- Complete investigator export with historical reason codes
- Controlled experiment framework for GraphSAGE / LightGBM

## v2.5 / v2.5.1-FIXED (`mulenet_v25.py`, `mulenet_v25_1_FIXED.py`)

Key additions over v2.4:
- Explanation layer integrated into training pipeline
- Validation-ECDF rank transformation
- Target-FPR threshold calibration
- Reason code framework (7 code types)

## v2.4 (`mulenet_v24.py`)

Key additions over v2.3:
- Multi-seed training (default: 42, 52, 62)
- Seed stability CSV output
- Mean-across-seeds final scores

## v2.3 (`mulenet_v23.py`)

Key additions over v2.2:
- Validation-gated RF blend weight selection
- RF candidate with `balanced_subsample`
- `all-windows-non-degrading` safeguard for RF inclusion

## v2.2 / v2.2.1 (`mulenet_v22.py`, `mulenet_v22_1.py`)

Key additions over v2.1:
- Strict chronological split (replaced random account split)
- 6-window rolling snapshot design
- Leakage audit: imputer/scaler fit on training accounts only

## v2.1 (`mulenet_v21.py`)

Key additions over v2:
- Added RF baseline model
- Initial validation gating concept (incomplete)

## v2 (`mulenet_v2.py`)

First complete implementation:
- GraphSAGE with 2-hop mean aggregation
- Random train/test split (later superseded by chronological)
- Basic AP/AUC metrics

## Standalone Post-Processing Modules

### explain_v1 (`mulenet_explain_v1.py` → `mulenet/explain.py`)

Vectorised rewrite of `mulenet_v27.py`. Fixes:
- `fillna(0)` on datetime columns
- O(n²) throughput loop → 6-h bucket groupby
- O(n²) rapid-forward loop → merge_asof + time filter
- O(n²) dwell-time loop → filtered to flagged accounts

### louvain_v1 (`mulenet_louvain_v1.py` → `mulenet/rings.py`)

Standalone ring detection added after v2.6 research run. Fallback chain:
networkx Louvain → python-louvain → label propagation.

### bootstrap_v291_v292 (`mulenet_bootstrap_v291_v292.py` → `mulenet/bootstrap.py`)

Paired bootstrap for statistical significance of AP improvements.
Default: 10,000 iterations, seed 0.

### dashboard_builder (`mulenet_dashboard_builder.py` → `mulenet/dashboard.py`)

HTML dashboard using `_dash_template.html` (self-contained, D3 via CDN).
Fixed: template path resolves to `dashboard/` directory, not module directory.
