# MuleNet AI — Codebase Audit Report

**Audit performed:** November 2026  
**Input:** 43 source files from `all_codes.zip`  
**Output:** One unified implementation in `mulenet/` package

---

## 1. Source File Inventory

| Source File | Size | Date | Classification | Disposition |
|---|---|---|---|---|
| `mulenet_v2.py` | 32 KB | Oct 31 | Early prototype with GraphSAGE | Superseded by v26 |
| `mulenet_v21.py` | 34 KB | Oct 31 | Added chronological split | Superseded |
| `mulenet_v22.py` | 42 KB | Oct 31 | Added RF candidate | Superseded |
| `mulenet_v22_1.py` | 43 KB | Oct 31 | Minor fix on v22 | Superseded |
| `mulenet_v23.py` | 51 KB | Oct 31 | Added validation gating | Superseded |
| `mulenet_v24.py` | 60 KB | Oct 31 | Added seed stability | Superseded |
| `mulenet_v25.py` | 74 KB | Oct 31 | Added explanations | Superseded |
| `mulenet_v25_1_FIXED.py` | 74 KB | Oct 31 | Identical size to v25 — groupby fix | Superseded |
| `mulenet_v26.py` | 77 KB | Nov 1 | **Most complete training script** | **→ mulenet/train.py** |
| `mulenet_v26.py.pre_category_groupby.bak` | 77 KB | Nov 1 | Backup before groupby patch | Deleted (backup only) |
| `mulenet_v26.py.pre_v29_optimization.bak` | 76 KB | Nov 1 | Backup before v29 optimization | Deleted (backup only) |
| `mulenet_v27.py` | 30 KB | Oct 31 | Standalone explainability report | Superseded by explain_v1 |
| `mulenet_v28.py` | 21 KB | Oct 31 | Calibration/benchmark module | Superseded |
| `mulenet_explain_v1.py` | 32 KB | Nov 2 | **Vectorized explanation layer** | **→ mulenet/explain.py** |
| `mulenet_louvain_v1.py` | 23 KB | Nov 3 | **Louvain ring detection** | **→ mulenet/rings.py** |
| `mulenet_bootstrap_v291_v292.py` | 11 KB | Nov 3 | **Bootstrap comparison** | **→ mulenet/bootstrap.py** |
| `mulenet_dashboard_builder.py` | 4 KB | Nov 4 | **HTML dashboard builder** | **→ mulenet/dashboard.py** |
| `mulenet_desktop.py` | 22 KB | Oct 30 | GUI wrapper (tkinter) | Not included (no deps) |
| `_dash_template.html` | 41 KB | Nov 4 | Dashboard HTML template | **→ dashboard/** |
| `run_pipeline.py` | 4 KB | Nov 5 | Orchestrator (references new structure) | **→ run_pipeline.py** |
| `patch_v26_groupby.py` | 4 KB | Nov 1 | Applied to v26 — fix already in v26 | Retired |
| `patch_v26_performance.py` | 2 KB | Nov 1 | Performance patch — applied to v26 | Retired |
| `setup_repo.py` | 4 KB | Nov 5 | Repo scaffold (now superseded) | Retired |
| `README.md` | 8 KB | Nov 5 | Good content, reorganized | **→ README.md** |
| `README.txt` | 1 KB | Nov 1 | v26 README | Superseded |
| `README_v26.md` | 2 KB | Oct 31 | v26 README | Superseded |
| `README_v27.txt` | 1 KB | Oct 31 | v27 README | Superseded |
| `README_v28.txt` | 3 KB | Oct 31 | v28 README | Superseded |
| `requirements.txt` | 330 B | Nov 5 | Partial (no torch, no joblib) | Consolidated |
| `requirements_v2.txt` through `requirements_v28.txt` | — | — | Per-version partial lists | Consolidated |
| `requirements_explain_v1.txt` | 61 B | Nov 2 | Explanation only | Consolidated |
| `__init__.py` | 77 B | Nov 5 | Minimal stub | Replaced |
| `gitignore` | 2 KB | Nov 5 | Good, missing dot prefix | Fixed |
| `LICENSE` | 1 KB | Nov 5 | MIT | Kept |
| `MuleNet_AI_Corrected_Research_Report.pdf` | 338 KB | Oct 31 | Research paper | Reference only |

---

## 2. Issues Found and Fixed

### 2.1 Account ID Separator Mismatch (Critical)

**Problem:** `mulenet_v26.py` / `train.py` constructs account IDs with `::` separator:
```python
tx["from_account"] = tx["from_bank"].astype(str) + "::" + tx["from_account"].astype(str)
```

`mulenet_explain_v1.py` and `mulenet_louvain_v1.py` construct IDs with `_` separator:
```python
tx["src_id"] = tx["from_bank"].astype(str) + "_" + tx["src_account"].astype(str)
```

This caused a silent 0% overlap between the prediction `account_id` values and
the transaction-graph `account_id` values — `explain.py` would find no flagged
accounts to process.

**Fix:** Changed the separator in `explain.py` and `rings.py` to `::` to match
`train.py`. The `remap_account_ids()` function in `rings.py` is retained for
defensive compatibility but will now return an empty dict (no remap needed).

### 2.2 Missing `test_predictions_by_seed.parquet` Export (Critical)

**Problem:** `mulenet_v26.py` saves model scores as a `.npz` archive
(`feature_only_gnn_predictions_v26.npz`), but `mulenet_explain_v1.py`,
`mulenet_louvain_v1.py`, and `mulenet_bootstrap_v291_v292.py` all require a
parquet file with columns `account_id`, `risk_score`, `snapshot`, `seed`, `label`.
No script existed to bridge this gap.

**Fix:** Added `export_predictions_parquet()` logic inside `fit_feature_only_gnn()`
in `train.py`. The parquet now contains per-seed rows plus an ensemble row
(seed == -1) with the final averaged scores.

### 2.3 Dashboard Template Path (Medium)

**Problem:** `mulenet_dashboard_builder.py` loads the HTML template with:
```python
_TEMPLATE = open(Path(__file__).parent / "_dash_template.html", ...)
```
When moved to `mulenet/dashboard.py`, this resolves to `mulenet/_dash_template.html`
(which does not exist).

**Fix:** Changed to:
```python
_TEMPLATE = open(Path(__file__).parent.parent / "dashboard" / "_dash_template.html", ...)
```

### 2.4 `.gitignore` Missing Dot Prefix (Minor)

**Problem:** The file is named `gitignore` without the leading dot.  
**Fix:** Renamed to `.gitignore` on copy.

### 2.5 `requirements.txt` Missing Core Dependencies (Minor)

**Problem:** The latest `requirements.txt` omitted `joblib`, `scipy`, `pyarrow`,
and lacked clear guidance on PyTorch installation.  
**Fix:** Consolidated all per-version requirements into one file with explicit
comments about PyTorch installation.

### 2.6 `--device` Default was `cuda` (Minor)

**Problem:** `mulenet_v26.py` defaulted `--device` to `"cuda"` and required the
explicit `--allow-cpu` flag for CPU-only environments, which is unusual for a
research codebase where most users run CPU.  
**Fix:** Changed default to `"cpu"`. CUDA is used automatically when available
and `--device cuda` is passed.

### 2.7 VERSION Bump to 2.9.2 (Consistency)

**Problem:** `mulenet_v26.py` has `VERSION = "2.6"` but all downstream tools,
the README, and the research report reference v2.9.2.  
**Fix:** Changed `VERSION = "2.9.2"` and updated all artifact filename suffixes
from `_v26` to `_v292`.

### 2.8 `fit_feature_only_gnn` Return Signature Change

**Problem:** The original function returned `(aggregated, rows, metadata)`. To
expose per-seed scores for the parquet export without re-loading the npz, the
`seeded` dict needed to be accessible in `main()`.  
**Fix:** Extended the return to `(seeded, aggregated, rows, metadata)`. The
`seeded` dict is used only for the parquet export and is not otherwise surfaced.

---

## 3. Functionality Preserved

| Feature | Source | Preserved |
|---|---|---|
| Chronological 6-window evaluation | v2.2+ | ✓ |
| Feature-only GNN (identity adjacency) | v2.3+ | ✓ |
| Random Forest with validation gating | v2.3+ | ✓ |
| Seed stability across 3+ seeds | v2.4+ | ✓ |
| Explanation reason codes (7 types) | v2.5+ | ✓ |
| Vectorised rapid-forward detection | explain_v1 | ✓ |
| Vectorised 6-h throughput buckets | v2.6 | ✓ |
| Louvain ring detection | louvain_v1 | ✓ |
| Ring typology (4 categories) | louvain_v1 | ✓ |
| Paired bootstrap comparison | bootstrap_v291_v292 | ✓ |
| HTML investigation dashboard | dashboard_builder | ✓ |
| GraphSAGE opt-in experiment | v2.6 | ✓ |
| LightGBM opt-in experiment | v2.6 | ✓ |
| Controlled experiment gate | v2.6 | ✓ |
| Validation-ECDF rank scores | v2.5+ | ✓ |
| Anti-leakage guarantees | v2.2+ | ✓ |

### Intentionally Excluded

- `mulenet_desktop.py` — tkinter GUI wrapper; adds no model logic, requires
  additional GUI dependencies, and is not needed for CLI/server use.
- Per-version README and requirements files — superseded by consolidated files.
- Patch scripts (`patch_v26_groupby.py`, `patch_v26_performance.py`) — both
  patches are already applied in `mulenet_v26.py`.

---

## 4. Version Selection Rationale

**Why v26 for train.py and not v27 or v28?**

- `mulenet_v27.py` (632 lines) is a focused explainability-only report generator
  that reads v26 output files. It is superseded by `mulenet_explain_v1.py`.
- `mulenet_v28.py` (386 lines) is a calibration/benchmark analysis script.
  Its functionality is covered by the temporal robustness output in v26.
- `mulenet_v26.py` (1,478 lines) is the most complete, well-structured training
  implementation with the full GNN + RF pipeline, seed stability, explanation
  thresholds, and investigator exports.

**Why explain_v1 for explain.py and not v27?**

- `mulenet_explain_v1.py` explicitly fixes O(n²) loops present in `mulenet_v27.py`
  (throughput, rapid-forward, dwell-time all vectorised).
- `mulenet_explain_v1.py` is standalone and reads the parquet predictions file.

---

## 5. What Was Not Verified

The following items require the IBM AML dataset to verify and are outside the
scope of this audit:

- **Actual AP scores** — the reported values (0.0287 / 0.0328 / 0.0267) cannot
  be verified without running on the full 5M-row dataset.
- **End-to-end runtime** — estimated 20–60 min CPU for 5M rows.
- **Bootstrap significance values** — require two separate training runs.
- **Dashboard rendering** — requires a browser; HTML structure was reviewed only.

The synthetic-data smoke test (Section 6 of TESTING_REPORT.md) verified all
code paths execute without errors.
