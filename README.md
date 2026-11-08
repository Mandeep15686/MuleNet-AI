# MuleNet AI

**Graph-based money-mule detection on the IBM AML HI-Small dataset.**

A research pipeline that combines a leakage-safe feature-only GNN with
Louvain community detection to flag and explain suspected money-mule accounts
in synthetic transaction data.

> **Dataset required:** [IBM Transactions for Anti-Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
> Place `HI-Small_Trans.csv` in `data/raw/` before running.

---

## Reported Results (reproduced from research report)

| Model | Eval | Test 1 AP | Test 2 AP | Test 3 AP | Macro AP |
|---|---|---|---|---|---|
| Random Forest (baseline) | Chronological | 0.0177 | 0.0197 | 0.0111 | 0.0162 |
| GraphSAGE (experiment) | Chronological | 0.0263 | 0.0282 | 0.0217 | 0.0254 |
| **Feature-only GNN v2.9.2** | **Chronological** | **0.0287** | **0.0328** | **0.0267** | **0.0294** |

**Bootstrap comparison (V2.9.1 vs V2.9.2, 10,000 iterations):**

| Window | Δ AP | 95% CI | p-value |
|---|---|---|---|
| Test 1 | +0.87% | [−0.28%, +0.37%] | 0.467 |
| Test 2 | **+28.8%** | [+0.24%, +1.37%] | **0.0006** |
| Test 3 | **+17.3%** | [+0.15%, +0.73%] | **0.0006** |
| Macro | **+15.1%** | — | 2/3 windows |

> AP scores are low by design: the IBM HI-Small dataset has ~1:1,000 class
> imbalance and uses weak transaction-derived proxy labels — not verified mule identities.

**Investigation layer (from research run):**
- 2,984 / 3,000 flagged accounts with specific behavioural reason codes
- 70 mule-network rings detected (19 LAYERING_RING, 22 FAN_OUT_HUB, 2 RAPID_TRANSFER_CHAIN, 27 MIXED)
- 9,469 rapid-forwarding events captured

---

## Architecture

```
IBM HI-Small dataset (5.07M transactions)
         │
         ▼
┌─────────────────────────────────────────┐
│          Feature Engineering            │
│  6-hr throughput · rapid forwarding     │
│  dwell time · fan-in/out · 2-hop flow   │
└─────────────────────┬───────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                        ▼
┌────────────────┐      ┌──────────────────┐
│  Tabular (RF/  │      │  Feature-only    │
│  LightGBM)     │      │  GNN (2-layer)   │
│  [optional]    │      │  [primary]       │
└────────────────┘      └──────────────────┘
         │                        │
         └────────────┬────────────┘
                      ▼
           ┌──────────────────┐
           │  Risk Scores     │
           │  (chronological  │
           │   evaluation)    │
           └────────┬─────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
┌────────────────┐   ┌──────────────────────┐
│  Explanation   │   │  Louvain Ring        │
│  Layer         │   │  Detection           │
│  (7 reason     │   │  (community-based    │
│   codes)       │   │   network grouping)  │
└────────────────┘   └──────────────────────┘
         │                     │
         └──────────┬──────────┘
                    ▼
         ┌──────────────────────┐
         │  Investigation       │
         │  Dashboard (HTML)    │
         └──────────────────────┘
```

---

## Project Structure

```
mulenet-ai/
├── README.md
├── LICENSE                              MIT licence
├── .gitignore
├── requirements.txt                     Python dependencies (PyTorch separate)
├── run_pipeline.py                      Post-processing orchestrator
│
├── mulenet/                             Core package
│   ├── __init__.py
│   ├── train.py                         Feature engineering + GNN training
│   ├── explain.py                       Reason-code post-processor
│   ├── rings.py                         Louvain ring detection
│   ├── bootstrap.py                     Paired bootstrap comparison
│   └── dashboard.py                     HTML dashboard builder
│
├── dashboard/
│   └── _dash_template.html              Self-contained dashboard template
│
├── data/
│   └── raw/                             ← place HI-Small_Trans.csv here
│
├── docs/
│   ├── ARCHITECTURE.md                  Detailed design notes
│   ├── AUDIT_REPORT.md                  Codebase audit and version history
│   └── VERSION_HISTORY.md              Changelog across all versions
│
└── outputs/                             All run artefacts (gitignored)
    ├── neural_v292/                     Training outputs + predictions parquet
    ├── explain_v1/                      Dossiers and reason codes
    ├── louvain_v1/                      Ring detection results
    ├── bootstrap/                       Statistical comparison results
    └── dashboard/                       HTML dashboard
```

---

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/<your-username>/mulenet-ai.git
cd mulenet-ai

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install PyTorch first (CPU-only example):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies:
pip install -r requirements.txt
```

### 2. Download the dataset

Get the IBM AML HI-Small dataset from Kaggle:
```
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
```
Place `HI-Small_Trans.csv` in `data/raw/`.

### 3. Train the model

```bash
python mulenet/train.py \
  --data   data/raw/HI-Small_Trans.csv \
  --output outputs/neural_v292
```

This takes approximately 20–60 minutes on CPU (full 5M-row dataset).
Use `--max-rows 500000` for a faster smoke test.

Outputs written to `outputs/neural_v292/`:
- `test_predictions_by_seed.parquet` — per-account risk scores (required by pipeline)
- `temporal_robustness_v292.csv` — AP / FPR per test window
- `account_explanations_v292.csv` — investigator evidence (top-N per window)
- `investigator_review_queue_v292.csv` — prioritised review list
- `model_comparison_v292.csv` — model selection summary
- `metrics_v292.json` — full metrics manifest

### 4. Run the full post-processing pipeline

```bash
python run_pipeline.py \
  --data        data/raw/HI-Small_Trans.csv \
  --predictions outputs/neural_v292/test_predictions_by_seed.parquet \
  --output-root outputs
```

Opens the HTML dashboard when `--open-browser` is added.

### 5. Run steps individually

```bash
# Explanation layer
python mulenet/explain.py \
  --data        data/raw/HI-Small_Trans.csv \
  --predictions outputs/neural_v292/test_predictions_by_seed.parquet \
  --output      outputs/explain_v1 \
  --top-n       1000

# Ring detection
python mulenet/rings.py \
  --data        data/raw/HI-Small_Trans.csv \
  --predictions outputs/neural_v292/test_predictions_by_seed.parquet \
  --dossiers    outputs/explain_v1/improved_dossiers.csv \
  --output      outputs/louvain_v1 \
  --top-n       1000

# Dashboard
python mulenet/dashboard.py \
  --explain outputs/explain_v1 \
  --louvain outputs/louvain_v1 \
  --output  outputs/dashboard

# Bootstrap comparison (requires two separate training runs)
python mulenet/bootstrap.py \
  --v291   outputs/run_a/test_predictions_by_seed.parquet \
  --v292   outputs/neural_v292/test_predictions_by_seed.parquet \
  --output outputs/bootstrap \
  --n-iter 10000
```

### 6. Optional experiments (GraphSAGE / LightGBM)

```bash
# GraphSAGE controlled comparison (included in train.py, no extra install)
python mulenet/train.py \
  --data   data/raw/HI-Small_Trans.csv \
  --output outputs/neural_v292 \
  --experiment-graphsage

# LightGBM comparison (requires: pip install lightgbm>=4.0)
python mulenet/train.py \
  --data   data/raw/HI-Small_Trans.csv \
  --output outputs/neural_v292 \
  --experiment-lightgbm
```

---

## Key Design Decisions

**Why chronological split?** Random account splits leak future transaction
patterns into training features. All evaluations use a strict chronological
6-window split (40/50/60/70/80/90th percentile cut-points).

**Why feature-only GNN beats GraphSAGE here?** On IBM HI-Small, the
account-level temporal features (6-hour throughput, dwell time, rapid
forwarding) carry most of the signal. Graph message-passing adds marginal
value because mule accounts connect to many legitimate accounts, diluting
neighbourhood representations.

**Why AP instead of AUC?** At ~1:1,000 class imbalance, AUC is dominated by
the majority class and masks precision-recall trade-offs. Average Precision
(area under the PR curve) is the primary metric.

**Why validation-gated RF blending?** The RF candidate is only included in
the final scorer if it improves AP on *both* validation windows simultaneously.
This prevents leaking test-time knowledge into the blending decision.

---

## Known Limitations

- IBM AML HI-Small is **fully synthetic** — labels are transaction-derived
  proxies, not independently verified mule identities
- **Single-bank, single-dataset evaluation** — federated multi-bank not implemented
- **Class imbalance (~1:1,000)** limits absolute AP scores regardless of model
- Reason codes are **historical behavioural descriptions**, not proof of
  illicit activity or fund tracing
- No device identifiers are available; device-sharing signals are absent

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@misc{mulenet2026,
  title  = {MuleNet AI: Graph-based Money-Mule Detection},
  author = {<Your Name>},
  year   = {2026},
  url    = {https://github.com/<your-username>/mulenet-ai}
}
```

---

## Licence

MIT — see `LICENSE`.
