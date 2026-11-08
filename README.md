# MuleNet AI

### Graph-Based Money-Mule Detection on the IBM AML HI-Small Dataset

MuleNet AI is a research pipeline for detecting suspicious money-mule accounts using transaction-derived behavioural features, Graph Neural Networks (GNNs), and Louvain community detection.

It combines a leakage-safe, feature-only GNN with an investigation layer to identify, rank, and explain potentially suspicious accounts in synthetic financial transaction data.

**Dataset:** [IBM Transactions for Anti-Money Laundering (AML) – Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)

Place `HI-Small_Trans.csv` inside `data/raw/` before running the project.

---

## 1. Research Results

The following results are reproduced from the project's research report.

### Model Performance

| Model                       | Evaluation        |  Test 1 AP |  Test 2 AP |  Test 3 AP |   Macro AP |
| --------------------------- | ----------------- | ---------: | ---------: | ---------: | ---------: |
| Random Forest (Baseline)    | Chronological     |     0.0177 |     0.0197 |     0.0111 |     0.0162 |
| GraphSAGE (Experiment)      | Chronological     |     0.0263 |     0.0282 |     0.0217 |     0.0254 |
| **Feature-only GNN v2.9.2** | **Chronological** | **0.0287** | **0.0328** | **0.0267** | **0.0294** |

### Bootstrap Comparison: v2.9.1 vs v2.9.2

10,000 bootstrap iterations.

| Window |       Δ AP | 95% CI           |     p-value |
| ------ | ---------: | ---------------- | ----------: |
| Test 1 |     +0.87% | [-0.28%, +0.37%] |       0.467 |
| Test 2 | **+28.8%** | [+0.24%, +1.37%] |  **0.0006** |
| Test 3 | **+17.3%** | [+0.15%, +0.73%] |  **0.0006** |
| Macro  | **+15.1%** | —                | 2/3 windows |

### Investigation Results

Reported investigation-layer results:

- 2,984 / 3,000 flagged accounts with specific behavioural reason codes.
- 70 mule-network rings detected:
  - 19 LAYERING_RING
  - 22 FAN_OUT_HUB
  - 2 RAPID_TRANSFER_CHAIN
  - 27 MIXED
- 9,469 rapid-forwarding events captured.

**Important:** Average Precision (AP) scores are low by design because the IBM HI-Small dataset has approximately 1:1,000 class imbalance and uses transaction-derived proxy labels rather than independently verified mule identities.

---

## 2. Architecture

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

### Main Components

| Component           | Responsibility                                 |
| ------------------- | ---------------------------------------------- |
| Feature Engineering | Extract transaction-based behavioural features |
| Feature-only GNN    | Generate account-level risk scores             |
| Random Forest       | Optional tabular baseline                      |
| LightGBM            | Optional model comparison                      |
| Explanation Layer   | Generate behavioural reason codes              |
| Louvain Detection   | Identify suspicious transaction communities    |
| Bootstrap           | Statistically compare model versions           |
| Dashboard           | Present investigation results through HTML     |

---

## 3. Project Structure

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

**Note:** The `outputs/` directory contains generated artifacts and is gitignored.

---

## 4. Installation and Setup

### Prerequisites

- Windows 10/11
- Python 3.10 or later (use a version supported by your installed PyTorch build and dependencies)
- Git
- pip
- Sufficient RAM and storage for the full dataset
- NVIDIA GPU recommended for accelerated training

### Step 1: Clone the Repository

```powershell
git clone https://github.com/<your-username>/mulenet-ai.git

cd mulenet-ai
```

### Step 2: Create Virtual Environment

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then activate:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install PyTorch

For CPU-only installation:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For NVIDIA GPU installation, select the compatible CUDA-enabled PyTorch installation command from:

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### Step 4: Install Project Dependencies

```powershell
python -m pip install --upgrade pip

pip install -r requirements.txt
```

For the optional LightGBM experiment:

```powershell
pip install "lightgbm>=4.0"
```

---

## 5. Dataset Installation

MuleNet AI uses the IBM Transactions for Anti-Money Laundering dataset.

Download it from:

[https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)

Locate the file:

```text
HI-Small_Trans.csv
```

Place it in:

```text
data/raw/HI-Small_Trans.csv
```

Verify that the dataset exists:

```powershell
Test-Path data/raw/HI-Small_Trans.csv
```

Expected output:

```text
True
```

---

## 6. Model Training

### Train Feature-only GNN v2.9.2

Run:

```powershell
python mulenet/train.py --data data/raw/HI-Small_Trans.csv --output outputs/neural_v292
```

The training process performs feature engineering, model training, and chronological evaluation.

### Faster Smoke Test

For testing the pipeline with a smaller dataset sample:

```powershell
python mulenet/train.py --data data/raw/HI-Small_Trans.csv --output outputs/smoke_test --max-rows 500000
```

The README's original estimate for full-dataset CPU training is approximately 20–60 minutes. Actual runtime depends on hardware and implementation.

### Expected Training Outputs

```text
outputs/neural_v292/
│
├── test_predictions_by_seed.parquet
├── temporal_robustness_v292.csv
├── account_explanations_v292.csv
├── investigator_review_queue_v292.csv
├── model_comparison_v292.csv
└── metrics_v292.json
```

The most important file for subsequent pipeline execution is:

`test_predictions_by_seed.parquet`

---

## 7. Run the Complete Pipeline

After successful model training, execute the post-processing pipeline.

```powershell
python run_pipeline.py --data data/raw/HI-Small_Trans.csv --predictions outputs/neural_v292/test_predictions_by_seed.parquet --output-root outputs
```

### Automatically Open Dashboard

```powershell
python run_pipeline.py --data data/raw/HI-Small_Trans.csv --predictions outputs/neural_v292/test_predictions_by_seed.parquet --output-root outputs --open-browser
```

The pipeline performs the post-processing stages using the trained predictions.

Expected output directories:

```text
outputs/
│
├── neural_v292/
│
├── explain_v1/
│
├── louvain_v1/
│
├── bootstrap/
│
└── dashboard/
```

---

## 8. Run Individual Modules

Each module can also be executed independently.

### A. Explanation Layer

Generates behavioural reason codes and investigator dossiers.

```powershell
python mulenet/explain.py --data data/raw/HI-Small_Trans.csv --predictions outputs/neural_v292/test_predictions_by_seed.parquet --output outputs/explain_v1 --top-n 1000
```

### B. Louvain Ring Detection

Detects suspicious communities and transaction-network patterns.

```powershell
python mulenet/rings.py --data data/raw/HI-Small_Trans.csv --predictions outputs/neural_v292/test_predictions_by_seed.parquet --dossiers outputs/explain_v1/improved_dossiers.csv --output outputs/louvain_v1 --top-n 1000
```

### C. Dashboard Generation

```powershell
python mulenet/dashboard.py --explain outputs/explain_v1 --louvain outputs/louvain_v1 --output outputs/dashboard
```

### D. Bootstrap Comparison

Requires two separate training runs with valid prediction files.

```powershell
python mulenet/bootstrap.py --v291 outputs/run_a/test_predictions_by_seed.parquet --v292 outputs/neural_v292/test_predictions_by_seed.parquet --output outputs/bootstrap --n-iter 10000
```

Replace `outputs/run_a/` with the actual directory containing the v2.9.1 predictions.

---

## 9. Optional Experiments

### A. GraphSAGE Comparison

GraphSAGE controlled comparison:

```powershell
python mulenet/train.py --data data/raw/HI-Small_Trans.csv --output outputs/neural_v292 --experiment-graphsage
```

No additional installation is specified in the original project instructions.

### B. LightGBM Comparison

Install LightGBM:

```powershell
pip install "lightgbm>=4.0"
```

Run the experiment:

```powershell
python mulenet/train.py --data data/raw/HI-Small_Trans.csv --output outputs/neural_v292 --experiment-lightgbm
```

---

## 10. Key Design Decisions

### Why Chronological Splitting?

Random account splits can leak future transaction patterns into training features.

MuleNet AI uses a strict chronological six-window evaluation with 40/50/60/70/80/90th percentile cut-points.

### Why Does Feature-only GNN Perform Better Than GraphSAGE?

On the IBM HI-Small dataset, account-level temporal features such as:

- 6-hour transaction throughput
- Dwell time
- Rapid forwarding

carry substantial predictive information.

Graph message-passing can add marginal value because mule accounts connect to many legitimate accounts, potentially diluting neighbourhood representations.

### Why Average Precision Instead of AUC?

Under approximately 1:1,000 class imbalance, ROC-AUC can obscure the precision-recall trade-off.

Average Precision (AP), based on the precision-recall curve, is the primary evaluation metric.

### Why Validation-gated Random Forest Blending?

The Random Forest candidate is included in the final scorer only when it improves AP on both validation windows simultaneously.

This prevents test-time information from influencing the model blending decision.

---

## 11. Known Limitations

1. **Synthetic Dataset:** IBM AML HI-Small is fully synthetic. Labels are transaction-derived proxies, not independently verified money-mule identities.

2. **Single Dataset Evaluation:** The current implementation evaluates one dataset and does not implement federated multi-bank learning.

3. **Class Imbalance:** Approximately 1:1,000 class imbalance limits absolute AP scores.

4. **Reason-code Limitations:** Behavioural reason codes describe historical transaction patterns. They are not proof of illicit activity or definitive fund tracing.

5. **Missing Device Information:** The dataset does not provide device identifiers, so device-sharing signals cannot be incorporated.

6. **Investigation Requirement:** Model predictions and network communities should be treated as investigation leads, not automatic determinations of criminal activity.

---

## 12. Reproducibility and Verification

Before considering an experiment complete, verify the following:

- [ ] Dataset is present in `data/raw/`.
- [ ] Virtual environment is activated.
- [ ] Dependencies are installed successfully.
- [ ] PyTorch installation is working.
- [ ] Model training completes without errors.
- [ ] `test_predictions_by_seed.parquet` is generated.
- [ ] Explanation layer generates its expected outputs.
- [ ] Louvain ring detection completes.
- [ ] Dashboard is generated successfully.
- [ ] Metrics and evaluation files are available.
- [ ] Results are checked against the reported research results.

---

## 13. Citation

If you use this code or methodology, please cite:

```bibtex
@misc{mulenet2026,
  title  = {MuleNet AI: Graph-based Money-Mule Detection},
  author = {Mandeep Singh},
  year   = {2026},
  url    = {https://github.com/Mandeep15686/MuleNet-AI}
}
```

---

## 14. License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.

---

## 15. Acknowledgements

- IBM for the synthetic Anti-Money Laundering transaction dataset.
- PyTorch for deep learning infrastructure.
- The open-source graph analytics ecosystem.
- Kaggle for dataset distribution.

---

**MuleNet AI — Research-driven Graph-based Money-Mule Detection**

*Detect suspicious behavioural patterns. Identify potentially connected networks. Support financial investigations.*
