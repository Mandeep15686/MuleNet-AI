"""
MuleNet AI — Paired Bootstrap: V2.9.1 vs V2.9.2
=================================================
Tests whether V2.9.2's AP improvement over V2.9.1 is statistically
significant using a paired bootstrap at the account level.

Method
------
For each test snapshot (test1 / test2 / test3):
  1. Average scores across 5 seeds per model → one score per account
  2. Run 10,000 bootstrap iterations:
       - Resample accounts WITH REPLACEMENT (same sample for both models)
       - Compute Average Precision for each model on the resample
       - Record AP_v292 − AP_v291
  3. Report:
       - Observed AP for each model (no resampling)
       - Bootstrap mean difference + 95 % CI
       - One-tailed p-value: P(bootstrap_diff ≤ 0)  →  "is v292 actually better?"

Usage
-----
python mulenet_bootstrap_v291_v292.py ^
    --v291  ".\\outputs_neural_v291\\test_predictions_by_seed.parquet" ^
    --v292  ".\\outputs_neural_v292\\test_predictions_by_seed.parquet" ^
    --output ".\\outputs_bootstrap" ^
    --n-iter 10000 ^
    --seed   0
"""

import argparse, json, logging, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_preds(path: Path, label: str) -> pd.DataFrame:
    log.info("Loading %s: %s", label, path)
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]

    # normalise score column
    for cand in ["score", "risk_score", "prob", "gnn_score"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "score"})
            break
    else:
        raise ValueError(f"{label}: no score column found in {list(df.columns)}")

    # normalise label column
    for cand in ["label", "is_laundering", "y", "target"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "label"})
            break
    else:
        raise ValueError(f"{label}: no label column found in {list(df.columns)}")

    # normalise snapshot column
    for cand in ["snapshot", "window", "period", "split"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "snapshot"})
            break
    else:
        raise ValueError(f"{label}: no snapshot/window column found in {list(df.columns)}")

    log.info("  %d rows | snapshots: %s | seeds: %s",
             len(df),
             sorted(df["snapshot"].unique()),
             sorted(df["seed"].unique()) if "seed" in df.columns else "n/a")
    return df


def mean_score_per_account(df: pd.DataFrame) -> pd.DataFrame:
    """Average score across seeds → one row per (snapshot, account_id)."""
    grp_cols = ["snapshot", "account_id"]
    agg = (df.groupby(grp_cols, observed=True, as_index=False)
             .agg(score=("score", "mean"), label=("label", "first")))
    return agg


def bootstrap_ap_diff(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels:   np.ndarray,
    n_iter:   int,
    rng:      np.random.Generator,
) -> np.ndarray:
    """
    Paired bootstrap: resample the same indices for both models.
    Returns array of shape (n_iter,) with AP_b − AP_a per iteration.
    """
    n = len(labels)
    diffs = np.empty(n_iter, dtype=np.float64)

    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        y   = labels[idx]
        if y.sum() == 0:          # no positives in resample → skip
            diffs[i] = 0.0
            continue
        ap_a = average_precision_score(y, scores_a[idx])
        ap_b = average_precision_score(y, scores_b[idx])
        diffs[i] = ap_b - ap_a

    return diffs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v291",    required=True, help="V2.9.1 predictions parquet")
    ap.add_argument("--v292",    required=True, help="V2.9.2 predictions parquet")
    ap.add_argument("--output",  required=True, help="Output directory")
    ap.add_argument("--n-iter",  type=int, default=10_000)
    ap.add_argument("--seed",    type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── load & merge ──────────────────────────────────────────────────────────
    v291 = load_preds(Path(args.v291), "V2.9.1")
    v292 = load_preds(Path(args.v292), "V2.9.2")

    v291_avg = mean_score_per_account(v291).rename(columns={"score": "score_291"})
    v292_avg = mean_score_per_account(v292).rename(columns={"score": "score_292"})

    merged = v291_avg.merge(
        v292_avg[["snapshot", "account_id", "score_292"]],
        on=["snapshot", "account_id"],
        how="inner",
    )

    n_total   = len(merged)
    n_pos     = int(merged["label"].sum())
    snapshots = sorted(merged["snapshot"].unique())

    log.info("Merged: %d accounts (%d positives) across %d snapshots",
             n_total, n_pos, len(snapshots))

    # ── per-snapshot bootstrap ────────────────────────────────────────────────
    all_rows   = []
    dist_data  = {}

    for snap in snapshots:
        sub = merged[merged["snapshot"] == snap].reset_index(drop=True)
        n_snap = len(sub)
        n_pos_snap = int(sub["label"].sum())

        scores_a = sub["score_291"].to_numpy()
        scores_b = sub["score_292"].to_numpy()
        labels   = sub["label"].to_numpy()

        obs_ap_a = average_precision_score(labels, scores_a)
        obs_ap_b = average_precision_score(labels, scores_b)
        obs_diff = obs_ap_b - obs_ap_a
        obs_pct  = obs_diff / obs_ap_a * 100

        log.info("Bootstrap [%s] — %d accounts, %d positives…", snap, n_snap, n_pos_snap)
        t0 = time.time()
        diffs = bootstrap_ap_diff(scores_a, scores_b, labels, args.n_iter, rng)
        elapsed = time.time() - t0

        ci_lo = float(np.percentile(diffs, 2.5))
        ci_hi = float(np.percentile(diffs, 97.5))
        p_val = float(np.mean(diffs <= 0))   # one-tailed: P(v292 NOT better)
        boot_mean = float(diffs.mean())
        boot_std  = float(diffs.std())

        significant = (p_val < 0.05) and (ci_lo > 0)
        verdict = "SIGNIFICANT ✓" if significant else "NOT SIGNIFICANT ✗"

        log.info("  V2.9.1 AP = %.5f | V2.9.2 AP = %.5f | diff = %+.5f (%+.2f%%)",
                 obs_ap_a, obs_ap_b, obs_diff, obs_pct)
        log.info("  95%% CI  = [%+.5f, %+.5f]", ci_lo, ci_hi)
        log.info("  p-value = %.4f  →  %s  (%.1f s)", p_val, verdict, elapsed)

        row = {
            "snapshot":          snap,
            "n_accounts":        n_snap,
            "n_positives":       n_pos_snap,
            "ap_v291":           round(obs_ap_a, 6),
            "ap_v292":           round(obs_ap_b, 6),
            "observed_diff":     round(obs_diff, 6),
            "observed_diff_pct": round(obs_pct,  3),
            "ci_95_lo":          round(ci_lo, 6),
            "ci_95_hi":          round(ci_hi, 6),
            "p_value":           round(p_val, 5),
            "bootstrap_mean_diff": round(boot_mean, 6),
            "bootstrap_std_diff":  round(boot_std,  6),
            "significant_at_05": significant,
            "verdict":           verdict,
            "n_bootstrap_iter":  args.n_iter,
            "elapsed_s":         round(elapsed, 1),
        }
        all_rows.append(row)
        dist_data[snap] = diffs.tolist()

    # ── global summary ────────────────────────────────────────────────────────
    macro_ap_291 = float(np.mean([r["ap_v291"] for r in all_rows]))
    macro_ap_292 = float(np.mean([r["ap_v292"] for r in all_rows]))
    macro_diff   = macro_ap_292 - macro_ap_291
    macro_pct    = macro_diff / macro_ap_291 * 100
    n_sig        = sum(1 for r in all_rows if r["significant_at_05"])

    log.info("─" * 60)
    log.info("GLOBAL SUMMARY")
    log.info("  Macro AP V2.9.1 : %.5f", macro_ap_291)
    log.info("  Macro AP V2.9.2 : %.5f", macro_ap_292)
    log.info("  Macro Δ AP      : %+.5f  (%+.2f%%)", macro_diff, macro_pct)
    log.info("  Significant windows: %d / %d", n_sig, len(snapshots))
    log.info("─" * 60)

    # ── save outputs ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(out_dir / "bootstrap_results.csv", index=False)
    log.info("bootstrap_results.csv written")

    # bootstrap distributions (for plotting / audit)
    dist_rows = []
    for snap, diffs in dist_data.items():
        for d in diffs:
            dist_rows.append({"snapshot": snap, "diff": d})
    pd.DataFrame(dist_rows).to_parquet(out_dir / "bootstrap_distributions.parquet", index=False)
    log.info("bootstrap_distributions.parquet written")

    summary = {
        "script":           "mulenet_bootstrap_v291_v292.py",
        "run_utc":          datetime.now(timezone.utc).isoformat(),
        "v291_path":        str(args.v291),
        "v292_path":        str(args.v292),
        "n_bootstrap_iter": args.n_iter,
        "rng_seed":         args.seed,
        "snapshots":        snapshots,
        "macro_ap_v291":    round(macro_ap_291, 6),
        "macro_ap_v292":    round(macro_ap_292, 6),
        "macro_delta_ap":   round(macro_diff, 6),
        "macro_delta_pct":  round(macro_pct, 3),
        "significant_windows": n_sig,
        "total_windows":       len(snapshots),
        "per_snapshot":     all_rows,
        "interpretation": (
            f"V2.9.2 {'IS' if n_sig == len(snapshots) else 'is NOT'} significantly "
            f"better than V2.9.1 across all {len(snapshots)} test windows at α=0.05. "
            f"Macro AP improvement: {macro_pct:+.2f}%."
        ),
    }
    with open(out_dir / "bootstrap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("bootstrap_summary.json written")
    log.info("All outputs → %s", out_dir.resolve())
    log.info("Done.")


if __name__ == "__main__":
    main()
