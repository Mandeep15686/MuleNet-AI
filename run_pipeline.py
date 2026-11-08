#!/usr/bin/env python
"""
MuleNet AI — End-to-End Pipeline Orchestrator
=============================================
Runs the full post-processing pipeline in sequence after training:

  1. Explanation layer   (reason codes per flagged account)
  2. Ring detection      (Louvain community detection)
  3. Dashboard builder   (self-contained HTML investigation dashboard)

This script assumes that mulenet/train.py has already been run and that
test_predictions_by_seed.parquet exists in the specified predictions path.

Usage
-----
python run_pipeline.py \\
    --data        data/raw/HI-Small_Trans.csv \\
    --predictions outputs/neural_v292/test_predictions_by_seed.parquet \\
    --output-root outputs \\
    --top-n       1000 \\
    --open-browser

One-command end-to-end (train + post-process)
---------------------------------------------
python mulenet/train.py --data data/raw/HI-Small_Trans.csv --output outputs/neural_v292
python run_pipeline.py  --data data/raw/HI-Small_Trans.csv \\
    --predictions outputs/neural_v292/test_predictions_by_seed.parquet \\
    --output-root outputs
"""

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path


def run(cmd: list, label: str) -> None:
    """Run a subprocess step, exit on failure."""
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{label}' failed (exit {result.returncode}). Stopping.")
        sys.exit(result.returncode)
    print(f"  ✓ {label} complete")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MuleNet AI pipeline orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", required=True, help="Path to HI-Small_Trans.csv")
    ap.add_argument(
        "--predictions", required=True,
        help="Path to test_predictions_by_seed.parquet (output of mulenet/train.py)",
    )
    ap.add_argument("--output-root", default="outputs", help="Root directory for all outputs")
    ap.add_argument("--top-n", type=int, default=1000, help="Top-N accounts to process")
    ap.add_argument("--rapid-window", type=int, default=60, help="Rapid-forward window in minutes")
    ap.add_argument("--min-ring-size", type=int, default=2, help="Minimum ring size for Louvain")
    ap.add_argument(
        "--open-browser", action="store_true",
        help="Open the HTML dashboard in the default browser when done",
    )
    args = ap.parse_args()

    root = Path(args.output_root)
    explain = root / "explain_v1"
    louvain = root / "louvain_v1"
    dashboard = root / "dashboard"
    py = sys.executable

    # Validate inputs
    data_path = Path(args.data)
    preds_path = Path(args.predictions)
    if not data_path.is_file():
        print(f"[ERROR] Dataset not found: {data_path}")
        sys.exit(1)
    if not preds_path.is_file():
        print(f"[ERROR] Predictions file not found: {preds_path}")
        print("  Run mulenet/train.py first to generate predictions.")
        sys.exit(1)

    # ── Step 1: Explanation layer ─────────────────────────────────────────────
    run(
        [
            py, "mulenet/explain.py",
            "--data", str(args.data),
            "--predictions", str(args.predictions),
            "--output", str(explain),
            "--top-n", str(args.top_n),
            "--rapid-window-min", str(args.rapid_window),
        ],
        "Explanation layer (reason codes)",
    )

    # ── Step 2: Ring detection ────────────────────────────────────────────────
    dossiers = explain / "improved_dossiers.csv"
    run(
        [
            py, "mulenet/rings.py",
            "--data", str(args.data),
            "--predictions", str(args.predictions),
            "--dossiers", str(dossiers),
            "--output", str(louvain),
            "--top-n", str(args.top_n),
            "--min-ring-size", str(args.min_ring_size),
        ],
        "Ring detection (Louvain)",
    )

    # ── Step 3: Dashboard ─────────────────────────────────────────────────────
    run(
        [
            py, "mulenet/dashboard.py",
            "--explain", str(explain),
            "--louvain", str(louvain),
            "--output", str(dashboard),
            "--top-rings", "30",
        ],
        "Dashboard builder",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    dash_html = dashboard / "mulenet_dashboard.html"
    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Explain outputs : {explain}")
    print(f"  Ring outputs    : {louvain}")
    print(f"  Dashboard       : {dash_html}")

    if args.open_browser and dash_html.exists():
        print("\n  Opening dashboard in browser…")
        webbrowser.open(dash_html.as_uri())
    elif not dash_html.exists():
        print("\n[WARNING] Dashboard HTML not found — dashboard step may have failed.")


if __name__ == "__main__":
    main()
