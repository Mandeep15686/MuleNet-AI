"""
MuleNet AI — Explainability Post-Processor v1  (vectorised rewrite)
====================================================================
Fixes:
  - fillna(0) on datetime columns → separate numeric fillna
  - throughput loop was O(n²) on 5 M rows → replaced with 6-h bucket groupby
  - rapid forwarding loop was O(n²)       → replaced with merge + time filter
  - dwell time loop was O(n²)             → filtered to flagged accounts only

Usage
-----
python mulenet_explain_v1.py ^
    --data        ".\\data\\HI-Small_Trans.csv"                         ^
    --predictions ".\\outputs_neural_v292\\test_predictions_by_seed.parquet" ^
    --output      ".\\outputs_explain_v1"                               ^
    --top-n       1000                                                  ^
    --rapid-window-min 60
"""

import argparse, json, logging, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _first(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"None of {candidates} found in columns: {list(cols)}")


def _fill_numeric(df):
    """Fill only numeric columns with 0 — avoids touching datetime cols."""
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].fillna(0)
    return df


def load_transactions(path: Path) -> pd.DataFrame:
    log.info("Loading transactions: %s", path)
    df = pd.read_csv(path, low_memory=False)
    c  = df.columns.tolist()

    src_acct = _first(c, ["Account",   "from_account", "From Account"])
    dst_acct = _first(c, ["Account.1", "to_account",   "To Account"])
    src_bank = _first(c, ["From Bank", "from_bank"])
    dst_bank = _first(c, ["To Bank",   "to_bank"])
    ts_col   = _first(c, ["Timestamp", "timestamp", "Date"])
    amt_paid = _first(c, ["Amount Paid",     "amount_paid",     "amount"])
    amt_recv = _first(c, ["Amount Received", "amount_received"])
    pay_cur  = _first(c, ["Payment Currency",   "payment_currency",   "currency"])
    recv_cur = _first(c, ["Receiving Currency", "receiving_currency"])

    tx = df.rename(columns={
        src_acct: "src_account", dst_acct: "dst_account",
        src_bank: "from_bank",   dst_bank: "to_bank",
        ts_col:   "timestamp",
        amt_paid: "amount_paid", amt_recv: "amount_received",
        pay_cur:  "payment_currency", recv_cur: "receiving_currency",
    }).copy()

    tx["timestamp"]       = pd.to_datetime(tx["timestamp"])
    tx["src_id"]          = tx["from_bank"].astype(str) + "::" + tx["src_account"].astype(str)
    tx["dst_id"]          = tx["to_bank"].astype(str) + "::" + tx["dst_account"].astype(str)
    tx["amount_paid"]     = pd.to_numeric(tx["amount_paid"],     errors="coerce").fillna(0.0)
    tx["amount_received"] = pd.to_numeric(tx["amount_received"], errors="coerce").fillna(0.0)
    log.info("Loaded %d transactions", len(tx))
    return tx


def load_predictions(path: Path) -> pd.DataFrame:
    log.info("Loading predictions: %s", path)
    suf = path.suffix.lower()
    df  = pd.read_parquet(path) if suf == ".parquet" else pd.read_csv(path)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    alts = {"account_id": ["account", "acct_id", "id"],
            "risk_score":  ["score", "gnn_score", "prob"]}
    for tgt, opts in alts.items():
        if tgt not in df.columns:
            for o in opts:
                if o in df.columns:
                    df = df.rename(columns={o: tgt}); break

    missing = {"account_id", "risk_score"} - set(df.columns)
    if missing:
        raise ValueError(f"Prediction file missing columns: {missing}")

    log.info("Loaded %d prediction rows", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — all vectorised
# ─────────────────────────────────────────────────────────────────────────────

def simple_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Fast aggregations for ALL accounts — used for population percentiles."""
    log.info("  simple features (all accounts)…")

    out = tx.groupby("src_id", observed=True).agg(
        total_out_count  = ("amount_paid",       "count"),
        total_out_amount = ("amount_paid",       "sum"),
        unique_out_ctpy  = ("dst_id",            "nunique"),
        out_currency_div = ("payment_currency",  "nunique"),
    ).reset_index().rename(columns={"src_id": "account_id"})

    inc = tx.groupby("dst_id", observed=True).agg(
        total_in_count   = ("amount_received",   "count"),
        total_in_amount  = ("amount_received",   "sum"),
        unique_in_ctpy   = ("src_id",            "nunique"),
        in_currency_div  = ("receiving_currency","nunique"),
    ).reset_index().rename(columns={"dst_id": "account_id"})

    feats = pd.merge(out, inc, on="account_id", how="outer")
    return _fill_numeric(feats)


def throughput_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Vectorised 6-hour throughput via time-bucket groupby."""
    log.info("  6-hour throughput (vectorised)…")

    tx = tx.copy()
    tx["bucket_6h"] = tx["timestamp"].dt.floor("6h")

    out_b = (tx.groupby(["src_id", "payment_currency",   "bucket_6h"], observed=True)
               ["amount_paid"].sum()
               .reset_index()
               .rename(columns={"src_id": "account_id",
                                 "payment_currency": "currency",
                                 "amount_paid": "out_amt"}))

    in_b  = (tx.groupby(["dst_id", "receiving_currency", "bucket_6h"], observed=True)
               ["amount_received"].sum()
               .reset_index()
               .rename(columns={"dst_id": "account_id",
                                 "receiving_currency": "currency",
                                 "amount_received": "in_amt"}))

    ft = out_b.merge(in_b, on=["account_id", "currency", "bucket_6h"], how="inner")
    ft["throughput"] = ft[["out_amt", "in_amt"]].min(axis=1)

    ft_agg = (ft.groupby("account_id", observed=True)["throughput"]
                .agg(ft_total="sum", ft_max="max", ft_windows="count")
                .reset_index())

    return ft_agg


def rapid_forward_features(
    tx: pd.DataFrame,
    account_ids: set,
    window_min: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Vectorised rapid-forwarding for FLAGGED accounts only.
    Returns (summary_df, events_df).
    """
    log.info("  rapid forwarding (flagged accounts only, window=%d min)…", window_min)

    in_tx = (tx[tx["dst_id"].isin(account_ids)]
             [["dst_id", "timestamp", "amount_received", "receiving_currency"]]
             .rename(columns={"dst_id": "account_id",
                               "timestamp": "ts_in",
                               "amount_received": "amt_in",
                               "receiving_currency": "cur"})
             .reset_index(drop=True))

    out_tx = (tx[tx["src_id"].isin(account_ids)]
              [["src_id", "timestamp", "amount_paid", "payment_currency"]]
              .rename(columns={"src_id": "account_id",
                                "timestamp": "ts_out",
                                "amount_paid": "amt_out",
                                "payment_currency": "cur"})
              .reset_index(drop=True))

    if in_tx.empty or out_tx.empty:
        empty_ev  = pd.DataFrame(columns=["account_id","in_timestamp","in_amount",
                                           "in_currency","forwarded_amount",
                                           "forward_ratio","n_out_events"])
        empty_sum = pd.DataFrame(columns=["account_id","rapid_event_count",
                                           "rapid_total_forwarded",
                                           "rapid_forward_ratio_max",
                                           "rapid_forward_ratio_mean"])
        return empty_sum, empty_ev

    window = pd.Timedelta(minutes=window_min)

    # Use sorted timestamps and prefix sums instead of a many-to-many merge.
    # A busy account/currency pair can otherwise create hundreds of millions
    # of intermediate rows before the time-window filter is applied.
    out_groups = {
        key: group.sort_values("ts_out")
        for key, group in out_tx.groupby(["account_id", "cur"], sort=False, observed=True)
    }
    event_frames = []
    window_ns = window.value
    for key, incoming in in_tx.groupby(["account_id", "cur"], sort=False, observed=True):
        outgoing = out_groups.get(key)
        if outgoing is None or outgoing.empty:
            continue
        incoming = incoming.sort_values("ts_in")
        out_times = outgoing["ts_out"].to_numpy(dtype="datetime64[ns]").astype("int64")
        out_amounts = outgoing["amt_out"].to_numpy(dtype="float64")
        prefix = np.concatenate(([0.0], np.cumsum(out_amounts, dtype="float64")))
        in_times = incoming["ts_in"].to_numpy(dtype="datetime64[ns]").astype("int64")
        # right-sided bounds exclude outputs at the same instant and include
        # outputs exactly at the end of the requested window.
        lo = np.searchsorted(out_times, in_times, side="right")
        hi = np.searchsorted(out_times, in_times + window_ns, side="right")
        counts = hi - lo
        selected = counts > 0
        if not selected.any():
            continue
        block = incoming.loc[selected, ["account_id", "ts_in", "amt_in", "cur"]].copy()
        block["forwarded_amount"] = prefix[hi[selected]] - prefix[lo[selected]]
        block["n_out_events"] = counts[selected]
        event_frames.append(block)

    if not event_frames:
        empty_ev  = pd.DataFrame(columns=["account_id","in_timestamp","in_amount",
                                           "in_currency","forwarded_amount",
                                           "forward_ratio","n_out_events"])
        empty_sum = pd.DataFrame(columns=["account_id","rapid_event_count",
                                           "rapid_total_forwarded",
                                           "rapid_forward_ratio_max",
                                           "rapid_forward_ratio_mean"])
        return empty_sum, empty_ev

    # Each incoming event already has its window aggregate.
    ev = pd.concat(event_frames, ignore_index=True)
    ev["forward_ratio"] = ev["forwarded_amount"] / ev["amt_in"].clip(lower=1e-9)
    ev = ev.rename(columns={"ts_in":"in_timestamp","amt_in":"in_amount","cur":"in_currency"})

    summary = (ev.groupby("account_id", observed=True)
                 .agg(rapid_event_count       = ("in_timestamp",    "count"),
                      rapid_total_forwarded   = ("forwarded_amount","sum"),
                      rapid_forward_ratio_max = ("forward_ratio",   "max"),
                      rapid_forward_ratio_mean= ("forward_ratio",   "mean"))
                 .reset_index())

    return summary, ev


def dwell_features(
    tx: pd.DataFrame,
    account_ids: set,
) -> pd.DataFrame:
    """Dwell time for FLAGGED accounts only — avoids cross-join explosion."""
    log.info("  dwell time (flagged accounts only)…")

    in_tx = (tx[tx["dst_id"].isin(account_ids)]
             [["dst_id", "timestamp"]]
             .rename(columns={"dst_id": "account_id", "timestamp": "ts_in"})
             .sort_values(["account_id","ts_in"])
             .reset_index(drop=True))

    out_tx = (tx[tx["src_id"].isin(account_ids)]
              [["src_id", "timestamp"]]
              .rename(columns={"src_id": "account_id", "timestamp": "ts_out"})
              .sort_values(["account_id","ts_out"])
              .reset_index(drop=True))

    if in_tx.empty or out_tx.empty:
        return pd.DataFrame(columns=["account_id","median_dwell_min","min_dwell_min"])

    # merge + filter: for each in-event find all out-events in same account that come after
    dwell = in_tx.merge(out_tx, on="account_id", how="inner")
    dwell = dwell[dwell["ts_out"] > dwell["ts_in"]].copy()
    dwell["dwell_min"] = (dwell["ts_out"] - dwell["ts_in"]).dt.total_seconds() / 60

    # take the *minimum* out-time per in-event, then aggregate per account
    dwell_min_per_event = (dwell.groupby(["account_id","ts_in"], observed=True)
                                ["dwell_min"].min().reset_index())

    agg = (dwell_min_per_event.groupby("account_id", observed=True)["dwell_min"]
                              .agg(median_dwell_min="median", min_dwell_min="min")
                              .reset_index())
    return agg


def build_all_features(
    tx: pd.DataFrame,
    flagged_accounts: set,
    rapid_window_min: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (full_feats, rf_events)
      full_feats : simple features for all accounts + temporal for flagged only
      rf_events  : raw rapid-forward events for evidence output
    """
    log.info("Building features…")

    # ── population-wide simple features ──────────────────────────────────────
    feats = simple_features(tx)

    # ── vectorised 6-hour throughput (all accounts) ───────────────────────────
    ft_agg = throughput_features(tx)
    feats  = feats.merge(ft_agg, on="account_id", how="left")
    feats[["ft_total","ft_max","ft_windows"]] = (
        feats[["ft_total","ft_max","ft_windows"]].fillna(0)
    )
    feats["ft_log_max"] = np.log1p(feats["ft_max"])

    # ── rapid forwarding — flagged only ──────────────────────────────────────
    rf_sum, rf_ev = rapid_forward_features(tx, flagged_accounts, rapid_window_min)
    feats = feats.merge(rf_sum, on="account_id", how="left")
    feats[["rapid_event_count","rapid_total_forwarded",
           "rapid_forward_ratio_max","rapid_forward_ratio_mean"]] = (
        feats[["rapid_event_count","rapid_total_forwarded",
               "rapid_forward_ratio_max","rapid_forward_ratio_mean"]].fillna(0)
    )

    # ── dwell time — flagged only ─────────────────────────────────────────────
    dwell = dwell_features(tx, flagged_accounts)
    feats = feats.merge(dwell, on="account_id", how="left")
    feats["median_dwell_min"] = feats["median_dwell_min"].fillna(99999)
    feats["min_dwell_min"]    = feats["min_dwell_min"].fillna(99999)

    log.info("Features built for %d accounts", len(feats))
    return feats, rf_ev


# ─────────────────────────────────────────────────────────────────────────────
# Reason-code assignment
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_COLS = [
    # (column,                     high_is_suspicious, pct_threshold)
    ("total_out_count",            True,  90),
    ("total_in_count",             True,  90),
    ("total_out_amount",           True,  90),
    ("total_in_amount",            True,  90),
    ("unique_out_ctpy",            True,  90),
    ("unique_in_ctpy",             True,  90),
    ("ft_log_max",                 True,  90),
    ("ft_windows",                 True,  90),
    ("rapid_event_count",          True,  90),
    ("rapid_forward_ratio_mean",   True,  90),
    ("median_dwell_min",           False, 10),
    ("out_currency_div",           True,  90),
    ("in_currency_div",            True,  90),
]

REASON_MAP = {
    "total_out_count":           "HIGH_OUTGOING_TRANSACTION_VOLUME",
    "total_in_count":            "HIGH_INCOMING_TRANSACTION_VOLUME",
    "total_out_amount":          "HIGH_TOTAL_OUTGOING_AMOUNT",
    "total_in_amount":           "HIGH_TOTAL_INCOMING_AMOUNT",
    "unique_out_ctpy":           "HIGH_OUTGOING_COUNTERPARTY_DIVERSITY",
    "unique_in_ctpy":            "HIGH_INCOMING_COUNTERPARTY_DIVERSITY",
    "ft_log_max":                "HIGH_PEAK_THROUGHPUT_WINDOW",
    "ft_windows":                "MANY_ACTIVE_THROUGHPUT_WINDOWS",
    "rapid_event_count":         "HIGH_RAPID_FORWARDING_EVENT_COUNT",
    "rapid_forward_ratio_mean":  "HIGH_RAPID_FORWARDING_RATIO",
    "median_dwell_min":          "SHORT_MEDIAN_DWELL_TIME",
    "out_currency_div":          "MULTI_CURRENCY_OUTGOING",
    "in_currency_div":           "MULTI_CURRENCY_INCOMING",
}

REASON_DESC = {
    "HIGH_OUTGOING_TRANSACTION_VOLUME":    "Sent transactions more frequently than 90 % of accounts",
    "HIGH_INCOMING_TRANSACTION_VOLUME":    "Received transactions more frequently than 90 % of accounts",
    "HIGH_TOTAL_OUTGOING_AMOUNT":          "Total outgoing amount exceeds 90th-percentile",
    "HIGH_TOTAL_INCOMING_AMOUNT":          "Total incoming amount exceeds 90th-percentile",
    "HIGH_OUTGOING_COUNTERPARTY_DIVERSITY":"Sent to more unique counterparties than 90 % of accounts",
    "HIGH_INCOMING_COUNTERPARTY_DIVERSITY":"Received from more unique counterparties than 90 % of accounts",
    "HIGH_PEAK_THROUGHPUT_WINDOW":         "Peak 6-hour in+out throughput exceeds 90th-percentile",
    "MANY_ACTIVE_THROUGHPUT_WINDOWS":      "Unusually many active 6-hour throughput windows",
    "HIGH_RAPID_FORWARDING_EVENT_COUNT":   "Forwarded funds within the rapid window more times than 90 % of accounts",
    "HIGH_RAPID_FORWARDING_RATIO":         "Average proportion of incoming funds forwarded quickly exceeds 90th-percentile",
    "SHORT_MEDIAN_DWELL_TIME":             "Funds leave the account faster than 90 % of accounts",
    "MULTI_CURRENCY_OUTGOING":             "Outgoing transactions span multiple currencies",
    "MULTI_CURRENCY_INCOMING":             "Incoming transactions span multiple currencies",
}


def compute_percentiles(feats: pd.DataFrame) -> dict:
    thresholds = {}
    for col, high_sus, pct in SIGNAL_COLS:
        if col not in feats.columns:
            continue
        vals = feats[col].replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            continue
        thresholds[col] = {
            "percentile": pct,
            "threshold":  float(np.percentile(vals, pct)),
            "high_is_suspicious": high_sus,
            "mean":   float(vals.mean()),
            "median": float(vals.median()),
        }
    return thresholds


def assign_reason_codes(row: pd.Series, thresholds: dict, max_codes: int = 5) -> list:
    hits = []
    for col, high_sus, _ in SIGNAL_COLS:
        if col not in thresholds or col not in row.index:
            continue
        val = row[col]
        thr = thresholds[col]["threshold"]
        triggered = (val > thr) if high_sus else (val < thr and val > 0)
        if triggered:
            strength = abs(val - thresholds[col]["median"])
            hits.append((strength, REASON_MAP[col], col, val, thr))
    hits.sort(key=lambda x: -x[0])
    return hits[:max_codes]


# ─────────────────────────────────────────────────────────────────────────────
# Evidence extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_evidence(accounts: list, tx: pd.DataFrame, max_per: int = 10) -> pd.DataFrame:
    log.info("Extracting supporting transactions for %d accounts…", len(accounts))
    accs = set(accounts)

    src = tx[tx["src_id"].isin(accs)].copy()
    src["role"]          = "sender"
    src["account_id"]    = src["src_id"]
    src["counterparty"]  = src["dst_id"]
    src["amount"]        = src["amount_paid"]
    src["currency"]      = src["payment_currency"]

    dst = tx[tx["dst_id"].isin(accs)].copy()
    dst["role"]          = "receiver"
    dst["account_id"]    = dst["dst_id"]
    dst["counterparty"]  = dst["src_id"]
    dst["amount"]        = dst["amount_received"]
    dst["currency"]      = dst["receiving_currency"]

    combined = pd.concat([src, dst], ignore_index=True)[
        ["account_id","timestamp","role","counterparty","amount","currency"]
    ].sort_values(["account_id","timestamp"], ascending=[True, False])

    return combined.groupby("account_id", observed=True).head(max_per).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",                required=True)
    ap.add_argument("--predictions",         required=True)
    ap.add_argument("--output",              required=True)
    ap.add_argument("--top-n",               type=int, default=1000)
    ap.add_argument("--rapid-window-min",    type=int, default=60)
    ap.add_argument("--evidence-per-account",type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load ─────────────────────────────────────────────────────────────────
    tx   = load_transactions(Path(args.data))
    pred = load_predictions(Path(args.predictions))

    # ── select top-N flagged accounts ─────────────────────────────────────────
    group_cols = ["account_id"]
    for c in ["window", "snapshot"]:
        if c in pred.columns:
            group_cols.append(c)

    pred_agg = (pred.groupby(group_cols, as_index=False)["risk_score"].mean()
                    .sort_values("risk_score", ascending=False))

    if len(group_cols) > 1:
        top = pred_agg.groupby(group_cols[1:], group_keys=False).head(args.top_n)
    else:
        top = pred_agg.head(args.top_n)

    top = top.copy()
    if len(group_cols) > 1:
        top["rank"] = (top.groupby(group_cols[1:])["risk_score"]
                          .rank(ascending=False, method="first").astype(int))
    else:
        top["rank"] = range(1, len(top) + 1)

    flagged = set(top["account_id"].unique())
    log.info("Top-%d flagged accounts selected (%d unique)", args.top_n, len(flagged))

    # ── account_id format reconciliation ─────────────────────────────────────
    # pred may store bare account numbers while tx uses BANK_ACCOUNT format.
    # Check overlap on a fast unique-set before running expensive features.
    tx_ids = set(tx["src_id"].unique()) | set(tx["dst_id"].unique())
    overlap_pre = flagged & tx_ids

    if len(overlap_pre) == 0:
        log.warning("Zero overlap between prediction account_ids and transaction ids.")
        pred_sample = sorted(pred["account_id"].dropna().unique())[:5]
        tx_sample   = sorted(list(tx_ids))[:5]
        log.warning("  pred sample : %s", pred_sample)
        log.warning("  tx   sample : %s", tx_sample)

        # Strategy 1 — pred is bare account number, tx is BANK_ACCOUNT
        raw_to_full = {}
        for fid in tx_ids:
            parts = fid.split("_", 1)
            if len(parts) == 2:
                raw_to_full.setdefault(parts[1], fid)

        remapped_count = sum(1 for a in flagged if a in raw_to_full)
        if remapped_count > 0:
            log.info("Strategy 1 (bare-account → BANK_ACCOUNT): %d / %d accounts remapped",
                     remapped_count, len(flagged))
            pred["account_id"] = pred["account_id"].map(lambda x: raw_to_full.get(x, x))
            top["account_id"]  = top["account_id"].map(lambda x: raw_to_full.get(x, x))
            flagged = set(top["account_id"].unique())
        else:
            # Strategy 2 — tx is BANK_ACCOUNT, pred might use a different separator
            # e.g. "BANKX-A001" or "BANKX/A001" or "BANKX A001"
            for sep in ["::", "-", "/", " ", ":"]:
                alt_to_full = {}
                for fid in tx_ids:
                    alt_to_full[fid.replace("_", sep)] = fid
                hits = sum(1 for a in flagged if a in alt_to_full)
                if hits > 0:
                    log.info("Strategy 2 (sep='%s'): %d accounts remapped", sep, hits)
                    pred["account_id"] = pred["account_id"].map(lambda x: alt_to_full.get(x, x))
                    top["account_id"]  = top["account_id"].map(lambda x: alt_to_full.get(x, x))
                    flagged = set(top["account_id"].unique())
                    break
            else:
                log.error("Could not auto-resolve account_id format mismatch.")
                log.error("Please check that --data and --predictions use the same account format.")
                sys.exit(1)

        new_overlap = flagged & tx_ids
        log.info("After remap: %d / %d flagged accounts matched in transactions",
                 len(new_overlap), len(flagged))
    else:
        log.info("account_id overlap: %d / %d flagged accounts matched (%.1f%%)",
                 len(overlap_pre), len(flagged), 100 * len(overlap_pre) / max(len(flagged), 1))

    # ── build features ────────────────────────────────────────────────────────
    feats, rf_ev = build_all_features(tx, flagged, args.rapid_window_min)

    # ── population percentiles ────────────────────────────────────────────────
    log.info("Computing population percentiles…")
    thresholds = compute_percentiles(feats)

    pd.DataFrame([
        {"signal": col, "reason_code": REASON_MAP.get(col, col),
         "percentile": v["percentile"], "threshold_value": v["threshold"],
         "population_mean": v["mean"], "population_median": v["median"]}
        for col, v in thresholds.items()
    ]).to_csv(out_dir / "population_percentiles.csv", index=False)

    # ── dossiers ──────────────────────────────────────────────────────────────
    log.info("Assigning reason codes…")
    feats_f = feats[feats["account_id"].isin(flagged)].copy()

    rows = []
    for _, row in feats_f.iterrows():
        codes = assign_reason_codes(row, thresholds)
        rows.append({
            "account_id":               row["account_id"],
            "total_out_count":          row.get("total_out_count", 0),
            "total_in_count":           row.get("total_in_count", 0),
            "total_out_amount":         row.get("total_out_amount", 0),
            "total_in_amount":          row.get("total_in_amount", 0),
            "unique_out_ctpy":          row.get("unique_out_ctpy", 0),
            "unique_in_ctpy":           row.get("unique_in_ctpy", 0),
            "ft_log_max":               row.get("ft_log_max", 0),
            "rapid_event_count":        row.get("rapid_event_count", 0),
            "rapid_forward_ratio_mean": row.get("rapid_forward_ratio_mean", 0),
            "median_dwell_min":         row.get("median_dwell_min", 99999),
            "n_reason_codes":           len(codes),
            "reason_1": codes[0][1] if len(codes) > 0 else "INSUFFICIENT_HISTORY",
            "reason_2": codes[1][1] if len(codes) > 1 else "",
            "reason_3": codes[2][1] if len(codes) > 2 else "",
            "reason_4": codes[3][1] if len(codes) > 3 else "",
            "reason_5": codes[4][1] if len(codes) > 4 else "",
            "reason_1_value": round(codes[0][3], 4) if len(codes) > 0 else None,
            "reason_2_value": round(codes[1][3], 4) if len(codes) > 1 else None,
            "reason_3_value": round(codes[2][3], 4) if len(codes) > 2 else None,
        })

    dossiers = pd.DataFrame(rows)
    dossiers = dossiers.merge(
        top[group_cols + ["risk_score", "rank"]], on="account_id", how="left"
    )
    dossiers["reason_1_description"] = dossiers["reason_1"].map(REASON_DESC).fillna("")
    dossiers["reason_2_description"] = dossiers["reason_2"].map(REASON_DESC).fillna("")
    dossiers["reason_3_description"] = dossiers["reason_3"].map(REASON_DESC).fillna("")
    dossiers.to_csv(out_dir / "improved_dossiers.csv", index=False)
    log.info("Dossiers written: %d", len(dossiers))

    # ── reason code distribution ──────────────────────────────────────────────
    rc = (dossiers[["reason_1","reason_2","reason_3","reason_4","reason_5"]]
          .stack().replace("", pd.NA).dropna().value_counts()
          .reset_index())
    rc.columns = ["reason_code", "count"]
    rc["description"]    = rc["reason_code"].map(REASON_DESC).fillna("")
    rc["pct_of_flagged"] = (rc["count"] / len(dossiers) * 100).round(1)
    rc.to_csv(out_dir / "reason_code_distribution.csv", index=False)

    # ── supporting evidence ───────────────────────────────────────────────────
    evidence = extract_evidence(list(flagged), tx, args.evidence_per_account)
    evidence = evidence.merge(
        dossiers[["account_id","risk_score","rank","reason_1"]], on="account_id", how="left"
    )
    evidence.to_csv(out_dir / "supporting_evidence.csv", index=False)

    # ── rapid forward events ──────────────────────────────────────────────────
    if not rf_ev.empty:
        rf_out = rf_ev[rf_ev["account_id"].isin(flagged)].copy()
        rf_out = rf_out.merge(
            dossiers[["account_id","risk_score","rank"]], on="account_id", how="left"
        ).sort_values(["rank","in_timestamp"])
        rf_out.to_csv(out_dir / "rapid_forward_events.csv", index=False)
        log.info("Rapid-forward events: %d", len(rf_out))

    # ── summary ───────────────────────────────────────────────────────────────
    n_specific = int((dossiers["reason_1"] != "INSUFFICIENT_HISTORY").sum())
    n_insuf    = int((dossiers["reason_1"] == "INSUFFICIENT_HISTORY").sum())

    log.info("─" * 60)
    log.info("BEFORE (v2.7):  ~2817 / 3000 → MODEL_RANKING_CONTEXT (useless)")
    log.info("AFTER  (v1):    %d / %d → specific behavioural reason", n_specific, len(dossiers))
    log.info("Insufficient history: %d accounts", n_insuf)
    log.info("─" * 60)

    summary = {
        "script":               "mulenet_explain_v1.py (vectorised)",
        "run_utc":              datetime.now(timezone.utc).isoformat(),
        "transactions":         int(len(tx)),
        "population_accounts":  int(len(feats)),
        "flagged_accounts":     int(len(flagged)),
        "dossiers_written":     int(len(dossiers)),
        "specific_reasons":     n_specific,
        "insufficient_history": n_insuf,
        "model_ranking_context_used": 0,
        "rapid_window_minutes": args.rapid_window_min,
        "top_n":                args.top_n,
        "outputs": {
            "improved_dossiers":      "improved_dossiers.csv",
            "reason_distribution":    "reason_code_distribution.csv",
            "supporting_evidence":    "supporting_evidence.csv",
            "rapid_forward_events":   "rapid_forward_events.csv",
            "population_percentiles": "population_percentiles.csv",
        },
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("All outputs → %s", out_dir.resolve())
    log.info("Done.")


if __name__ == "__main__":
    main()
