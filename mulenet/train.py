#!/usr/bin/env python
"""
MuleNet AI v2.9.2 — Explainable, temporally robust AML risk ranking.

The default pipeline is intentionally small:
  * a feature-only neural ranker (the primary model)
  * a Random Forest candidate, retained only after validation-only gating

LightGBM and GraphSAGE are excluded from default ranking. They are available
only as controlled, opt-in experiments and cannot silently affect production
scores. All account features and explanatory evidence use transactions
strictly before the relevant prediction cutoff. Labels come only from the
following target interval and are weak, synthetic proxy labels.

Usage
-----
python mulenet/train.py \\
    --data   data/raw/HI-Small_Trans.csv \\
    --output outputs/neural_v292

The final output includes test_predictions_by_seed.parquet which is consumed
by the explain, rings, and bootstrap post-processing modules.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn


VERSION = "2.9.2"
LOG = logging.getLogger("mulenet.train")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Seed all stochastic components used by the implemented models."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_dataset(data_arg: str) -> Path:
    path = Path(data_arg).expanduser()
    if path.is_file() and path.suffix.lower() == ".csv":
        return path
    if path.is_dir():
        candidates = list(path.rglob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV files found under {path}")
        transaction_files = sorted(p for p in candidates if "trans" in p.stem.lower())
        if transaction_files:
            hi_small = [p for p in transaction_files if "hi-small" in p.name.lower()]
            return (hi_small or transaction_files)[0]
        return sorted(candidates)[0]
    raise FileNotFoundError(f"Dataset CSV or directory not found: {path}")


def _norm_col(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    lookup = {_norm_col(column): column for column in df.columns}
    aliases = {
        "timestamp": ["timestamp", "datetime", "date"],
        "from_bank": ["frombank", "senderbank"],
        "from_account": ["account", "fromaccount", "senderaccount"],
        "to_bank": ["tobank", "receiverbank"],
        "to_account": ["account1", "toaccount", "receiveraccount"],
        "amount_received": ["amountreceived", "receivedamount"],
        "receiving_currency": ["receivingcurrency", "currencyreceived"],
        "amount_paid": ["amountpaid", "paidamount"],
        "payment_currency": ["paymentcurrency", "currency"],
        "payment_format": ["paymentformat", "format"],
        "is_laundering": ["islaundering", "laundering", "label"],
    }
    resolved: Dict[str, str] = {}
    missing = []
    for key, names in aliases.items():
        found = next((lookup[name] for name in names if name in lookup), None)
        if found is None:
            missing.append(key)
        else:
            resolved[key] = found
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    return resolved


def load_transactions(csv_path: Path, max_rows: Optional[int]) -> pd.DataFrame:
    LOG.info("Loading %s", csv_path)
    tx = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)
    tx = tx.rename(columns={value: key for key, value in resolve_columns(tx).items()})
    tx["timestamp"] = pd.to_datetime(tx["timestamp"], errors="coerce")
    for name in ("amount_received", "amount_paid", "is_laundering"):
        tx[name] = pd.to_numeric(tx[name], errors="coerce")
    # Account IDs are scoped to a bank. Avoids merging distinct account numbers
    # that happen to be reused at different institutions.
    tx["from_account"] = tx["from_bank"].astype(str) + "::" + tx["from_account"].astype(str)
    tx["to_account"] = tx["to_bank"].astype(str) + "::" + tx["to_account"].astype(str)
    required = ["timestamp", "from_account", "to_account", "amount_received", "amount_paid", "is_laundering"]
    before = len(tx)
    tx = tx.dropna(subset=required).copy()
    tx["is_laundering"] = tx["is_laundering"].astype(np.int8).clip(0, 1)
    for name in ("receiving_currency", "payment_currency", "payment_format"):
        tx[name] = tx[name].fillna("UNKNOWN").astype(str)
    tx["amount_received"] = tx["amount_received"].clip(lower=0)
    tx["amount_paid"] = tx["amount_paid"].clip(lower=0)
    tx = tx.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    LOG.info("Retained %d / %d valid transactions", len(tx), before)
    if tx.empty:
        raise ValueError("No valid transactions remain after cleaning.")
    return tx


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def build_events(tx: pd.DataFrame) -> pd.DataFrame:
    """Create account-centric, directional event rows from transaction records."""
    outgoing = pd.DataFrame({
        "account": tx["from_account"].to_numpy(),
        "counterparty": tx["to_account"].to_numpy(),
        "direction": "out",
        "amount": tx["amount_paid"].to_numpy(dtype=np.float64),
        "currency": tx["payment_currency"].to_numpy(),
        "timestamp": tx["timestamp"].to_numpy(),
        "payment_format": tx["payment_format"].to_numpy(),
    })
    incoming = pd.DataFrame({
        "account": tx["to_account"].to_numpy(),
        "counterparty": tx["from_account"].to_numpy(),
        "direction": "in",
        "amount": tx["amount_received"].to_numpy(dtype=np.float64),
        "currency": tx["receiving_currency"].to_numpy(),
        "timestamp": tx["timestamp"].to_numpy(),
        "payment_format": tx["payment_format"].to_numpy(),
    })
    events = pd.concat([outgoing, incoming], ignore_index=True)
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    events["log_amount"] = np.log1p(events["amount"])
    events["time_bucket_6h"] = events["timestamp"].dt.floor("6h")
    for column in ("account", "counterparty", "currency", "payment_format", "direction"):
        events[column] = events[column].astype("category")
    return events


def add_rapid_forward_features(events: pd.DataFrame) -> pd.DataFrame:
    """Summarise outgoing events following a same-currency incoming event.

    The latest strictly earlier incoming event is matched within one hour. It
    is a behavioural association, not a funds-tracing or causality claim.
    """
    incoming = events.loc[
        events["direction"].eq("in"),
        ["account", "currency", "timestamp", "amount"]
    ].rename(
        columns={"timestamp": "incoming_timestamp", "amount": "incoming_amount"}
    ).sort_values("incoming_timestamp")
    outgoing = events.loc[
        events["direction"].eq("out"),
        ["account", "currency", "timestamp", "amount", "counterparty"]
    ].rename(
        columns={"timestamp": "outgoing_timestamp", "amount": "outgoing_amount"}
    ).sort_values("outgoing_timestamp")
    matched = pd.merge_asof(
        outgoing,
        incoming,
        left_on="outgoing_timestamp",
        right_on="incoming_timestamp",
        by=["account", "currency"],
        direction="backward",
        tolerance=pd.Timedelta(hours=1),
        allow_exact_matches=False,
    ).dropna(subset=["incoming_timestamp"]).copy()
    columns = [
        "rapid_out_1h_count", "rapid_out_1h_log_amount_sum", "rapid_out_1h_log_amount_mean",
        "rapid_forward_ratio_mean", "median_in_to_out_minutes", "rapid_out_counterparties",
    ]
    if matched.empty:
        return pd.DataFrame(columns=columns)
    matched["gap_minutes"] = (
        matched["outgoing_timestamp"] - matched["incoming_timestamp"]
    ).dt.total_seconds() / 60.0
    matched["log_outgoing_amount"] = np.log1p(matched["outgoing_amount"])
    per_currency = matched.groupby(["account", "currency"], observed=True).agg(
        rapid_out_1h_count=("outgoing_amount", "size"),
        rapid_out_1h_log_amount_sum=("log_outgoing_amount", "sum"),
        rapid_out_1h_log_amount_mean=("log_outgoing_amount", "mean"),
        rapid_out_amount=("outgoing_amount", "sum"),
        median_in_to_out_minutes=("gap_minutes", "median"),
        rapid_out_counterparties=("counterparty", "nunique"),
    )
    incoming_totals = incoming.groupby(["account", "currency"], observed=True)["incoming_amount"].sum()
    per_currency = per_currency.join(incoming_totals.rename("total_incoming_amount"))
    per_currency["rapid_forward_ratio"] = (
        per_currency["rapid_out_amount"] / per_currency["total_incoming_amount"].clip(lower=1e-9)
    )
    return per_currency.groupby(level="account", observed=True).agg(
        rapid_out_1h_count=("rapid_out_1h_count", "sum"),
        rapid_out_1h_log_amount_sum=("rapid_out_1h_log_amount_sum", "sum"),
        rapid_out_1h_log_amount_mean=("rapid_out_1h_log_amount_mean", "mean"),
        rapid_forward_ratio_mean=("rapid_forward_ratio", "mean"),
        median_in_to_out_minutes=("median_in_to_out_minutes", "median"),
        rapid_out_counterparties=("rapid_out_counterparties", "sum"),
    )


def build_account_features(
    history_tx: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int], np.ndarray, np.ndarray, pd.DataFrame]:
    """Build account features and historical-only graph edges."""
    events = build_events(history_tx)
    blocks = []
    for direction in ("in", "out"):
        subset = events.loc[events["direction"].eq(direction)]
        blocks.append(subset.groupby("account", observed=True).agg(**{
            f"{direction}_count": ("amount", "size"),
            f"{direction}_log_amount_mean": ("log_amount", "mean"),
            f"{direction}_log_amount_std": ("log_amount", "std"),
            f"{direction}_log_amount_max": ("log_amount", "max"),
            f"{direction}_counterparties": ("counterparty", "nunique"),
            f"{direction}_currency_count": ("currency", "nunique"),
            f"{direction}_format_count": ("payment_format", "nunique"),
        }))
    features = pd.concat(blocks, axis=1)
    bucket = events.groupby(
        ["account", "currency", "time_bucket_6h", "direction"], observed=True
    )["amount"].sum().unstack("direction", fill_value=0.0)
    for direction in ("in", "out"):
        if direction not in bucket:
            bucket[direction] = 0.0
    bucket["throughput"] = np.minimum(bucket["in"], bucket["out"])
    bucket["log_throughput"] = np.log1p(bucket["throughput"])
    bucket["active_6h"] = (bucket["throughput"] > 0).astype(np.int8)
    per_currency = bucket.groupby(["account", "currency"], observed=True).agg(
        ft_log_sum=("log_throughput", "sum"),
        ft_log_max=("log_throughput", "max"),
        ft_active_6h_windows=("active_6h", "sum"),
    )
    features = features.join(per_currency.groupby(level="account", observed=True).agg(
        ft_log_sum_mean_currency=("ft_log_sum", "mean"),
        ft_log_max=("ft_log_max", "max"),
        ft_active_6h_windows=("ft_active_6h_windows", "sum"),
    ), how="left")
    features = features.join(add_rapid_forward_features(events), how="left")
    features["fan_in_degree"] = features["in_counterparties"]
    features["fan_out_degree"] = features["out_counterparties"]
    features["out_in_count_ratio"] = features["out_count"] / features["in_count"].clip(lower=1)
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.where(np.isfinite(features), np.nan)
    features.index = features.index.astype(str)
    accounts = features.index.tolist()
    node_map = {account: index for index, account in enumerate(accounts)}
    source = history_tx["from_account"].map(node_map).to_numpy(dtype=np.int64)
    destination = history_tx["to_account"].map(node_map).to_numpy(dtype=np.int64)
    return features, node_map, np.concatenate([source, destination]), np.concatenate([destination, source]), events


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Snapshot Construction
# ─────────────────────────────────────────────────────────────────────────────

def prepare_temporal_snapshot(
    history_tx: pd.DataFrame, target_tx: pd.DataFrame, feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Produce one leakage-controlled snapshot."""
    if history_tx.empty or target_tx.empty:
        raise ValueError("Temporal snapshot needs non-empty history and target windows.")
    features, node_map, edge_src, edge_dst, events = build_account_features(history_tx)
    historical_ids = features.index.astype(str).to_numpy()
    if feature_names is not None:
        features = features.reindex(columns=feature_names)
    target_accounts = set(target_tx["from_account"].astype(str)) | set(target_tx["to_account"].astype(str))
    positives = set(
        target_tx.loc[target_tx["is_laundering"].eq(1), "from_account"].astype(str)
    ) | set(target_tx.loc[target_tx["is_laundering"].eq(1), "to_account"].astype(str))
    eligible_ids = [account for account in historical_ids if account in target_accounts]
    sample_idx = np.asarray([node_map[account] for account in eligible_ids], dtype=np.int64)
    labels = np.zeros(len(historical_ids), dtype=np.int8)
    if len(sample_idx):
        labels[sample_idx] = np.fromiter(
            (account in positives for account in eligible_ids), dtype=np.int8
        )
    activity = events.groupby("account", observed=True).agg(
        historical_event_count=("amount", "size"),
        latest_historical_activity=("timestamp", "max"),
    )
    return {
        "features": features, "account_ids": historical_ids, "node_map": node_map,
        "edge_src": edge_src, "edge_dst": edge_dst, "labels": labels, "sample_idx": sample_idx,
        "sample_ids": np.asarray(eligible_ids, dtype=object), "activity": activity,
        "unseen_target_accounts": int(len(target_accounts - set(historical_ids))),
        "target_accounts": int(len(target_accounts)),
        "positive_accounts": int(sum(account in positives for account in eligible_ids)),
        "history_transactions": int(len(history_tx)), "target_transactions": int(len(target_tx)),
    }


def build_rolling_snapshots(
    tx: pd.DataFrame,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """Create one train, two validation and three forward test snapshots."""
    counts = tx.groupby("timestamp", sort=True).size()
    unique_times = counts.index.to_numpy()
    cumulative = counts.to_numpy(dtype=np.int64).cumsum()
    if len(unique_times) < 8:
        raise ValueError(f"Need at least eight distinct timestamps; found {len(unique_times)}.")
    quantiles = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    raw_indices = [
        int(np.searchsorted(cumulative, fraction * len(tx), side="left"))
        for fraction in quantiles
    ]
    indices: List[int] = []
    previous = 0
    for position, raw in enumerate(raw_indices):
        upper = len(unique_times) - (len(raw_indices) - position)
        bounded = min(max(raw, previous + 1), upper)
        indices.append(bounded)
        previous = bounded
    cuts = [pd.Timestamp(unique_times[index]) for index in indices]
    names = ["train", "validation_1", "validation_2", "test_1", "test_2", "test_3"]
    packs = []
    feature_names: Optional[List[str]] = None
    for index, name in enumerate(names):
        cutoff = cuts[index]
        end = cuts[index + 1] if index + 1 < len(cuts) else None
        history = tx.loc[tx["timestamp"] < cutoff].copy()
        target = (
            tx.loc[tx["timestamp"] >= cutoff].copy()
            if end is None
            else tx.loc[(tx["timestamp"] >= cutoff) & (tx["timestamp"] < end)].copy()
        )
        pack = prepare_temporal_snapshot(history, target, feature_names)
        if feature_names is None:
            feature_names = list(pack["features"].columns)
        pack.update({
            "window_name": name,
            "cutoff": str(cutoff),
            "target_end": str(end) if end is not None else "dataset_end",
        })
        if len(pack["sample_idx"]) < 10 or pack["positive_accounts"] < 1:
            raise ValueError(
                f"{name} has too few eligible accounts or no positive proxy accounts."
            )
        packs.append(pack)
        LOG.info(
            "%s | history_tx=%d target_tx=%d nodes=%d scored=%d positives=%d unseen_target=%d",
            name, pack["history_transactions"], pack["target_transactions"], len(pack["account_ids"]),
            len(pack["sample_idx"]), pack["positive_accounts"], pack["unseen_target_accounts"],
        )
    metadata = {
        pack["window_name"]: {
            "cutoff": pack["cutoff"], "target_end": pack["target_end"],
            "history_transactions": pack["history_transactions"],
            "target_transactions": pack["target_transactions"],
            "target_accounts": pack["target_accounts"],
            "scored_accounts": int(len(pack["sample_idx"])),
            "positive_accounts": pack["positive_accounts"],
            "unseen_target_accounts": pack["unseen_target_accounts"],
        }
        for pack in packs
    }
    return packs[0], packs[1:3], packs[3:], feature_names or [], metadata


def transform_snapshot(
    pack: Dict[str, Any],
    feature_names: List[str],
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> None:
    values = pack["features"].reindex(columns=feature_names).astype(np.float32)
    pack["X"] = scaler.transform(imputer.transform(values)).astype(np.float32)


def concat_labels(packs: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.concatenate([pack["labels"][pack["sample_idx"]] for pack in packs]).astype(np.int8)


# ─────────────────────────────────────────────────────────────────────────────
# Metric Helpers
# ─────────────────────────────────────────────────────────────────────────────

def topk_metrics(
    y: np.ndarray, scores: np.ndarray, k_values: Sequence[int]
) -> Dict[str, float]:
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    total_positive = int(np.sum(y))
    prevalence = float(np.mean(y)) if len(y) else 0.0
    metrics: Dict[str, float] = {"positive_prevalence": prevalence}
    for requested_k in k_values:
        k = min(max(1, int(requested_k)), len(y))
        hits = int(np.sum(y[order[:k]]))
        precision = hits / k
        metrics[f"precision_at_{requested_k}"] = float(precision)
        metrics[f"recall_at_{requested_k}"] = float(hits / total_positive) if total_positive else 0.0
        metrics[f"lift_at_{requested_k}"] = float(precision / prevalence) if prevalence else 0.0
        metrics[f"review_count_at_{requested_k}"] = int(k)
    return metrics


def metric_dict(y: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    prediction = np.asarray(scores >= threshold, dtype=np.int8)
    negatives = max(1, int(np.sum(y == 0)))
    return {
        "pr_auc_average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else float("nan"),
        "threshold": float(threshold),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "fpr": float(np.sum((prediction == 1) & (y == 0)) / negatives),
        "positive_count": int(np.sum(y)),
        "negative_count": int(np.sum(y == 0)),
    }


def choose_threshold_at_fpr(
    y_true: np.ndarray, scores: np.ndarray, target_fpr: float
) -> float:
    negatives = np.asarray(scores)[np.asarray(y_true) == 0]
    if len(negatives) == 0:
        raise ValueError("Cannot select an FPR threshold without validation negatives.")
    return float(np.quantile(negatives, 1.0 - target_fpr))


def rank_against_reference(reference_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Validation-ECDF rank transformation. It is not probability calibration."""
    reference = np.sort(np.asarray(reference_scores, dtype=np.float64))
    return np.searchsorted(reference, np.asarray(scores, dtype=np.float64), side="right") / max(1, len(reference))


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class FeatureOnlyGNN(nn.Module):
    """Identity-adjacency GNN control: no cross-account neighbour messages."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.30) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(input_dim * 2, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(torch.relu(self.linear_1(torch.cat([x, x], dim=1))))
        h = self.dropout(torch.relu(self.linear_2(torch.cat([h, h], dim=1))))
        return self.classifier(h).squeeze(1)


class MeanGraphSAGEExperiment(nn.Module):
    """Opt-in historical-edge experiment; never part of the default scorer."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(input_dim * 2, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    @staticmethod
    def aggregate(
        x: torch.Tensor, adjacency: torch.Tensor, degree: torch.Tensor
    ) -> torch.Tensor:
        return torch.sparse.mm(adjacency, x) / degree.clamp_min(1.0)

    def forward(
        self, x: torch.Tensor, adjacency: torch.Tensor, degree: torch.Tensor
    ) -> torch.Tensor:
        first = self.aggregate(x, adjacency, degree)
        h = self.dropout(torch.relu(self.linear_1(torch.cat([x, first], dim=1))))
        second = self.aggregate(h, adjacency, degree)
        h = self.dropout(torch.relu(self.linear_2(torch.cat([h, second], dim=1))))
        return self.classifier(h).squeeze(1)


def build_sparse_adjacency(
    edge_src: np.ndarray, edge_dst: np.ndarray, node_count: int, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct a coalesced weighted historical graph only for an opt-in test."""
    try:
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise RuntimeError("GraphSAGE experiment requires scipy.") from exc
    matrix = coo_matrix(
        (np.ones(len(edge_src), dtype=np.float32), (edge_dst, edge_src)),
        shape=(node_count, node_count), dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    coo = matrix.tocoo()
    adjacency = torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64)),
        torch.from_numpy(coo.data.astype(np.float32)),
        (node_count, node_count), dtype=torch.float32,
    ).coalesce().to(device)
    degree = torch.from_numpy(np.asarray(matrix.sum(axis=1), dtype=np.float32)).to(device)
    return adjacency, degree


# ─────────────────────────────────────────────────────────────────────────────
# Training Loops
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SeedPredictions:
    validation: np.ndarray
    tests: Dict[str, np.ndarray]
    best_epoch: int
    validation_ap: float


def train_feature_only_seed(
    train_pack: Dict[str, Any],
    validation_packs: Sequence[Dict[str, Any]],
    test_packs: Sequence[Dict[str, Any]],
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    feature_names: Sequence[str],
) -> SeedPredictions:
    """Fit one primary feature-only GNN using validation-only early stopping."""
    set_seed(seed)
    train_x = torch.as_tensor(train_pack["X"], dtype=torch.float32, device=device)
    train_y = torch.as_tensor(train_pack["labels"], dtype=torch.float32, device=device)
    train_idx = torch.as_tensor(train_pack["sample_idx"], dtype=torch.long, device=device)
    validation = [
        (
            pack,
            torch.as_tensor(pack["X"], dtype=torch.float32, device=device),
            torch.as_tensor(pack["sample_idx"], dtype=torch.long, device=device),
            pack["labels"][pack["sample_idx"]],
        )
        for pack in validation_packs
    ]
    positive_count = max(1, int(train_pack["labels"][train_pack["sample_idx"]].sum()))
    negative_count = max(1, len(train_pack["sample_idx"]) - positive_count)
    pos_weight = min(negative_count / positive_count, args.max_pos_weight)
    model = FeatureOnlyGNN(train_x.shape[1], args.hidden_dim, args.dropout).to(device)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_ap, best_epoch, wait, best_state = -1.0, 0, 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = loss_fn(logits[train_idx], train_y[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.eval()
        validation_aps = []
        with torch.no_grad():
            for _, x, idx, y in validation:
                scores = torch.sigmoid(model(x)[idx]).cpu().numpy()
                validation_aps.append(float(average_precision_score(y, scores)))
        validation_ap = float(np.mean(validation_aps))
        LOG.info(
            "feature_only_gnn seed=%d epoch=%03d loss=%.5f val_AP=%.6f",
            seed, epoch, float(loss.item()), validation_ap,
        )
        if validation_ap > best_ap:
            best_ap, best_epoch, wait = validation_ap, epoch, 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            wait += 1
            if wait >= args.patience:
                LOG.info("feature_only_gnn seed=%d early stopped at epoch %d", seed, epoch)
                break
    if best_state is None:
        raise RuntimeError("Feature-only GNN did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_scores = np.concatenate([
            torch.sigmoid(model(x)[idx]).cpu().numpy() for _, x, idx, _ in validation
        ]).astype(np.float64)
    test_scores: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for pack in test_packs:
            x = torch.as_tensor(pack["X"], dtype=torch.float32, device=device)
            idx = torch.as_tensor(pack["sample_idx"], dtype=torch.long, device=device)
            test_scores[pack["window_name"]] = (
                torch.sigmoid(model(x)[idx]).cpu().numpy().astype(np.float64)
            )
    checkpoint = {
        "pipeline_version": VERSION,
        "model_type": "feature_only_gnn",
        "model_state_dict": model.state_dict(),
        "input_dim": int(train_x.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "feature_names": list(feature_names),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_ap_mean": float(best_ap),
        "graph_policy": "identity adjacency; no cross-account neighbour messages",
        "history_policy": "features built only from transactions before each snapshot cutoff",
    }
    torch.save(checkpoint, out_dir / f"feature_only_gnn_seed_{seed}_v292.pt")
    del model, train_x, train_y, train_idx
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return SeedPredictions(validation_scores, test_scores, best_epoch, best_ap)


def fit_feature_only_gnn(
    train_pack: Dict[str, Any],
    validation_packs: Sequence[Dict[str, Any]],
    test_packs: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    feature_names: Sequence[str],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    """Run repeated primary-model fits and persist reproducible score arrays.

    Returns
    -------
    seeded : per-seed SeedPredictions dict
    aggregated : {"validation": np.ndarray, "test": {window: np.ndarray}}
    rows : seed stability metric rows
    metadata : per-seed training metadata
    """
    seeded: Dict[int, SeedPredictions] = {}
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        result = train_feature_only_seed(
            train_pack, validation_packs, test_packs, seed, args, device, out_dir, feature_names
        )
        seeded[seed] = result
        for pack in test_packs:
            y = pack["labels"][pack["sample_idx"]]
            threshold = choose_threshold_at_fpr(
                concat_labels(validation_packs), result.validation, args.target_fpr
            )
            row = {
                "row_type": "seed_window",
                "model": "feature_only_gnn",
                "seed": seed,
                "window": pack["window_name"],
                "best_epoch": result.best_epoch,
                "validation_ap_mean": result.validation_ap,
            }
            row.update(metric_dict(y, result.tests[pack["window_name"]], threshold))
            row.update(topk_metrics(y, result.tests[pack["window_name"]], args.top_k_values))
            rows.append(row)
    validation = np.mean(np.vstack([seeded[seed].validation for seed in seeds]), axis=0)
    tests = {
        pack["window_name"]: np.mean(
            np.vstack([seeded[seed].tests[pack["window_name"]] for seed in seeds]), axis=0
        )
        for pack in test_packs
    }
    # Archive per-seed arrays for reproducibility and downstream bootstrap use
    archive: Dict[str, np.ndarray] = {"validation_labels": concat_labels(validation_packs)}
    for seed, result in seeded.items():
        archive[f"seed_{seed}__validation"] = result.validation.astype(np.float32)
        for window, scores in result.tests.items():
            archive[f"seed_{seed}__{window}"] = scores.astype(np.float32)
    np.savez_compressed(out_dir / "feature_only_gnn_predictions_v292.npz", **archive)
    # ── Parquet export for explain / rings / bootstrap ─────────────────────
    pred_rows = []
    for seed, result in seeded.items():
        for pack in test_packs:
            window = pack["window_name"]
            scores = result.tests[window]
            y = pack["labels"][pack["sample_idx"]]
            ids = pack["sample_ids"]
            for acct, sc, lbl in zip(ids, scores, y):
                pred_rows.append({
                    "seed": int(seed),
                    "snapshot": window,
                    "account_id": str(acct),
                    "risk_score": float(sc),
                    "label": int(lbl),
                })
    # Also add ensemble (seed == -1) which explain/rings prefer by default
    for pack in test_packs:
        window = pack["window_name"]
        ensemble_scores = tests[window]
        y = pack["labels"][pack["sample_idx"]]
        ids = pack["sample_ids"]
        for acct, sc, lbl in zip(ids, ensemble_scores, y):
            pred_rows.append({
                "seed": -1,
                "snapshot": window,
                "account_id": str(acct),
                "risk_score": float(sc),
                "label": int(lbl),
            })
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_parquet(out_dir / "test_predictions_by_seed.parquet", index=False)
    LOG.info(
        "test_predictions_by_seed.parquet written: %d rows (%d seeds + 1 ensemble, %d windows)",
        len(pred_df), len(seeds), len(test_packs),
    )
    stability = pd.DataFrame(rows)
    summary_rows = []
    metric_columns = ["pr_auc_average_precision", "roc_auc", "precision", "recall", "fpr"]
    for window, group in stability.groupby("window", sort=False):
        summary = {
            "row_type": "seed_summary",
            "model": "feature_only_gnn",
            "seed": "all",
            "window": window,
            "seed_count": int(len(group)),
        }
        for metric in metric_columns:
            summary[f"{metric}_mean"] = float(group[metric].mean())
            summary[f"{metric}_std"] = float(group[metric].std(ddof=0))
        summary_rows.append(summary)
    return seeded, {"validation": validation, "test": tests}, rows + summary_rows, {
        str(seed): {
            "best_epoch": result.best_epoch,
            "validation_ap_mean": result.validation_ap,
        }
        for seed, result in seeded.items()
    }


def fit_random_forest_candidate(
    train_pack: Dict[str, Any],
    validation_packs: Sequence[Dict[str, Any]],
    test_packs: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
    args: argparse.Namespace,
    out_dir: Path,
    feature_names: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Fit a secondary RF candidate on the same training snapshot and features."""
    train_idx = train_pack["sample_idx"]
    x_train = train_pack["X"][train_idx]
    y_train = train_pack["labels"][train_idx]
    validation_seed = []
    test_seed: Dict[str, List[np.ndarray]] = {
        pack["window_name"]: [] for pack in test_packs
    }
    per_seed: Dict[str, Any] = {}
    for seed in seeds:
        model = RandomForestClassifier(
            n_estimators=args.rf_trees,
            class_weight="balanced_subsample",
            min_samples_leaf=args.rf_min_samples_leaf,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(x_train, y_train)
        validation_scores = np.concatenate([
            model.predict_proba(pack["X"][pack["sample_idx"]])[:, 1]
            for pack in validation_packs
        ]).astype(np.float64)
        validation_seed.append(validation_scores)
        for pack in test_packs:
            test_seed[pack["window_name"]].append(
                model.predict_proba(pack["X"][pack["sample_idx"]])[:, 1].astype(np.float64)
            )
        joblib.dump(
            {
                "pipeline_version": VERSION,
                "model": model,
                "feature_names": list(feature_names),
                "seed": seed,
                "history_policy": "training snapshot only; preprocessing fitted on training accounts only",
            },
            out_dir / f"random_forest_candidate_seed_{seed}_v292.joblib",
        )
        per_seed[str(seed)] = {
            "validation_ap": float(
                average_precision_score(concat_labels(validation_packs), validation_scores)
            ),
        }
    return {
        "validation": np.mean(np.vstack(validation_seed), axis=0),
        "test": {
            window: np.mean(np.vstack(values), axis=0)
            for window, values in test_seed.items()
        },
    }, per_seed


def _window_ap(
    labels: np.ndarray,
    scores: np.ndarray,
    packs: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    offset = 0
    for pack in packs:
        length = len(pack["sample_idx"])
        results[pack["window_name"]] = float(
            average_precision_score(
                labels[offset : offset + length], scores[offset : offset + length]
            )
        )
        offset += length
    return results


def validation_gate_random_forest(
    primary_validation: np.ndarray,
    rf_validation: np.ndarray,
    validation_packs: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, np.ndarray]]:
    """Select RF weight with validation labels only and an all-window safeguard."""
    labels = concat_labels(validation_packs)
    rank_maps = {
        "feature_only_gnn": np.sort(np.asarray(primary_validation, dtype=np.float64)),
        "random_forest": np.sort(np.asarray(rf_validation, dtype=np.float64)),
    }
    primary_rank = rank_against_reference(rank_maps["feature_only_gnn"], primary_validation)
    rf_rank = rank_against_reference(rank_maps["random_forest"], rf_validation)
    primary_ap = float(average_precision_score(labels, primary_rank))
    primary_windows = _window_ap(labels, primary_rank, validation_packs)
    candidates = []
    for rf_weight in np.arange(0.0, 0.51, 0.10):
        blended = (1.0 - rf_weight) * primary_rank + rf_weight * rf_rank
        ap = float(average_precision_score(labels, blended))
        window_aps = _window_ap(labels, blended, validation_packs)
        candidates.append({
            "rf_weight": float(round(rf_weight, 2)),
            "validation_ap": ap,
            "validation_ap_gain_vs_primary": ap - primary_ap,
            "validation_ap_by_window": window_aps,
            "all_validation_windows_non_degrading": bool(all(
                window_aps[name] >= primary_windows[name] - args.rf_validation_tolerance
                for name in primary_windows
            )),
        })
    nonzero = [row for row in candidates if row["rf_weight"] > 0]
    best = max(nonzero, key=lambda row: (row["validation_ap"], -row["rf_weight"]))
    accepted = bool(
        best["validation_ap_gain_vs_primary"] >= args.rf_min_validation_ap_gain
        and best["all_validation_windows_non_degrading"]
    )
    weights = (
        {"feature_only_gnn": 1.0 - best["rf_weight"], "random_forest": best["rf_weight"]}
        if accepted
        else {"feature_only_gnn": 1.0, "random_forest": 0.0}
    )
    selected = "feature_only_gnn" if not accepted else (
        "feature_only_gnn_rf_blend" if best["rf_weight"] < 1.0 else "random_forest"
    )
    decision = {
        "rf_retained": accepted,
        "best_rf_weight": best["rf_weight"],
        "validation_ap_gain": best["validation_ap_gain_vs_primary"],
        "all_windows_non_degrading": best["all_validation_windows_non_degrading"],
        "selected_model": selected,
        "decision_reason": (
            f"RF retained with weight {best['rf_weight']:.2f} "
            f"(gain={best['validation_ap_gain_vs_primary']:.5f})"
            if accepted
            else f"RF excluded: gain={best['validation_ap_gain_vs_primary']:.5f} below threshold "
                 f"or a validation window degraded."
        ),
        "candidates": candidates,
    }
    return weights, decision, rank_maps


def combine_scores(
    primary_scores: np.ndarray,
    rf_scores: np.ndarray,
    rank_maps: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    components = {
        "feature_only_gnn": rank_against_reference(rank_maps["feature_only_gnn"], primary_scores),
        "random_forest": rank_against_reference(rank_maps["random_forest"], rf_scores),
    }
    combined = sum(weights[name] * components[name] for name in weights)
    return np.asarray(combined, dtype=np.float64), components


# ─────────────────────────────────────────────────────────────────────────────
# Optional Experiments
# ─────────────────────────────────────────────────────────────────────────────

def run_graphsage_experiment(
    train_pack: Dict[str, Any],
    validation_packs: Sequence[Dict[str, Any]],
    test_packs: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    """Run a controlled GraphSAGE comparison with the same snapshots and seeds."""
    per_seed = []
    validation_seed = []
    test_seed: Dict[str, List[np.ndarray]] = {pack["window_name"]: [] for pack in test_packs}
    for seed in seeds:
        set_seed(seed)
        train_x = torch.as_tensor(train_pack["X"], dtype=torch.float32, device=device)
        train_y = torch.as_tensor(train_pack["labels"], dtype=torch.float32, device=device)
        train_idx = torch.as_tensor(train_pack["sample_idx"], dtype=torch.long, device=device)
        train_adj, train_degree = build_sparse_adjacency(
            train_pack["edge_src"], train_pack["edge_dst"],
            len(train_pack["account_ids"]), device,
        )
        validation = []
        for pack in validation_packs:
            x = torch.as_tensor(pack["X"], dtype=torch.float32, device=device)
            idx = torch.as_tensor(pack["sample_idx"], dtype=torch.long, device=device)
            adjacency, degree = build_sparse_adjacency(
                pack["edge_src"], pack["edge_dst"], len(pack["account_ids"]), device
            )
            validation.append((pack, x, idx, adjacency, degree, pack["labels"][pack["sample_idx"]]))
        positive = max(1, int(train_pack["labels"][train_pack["sample_idx"]].sum()))
        negative = max(1, len(train_pack["sample_idx"]) - positive)
        model = MeanGraphSAGEExperiment(train_x.shape[1], args.hidden_dim, args.dropout).to(device)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                min(negative / positive, args.max_pos_weight), dtype=torch.float32, device=device
            )
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        best_ap, best_epoch, wait, state = -1.0, 0, 0, None
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(
                model(train_x, train_adj, train_degree)[train_idx], train_y[train_idx]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                aps = [
                    float(average_precision_score(
                        y, torch.sigmoid(model(x, adj, deg)[idx]).cpu().numpy()
                    ))
                    for _, x, idx, adj, deg, y in validation
                ]
            ap = float(np.mean(aps))
            if ap > best_ap:
                best_ap, best_epoch, wait = ap, epoch, 0
                state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= args.patience:
                    break
        if state is None:
            raise RuntimeError("GraphSAGE experiment did not produce a checkpoint.")
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            validation_scores = np.concatenate([
                torch.sigmoid(model(x, adj, deg)[idx]).cpu().numpy()
                for _, x, idx, adj, deg, _ in validation
            ]).astype(np.float64)
        validation_seed.append(validation_scores)
        for pack in test_packs:
            x = torch.as_tensor(pack["X"], dtype=torch.float32, device=device)
            idx = torch.as_tensor(pack["sample_idx"], dtype=torch.long, device=device)
            adjacency, degree = build_sparse_adjacency(
                pack["edge_src"], pack["edge_dst"], len(pack["account_ids"]), device
            )
            with torch.no_grad():
                scores = torch.sigmoid(model(x, adjacency, degree)[idx]).cpu().numpy().astype(np.float64)
            test_seed[pack["window_name"]].append(scores)
            del x, idx, adjacency, degree
        torch.save(
            {
                "pipeline_version": VERSION,
                "model_type": "graphsage_controlled_experiment",
                "model_state_dict": model.state_dict(),
                "input_dim": int(train_x.shape[1]),
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "feature_names": list(feature_names),
                "seed": seed,
                "best_epoch": best_epoch,
                "best_validation_ap_mean": best_ap,
                "graph_policy": "historical transaction edges only for each snapshot",
            },
            out_dir / f"graphsage_experiment_seed_{seed}_v292.pt",
        )
        per_seed.append({"seed": seed, "best_epoch": best_epoch, "validation_ap_mean": best_ap})
        del model, train_x, train_y, train_idx, train_adj, train_degree, validation
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "model": "graphsage_experiment",
        "validation": np.mean(np.vstack(validation_seed), axis=0),
        "test": {name: np.mean(np.vstack(values), axis=0) for name, values in test_seed.items()},
        "per_seed": per_seed,
        "policy": "opt-in controlled experiment; not eligible for default production ranking",
    }


def run_lightgbm_experiment(
    train_pack: Dict[str, Any],
    validation_packs: Sequence[Dict[str, Any]],
    test_packs: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
    args: argparse.Namespace,
    out_dir: Path,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    """Run an explicit optional LightGBM comparison without adding it by default."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError(
            "--experiment-lightgbm was requested, but LightGBM is not installed. "
            "Install it with: pip install lightgbm>=4.0"
        ) from exc
    train_idx = train_pack["sample_idx"]
    validation_seed = []
    test_seed: Dict[str, List[np.ndarray]] = {pack["window_name"]: [] for pack in test_packs}
    per_seed = []
    for seed in seeds:
        model = LGBMClassifier(
            objective="binary", n_estimators=args.lgbm_trees, learning_rate=0.04,
            num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=2.0, class_weight="balanced",
            random_state=seed, n_jobs=-1, verbosity=-1,
        )
        model.fit(train_pack["X"][train_idx], train_pack["labels"][train_idx])
        validation_scores = np.concatenate([
            model.predict_proba(pack["X"][pack["sample_idx"]])[:, 1]
            for pack in validation_packs
        ]).astype(np.float64)
        validation_seed.append(validation_scores)
        for pack in test_packs:
            test_seed[pack["window_name"]].append(
                model.predict_proba(pack["X"][pack["sample_idx"]])[:, 1].astype(np.float64)
            )
        joblib.dump(
            {
                "pipeline_version": VERSION, "model": model,
                "feature_names": list(feature_names), "seed": seed,
                "policy": "optional controlled experiment only; excluded from default production ranking",
            },
            out_dir / f"lightgbm_experiment_seed_{seed}_v292.joblib",
        )
        per_seed.append({
            "seed": seed,
            "validation_ap": float(
                average_precision_score(concat_labels(validation_packs), validation_scores)
            ),
        })
    return {
        "model": "lightgbm_experiment",
        "validation": np.mean(np.vstack(validation_seed), axis=0),
        "test": {
            name: np.mean(np.vstack(values), axis=0) for name, values in test_seed.items()
        },
        "per_seed": per_seed,
        "policy": "opt-in controlled experiment; not eligible for default production ranking",
    }


def controlled_experiment_summary(
    experiment: Dict[str, Any],
    primary_validation: np.ndarray,
    validation_packs: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    labels = concat_labels(validation_packs)
    primary_ap = float(average_precision_score(labels, primary_validation))
    exp_ap = float(average_precision_score(labels, experiment["validation"]))
    gain = exp_ap - primary_ap
    primary_windows = _window_ap(labels, primary_validation, validation_packs)
    exp_windows = _window_ap(labels, experiment["validation"], validation_packs)
    consistent = all(
        exp_windows[name] >= primary_windows[name] - args.experiment_min_validation_ap_gain
        for name in primary_windows
    )
    return {
        "model": experiment["model"],
        "primary_validation_ap": primary_ap,
        "experiment_validation_ap": exp_ap,
        "pooled_validation_ap_gain_vs_primary": gain,
        "consistent_gain_observed": consistent,
        "primary_validation_ap_by_window": primary_windows,
        "experiment_validation_ap_by_window": exp_windows,
        "promotion_note": (
            f"Consistent validation AP gain of +{gain:.5f} observed across all windows. "
            "Requires explicit promotion to production."
            if consistent and gain >= args.experiment_min_validation_ap_gain
            else f"No consistent AP gain observed (gain={gain:.5f}). Do not promote."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Explanation Thresholds and Reason Codes
# ─────────────────────────────────────────────────────────────────────────────

def make_reason_thresholds(
    train_pack: Dict[str, Any], percentile: float
) -> Dict[str, float]:
    """Fit deterministic explanation thresholds from training-history values only."""
    frame = train_pack["features"].iloc[train_pack["sample_idx"]]
    high_features = [
        "rapid_out_1h_count", "rapid_forward_ratio_mean", "ft_active_6h_windows",
        "ft_log_sum_mean_currency", "fan_out_degree", "fan_in_degree",
        "out_in_count_ratio", "in_count", "out_count",
    ]
    thresholds = {}
    for feature in high_features:
        if feature in frame:
            values = pd.to_numeric(frame[feature], errors="coerce").dropna()
            thresholds[feature] = float(values.quantile(percentile)) if len(values) else float("nan")
    dwell = pd.to_numeric(
        frame.get("median_in_to_out_minutes", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    thresholds["median_in_to_out_minutes_low"] = (
        float(dwell.quantile(1.0 - percentile)) if len(dwell) else float("nan")
    )
    thresholds["percentile"] = float(percentile)
    thresholds["fit_scope"] = "eligible accounts in the training history snapshot only"
    return thresholds


def _number(value: Any, digits: int = 2) -> str:
    if pd.isna(value):
        return "not available"
    return f"{float(value):.{digits}f}"


def account_reason_codes(
    row: pd.Series, thresholds: Dict[str, float]
) -> List[Tuple[str, str]]:
    """Return descriptive, non-causal reason codes backed by history-only values."""
    def above(name: str) -> bool:
        value, threshold = row.get(name, np.nan), thresholds.get(name, np.nan)
        return bool(pd.notna(value) and pd.notna(threshold) and float(value) >= float(threshold))

    reasons: List[Tuple[str, str]] = []
    rapid_count = row.get("rapid_out_1h_count", np.nan)
    dwell = row.get("median_in_to_out_minutes", np.nan)
    rapid = above("rapid_out_1h_count") or above("rapid_forward_ratio_mean")
    short_dwell = pd.notna(dwell) and pd.notna(thresholds.get("median_in_to_out_minutes_low")) and (
        float(dwell) <= float(thresholds["median_in_to_out_minutes_low"])
    )
    if rapid:
        reasons.append((
            "HISTORICAL_RAPID_FORWARDING_PATTERN",
            "Historical same-currency matching found "
            f"{_number(rapid_count, 0)} outgoing event(s) within one hour after an earlier incoming event; "
            f"median observed gap {_number(dwell, 1)} minutes. This is an association, not proof of fund tracing.",
        ))
    if above("ft_active_6h_windows") or above("ft_log_sum_mean_currency"):
        reasons.append((
            "HIGH_HISTORICAL_THROUGHPUT",
            "Historical six-hour activity shows "
            f"{_number(row.get('ft_active_6h_windows'), 0)} active throughput window(s) and "
            f"log-throughput summary {_number(row.get('ft_log_sum_mean_currency'))}; both are compared to "
            "training-history percentile thresholds.",
        ))
    if above("fan_out_degree"):
        reasons.append((
            "HIGH_HISTORICAL_FAN_OUT",
            f"Historical outgoing activity involved {_number(row.get('fan_out_degree'), 0)} distinct counterparties.",
        ))
    if above("fan_in_degree"):
        reasons.append((
            "HIGH_HISTORICAL_FAN_IN",
            f"Historical incoming activity involved {_number(row.get('fan_in_degree'), 0)} distinct counterparties.",
        ))
    if above("out_in_count_ratio"):
        reasons.append((
            "HISTORICAL_OUT_IN_ACTIVITY_IMBALANCE",
            "Historical outgoing/incoming event-count ratio was "
            f"{_number(row.get('out_in_count_ratio'))}.",
        ))
    if above("in_count") or above("out_count"):
        reasons.append((
            "HIGH_HISTORICAL_TRANSACTION_ACTIVITY",
            f"Historical ledger entries: {_number(row.get('in_count'), 0)} incoming and "
            f"{_number(row.get('out_count'), 0)} outgoing.",
        ))
    if short_dwell and not rapid:
        reasons.append((
            "SHORT_HISTORICAL_IN_TO_OUT_GAP",
            f"Median historical matched incoming-to-outgoing gap was {_number(dwell, 1)} minutes.",
        ))
    return reasons


def review_priority(rank: int) -> str:
    if rank <= 50:
        return "urgent"
    if rank <= 100:
        return "high"
    if rank <= 500:
        return "standard"
    return "routine"


# ─────────────────────────────────────────────────────────────────────────────
# Output Writers
# ─────────────────────────────────────────────────────────────────────────────

def build_explanations_and_queue(
    test_packs: Sequence[Dict[str, Any]],
    final_scores: Dict[str, np.ndarray],
    component_scores: Dict[str, Dict[str, np.ndarray]],
    thresholds: Dict[str, float],
    selected_model: str,
    output_dir: Path,
    review_top_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Export investigator-facing history-only evidence for ranked accounts.

    No future proxy label is written to either investigator artifact. Labels are
    reserved for offline evaluation and never used to construct reason codes.
    """
    explanation_rows: List[Dict[str, Any]] = []
    queue_rows: List[Dict[str, Any]] = []
    displayed_features = [
        "in_count", "out_count", "fan_in_degree", "fan_out_degree", "rapid_out_1h_count",
        "rapid_forward_ratio_mean", "median_in_to_out_minutes", "ft_active_6h_windows",
        "ft_log_sum_mean_currency", "out_in_count_ratio",
    ]
    for pack in test_packs:
        window = pack["window_name"]
        score = final_scores[window]
        order = np.argsort(-score, kind="mergesort")[: min(review_top_n, len(score))]
        raw = pack["features"].iloc[pack["sample_idx"]].copy()
        raw["account_id"] = pack["sample_ids"]
        raw = raw.set_index("account_id", drop=False)
        activity = pack["activity"]
        for local_rank, index in enumerate(order, start=1):
            account = str(pack["sample_ids"][index])
            feature_row = raw.loc[account]
            reasons = account_reason_codes(feature_row, thresholds)[:3]
            if not reasons:
                reasons = [(
                    "MODEL_RANKING_CONTEXT",
                    "This account ranked highly from the learned combination of its historical features; "
                    "no single rule exceeded the configured training-history explanation threshold.",
                )]
            history_event_count = (
                activity.loc[account, "historical_event_count"]
                if account in activity.index
                else np.nan
            )
            latest_activity = (
                activity.loc[account, "latest_historical_activity"]
                if account in activity.index
                else pd.NaT
            )
            feature_evidence = {
                name: (
                    None if pd.isna(feature_row.get(name)) else float(feature_row.get(name))
                )
                for name in displayed_features
                if name in feature_row.index
            }
            reason_codes = [code for code, _ in reasons]
            reason_evidence = [evidence for _, evidence in reasons]
            explanation = {
                "window": window,
                "account_id": account,
                "rank": local_rank,
                "relative_risk_score": float(score[index]),
                "selected_model": selected_model,
                "historical_cutoff": pack["cutoff"],
                "target_window_end": pack["target_end"],
                "historical_event_count": (
                    None if pd.isna(history_event_count) else int(history_event_count)
                ),
                "latest_historical_activity": (
                    "" if pd.isna(latest_activity) else str(latest_activity)
                ),
                "reason_codes": "|".join(reason_codes),
                "reason_1_code": reason_codes[0],
                "reason_1_evidence": reason_evidence[0],
                "reason_2_code": reason_codes[1] if len(reason_codes) > 1 else "",
                "reason_2_evidence": reason_evidence[1] if len(reason_evidence) > 1 else "",
                "reason_3_code": reason_codes[2] if len(reason_codes) > 2 else "",
                "reason_3_evidence": reason_evidence[2] if len(reason_evidence) > 2 else "",
                "historical_feature_evidence_json": json.dumps(feature_evidence, sort_keys=True),
                "explanation_caveat": (
                    "Historical descriptive evidence only; reason codes are not causal attributions, "
                    "proof of laundering, or confirmation of a mule account."
                ),
            }
            explanation_rows.append(explanation)
            queue_rows.append({
                "window": window,
                "rank": local_rank,
                "review_priority": review_priority(local_rank),
                "account_id": account,
                "relative_risk_score": float(score[index]),
                "selected_model": selected_model,
                "feature_only_gnn_rank_score": float(component_scores["feature_only_gnn"][window][index]),
                "random_forest_rank_score": float(component_scores["random_forest"][window][index]),
                "reason_codes": explanation["reason_codes"],
                "supporting_historical_evidence": explanation["reason_1_evidence"],
                "historical_cutoff": pack["cutoff"],
                "score_interpretation": "Relative rank score, not a calibrated probability or a case finding.",
            })
    explanations = pd.DataFrame(explanation_rows)
    queue = pd.DataFrame(queue_rows)
    explanations.to_csv(output_dir / "account_explanations_v292.csv", index=False)
    queue.to_csv(output_dir / "investigator_review_queue_v292.csv", index=False)
    return explanations, queue


def average_test_metrics(
    packs: Sequence[Dict[str, Any]],
    scores: Dict[str, np.ndarray],
    threshold: float,
    k_values: Sequence[int],
) -> Dict[str, float]:
    rows = []
    for pack in packs:
        y = pack["labels"][pack["sample_idx"]]
        metric = metric_dict(y, scores[pack["window_name"]], threshold)
        metric.update(topk_metrics(y, scores[pack["window_name"]], k_values))
        rows.append(metric)
    numeric_columns = {
        key for row in rows for key, value in row.items()
        if isinstance(value, (int, float, np.number))
    }
    return {
        f"mean_{key}": float(np.nanmean([row.get(key, np.nan) for row in rows]))
        for key in numeric_columns
    }


def write_temporal_robustness(
    test_packs: Sequence[Dict[str, Any]],
    scores: Dict[str, np.ndarray],
    threshold: float,
    selected_model: str,
    k_values: Sequence[int],
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    for pack in test_packs:
        y = pack["labels"][pack["sample_idx"]]
        metrics = metric_dict(y, scores[pack["window_name"]], threshold)
        metrics.update(topk_metrics(y, scores[pack["window_name"]], k_values))
        rows.append({
            "row_type": "test_window",
            "model": selected_model,
            "window": pack["window_name"],
            "historical_cutoff": pack["cutoff"],
            "target_end": pack["target_end"],
            "history_transactions": pack["history_transactions"],
            "target_transactions": pack["target_transactions"],
            "scored_accounts": len(pack["sample_idx"]),
            "unseen_target_accounts": pack["unseen_target_accounts"],
            **metrics,
        })
    frame = pd.DataFrame(rows)
    summary = {
        "row_type": "summary",
        "model": selected_model,
        "window": "all_test_windows",
        "mean_pr_auc_average_precision": float(frame["pr_auc_average_precision"].mean()),
        "std_pr_auc_average_precision": float(frame["pr_auc_average_precision"].std(ddof=0)),
        "mean_fpr": float(frame["fpr"].mean()),
        "std_fpr": float(frame["fpr"].std(ddof=0)),
        "fpr_drift_range": float(frame["fpr"].max() - frame["fpr"].min()),
        "mean_precision": float(frame["precision"].mean()),
        "mean_recall": float(frame["recall"].mean()),
    }
    frame = pd.concat([frame, pd.DataFrame([summary])], ignore_index=True, sort=False)
    frame.to_csv(output_dir / "temporal_robustness_v292.csv", index=False)
    return frame, summary


def json_safe(value: Any) -> Any:
    """Convert numpy values and non-finite metrics to strict JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


# ─────────────────────────────────────────────────────────────────────────────
# Argument Parsing and Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_integer_list(value: str, option_name: str) -> List[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{option_name} must be a comma-separated list of integers.") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"{option_name} must contain positive integers.")
    return parsed


def model_comparison_row(
    model: str,
    status: str,
    validation_labels: np.ndarray,
    validation_scores: np.ndarray,
    test_packs: Sequence[Dict[str, Any]],
    test_scores: Dict[str, np.ndarray],
    k_values: Sequence[int],
    target_fpr: float,
    note: str,
) -> Dict[str, Any]:
    threshold = choose_threshold_at_fpr(validation_labels, validation_scores, target_fpr)
    test_summary = average_test_metrics(test_packs, test_scores, threshold, k_values)
    return {
        "model": model,
        "selection_status": status,
        "validation_pr_auc": float(average_precision_score(validation_labels, validation_scores)),
        "validation_threshold_at_target_fpr": threshold,
        **test_summary,
        "note": note,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"MuleNet AI v{VERSION}: leakage-controlled, explainable temporal AML risk ranking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to HI-Small_Trans.csv or the containing folder.",
    )
    parser.add_argument(
        "--output", default="outputs/neural_v292",
        help="Directory for training outputs.",
    )
    parser.add_argument(
        "--max-rows", type=int, default=0,
        help="Optional row cap for smoke tests; 0 reads all rows.",
    )
    parser.add_argument("--seeds", default="42,52,62", help="Comma-separated training seeds.")
    parser.add_argument(
        "--target-fpr", type=float, default=0.01,
        help="Validation-only FPR target for the final alert threshold.",
    )
    parser.add_argument("--top-k", default="50,100,500,1000", help="Review queue size metrics.")
    parser.add_argument("--review-top-n", type=int, default=1000, help="Rows per test window.")
    parser.add_argument(
        "--reason-percentile", type=float, default=0.95,
        help="Training-only percentile for descriptive reason codes.",
    )
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)
    parser.add_argument("--rf-min-validation-ap-gain", type=float, default=0.0005)
    parser.add_argument("--rf-validation-tolerance", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--max-pos-weight", type=float, default=100.0)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default="cpu",
        help="Training device. Defaults to 'cpu'; use 'cuda' when a GPU is available.",
    )
    parser.add_argument(
        "--experiment-graphsage", action="store_true",
        help="Run the excluded GraphSAGE model as a controlled comparison only.",
    )
    parser.add_argument(
        "--experiment-lightgbm", action="store_true",
        help="Run LightGBM as an optional controlled comparison only.",
    )
    parser.add_argument("--lgbm-trees", type=int, default=700)
    parser.add_argument(
        "--experiment-min-validation-ap-gain", type=float, default=0.0005,
        help="Required AP improvement for an opt-in experiment.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if not 0.0 < args.target_fpr < 1.0:
        raise ValueError("--target-fpr must be between 0 and 1.")
    if not 0.0 < args.reason_percentile < 1.0:
        raise ValueError("--reason-percentile must be between 0 and 1.")
    if args.review_top_n <= 0 or args.rf_trees <= 0 or args.epochs <= 0 or args.patience <= 0:
        raise ValueError("Review size, RF trees, epochs, and patience must be positive.")
    seeds = parse_integer_list(args.seeds, "--seeds")
    args.top_k_values = parse_integer_list(args.top_k, "--top-k")
    set_seed(seeds[0])

    if args.device == "cuda" and not torch.cuda.is_available():
        LOG.warning("CUDA requested but unavailable — falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    LOG.info("Device: %s", device)
    if device.type == "cuda":
        LOG.info(
            "GPU: %s | VRAM: %.2f GiB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
        )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = find_dataset(args.data)
    LOG.info("Dataset: %s", csv_path)
    tx = load_transactions(csv_path, None if args.max_rows == 0 else args.max_rows)

    LOG.info("Building rolling temporal snapshots…")
    train_pack, validation_packs, test_packs, feature_names, windows = build_rolling_snapshots(tx)

    # ── Fit preprocessing on training snapshot accounts only ─────────────────
    train_frame = train_pack["features"].iloc[train_pack["sample_idx"]].astype(np.float32)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    imputer.fit(train_frame)
    scaler.fit(imputer.transform(train_frame))
    for pack in [train_pack, *validation_packs, *test_packs]:
        transform_snapshot(pack, feature_names, imputer, scaler)
    joblib.dump(imputer, out_dir / "feature_imputer_v292.joblib")
    joblib.dump(scaler, out_dir / "feature_scaler_v292.joblib")
    joblib.dump(feature_names, out_dir / "feature_names_v292.joblib")
    reason_thresholds = make_reason_thresholds(train_pack, args.reason_percentile)
    with open(out_dir / "explanation_thresholds_v292.json", "w", encoding="utf-8") as fh:
        json.dump(json_safe(reason_thresholds), fh, indent=2)

    # ── Train primary feature-only GNN ────────────────────────────────────────
    LOG.info("Training primary feature-only GNN across seeds=%s", seeds)
    seeded, primary, seed_rows, primary_seed_metadata = fit_feature_only_gnn(
        train_pack, validation_packs, test_packs, seeds, args, device, out_dir, feature_names
    )

    # ── Train validation-gated Random Forest ─────────────────────────────────
    LOG.info("Training validation-gated Random Forest candidate across seeds=%s", seeds)
    rf, rf_seed_metadata = fit_random_forest_candidate(
        train_pack, validation_packs, test_packs, seeds, args, out_dir, feature_names
    )
    weights, rf_decision, rank_maps = validation_gate_random_forest(
        primary["validation"], rf["validation"], validation_packs, args
    )
    final_validation, _ = combine_scores(
        primary["validation"], rf["validation"], rank_maps, weights
    )
    validation_labels = concat_labels(validation_packs)
    final_threshold = choose_threshold_at_fpr(validation_labels, final_validation, args.target_fpr)
    selected_model = rf_decision["selected_model"]

    # ── Compute final combined test scores ────────────────────────────────────
    final_scores: Dict[str, np.ndarray] = {}
    component_scores: Dict[str, Dict[str, np.ndarray]] = {
        "feature_only_gnn": {}, "random_forest": {}
    }
    for pack in test_packs:
        window = pack["window_name"]
        combined, components = combine_scores(
            primary["test"][window], rf["test"][window], rank_maps, weights
        )
        final_scores[window] = combined
        for name, values in components.items():
            component_scores[name][window] = values

    with open(out_dir / "model_selection_v292.json", "w", encoding="utf-8") as fh:
        json.dump(json_safe({
            "pipeline_version": VERSION,
            "default_primary_model": "feature_only_gnn",
            "selected_weights": weights,
            "final_validation_threshold": final_threshold,
            "threshold_source": "pooled negatives from validation_1 and validation_2 only",
            "random_forest_gate": rf_decision,
        }), fh, indent=2)

    # ── Stability and robustness reports ─────────────────────────────────────
    seed_stability = pd.DataFrame(seed_rows)
    seed_stability.to_csv(out_dir / "seed_stability_v292.csv", index=False)
    temporal_frame, temporal_summary = write_temporal_robustness(
        test_packs, final_scores, final_threshold, selected_model, args.top_k_values, out_dir
    )
    explanations, queue = build_explanations_and_queue(
        test_packs, final_scores, component_scores, reason_thresholds,
        selected_model, out_dir, args.review_top_n,
    )

    # ── Model comparison table ────────────────────────────────────────────────
    primary_validation_rank = rank_against_reference(primary["validation"], primary["validation"])
    rf_validation_rank = rank_against_reference(rf["validation"], rf["validation"])
    primary_test_rank = {
        name: rank_against_reference(primary["validation"], values)
        for name, values in primary["test"].items()
    }
    rf_test_rank = {
        name: rank_against_reference(rf["validation"], values)
        for name, values in rf["test"].items()
    }
    comparison = [
        model_comparison_row(
            "feature_only_gnn", "primary_default", validation_labels, primary_validation_rank,
            test_packs, primary_test_rank, args.top_k_values, args.target_fpr,
            "Primary model; identity-adjacency GNN with no cross-account neighbour messages.",
        ),
        model_comparison_row(
            "random_forest",
            "secondary_candidate_retained" if rf_decision["rf_retained"] else "secondary_candidate_excluded",
            validation_labels, rf_validation_rank, test_packs, rf_test_rank, args.top_k_values,
            args.target_fpr, rf_decision["decision_reason"],
        ),
        model_comparison_row(
            selected_model, "selected_default_scorer", validation_labels, final_validation,
            test_packs, final_scores, args.top_k_values, args.target_fpr,
            "Selection, weights, and threshold use validation labels only; "
            "test values are post-selection observations.",
        ),
        {
            "model": "graphsage",
            "selection_status": "excluded_default_not_run",
            "note": "Excluded from default pipeline. Use --experiment-graphsage for a controlled comparison.",
        },
        {
            "model": "lightgbm",
            "selection_status": "excluded_default_not_run",
            "note": "Excluded from default pipeline. Use --experiment-lightgbm after installing LightGBM.",
        },
    ]
    experiments: Dict[str, Any] = {}
    if args.experiment_graphsage:
        LOG.info("Running opt-in GraphSAGE controlled experiment")
        experiment = run_graphsage_experiment(
            train_pack, validation_packs, test_packs, seeds, args, device, out_dir, feature_names
        )
        summary = controlled_experiment_summary(
            experiment, primary["validation"], validation_packs, args
        )
        comparison.append({
            **model_comparison_row(
                "graphsage_experiment", "experiment_only", validation_labels,
                rank_against_reference(experiment["validation"], experiment["validation"]),
                test_packs,
                {
                    name: rank_against_reference(experiment["validation"], values)
                    for name, values in experiment["test"].items()
                },
                args.top_k_values, args.target_fpr, summary["promotion_note"],
            ),
            "consistent_gain_observed": summary["consistent_gain_observed"],
            "pooled_validation_ap_gain_vs_primary": summary["pooled_validation_ap_gain_vs_primary"],
        })
        experiments["graphsage"] = {**summary, "per_seed": experiment["per_seed"]}
    if args.experiment_lightgbm:
        LOG.info("Running opt-in LightGBM controlled experiment")
        experiment = run_lightgbm_experiment(
            train_pack, validation_packs, test_packs, seeds, args, out_dir, feature_names
        )
        summary = controlled_experiment_summary(
            experiment, primary["validation"], validation_packs, args
        )
        comparison.append({
            **model_comparison_row(
                "lightgbm_experiment", "experiment_only", validation_labels,
                rank_against_reference(experiment["validation"], experiment["validation"]),
                test_packs,
                {
                    name: rank_against_reference(experiment["validation"], values)
                    for name, values in experiment["test"].items()
                },
                args.top_k_values, args.target_fpr, summary["promotion_note"],
            ),
            "consistent_gain_observed": summary["consistent_gain_observed"],
            "pooled_validation_ap_gain_vs_primary": summary["pooled_validation_ap_gain_vs_primary"],
        })
        experiments["lightgbm"] = {**summary, "per_seed": experiment["per_seed"]}
    comparison_frame = pd.DataFrame(comparison)
    comparison_frame.to_csv(out_dir / "model_comparison_v292.csv", index=False)

    # ── Final metrics manifest ────────────────────────────────────────────────
    metrics = {
        "pipeline_version": VERSION,
        "result_status": "observed_outputs_from_this_execution; no result values are pre-populated or fabricated",
        "dataset": str(csv_path.resolve()),
        "device": str(device),
        "seeds": seeds,
        "target_fpr": args.target_fpr,
        "windows": windows,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "evaluation_design": (
            "One training, two validation, and three non-overlapping forward test target windows. "
            "Every snapshot's features, explanations, and optional graph edges use only transactions "
            "strictly before its prediction cutoff. Accounts first appearing in a target window are "
            "excluded from scoring."
        ),
        "model_selection": {
            "default_primary": "feature_only_gnn",
            "selected_model": selected_model,
            "weights": weights,
            "validation_threshold": final_threshold,
            "random_forest_gate": rf_decision,
            "lightgbm_and_graphsage": "excluded from default scorer; optional experiment results are separate",
        },
        "primary_seed_runs": primary_seed_metadata,
        "random_forest_seed_runs": rf_seed_metadata,
        "temporal_robustness_summary": temporal_summary,
        "investigator_exports": {
            "queue_rows": int(len(queue)),
            "explanation_rows": int(len(explanations)),
            "scope": f"Top {args.review_top_n} accounts per held-out test window",
        },
        "optional_experiments": experiments,
        "methodology_and_limits": {
            "labels": (
                "Weak account proxy: involvement in a transaction labelled laundering "
                "in the following window; not a verified mule identity."
            ),
            "preprocessing": "Median imputer and scaler fitted only on eligible training-snapshot accounts.",
            "scores": "Validation-ECDF relative ranking scores, not calibrated probabilities.",
            "threshold": "Chosen only from pooled validation-window negatives at the configured target FPR.",
            "limitations": [
                "IBM AML HI-Small is synthetic.",
                "Results do not establish real-world or cross-bank generalisation.",
                "Reason codes are historical behavioral descriptions, not proof of illicit activity.",
                "The batch score is for human prioritisation and requires governance before action.",
            ],
        },
        "artifacts": [
            "metrics_v292.json",
            "test_predictions_by_seed.parquet",
            "temporal_robustness_v292.csv",
            "seed_stability_v292.csv",
            "account_explanations_v292.csv",
            "investigator_review_queue_v292.csv",
            "model_comparison_v292.csv",
            "feature_imputer_v292.joblib",
            "feature_scaler_v292.joblib",
            "feature_names_v292.joblib",
            "explanation_thresholds_v292.json",
            "model_selection_v292.json",
        ],
    }
    with open(out_dir / "metrics_v292.json", "w", encoding="utf-8") as fh:
        json.dump(json_safe(metrics), fh, indent=2)

    LOG.info("MuleNet AI v%s complete. Outputs: %s", VERSION, out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
