"""
MuleNet AI — Louvain Ring Detection v1
=======================================
Groups flagged accounts into mule-network rings using community
detection on the transaction subgraph, then scores each ring.

Method
------
1. Load transactions + V2.9.2 predictions
2. Build a directed transaction graph on ALL accounts (nodes) and
   transactions (edges), weighted by total amount transferred
3. Extract the subgraph induced by the top-N flagged accounts
   plus their 1-hop neighbours (to capture ring structure)
4. Run Louvain community detection on the undirected projection
5. Filter to communities with ≥ 2 flagged members
6. Score each ring:
     ring_score  = mean GNN score of flagged members
     ring_size   = total nodes in community
     flag_density = flagged / total nodes in community
     total_flow  = sum of edge weights inside the community
7. Assign typology labels: LAYERING_RING, FAN_OUT_HUB,
   RAPID_TRANSFER_CHAIN, MIXED

Usage
-----
python mulenet_louvain_v1.py ^
    --data        ".\\data\\HI-Small_Trans.csv"                         ^
    --predictions ".\\outputs_neural_v292\\test_predictions_by_seed.parquet" ^
    --dossiers    ".\\outputs_explain_v1\\improved_dossiers.csv"        ^
    --output      ".\\outputs_louvain_v1"                               ^
    --top-n       1000                                                  ^
    --min-ring-size 2

Outputs
-------
outputs_louvain_v1/
  rings.csv                  — one row per ring, sorted by ring_score desc
  ring_members.csv           — one row per account per ring
  ring_edges.csv             — edges inside each ring (for visualisation)
  run_summary.json
"""

import argparse, json, logging
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

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
    raise KeyError(f"None of {candidates} in {list(cols)}")


def load_transactions(path: Path) -> pd.DataFrame:
    log.info("Loading transactions: %s", path)
    df = pd.read_csv(path, low_memory=False)
    c  = df.columns.tolist()
    tx = df.rename(columns={
        _first(c, ["From Bank",          "from_bank"]):           "from_bank",
        _first(c, ["Account",            "from_account"]):        "from_account",
        _first(c, ["To Bank",            "to_bank"]):             "to_bank",
        _first(c, ["Account.1",          "to_account"]):          "to_account",
        _first(c, ["Amount Paid",        "amount_paid", "amount"]):"amount_paid",
        _first(c, ["Payment Currency",   "payment_currency",
                   "currency"]):                                  "payment_currency",
        _first(c, ["Timestamp",          "timestamp", "Date"]):   "timestamp",
    }).copy()
    tx["src_id"] = tx["from_bank"].astype(str) + "::" + tx["from_account"].astype(str)
    tx["dst_id"] = tx["to_bank"].astype(str) + "::" + tx["to_account"].astype(str)
    tx["amount_paid"] = pd.to_numeric(tx["amount_paid"], errors="coerce").fillna(0.0)
    log.info("Loaded %d transactions", len(tx))
    return tx


def load_predictions(path: Path) -> pd.DataFrame:
    log.info("Loading predictions: %s", path)
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    for cand in ["score", "risk_score", "prob"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "score"}); break
    for cand in ["snapshot", "window", "period"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "snapshot"}); break
    return df


def remap_account_ids(pred_ids: set, tx_ids: set) -> dict:
    """Return mapping pred_id → tx_id if formats differ."""
    if pred_ids & tx_ids:
        return {}   # no remap needed

    # try :: → _ separator swap
    alt = {fid.replace("_", "::"): fid for fid in tx_ids}
    hits = pred_ids & set(alt.keys())
    if hits:
        log.info("Auto-remap :: → _ for %d accounts", len(hits))
        return {k: alt[k] for k in hits}

    # try bare account number → BANK_ACCOUNT
    raw_map = {}
    for fid in tx_ids:
        parts = fid.split("_", 1)
        if len(parts) == 2:
            raw_map.setdefault(parts[1], fid)
    hits = pred_ids & set(raw_map.keys())
    if hits:
        log.info("Auto-remap bare → BANK_ACCOUNT for %d accounts", len(hits))
        return {k: raw_map[k] for k in hits}

    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_edge_list(tx: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions to weighted directed edges."""
    log.info("Building edge list…")
    edges = (tx.groupby(["src_id", "dst_id"], observed=True)
               .agg(total_amount=("amount_paid", "sum"),
                    n_transactions=("amount_paid", "count"))
               .reset_index())
    log.info("  %d unique directed edges", len(edges))
    return edges


def extract_subgraph(
    edges: pd.DataFrame,
    flagged: set,
    include_neighbours: bool = True,
) -> pd.DataFrame:
    """
    Return edges where at least one endpoint is flagged.
    Optionally include edges between 1-hop neighbours of flagged accounts.
    """
    log.info("Extracting subgraph (flagged=%d, neighbours=%s)…",
             len(flagged), include_neighbours)

    # edges touching at least one flagged node
    mask = edges["src_id"].isin(flagged) | edges["dst_id"].isin(flagged)
    sub  = edges[mask].copy()

    if include_neighbours:
        # 1-hop neighbours
        neighbours = (set(sub["src_id"].unique()) |
                      set(sub["dst_id"].unique())) - flagged
        # edges between neighbours (second-order connections)
        mask2 = (edges["src_id"].isin(neighbours) &
                 edges["dst_id"].isin(neighbours))
        sub = pd.concat([sub, edges[mask2]], ignore_index=True).drop_duplicates()

    nodes = set(sub["src_id"].unique()) | set(sub["dst_id"].unique())
    log.info("  Subgraph: %d nodes, %d edges", len(nodes), len(sub))
    return sub


# ─────────────────────────────────────────────────────────────────────────────
# Louvain community detection  (pure Python — no external graph library needed)
# ─────────────────────────────────────────────────────────────────────────────

def louvain_communities(edges: pd.DataFrame, seed: int = 42) -> dict:
    """
    Run Louvain on the UNDIRECTED projection of the edge list.
    Falls back to networkx.community.louvain_communities if available,
    otherwise uses python-louvain (community package).
    Returns {node_id: community_id}.
    """
    # Build undirected weighted adjacency
    # Merge parallel edges A→B and B→A into one weight
    pairs = edges[["src_id", "dst_id", "total_amount"]].copy()
    pairs["u"] = pairs[["src_id", "dst_id"]].min(axis=1)
    pairs["v"] = pairs[["src_id", "dst_id"]].max(axis=1)
    undirected = (pairs.groupby(["u", "v"], observed=True)["total_amount"]
                       .sum().reset_index())

    all_nodes = list(set(undirected["u"].unique()) | set(undirected["v"].unique()))
    node_idx  = {n: i for i, n in enumerate(all_nodes)}
    n         = len(all_nodes)

    try:
        import networkx as nx
        log.info("Using networkx Louvain (seed=%d)…", seed)
        G = nx.Graph()
        G.add_nodes_from(all_nodes)
        for _, row in undirected.iterrows():
            G.add_edge(row["u"], row["v"], weight=float(row["total_amount"]))
        comms = nx.community.louvain_communities(G, seed=seed, weight="weight")
        mapping = {}
        for cid, comm in enumerate(comms):
            for node in comm:
                mapping[node] = cid
        log.info("  Louvain found %d communities", len(comms))
        return mapping

    except (ImportError, AttributeError):
        pass

    try:
        import community as community_louvain   # python-louvain package
        import networkx as nx
        log.info("Using python-louvain…")
        G = nx.Graph()
        G.add_nodes_from(all_nodes)
        for _, row in undirected.iterrows():
            G.add_edge(row["u"], row["v"], weight=float(row["total_amount"]))
        mapping = community_louvain.best_partition(G, weight="weight",
                                                   random_state=seed)
        n_comms = len(set(mapping.values()))
        log.info("  Louvain found %d communities", n_comms)
        return mapping

    except ImportError:
        pass

    # ── fallback: greedy label propagation (no external deps) ────────────────
    log.warning("networkx Louvain not available — using label propagation fallback")
    return _label_propagation(undirected, all_nodes, seed)


def _label_propagation(undirected, all_nodes, seed):
    """Greedy label propagation — O(E log E), deterministic."""
    rng      = np.random.default_rng(seed)
    labels   = {n: i for i, n in enumerate(all_nodes)}

    # adjacency dict: node → [(neighbour, weight)]
    adj = defaultdict(list)
    for _, row in undirected.iterrows():
        adj[row["u"]].append((row["v"], row["total_amount"]))
        adj[row["v"]].append((row["u"], row["total_amount"]))

    for _ in range(20):   # max 20 passes
        changed = False
        order   = rng.permutation(len(all_nodes))
        for i in order:
            node = all_nodes[i]
            if not adj[node]:
                continue
            # choose label of heaviest-weighted neighbour group
            votes = defaultdict(float)
            for nb, w in adj[node]:
                votes[labels[nb]] += w
            best = max(votes, key=votes.get)
            if best != labels[node]:
                labels[node] = best
                changed = True
        if not changed:
            break

    # re-index community ids
    unique_labels = {v: i for i, v in enumerate(sorted(set(labels.values())))}
    return {n: unique_labels[l] for n, l in labels.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Ring scoring & typology
# ─────────────────────────────────────────────────────────────────────────────

def assign_typology(row: pd.Series) -> str:
    """
    Heuristic typology based on ring structure metrics.
    LAYERING_RING     : dense flow cycling through many accounts
    FAN_OUT_HUB       : one hub sending to many flagged members
    RAPID_TRANSFER_CHAIN : linear chain of rapid transfers
    MIXED             : doesn't fit a single pattern
    """
    if row["flag_density"] > 0.6 and row["ring_size"] >= 3:
        return "LAYERING_RING"
    if row["max_out_degree"] > 3 * row["avg_out_degree"] + 1:
        return "FAN_OUT_HUB"
    if row["flag_density"] > 0.4 and row["avg_out_degree"] < 1.5:
        return "RAPID_TRANSFER_CHAIN"
    return "MIXED"


def score_rings(
    community_map: dict,
    flagged_scores: dict,
    flagged_labels: dict,
    sub_edges: pd.DataFrame,
    min_ring_flagged: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (rings_df, members_df, edges_df).
    """
    log.info("Scoring rings…")

    # group nodes by community
    comm_nodes = defaultdict(set)
    for node, cid in community_map.items():
        comm_nodes[cid].add(node)

    # per-community edge stats
    out_degrees = defaultdict(lambda: defaultdict(int))
    comm_flow   = defaultdict(float)
    comm_edges  = []

    for _, row in sub_edges.iterrows():
        src, dst, amt = row["src_id"], row["dst_id"], row["total_amount"]
        cid_src = community_map.get(src)
        cid_dst = community_map.get(dst)
        if cid_src is not None and cid_src == cid_dst:
            comm_flow[cid_src]   += amt
            out_degrees[cid_src][src] += 1
            comm_edges.append({
                "community_id": cid_src,
                "src_id": src, "dst_id": dst,
                "total_amount": amt,
                "n_transactions": row.get("n_transactions", 1),
            })

    ring_rows   = []
    member_rows = []
    ring_id     = 0

    for cid, nodes in comm_nodes.items():
        flagged_members = {n for n in nodes if n in flagged_scores}
        if len(flagged_members) < min_ring_flagged:
            continue

        scores  = [flagged_scores[n] for n in flagged_members]
        labels  = [flagged_labels.get(n, 0) for n in flagged_members]
        od      = out_degrees[cid]
        avg_od  = float(np.mean(list(od.values()))) if od else 0.0
        max_od  = float(max(od.values())) if od else 0.0

        ring_row = {
            "ring_id":         ring_id,
            "community_id":    cid,
            "ring_size":       len(nodes),
            "n_flagged":       len(flagged_members),
            "flag_density":    round(len(flagged_members) / len(nodes), 4),
            "n_confirmed_mules": int(sum(labels)),
            "ring_score":      round(float(np.mean(scores)), 5),
            "max_score":       round(float(np.max(scores)), 5),
            "min_score":       round(float(np.min(scores)), 5),
            "total_flow":      round(comm_flow[cid], 2),
            "avg_out_degree":  round(avg_od, 3),
            "max_out_degree":  round(max_od, 1),
        }
        ring_row["typology"] = assign_typology(pd.Series(ring_row))
        ring_rows.append(ring_row)

        for node in nodes:
            member_rows.append({
                "ring_id":     ring_id,
                "account_id":  node,
                "is_flagged":  node in flagged_members,
                "risk_score":  round(flagged_scores.get(node, 0.0), 5),
                "label":       int(flagged_labels.get(node, 0)),
            })

        ring_id += 1

    rings_df   = pd.DataFrame(ring_rows).sort_values("ring_score", ascending=False).reset_index(drop=True)
    rings_df["ring_rank"] = range(1, len(rings_df) + 1)

    members_df = pd.DataFrame(member_rows)
    edges_df   = pd.DataFrame(comm_edges)

    # attach ring_rank to members and edges
    rank_map = dict(zip(rings_df["ring_id"], rings_df["ring_rank"]))
    if not members_df.empty:
        members_df["ring_rank"] = members_df["ring_id"].map(rank_map)
    if not edges_df.empty:
        edges_df["ring_rank"] = edges_df["community_id"].map(
            dict(zip(rings_df["community_id"], rings_df["ring_rank"]))
        )

    log.info("  %d rings found (%d communities had ≥ %d flagged members)",
             len(rings_df), len(rings_df), min_ring_flagged)
    return rings_df, members_df, edges_df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",           required=True)
    ap.add_argument("--predictions",    required=True)
    ap.add_argument("--dossiers",       default=None,
                    help="improved_dossiers.csv from mulenet_explain_v1 (optional)")
    ap.add_argument("--output",         required=True)
    ap.add_argument("--top-n",          type=int, default=1000)
    ap.add_argument("--min-ring-size",  type=int, default=2,
                    help="Minimum flagged members for a ring to be reported")
    ap.add_argument("--louvain-seed",   type=int, default=42)
    ap.add_argument("--snapshot",       default=None,
                    help="Restrict to one snapshot (test1/test2/test3). Default: use all.")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load ─────────────────────────────────────────────────────────────────
    tx   = load_transactions(Path(args.data))
    pred = load_predictions(Path(args.predictions))

    # filter snapshot
    if args.snapshot:
        pred = pred[pred["snapshot"] == args.snapshot]
        log.info("Filtered to snapshot: %s (%d rows)", args.snapshot, len(pred))

    # average scores across seeds
    avg = (pred.groupby("account_id", observed=True, as_index=False)
               .agg(score=("score", "mean"), label=("label", "first")))

    # select top-N by score
    avg = avg.sort_values("score", ascending=False)
    top = avg.head(args.top_n)
    flagged_raw = set(top["account_id"].unique())

    # ── account_id format reconciliation ─────────────────────────────────────
    tx_ids  = set(tx["src_id"].unique()) | set(tx["dst_id"].unique())
    remap   = remap_account_ids(flagged_raw, tx_ids)
    if remap:
        top["account_id"] = top["account_id"].map(lambda x: remap.get(x, x))
        avg["account_id"] = avg["account_id"].map(lambda x: remap.get(x, x))
    flagged = set(top["account_id"].unique())

    flagged_scores = dict(zip(top["account_id"], top["score"]))
    flagged_labels = dict(zip(top["account_id"], top["label"]))

    log.info("Flagged accounts: %d  |  positives in top-%d: %d",
             len(flagged), args.top_n, int(top["label"].sum()))

    # ── merge dossier reason codes if available ───────────────────────────────
    dossier_map = {}
    if args.dossiers and Path(args.dossiers).exists():
        dos = pd.read_csv(args.dossiers)
        dossier_map = dict(zip(dos["account_id"],
                               dos.get("reason_1", pd.Series(dtype=str))))
        log.info("Loaded %d dossier reason codes", len(dossier_map))

    # ── build graph ───────────────────────────────────────────────────────────
    edges    = build_edge_list(tx)
    sub_edges = extract_subgraph(edges, flagged, include_neighbours=True)

    # ── Louvain ───────────────────────────────────────────────────────────────
    comm_map = louvain_communities(sub_edges, seed=args.louvain_seed)

    # ── score rings ──────────────────────────────────────────────────────────
    rings, members, ring_edges = score_rings(
        comm_map, flagged_scores, flagged_labels,
        sub_edges, min_ring_flagged=args.min_ring_size,
    )

    if rings.empty:
        log.warning("No rings found with ≥ %d flagged members.", args.min_ring_size)
    else:
        log.info("Top 5 rings:")
        for _, r in rings.head(5).iterrows():
            log.info("  Ring #%d | %s | size=%d flagged=%d score=%.4f flow=%.0f",
                     r["ring_rank"], r["typology"], r["ring_size"],
                     r["n_flagged"], r["ring_score"], r["total_flow"])

    # attach dossier reason codes to members
    if dossier_map and not members.empty:
        members["primary_reason"] = members["account_id"].map(dossier_map).fillna("")

    # ── save ─────────────────────────────────────────────────────────────────
    rings.to_csv(out_dir / "rings.csv", index=False)
    members.to_csv(out_dir / "ring_members.csv", index=False)
    ring_edges.to_csv(out_dir / "ring_edges.csv", index=False)
    log.info("rings.csv: %d rings | ring_members.csv: %d rows | ring_edges.csv: %d rows",
             len(rings), len(members), len(ring_edges))

    # ── summary ───────────────────────────────────────────────────────────────
    typology_counts = rings["typology"].value_counts().to_dict() if not rings.empty else {}
    summary = {
        "script":           "mulenet_louvain_v1.py",
        "run_utc":          datetime.now(timezone.utc).isoformat(),
        "transactions":     int(len(tx)),
        "flagged_accounts": int(len(flagged)),
        "subgraph_nodes":   int(len(set(sub_edges["src_id"]) | set(sub_edges["dst_id"]))) if not sub_edges.empty else 0,
        "subgraph_edges":   int(len(sub_edges)),
        "total_communities": int(len(set(comm_map.values()))),
        "rings_reported":   int(len(rings)),
        "min_ring_size":    args.min_ring_size,
        "louvain_seed":     args.louvain_seed,
        "typology_counts":  typology_counts,
        "top_10_rings": rings.head(10)[
            ["ring_rank","typology","ring_size","n_flagged",
             "flag_density","ring_score","total_flow"]
        ].to_dict(orient="records") if not rings.empty else [],
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("All outputs → %s", out_dir.resolve())
    log.info("Done.")


if __name__ == "__main__":
    main()
