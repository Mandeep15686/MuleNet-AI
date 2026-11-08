"""
MuleNet AI — Investigation Dashboard Builder v2
"""
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

def df_json(df):
    return json.dumps(
        df.replace([np.inf,-np.inf],None).where(pd.notnull(df),None).to_dict(orient="records"),
        default=str)

def build(args):
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading data…")
    rings    = pd.read_csv(Path(args.louvain)/"rings.csv")
    members  = pd.read_csv(Path(args.louvain)/"ring_members.csv")
    edges    = pd.read_csv(Path(args.louvain)/"ring_edges.csv")
    dossiers = pd.read_csv(Path(args.explain)/"improved_dossiers.csv")
    ev_path  = Path(args.explain)/"supporting_evidence.csv"
    evidence = pd.read_csv(ev_path) if ev_path.exists() else pd.DataFrame()

    top      = rings.head(args.top_rings).copy()
    top_ids  = set(top["ring_id"]); top_cids = set(top["community_id"])
    mem_all  = members[members["ring_id"].isin(top_ids)][
        ["ring_id","account_id","is_flagged","risk_score","label","ring_rank"]].copy()
    flagged_accs = set(mem_all[mem_all["is_flagged"]==True]["account_id"])
    # All rings get edges; large rings capped at top-500 by flow
    edges_all   = edges[edges["community_id"].isin(top_cids)].copy()
    large_cids  = set(top[top["ring_size"] > 150]["community_id"])
    edges_small_part = edges_all[~edges_all["community_id"].isin(large_cids)]
    edges_large_part = (edges_all[edges_all["community_id"].isin(large_cids)]
                        .sort_values("total_amount", ascending=False)
                        .groupby("community_id", group_keys=False)
                        .head(500))
    edges_small = pd.concat([edges_small_part, edges_large_part], ignore_index=True)
    ev_filt = evidence[evidence["account_id"].isin(flagged_accs)].copy() \
        if not evidence.empty else pd.DataFrame()
    dos_cols = [c for c in [
        "account_id","risk_score","total_out_count","total_in_count",
        "total_out_amount","total_in_amount","unique_out_ctpy","unique_in_ctpy",
        "rapid_event_count","median_dwell_min","n_reason_codes",
        "reason_1","reason_2","reason_3","reason_4","reason_5",
        "reason_1_description","reason_2_description","reason_3_description"
    ] if c in dossiers.columns]
    dos_filt = dossiers[dossiers["account_id"].isin(flagged_accs)][dos_cols].copy()

    buckets = []
    if "risk_score" in dossiers.columns:
        sc = pd.to_numeric(dossiers["risk_score"], errors="coerce").dropna()
        for lo in [i/10 for i in range(10)]:
            buckets.append({"lo":round(lo,1),"hi":round(lo+0.1,1),
                            "count":int(((sc>=lo)&(sc<lo+0.1)).sum())})
    stats = {
        "total_flagged":     int(dossiers["account_id"].nunique()),
        "total_rings":       int(len(rings)),
        "accounts_in_rings": int(members[members["is_flagged"]==True]["account_id"].nunique()),
        "confirmed_mules":   int(members[members["label"]==1]["account_id"].nunique()),
        "typology_counts":   rings["typology"].value_counts().to_dict(),
        "score_buckets":     buckets,
        "total_flow_top10":  float(rings.head(10)["total_flow"].sum()),
    }
    print(f"  rings:{len(top)} members:{len(mem_all)} edges:{len(edges_small)} "
          f"dossiers:{len(dos_filt)} evidence:{len(ev_filt)}")
    html = _TEMPLATE \
        .replace("__RINGS__",   df_json(top)) \
        .replace("__MEMBERS__", df_json(mem_all)) \
        .replace("__EDGES__",   df_json(edges_small)) \
        .replace("__DOS__",     df_json(dos_filt)) \
        .replace("__EV__",      df_json(ev_filt) if not ev_filt.empty else "[]") \
        .replace("__STATS__",   json.dumps(stats))
    out = out_dir/"mulenet_dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard written: {out}  ({out.stat().st_size//1024} KB)")

_TEMPLATE = open(Path(__file__).parent.parent / "dashboard" / "_dash_template.html", encoding="utf-8").read()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explain", required=True)
    ap.add_argument("--louvain", required=True)
    ap.add_argument("--output",  required=True)
    ap.add_argument("--top-rings",       type=int, default=30)
    ap.add_argument("--max-graph-nodes", type=int, default=300)
    args = ap.parse_args()
    build(args)

if __name__ == "__main__":
    main()
