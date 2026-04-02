#!/usr/bin/env python3
"""
Audit stress_scan_raw.csv and plot C6-C8 with error bars (mean ± std over seeds).
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

REQ = ["scenario", "value", "policy", "seed", "p_c6_viol", "p_c7_viol", "p_c8_viol", "p_c6_exceed_ratio"]
SCENARIOS = ["user_num", "f_scale"]
METRICS = ["p_c6_viol", "p_c7_viol", "p_c8_viol"]
TITLES = ["C6 violation rate", "C7 violation rate", "C8 violation rate"]


def _map_cols(fieldnames: List[str]) -> Dict[str, str]:
    norm = {x.strip().lower(): x for x in fieldnames}
    out = {}
    miss = []
    for k in REQ:
        if k not in norm:
            miss.append(k)
        else:
            out[k] = norm[k]
    if miss:
        raise ValueError(f"Missing columns: {', '.join(miss)}")
    return out


def read_raw(path: str):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames is None:
            raise ValueError("empty csv header")
        col = _map_cols(rd.fieldnames)

        for r in rd:
            try:
                sc = r[col["scenario"]].strip()
                if sc not in SCENARIOS:
                    continue
                rows.append(
                    {
                        "scenario": sc,
                        "value": float(r[col["value"]]),
                        "policy": r[col["policy"]].strip(),
                        "seed": int(float(r[col["seed"]])),
                        "p_c6_viol": float(r[col["p_c6_viol"]]),
                        "p_c6_exceed_ratio": float(r[col["p_c6_exceed_ratio"]]),
                        "p_c7_viol": float(r[col["p_c7_viol"]]),
                        "p_c8_viol": float(r[col["p_c8_viol"]]),
                    }
                )
            except Exception:
                continue
    return rows


def dedup_seed_rows(rows, mode: str):
    by = defaultdict(list)
    for r in rows:
        by[(r["scenario"], r["value"], r["policy"], r["seed"])].append(r)

    conflict_rows = []
    dedup = []
    for k, rs in by.items():
        if len(rs) == 1:
            dedup.append(rs[0])
            continue

        # record conflicts for audit
        c6_set = sorted(set(x["p_c6_viol"] for x in rs))
        c6e_set = sorted(set(x["p_c6_exceed_ratio"] for x in rs))
        c7_set = sorted(set(x["p_c7_viol"] for x in rs))
        c8_set = sorted(set(x["p_c8_viol"] for x in rs))
        if len(c6_set) > 1 or len(c6e_set) > 1 or len(c7_set) > 1 or len(c8_set) > 1:
            conflict_rows.append(
                {
                    "scenario": k[0],
                    "value": k[1],
                    "policy": k[2],
                    "seed": k[3],
                    "dup_count": len(rs),
                    "c6_values": "|".join(f"{x:g}" for x in c6_set),
                    "c6_exceed_values": "|".join(f"{x:g}" for x in c6e_set),
                    "c7_values": "|".join(f"{x:g}" for x in c7_set),
                    "c8_values": "|".join(f"{x:g}" for x in c8_set),
                }
            )

        if mode == "first":
            dedup.append(rs[0])
        elif mode == "last":
            dedup.append(rs[-1])
        elif mode == "mean":
            dedup.append(
                {
                    "scenario": k[0],
                    "value": k[1],
                    "policy": k[2],
                    "seed": k[3],
                    "p_c6_viol": mean(x["p_c6_viol"] for x in rs),
                    "p_c6_exceed_ratio": mean(x["p_c6_exceed_ratio"] for x in rs),
                    "p_c7_viol": mean(x["p_c7_viol"] for x in rs),
                    "p_c8_viol": mean(x["p_c8_viol"] for x in rs),
                }
            )
        else:
            raise ValueError("invalid dedup mode")

    return dedup, conflict_rows, len(by)


def aggregate_seed_stats(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["scenario"], r["value"], r["policy"])].append(r)

    out = []
    for k, rs in by.items():
        out.append(
            {
                "scenario": k[0],
                "value": k[1],
                "policy": k[2],
                "n_seed": len(rs),
                "p_c6_viol_mean": mean(x["p_c6_viol"] for x in rs),
                "p_c6_viol_std": pstdev(x["p_c6_viol"] for x in rs) if len(rs) > 1 else 0.0,
                "p_c6_exceed_ratio_mean": mean(x["p_c6_exceed_ratio"] for x in rs),
                "p_c6_exceed_ratio_std": pstdev(x["p_c6_exceed_ratio"] for x in rs) if len(rs) > 1 else 0.0,
                "p_c7_viol_mean": mean(x["p_c7_viol"] for x in rs),
                "p_c7_viol_std": pstdev(x["p_c7_viol"] for x in rs) if len(rs) > 1 else 0.0,
                "p_c8_viol_mean": mean(x["p_c8_viol"] for x in rs),
                "p_c8_viol_std": pstdev(x["p_c8_viol"] for x in rs) if len(rs) > 1 else 0.0,
            }
        )
    out.sort(key=lambda r: (r["scenario"], r["value"], r["policy"]))
    return out


def save_csv(path: str, rows: List[dict], fields: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_errorbar(stats_rows: List[dict], scenario: str, out_path: str, keep_policies: List[str] | None):
    S = [r for r in stats_rows if r["scenario"] == scenario]
    values = sorted({r["value"] for r in S})
    policies = sorted({r["policy"] for r in S})
    if keep_policies:
        keep = set(keep_policies)
        policies = [p for p in policies if p in keep]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(policies), 3)))
    markers = ["o", "s", "d", "^", "v", ">", "<", "p", "h", "x", "+"]

    for i, title in enumerate(TITLES):
        ax = axes[i]
        m_mean = f"{METRICS[i]}_mean"
        m_std = f"{METRICS[i]}_std"
        for p_idx, p in enumerate(policies):
            subset = [r for r in S if r["policy"] == p]
            lookup = {r["value"]: r for r in subset}
            y = [lookup[v][m_mean] if v in lookup else np.nan for v in values]
            e = [lookup[v][m_std] if v in lookup else np.nan for v in values]
            ax.errorbar(
                values,
                y,
                yerr=e,
                linewidth=1.8,
                elinewidth=1.1,
                capsize=3,
                marker=markers[p_idx % len(markers)],
                markersize=5.5,
                color=colors[p_idx],
                label=p,
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(scenario)
        ax.set_ylabel("Violation rate")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(4, len(policies)), frameon=False)
    fig.suptitle(f"C6-C8 vs {scenario} (mean ± std over seeds)", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_c6_dual(stats_rows: List[dict], scenario: str, out_path: str, keep_policies: List[str] | None):
    S = [r for r in stats_rows if r["scenario"] == scenario]
    values = sorted({r["value"] for r in S})
    policies = sorted({r["policy"] for r in S})
    if keep_policies:
        keep = set(keep_policies)
        policies = [p for p in policies if p in keep]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(policies), 3)))
    markers = ["o", "s", "d", "^", "v", ">", "<", "p", "h", "x", "+"]

    metric_pairs = [
        ("p_c6_viol_mean", "p_c6_viol_std", "C6 overflow probability (new)"),
        ("p_c6_exceed_ratio_mean", "p_c6_exceed_ratio_std", "C6 exceed-ratio (legacy)"),
    ]

    for i, (m_mean, m_std, ttl) in enumerate(metric_pairs):
        ax = axes[i]
        for p_idx, p in enumerate(policies):
            subset = [r for r in S if r["policy"] == p]
            lookup = {r["value"]: r for r in subset}
            y = [lookup[v][m_mean] if v in lookup else np.nan for v in values]
            e = [lookup[v][m_std] if v in lookup else np.nan for v in values]
            ax.errorbar(
                values,
                y,
                yerr=e,
                linewidth=1.8,
                elinewidth=1.1,
                capsize=3,
                marker=markers[p_idx % len(markers)],
                markersize=5.5,
                color=colors[p_idx],
                label=p,
            )
        ax.set_title(ttl, fontsize=11, fontweight="bold")
        ax.set_xlabel(scenario)
        ax.set_ylabel("Rate")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(4, len(policies)), frameon=False)
    fig.suptitle(f"C6 metrics vs {scenario} (mean ± std over seeds)", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    base = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=os.path.join(base, "results_paper_c5", "stress_scan_raw.csv"))
    p.add_argument("--out-dir", default=os.path.join(base, "figures_paper_c5"))
    p.add_argument("--dedup-mode", default="last", choices=["first", "last", "mean"])
    p.add_argument("--policies", default="", help="comma-separated filter")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = read_raw(args.input)
    if not rows:
        raise RuntimeError("No valid rows from raw csv")

    dedup_rows, conflicts, group_count = dedup_seed_rows(rows, args.dedup_mode)
    print(f"raw groups(scenario,value,policy,seed): {group_count}")
    print(f"conflicting duplicate groups: {len(conflicts)}")

    conflict_csv = os.path.join(args.out_dir, "stress_scan_conflict_groups.csv")
    save_csv(
        conflict_csv,
        conflicts,
        ["scenario", "value", "policy", "seed", "dup_count", "c6_values", "c6_exceed_values", "c7_values", "c8_values"],
    )
    print(f"Saved: {conflict_csv}")

    stats_rows = aggregate_seed_stats(dedup_rows)
    stats_csv = os.path.join(args.out_dir, f"c6_c8_seed_stats_{args.dedup_mode}.csv")
    save_csv(
        stats_csv,
        stats_rows,
        [
            "scenario", "value", "policy", "n_seed",
            "p_c6_viol_mean", "p_c6_viol_std",
            "p_c6_exceed_ratio_mean", "p_c6_exceed_ratio_std",
            "p_c7_viol_mean", "p_c7_viol_std",
            "p_c8_viol_mean", "p_c8_viol_std",
        ],
    )
    print(f"Saved: {stats_csv}")

    keep = [x.strip() for x in args.policies.split(",") if x.strip()] or None
    for sc in SCENARIOS:
        out = os.path.join(args.out_dir, f"c6_c8_vs_{sc}_errorbar_{args.dedup_mode}.png")
        plot_errorbar(stats_rows, sc, out, keep)
        print(f"Saved: {out}")

        out_c6 = os.path.join(args.out_dir, f"c6_dual_vs_{sc}_errorbar_{args.dedup_mode}.png")
        plot_c6_dual(stats_rows, sc, out_c6, keep)
        print(f"Saved: {out_c6}")


if __name__ == "__main__":
    main()
