#!/usr/bin/env python3
"""
Plot C6-C8 violation rates from stress_scan_summary.csv.

Generates two figures by default:
1) scenario = user_num
2) scenario = f_scale

Supports two plot modes:
- line
- bar (grouped bar)
- both
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLS = [
    "scenario",
    "value",
    "policy",
    "p_c6_viol_mean",
    "p_c7_viol_mean",
    "p_c8_viol_mean",
]

TARGET_SCENARIOS = ["user_num", "f_scale"]
METRIC_KEYS = ["p_c6_viol_mean", "p_c7_viol_mean", "p_c8_viol_mean"]
METRIC_TITLES = ["C6 violation rate", "C7 violation rate", "C8 violation rate"]


def _find_columns(fieldnames: List[str]) -> Dict[str, str]:
    normalized = {name.strip().lower(): name for name in fieldnames}
    mapping: Dict[str, str] = {}
    missing = []
    for req in REQUIRED_COLS:
        key = req.lower()
        if key not in normalized:
            missing.append(req)
        else:
            mapping[req] = normalized[key]
    if missing:
        found = ", ".join(fieldnames)
        need = ", ".join(missing)
        raise ValueError(f"Missing required columns: {need}. Found columns: {found}")
    return mapping


def load_and_aggregate(
    csv_path: str,
    dedup_mode: str = "mean",
) -> Dict[Tuple[str, float, str], Dict[str, float]]:
    """Aggregate repeated (scenario, value, policy) rows.

    dedup_mode:
    - mean: average all duplicates
    - first: keep first row per key
    - last: keep last row per key
    """
    rows_by_key: Dict[Tuple[str, float, str], List[Dict[str, float]]] = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        col = _find_columns(reader.fieldnames)

        for row in reader:
            try:
                scenario = row[col["scenario"]].strip()
                if scenario not in TARGET_SCENARIOS:
                    continue

                policy = row[col["policy"]].strip()
                value = float(row[col["value"]])
                c6 = float(row[col["p_c6_viol_mean"]])
                c7 = float(row[col["p_c7_viol_mean"]])
                c8 = float(row[col["p_c8_viol_mean"]])
            except Exception:
                continue

            key = (scenario, value, policy)
            rows_by_key[key].append(
                {
                    "p_c6_viol_mean": c6,
                    "p_c7_viol_mean": c7,
                    "p_c8_viol_mean": c8,
                }
            )

    if not rows_by_key:
        raise ValueError("No valid rows found for scenario in {user_num, f_scale}.")

    dup_count = sum(1 for v in rows_by_key.values() if len(v) > 1)
    if dup_count > 0:
        print(f"[warn] found {dup_count} duplicated (scenario,value,policy) groups; dedup_mode={dedup_mode}")

    agg_mean: Dict[Tuple[str, float, str], Dict[str, float]] = {}
    for key, vals in rows_by_key.items():
        if dedup_mode == "first":
            agg_mean[key] = vals[0]
        elif dedup_mode == "last":
            agg_mean[key] = vals[-1]
        else:
            n = float(len(vals))
            agg_mean[key] = {
                "p_c6_viol_mean": sum(x["p_c6_viol_mean"] for x in vals) / n,
                "p_c7_viol_mean": sum(x["p_c7_viol_mean"] for x in vals) / n,
                "p_c8_viol_mean": sum(x["p_c8_viol_mean"] for x in vals) / n,
            }
    return agg_mean


def save_used_data(
    agg_mean: Dict[Tuple[str, float, str], Dict[str, float]],
    out_csv: str,
) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows = []
    for (scenario, value, policy), m in agg_mean.items():
        rows.append(
            {
                "scenario": scenario,
                "value": value,
                "policy": policy,
                "p_c6_viol_mean": m["p_c6_viol_mean"],
                "p_c7_viol_mean": m["p_c7_viol_mean"],
                "p_c8_viol_mean": m["p_c8_viol_mean"],
            }
        )
    rows.sort(key=lambda r: (r["scenario"], float(r["value"]), r["policy"]))

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "value",
                "policy",
                "p_c6_viol_mean",
                "p_c7_viol_mean",
                "p_c8_viol_mean",
            ],
        )
        w.writeheader()
        w.writerows(rows)


def _scenario_pack(
    agg_mean: Dict[Tuple[str, float, str], Dict[str, float]],
    scenario: str,
    policy_filter: List[str] | None,
):
    values = sorted({k[1] for k in agg_mean.keys() if k[0] == scenario})
    policies = sorted({k[2] for k in agg_mean.keys() if k[0] == scenario})

    if policy_filter:
        keep = set(policy_filter)
        policies = [p for p in policies if p in keep]

    if not values or not policies:
        raise ValueError(f"No data left for scenario={scenario} after filtering.")

    return values, policies


def plot_line(
    agg_mean: Dict[Tuple[str, float, str], Dict[str, float]],
    scenario: str,
    out_path: str,
    policy_filter: List[str] | None,
) -> None:
    values, policies = _scenario_pack(agg_mean, scenario, policy_filter)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(policies), 3)))
    markers = ["o", "s", "d", "^", "v", ">", "<", "p", "h", "x", "+"]

    for m_idx, (metric, title) in enumerate(zip(METRIC_KEYS, METRIC_TITLES)):
        ax = axes[m_idx]
        for p_idx, policy in enumerate(policies):
            y = []
            for v in values:
                key = (scenario, v, policy)
                y.append(agg_mean.get(key, {}).get(metric, np.nan))
            ax.plot(
                values,
                y,
                linewidth=1.8,
                marker=markers[p_idx % len(markers)],
                markersize=5.5,
                color=colors[p_idx],
                label=policy,
            )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(scenario)
        ax.set_ylabel("Violation rate")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(policies)), frameon=False)
    fig.suptitle(f"C6-C8 violations vs {scenario} (line)", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_bar(
    agg_mean: Dict[Tuple[str, float, str], Dict[str, float]],
    scenario: str,
    out_path: str,
    policy_filter: List[str] | None,
) -> None:
    values, policies = _scenario_pack(agg_mean, scenario, policy_filter)

    x = np.arange(len(values), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(policies), 3)))

    width = 0.82 / max(len(policies), 1)

    for m_idx, (metric, title) in enumerate(zip(METRIC_KEYS, METRIC_TITLES)):
        ax = axes[m_idx]
        for p_idx, policy in enumerate(policies):
            y = []
            for v in values:
                key = (scenario, v, policy)
                y.append(agg_mean.get(key, {}).get(metric, np.nan))

            offset = (p_idx - (len(policies) - 1) / 2.0) * width
            ax.bar(x + offset, y, width=width, color=colors[p_idx], label=policy)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(scenario)
        ax.set_ylabel("Violation rate")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in values])
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(policies)), frameon=False)
    fig.suptitle(f"C6-C8 violations vs {scenario} (grouped bar)", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(script_dir, "results_paper_c5", "stress_scan_summary.csv")
    default_out = os.path.join(script_dir, "figures_paper_c5")

    p = argparse.ArgumentParser(description="Plot C6-C8 violation rates for user_num and f_scale")
    p.add_argument("--input", default=default_csv, help="Path to stress_scan_summary.csv")
    p.add_argument("--out-dir", default=default_out, help="Output directory for figures")
    p.add_argument(
        "--mode",
        default="line",
        choices=["line", "bar", "both"],
        help="Plot style mode",
    )
    p.add_argument(
        "--policies",
        default="",
        help="Optional comma-separated policy names to keep, e.g. Full,IQL,No-EVT",
    )
    p.add_argument(
        "--dedup-mode",
        default="mean",
        choices=["mean", "first", "last"],
        help="How to handle duplicated (scenario,value,policy) rows",
    )
    p.add_argument(
        "--save-used-csv",
        default="",
        help="Optional path to save the exact aggregated data used for plotting",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    os.makedirs(args.out_dir, exist_ok=True)

    policy_filter = [x.strip() for x in args.policies.split(",") if x.strip()] or None

    agg_mean = load_and_aggregate(args.input, dedup_mode=args.dedup_mode)

    used_csv = args.save_used_csv.strip() or os.path.join(args.out_dir, "c6_c8_used_data.csv")
    save_used_data(agg_mean, used_csv)
    print(f"Saved: {used_csv}")

    for scenario in TARGET_SCENARIOS:
        if args.mode in ("line", "both"):
            out_line = os.path.join(args.out_dir, f"c6_c8_vs_{scenario}_line.png")
            plot_line(agg_mean, scenario, out_line, policy_filter)
            print(f"Saved: {out_line}")

        if args.mode in ("bar", "both"):
            out_bar = os.path.join(args.out_dir, f"c6_c8_vs_{scenario}_bar.png")
            plot_grouped_bar(agg_mean, scenario, out_bar, policy_filter)
            print(f"Saved: {out_bar}")

    print("Done.")


if __name__ == "__main__":
    main()
