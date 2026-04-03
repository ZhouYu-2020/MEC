import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def parse_policies(text: str) -> List[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def plot_energy_errorbar(df: pd.DataFrame, scenario: str, x_label: str, out_path: str, policies: List[str]) -> None:
    subset = df[df["scenario"] == scenario].copy()
    if subset.empty:
        raise ValueError(f"No data for scenario '{scenario}'")

    subset = subset[subset["policy"].isin(policies)]
    if subset.empty:
        raise ValueError("No data after policy filtering")

    group = (
        subset.groupby(["value", "policy"], as_index=False)
        .agg(total_energy_mean=("total_energy", "mean"), total_energy_std=("total_energy", "std"))
    )

    x_values = sorted(group["value"].unique())
    plt.figure(figsize=(8.5, 4.6))
    for policy in policies:
        policy_rows = group[group["policy"] == policy]
        if policy_rows.empty:
            continue
        y = []
        yerr = []
        for x in x_values:
            row = policy_rows[policy_rows["value"] == x]
            if row.empty:
                y.append(float("nan"))
                yerr.append(0.0)
            else:
                y.append(float(row["total_energy_mean"].iloc[0]))
                yerr.append(float(row["total_energy_std"].iloc[0] if pd.notna(row["total_energy_std"].iloc[0]) else 0.0))
        plt.errorbar(x_values, y, yerr=yerr, marker="o", capsize=3, linewidth=1.6, label=policy)

    plt.xlabel(x_label)
    plt.ylabel("总能耗")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", ncol=1, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to stress_scan_raw.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory for figures")
    parser.add_argument(
        "--policies",
        default="Full,IQL,No-EVT,C6-Only,Non-MEC,Stand-alone MEC,Random,Lyapunov-Greedy",
        help="Comma-separated list of policies to plot",
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    policies = parse_policies(args.policies)

    df = pd.read_csv(args.input)
    if "total_energy" not in df.columns:
        raise ValueError("Column 'total_energy' not found in input CSV")

    user_out = os.path.join(args.out_dir, "total_energy_vs_user_num_errorbar.png")
    plot_energy_errorbar(df, "user_num", "设备数", user_out, policies)

    fscale_out = os.path.join(args.out_dir, "total_energy_vs_f_scale_errorbar.png")
    plot_energy_errorbar(df, "f_scale", "计算资源缩放系数", fscale_out, policies)


if __name__ == "__main__":
    main()
