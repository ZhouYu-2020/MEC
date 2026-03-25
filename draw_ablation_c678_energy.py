import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def read_summary(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["value"] = float(row["value"])
            row["p_c6_viol_mean"] = float(row["p_c6_viol_mean"])
            row["p_c6_viol_std"] = float(row["p_c6_viol_std"])
            row["p_c7_viol_mean"] = float(row["p_c7_viol_mean"])
            row["p_c7_viol_std"] = float(row["p_c7_viol_std"])
            row["p_c8_viol_mean"] = float(row["p_c8_viol_mean"])
            row["p_c8_viol_std"] = float(row["p_c8_viol_std"])
            row["total_energy_mean"] = float(row["total_energy_mean"])
            row["total_energy_std"] = float(row["total_energy_std"])
            rows.append(row)
    return rows


def plot_one_scenario(rows, scenario, out_file):
    rows = [r for r in rows if r["scenario"] == scenario]
    if not rows:
        return

    ablations = ["Full", "-VQ", "-GPD", "-Lyapunov", "-Priority", "Random"]
    metrics = [
        ("total_energy_mean", "total_energy_std", "Total Energy", "Total Energy"),
        ("p_c6_viol_mean", "p_c6_viol_std", "C6 Violation Probability", "P(C6 violation)"),
        ("p_c7_viol_mean", "p_c7_viol_std", "C7 Violation Probability", "P(C7 violation)"),
        ("p_c8_viol_mean", "p_c8_viol_std", "C8 Violation Probability", "P(C8 violation)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    scenario_name = {
        "load_scale": "Input Load Scale",
        "f_scale": "MEC CPU Scale",
        "user_num": "Number of UEs",
    }.get(scenario, scenario)

    for m_idx, (mean_key, std_key, title, ylab) in enumerate(metrics):
        ax = axes[m_idx]
        for ab in ablations:
            sub = [r for r in rows if r["ablation"] == ab]
            if not sub:
                continue
            sub = sorted(sub, key=lambda x: x["value"])
            x = [r["value"] for r in sub]
            y = [r[mean_key] for r in sub]
            yerr = [r[std_key] for r in sub]
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.5, label=ab)

        ax.set_title(title)
        ax.set_xlabel(scenario_name)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    fig.suptitle(f"Ablation Study under {scenario_name}", fontsize=13)
    plt.tight_layout(rect=(0.02, 0.05, 1.0, 0.96))
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="picture/reliability/ablation_results.csv")
    parser.add_argument("--out_dir", type=str, default="picture/reliability/ablation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = read_summary(args.input)
    for scenario in ["load_scale", "f_scale", "user_num"]:
        out_file = os.path.join(args.out_dir, f"ablation_{scenario}.png")
        plot_one_scenario(rows, scenario, out_file)


if __name__ == "__main__":
    main()
