import argparse
import csv
import os

import matplotlib.pyplot as plt


PREFERRED_ORDER = [
    "Full",
    "IQL",
    "No-EVT",
    "Lyapunov-Greedy",
    "Non-MEC",
    "Stand-alone MEC",
    "Random",
]


def ordered_policies(rows):
    seen = sorted(list({r["policy"] for r in rows}))
    ordered = [p for p in PREFERRED_ORDER if p in seen]
    for p in seen:
        if p not in ordered:
            ordered.append(p)
    return ordered


def read_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if any(v is None for v in row.values()):
                continue
            rows.append(row)
    return rows


def to_float(rows, keys):
    for r in rows:
        for k in keys:
            if k in r:
                r[k] = float(r[k])


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def plot_stress_scan(summary_rows, out_dir):
    to_float(summary_rows, [
        "value",
        "p_c5_viol_mean", "p_c5_viol_std",
        "p_c6_viol_mean", "p_c6_viol_std",
        "p_c7_viol_mean", "p_c7_viol_std",
        "p_c8_viol_mean", "p_c8_viol_std",
        "mean_c5_excess_ratio_mean", "mean_c5_excess_ratio_std",
        "total_energy_mean", "total_energy_std",
        "mean_offload_ratio_mean", "mean_offload_ratio_std",
    ])

    scenarios = ["load_scale", "f_scale", "user_num"]
    metric_specs = [
        ("total_energy_mean", "total_energy_std", "Total Energy", "Total Energy"),
        ("p_c5_viol_mean", "p_c5_viol_std", "C5 Viol.", "P(C5 viol.)"),
        ("p_c6_viol_mean", "p_c6_viol_std", "C6 Viol.", "P(C6 viol.)"),
        ("p_c7_viol_mean", "p_c7_viol_std", "C7 Viol.", "P(C7 viol.)"),
        ("p_c8_viol_mean", "p_c8_viol_std", "C8 Viol.", "P(C8 viol.)"),
        ("mean_c5_excess_ratio_mean", "mean_c5_excess_ratio_std", "C5 Excess Ratio", "Mean C5 excess ratio"),
    ]

    name_map = {
        "load_scale": "Load Scale",
        "f_scale": "MEC CPU Scale",
        "user_num": "Number of UEs",
    }

    for scenario in scenarios:
        rows = [r for r in summary_rows if r["scenario"] == scenario]
        if not rows:
            continue
        policies = ordered_policies(rows)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for idx, (mean_key, std_key, title, ylab) in enumerate(metric_specs):
            ax = axes[idx]
            for p in policies:
                sub = sorted([r for r in rows if r["policy"] == p], key=lambda x: x["value"])
                if not sub:
                    continue
                x = [r["value"] for r in sub]
                y = [r[mean_key] for r in sub]
                yerr = [r[std_key] for r in sub]
                # Draw Full last so overlap does not hide it.
                z = 4 if p != "Full" else 10
                lw = 1.5 if p != "Full" else 2.6
                alpha = 0.85 if p != "Full" else 1.0
                markerfacecolor = None if p != "Full" else "white"
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    marker="o",
                    capsize=3,
                    linewidth=lw,
                    alpha=alpha,
                    label=p,
                    zorder=z,
                    markerfacecolor=markerfacecolor,
                )
            ax.set_title(title)
            ax.set_xlabel(name_map.get(scenario, scenario))
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.25)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=max(len(policies), 1), frameon=False)
        fig.suptitle(f"New Simulation: Stress Scan ({name_map.get(scenario, scenario)})", fontsize=13)
        plt.tight_layout(rect=(0.01, 0.06, 1.0, 0.95))
        out_path = os.path.join(out_dir, f"stress_scan_{scenario}.png")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)


def plot_weight_sweep(summary_rows, out_dir):
    to_float(summary_rows, [
        "weight",
        "p_c5_viol_mean", "p_c6_viol_mean", "p_c7_viol_mean", "p_c8_viol_mean",
        "mean_c5_excess_ratio_mean",
        "total_energy_mean", "mean_offload_ratio_mean",
    ])

    rows = sorted(summary_rows, key=lambda x: x["weight"])
    if not rows:
        return

    x = [r["weight"] for r in rows]
    p5 = [r["p_c5_viol_mean"] for r in rows]
    p6 = [r["p_c6_viol_mean"] for r in rows]
    p7 = [r["p_c7_viol_mean"] for r in rows]
    p8 = [r["p_c8_viol_mean"] for r in rows]
    c5r = [r["mean_c5_excess_ratio_mean"] for r in rows]
    en = [r["total_energy_mean"] for r in rows]
    off = [r["mean_offload_ratio_mean"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(x, p5, marker="o", label="C5")
    axes[0].plot(x, p6, marker="o", label="C6")
    axes[0].plot(x, p7, marker="o", label="C7")
    axes[0].plot(x, p8, marker="o", label="C8")
    axes[0].set_xlabel("Constraint Weight")
    axes[0].set_ylabel("Violation Probability")
    axes[0].set_title("Constraint Weight vs Violation")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    ax2 = axes[1]
    ax2.plot(x, en, marker="o", color="tab:red", label="Total Energy")
    ax2.set_xlabel("Constraint Weight")
    ax2.set_ylabel("Total Energy", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.grid(True, alpha=0.25)
    ax2.set_title("Constraint Weight Trade-off")

    ax3 = ax2.twinx()
    ax3.plot(x, off, marker="s", color="tab:blue", label="Offload Ratio")
    ax3.plot(x, c5r, marker="^", color="tab:green", label="C5 Excess")
    ax3.set_ylabel("Offload Ratio", color="tab:blue")
    ax3.tick_params(axis="y", labelcolor="tab:blue")

    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="best")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "weight_sweep_tradeoff.png"), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="new_simulation/results")
    parser.add_argument("--figures_dir", type=str, default="new_simulation/figures")
    args = parser.parse_args()

    ensure_dir(args.figures_dir)

    stress_summary = read_csv(os.path.join(args.results_dir, "stress_scan_summary.csv"))
    weight_summary = read_csv(os.path.join(args.results_dir, "weight_sweep_summary.csv"))

    plot_stress_scan(stress_summary, args.figures_dir)
    plot_weight_sweep(weight_summary, args.figures_dir)


if __name__ == "__main__":
    main()
