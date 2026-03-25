import argparse
import csv
import os

import matplotlib.pyplot as plt


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def load_rows(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="picture/reliability/strategy_compare_results.csv")
    parser.add_argument("--out_dir", type=str, default="picture/reliability/strategy_compare")
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        raise SystemExit("no rows found in input csv")

    ensure_dir(args.out_dir)

    scenarios = sorted({r["scenario"] for r in rows})
    policies = ["main", "non_mec", "mec_only"]
    policy_label = {
        "main": "Main",
        "non_mec": "non-MEC",
        "mec_only": "Stand-only MEC",
    }

    for scenario in scenarios:
        data_s = [r for r in rows if r["scenario"] == scenario]

        x_values = sorted({float(r["value"]) for r in data_s})

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        metrics = [
            ("total_energy", "Total Energy"),
            ("p_c6_viol", "C6 Violation Probability"),
            ("p_c7_viol", "C7 Violation Probability"),
            ("p_c8_viol", "C8 Violation Probability"),
        ]

        for ax, (metric_key, metric_title) in zip(axes.flatten(), metrics):
            for p in policies:
                data_p = [r for r in data_s if r["policy"] == p]
                data_p.sort(key=lambda r: float(r["value"]))
                x = [float(r["value"]) for r in data_p]
                y = [float(r[metric_key]) for r in data_p]
                ax.plot(x, y, marker="o", label=policy_label[p])

            ax.set_title(metric_title)
            ax.set_xlabel(scenario)
            ax.grid(True, linestyle=":", linewidth=0.5)

        axes[0, 1].legend()
        fig.suptitle("Strategy Comparison vs " + scenario)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

        out_path = os.path.join(args.out_dir, f"strategy_compare_{scenario}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
