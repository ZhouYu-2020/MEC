import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


ABLATIONS = ["Full", "-VQ", "-GPD", "-Lyapunov", "-Priority", "Random"]
SCENARIOS = ["load_scale", "f_scale", "user_num"]
ACTION_LABELS = [f"A{i}" for i in range(8)]


def read_rows(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["value"] = float(row["value"])
            for i in range(8):
                row[f"action_{i}_ratio_mean"] = float(row[f"action_{i}_ratio_mean"])
            rows.append(row)
    return rows


def build_matrix(rows, scenario, ablation):
    sub = [r for r in rows if r["scenario"] == scenario and r["ablation"] == ablation]
    if not sub:
        return None, None
    sub.sort(key=lambda x: x["value"])

    values = [r["value"] for r in sub]
    mat = np.zeros((8, len(values)), dtype=float)
    for c, r in enumerate(sub):
        for i in range(8):
            mat[i, c] = r[f"action_{i}_ratio_mean"]
    return values, mat


def draw_heatmap(rows, scenario, out_path):
    scenario_name = {
        "load_scale": "Load Scale",
        "f_scale": "MEC CPU Scale",
        "user_num": "Number of UEs",
    }.get(scenario, scenario)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=True)
    axes = axes.flatten()

    vmin = 0.0
    vmax = 1.0
    im = None

    for idx, ab in enumerate(ABLATIONS):
        ax = axes[idx]
        values, mat = build_matrix(rows, scenario, ab)
        if mat is None or values is None:
            ax.set_axis_off()
            continue

        im = ax.imshow(mat, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="YlOrRd")
        ax.set_title(ab)
        ax.set_yticks(np.arange(8))
        ax.set_yticklabels(ACTION_LABELS)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels([str(v) for v in values], rotation=45, ha="right", fontsize=8)

        if idx % 3 == 0:
            ax.set_ylabel("Action ID")
        if idx >= 3:
            ax.set_xlabel(scenario_name)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.92)
        cbar.set_label("Action Selection Ratio")

    fig.suptitle(f"Action Selection Heatmap by Ablation ({scenario_name})", fontsize=13)
    plt.tight_layout(rect=(0.01, 0.02, 1.0, 0.95))
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="picture/reliability/ablation_results_weighted.csv")
    parser.add_argument("--out_dir", type=str, default="picture/reliability/ablation_action_heatmap")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_rows(args.input)

    for scenario in SCENARIOS:
        out_path = os.path.join(args.out_dir, f"action_heatmap_{scenario}.png")
        draw_heatmap(rows, scenario, out_path)


if __name__ == "__main__":
    main()
