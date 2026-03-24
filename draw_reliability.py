import argparse
import csv
import os

import matplotlib.pyplot as plt


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def load_rows(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="picture/reliability/reliability_results.csv")
    parser.add_argument("--out_dir", type=str, default="picture/reliability")
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        raise SystemExit("no rows found in input csv")

    ensure_dir(args.out_dir)

    scenarios = sorted({r["scenario"] for r in rows})

    for scenario in scenarios:
        data = [r for r in rows if r["scenario"] == scenario]
        data.sort(key=lambda r: float(r["value"]))

        x = [float(r["value"]) for r in data]
        p_c5_viol = [float(r["p_c5_viol"]) for r in data]
        p_c6_viol = [float(r["p_c6_viol"]) for r in data]
        p_c7_m1_viol = [float(r["p_c7_m1_viol"]) for r in data]
        p_c7_m2_viol = [float(r["p_c7_m2_viol"]) for r in data]

        plt.figure()
        plt.plot(x, p_c5_viol, marker="o", label="C5 violation probability")
        plt.plot(x, p_c6_viol, marker="s", label="C6 violation probability")
        plt.plot(x, p_c7_m1_viol, marker="^", label="C7 violation probability")
        plt.plot(x, p_c7_m2_viol, marker="d", label="C8 violation probability")
        plt.xlabel(scenario)
        plt.ylabel("violation probability")
        plt.title("Constraint Violation Probability vs " + scenario)
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.legend()

        out_path = os.path.join(args.out_dir, f"reliability_{scenario}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
