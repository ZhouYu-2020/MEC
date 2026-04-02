import argparse
import csv
import os
import sys
from statistics import mean, pstdev

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from new_simulation.sim_core import EnvConfig, SimConfig, get_policy_variants, run_policy_once


def ensure_dir(path: str):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def guard_output_files(paths, overwrite: bool):
    existing = [p for p in paths if os.path.exists(p)]
    if existing and not overwrite:
        msg = "\n".join(existing)
        raise FileExistsError(
            "Output files already exist. Use a new --out_dir or enable --overwrite_outputs:\n" + msg
        )


def parse_seeds(seed_text: str):
    return [int(x.strip()) for x in seed_text.split(",") if x.strip()]


def aggregate(rows, keys):
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows]
        out[k + "_mean"] = float(mean(vals))
        out[k + "_std"] = float(pstdev(vals)) if len(vals) > 1 else 0.0
    return out


def run_stress_scan(sim: SimConfig, seeds, out_raw, out_summary, use_gpd, overwrite=False):
    scenarios = [
        ("load_scale", [0.5, 1.0, 1.5, 2.0]),
        ("f_scale", [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]),
        ("user_num", [5, 10, 15, 20, 25, 30, 35, 40]),
    ]

    policies = get_policy_variants()

    metric_keys = [
        "p_c5_viol", "p_c6_viol", "p_c6_exceed_ratio", "p_c7_viol", "p_c8_viol",
        "mean_c5_excess_ratio",
        "mean_overflow_prob", "mean_m1_excess_cond", "mean_m2_excess_cond",
        "mean_offload_ratio", "total_energy", "avg_energy_per_slot_user",
    ] + [f"action_{i}_ratio" for i in range(8)]

    gpd_estimator = None
    if use_gpd:
        try:
            from gpd import GPD
            gpd_estimator = GPD()
        except Exception as exc:
            print("[WARN] GPD disabled due to init failure:", exc)

    guard_output_files([out_raw, out_summary], overwrite=overwrite)

    with open(out_raw, "w", newline="") as f_raw, open(out_summary, "w", newline="") as f_sum:
        raw_fields = [
            "experiment", "scenario", "value", "policy", "seed",
            "user_num", "load_scale", "bw_scale", "f_scale", "tmax_scale",
            "episodes", "warmup_ratio",
        ] + metric_keys
        sum_fields = [
            "experiment", "scenario", "value", "policy", "n_seed",
            "user_num", "load_scale", "bw_scale", "f_scale", "tmax_scale",
            "episodes", "warmup_ratio",
        ] + [x for k in metric_keys for x in (k + "_mean", k + "_std")]

        raw_writer = csv.DictWriter(f_raw, fieldnames=raw_fields)
        sum_writer = csv.DictWriter(f_sum, fieldnames=sum_fields)
        raw_writer.writeheader()
        sum_writer.writeheader()
        f_raw.flush()
        f_sum.flush()

        for scenario, values in scenarios:
            for v in values:
                user_num = 15
                load_scale = 1.0
                bw_scale = 1.0
                f_scale = 1.0
                tmax_scale = 1.0

                if scenario == "user_num":
                    user_num = int(v)
                elif scenario == "load_scale":
                    load_scale = float(v)
                elif scenario == "f_scale":
                    f_scale = float(v)

                env = EnvConfig(
                    user_num=user_num,
                    load_scale=load_scale,
                    bw_scale=bw_scale,
                    f_scale=f_scale,
                    tmax_scale=tmax_scale,
                )

                for policy in policies:
                    seed_rows = []
                    for seed in seeds:
                        print(
                            f"[RUN] scenario={scenario} value={v} policy={policy['name']} seed={seed}",
                            flush=True,
                        )
                        res = run_policy_once(policy, env, sim, seed, gpd_estimator=gpd_estimator)
                        row = {
                            "experiment": "stress_scan",
                            "scenario": scenario,
                            "value": v,
                            "policy": policy["name"],
                            "seed": seed,
                            "user_num": user_num,
                            "load_scale": load_scale,
                            "bw_scale": bw_scale,
                            "f_scale": f_scale,
                            "tmax_scale": tmax_scale,
                            "episodes": sim.episodes,
                            "warmup_ratio": sim.warmup_ratio,
                        }
                        row.update(res)
                        raw_writer.writerow(row)
                        f_raw.flush()
                        seed_rows.append(row)

                    agg = aggregate(seed_rows, metric_keys)
                    sum_row = {
                        "experiment": "stress_scan",
                        "scenario": scenario,
                        "value": v,
                        "policy": policy["name"],
                        "n_seed": len(seeds),
                        "user_num": user_num,
                        "load_scale": load_scale,
                        "bw_scale": bw_scale,
                        "f_scale": f_scale,
                        "tmax_scale": tmax_scale,
                        "episodes": sim.episodes,
                        "warmup_ratio": sim.warmup_ratio,
                    }
                    sum_row.update(agg)
                    sum_writer.writerow(sum_row)
                    f_sum.flush()


def run_weight_sweep(sim: SimConfig, seeds, out_raw, out_summary, use_gpd, overwrite=False):
    weights = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
    policy = [x for x in get_policy_variants() if x["name"] == "Full"][0]

    metric_keys = [
        "p_c5_viol", "p_c6_viol", "p_c6_exceed_ratio", "p_c7_viol", "p_c8_viol",
        "mean_c5_excess_ratio",
        "mean_overflow_prob", "mean_m1_excess_cond", "mean_m2_excess_cond",
        "mean_offload_ratio", "total_energy", "avg_energy_per_slot_user",
    ] + [f"action_{i}_ratio" for i in range(8)]

    gpd_estimator = None
    if use_gpd:
        try:
            from gpd import GPD
            gpd_estimator = GPD()
        except Exception as exc:
            print("[WARN] GPD disabled due to init failure:", exc)

    guard_output_files([out_raw, out_summary], overwrite=overwrite)

    with open(out_raw, "w", newline="") as f_raw, open(out_summary, "w", newline="") as f_sum:
        raw_fields = [
            "experiment", "weight", "policy", "seed",
            "user_num", "load_scale", "bw_scale", "f_scale", "tmax_scale",
            "episodes", "warmup_ratio", "lyap_constraint_weight",
        ] + metric_keys
        sum_fields = [
            "experiment", "weight", "policy", "n_seed",
            "user_num", "load_scale", "bw_scale", "f_scale", "tmax_scale",
            "episodes", "warmup_ratio", "lyap_constraint_weight",
        ] + [x for k in metric_keys for x in (k + "_mean", k + "_std")]

        raw_writer = csv.DictWriter(f_raw, fieldnames=raw_fields)
        sum_writer = csv.DictWriter(f_sum, fieldnames=sum_fields)
        raw_writer.writeheader()
        sum_writer.writeheader()

        for w in weights:
            sim_local = SimConfig(
                episodes=sim.episodes,
                warmup_ratio=sim.warmup_ratio,
                gpd_update_interval=sim.gpd_update_interval,
                gpd_min_episode_for_update=sim.gpd_min_episode_for_update,
                lyap_constraint_weight=w,
                lyap_energy_weight=sim.lyap_energy_weight,
                lyap_constraint_scale=sim.lyap_constraint_scale,
                lyap_energy_scale=sim.lyap_energy_scale,
                wolf_epsilon=sim.wolf_epsilon,
                c5_penalty_weight=sim.c5_penalty_weight,
                c5_penalty_scale=sim.c5_penalty_scale,
                mec_bw_base=sim.mec_bw_base,
                mec_f_base=sim.mec_f_base,
            )
            env = EnvConfig(user_num=15, load_scale=1.0, bw_scale=1.0, f_scale=1.0, tmax_scale=1.0)

            seed_rows = []
            for seed in seeds:
                res = run_policy_once(policy, env, sim_local, seed, gpd_estimator=gpd_estimator)
                row = {
                    "experiment": "weight_sweep",
                    "weight": w,
                    "policy": policy["name"],
                    "seed": seed,
                    "user_num": env.user_num,
                    "load_scale": env.load_scale,
                    "bw_scale": env.bw_scale,
                    "f_scale": env.f_scale,
                    "tmax_scale": env.tmax_scale,
                    "episodes": sim_local.episodes,
                    "warmup_ratio": sim_local.warmup_ratio,
                    "lyap_constraint_weight": w,
                }
                row.update(res)
                raw_writer.writerow(row)
                seed_rows.append(row)

            agg = aggregate(seed_rows, metric_keys)
            sum_row = {
                "experiment": "weight_sweep",
                "weight": w,
                "policy": policy["name"],
                "n_seed": len(seeds),
                "user_num": env.user_num,
                "load_scale": env.load_scale,
                "bw_scale": env.bw_scale,
                "f_scale": env.f_scale,
                "tmax_scale": env.tmax_scale,
                "episodes": sim_local.episodes,
                "warmup_ratio": sim_local.warmup_ratio,
                "lyap_constraint_weight": w,
            }
            sum_row.update(agg)
            sum_writer.writerow(sum_row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--warmup_ratio", type=float, default=0.6)
    parser.add_argument("--seeds", type=str, default="1,2")
    parser.add_argument("--gpd_update_interval", type=int, default=100)
    parser.add_argument("--gpd_min_episode_for_update", type=int, default=150)
    parser.add_argument("--lyap_constraint_weight", type=float, default=20.0)
    parser.add_argument("--lyap_energy_weight", type=float, default=1.0)
    parser.add_argument("--lyap_constraint_scale", type=float, default=1e24)
    parser.add_argument("--lyap_energy_scale", type=float, default=1e25)
    parser.add_argument("--wolf_epsilon", type=float, default=0.1)
    parser.add_argument("--c5_penalty_weight", type=float, default=20.0)
    parser.add_argument("--c5_penalty_scale", type=float, default=1.0)
    parser.add_argument("--mec_bw_base", type=float, default=10e6)
    parser.add_argument("--mec_f_base", type=float, default=10e9)
    parser.add_argument("--enable_gpd", action="store_true")
    parser.add_argument("--overwrite_outputs", action="store_true")
    parser.add_argument("--out_dir", type=str, default="new_simulation/results")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    sim = SimConfig(
        episodes=args.episodes,
        warmup_ratio=args.warmup_ratio,
        gpd_update_interval=args.gpd_update_interval,
        gpd_min_episode_for_update=args.gpd_min_episode_for_update,
        lyap_constraint_weight=args.lyap_constraint_weight,
        lyap_energy_weight=args.lyap_energy_weight,
        lyap_constraint_scale=args.lyap_constraint_scale,
        lyap_energy_scale=args.lyap_energy_scale,
        wolf_epsilon=args.wolf_epsilon,
        c5_penalty_weight=args.c5_penalty_weight,
        c5_penalty_scale=args.c5_penalty_scale,
        mec_bw_base=args.mec_bw_base,
        mec_f_base=args.mec_f_base,
    )
    seeds = parse_seeds(args.seeds)

    try:
        run_stress_scan(
            sim=sim,
            seeds=seeds,
            out_raw=os.path.join(args.out_dir, "stress_scan_raw.csv"),
            out_summary=os.path.join(args.out_dir, "stress_scan_summary.csv"),
            use_gpd=args.enable_gpd,
            overwrite=args.overwrite_outputs,
        )
        run_weight_sweep(
            sim=sim,
            seeds=seeds,
            out_raw=os.path.join(args.out_dir, "weight_sweep_raw.csv"),
            out_summary=os.path.join(args.out_dir, "weight_sweep_summary.csv"),
            use_gpd=args.enable_gpd,
            overwrite=args.overwrite_outputs,
        )
    except Exception as exc:
        print(f"[ERROR] experiment failed: {exc}", flush=True)
        raise


if __name__ == "__main__":
    main()
