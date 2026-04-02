import argparse
import os
import sys
from statistics import mean

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from new_simulation.sim_core import EnvConfig, SimConfig, get_policy_variants, run_policy_once


def parse_list(text: str):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--warmup_ratio", type=float, default=0.7)
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--bw_list", type=str, default="1e7,2e7,5e7,1e8,2e8")
    p.add_argument("--f_list", type=str, default="1e10,2e10,5e10,1e11")
    p.add_argument("--target", type=float, default=0.2)
    args = p.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    bw_list = parse_list(args.bw_list)
    f_list = parse_list(args.f_list)

    full = [x for x in get_policy_variants() if x["name"] == "Full"][0]
    env = EnvConfig(user_num=40, load_scale=1.0, bw_scale=1.0, f_scale=1.0, tmax_scale=1.0)

    best = None
    for bw in bw_list:
        for f in f_list:
            sim = SimConfig(
                episodes=args.episodes,
                warmup_ratio=args.warmup_ratio,
                gpd_update_interval=100,
                gpd_min_episode_for_update=150,
                lyap_constraint_weight=20.0,
                lyap_energy_weight=1.0,
                lyap_constraint_scale=1e24,
                lyap_energy_scale=1e25,
                wolf_epsilon=0.1,
                c5_penalty_weight=20.0,
                c5_penalty_scale=1.0,
                mec_bw_base=bw,
                mec_f_base=f,
            )

            rows = [run_policy_once(full, env, sim, sd, gpd_estimator=None) for sd in seeds]
            p6 = mean(r["p_c6_viol"] for r in rows)
            p7 = mean(r["p_c7_viol"] for r in rows)
            p8 = mean(r["p_c8_viol"] for r in rows)
            score = max(p6, p7, p8)
            print(f"bw={bw:.3e}, f={f:.3e}, p6={p6:.3f}, p7={p7:.3f}, p8={p8:.3f}, max={score:.3f}")

            if best is None or score < best["score"]:
                best = {"bw": bw, "f": f, "p6": p6, "p7": p7, "p8": p8, "score": score}

    print("BEST:", best)
    if best["score"] <= args.target:
        print("TARGET_MET")
    else:
        print("TARGET_NOT_MET")


if __name__ == "__main__":
    main()
