import argparse
import csv
import os
import random

import numpy as np

from matrix_game import MatrixGame
from queue_relay import QueueRelay
from wolf_agent import WoLFAgent


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def build_lambda(user_num, load_scale=1.0):
    base = np.zeros(user_num)
    for i in range(user_num):
        if i % 5 == 0:
            base[i] = 0.001
        if i % 5 == 1:
            base[i] = 0.01
        if i % 5 == 2:
            base[i] = 0.1
        if i % 5 == 3:
            base[i] = 0.001
        if i % 5 == 4:
            base[i] = 0.01
    return base * load_scale


def valid_gpd_params(gpd1, gpd2):
    if not np.isfinite(gpd1) or not np.isfinite(gpd2):
        return False
    if gpd1 <= 0:
        return False
    if gpd2 <= -0.9 or gpd2 >= 0.49:
        return False
    return True


def compute_energy(game, actions, rf):
    energy = 0.0
    theta_pr = 0.0
    pr_q = 0.0

    for i in range(game.num_ue):
        theta_pr += game.action_space[actions[i]][0] * game.pr_n[i]
        pr_q += game.pr_n[i] * (1 if game.Q[i] > 0 else 0)

    for i in range(game.num_ue):
        action = actions[i]
        if action < 4:
            f_local = game.action_space[action][1]
            energy += game.kmob * game.tau * (f_local ** 3)
            continue

        tx_power = game.action_space[action][2]
        rw = (game.BW * game.pr_n[i]) / theta_pr if theta_pr > 0 else 0.0
        vn = 0.0
        if rw > 0:
            snr_term = 1 + (game.g0 * tx_power / (game.N0 * rw)) - pow(10, -5)
            if snr_term > 0:
                vn = rw * np.log2(snr_term)
        e_tx = tx_power * game.bn[i] / vn if vn > 0 and np.isfinite(vn) else 1e6

        rf_i = 0.0
        if pr_q > 0:
            rf_i = game.F * (game.pr_n[i] * (1 if game.Q[i] > 0 else 0)) / pr_q
        if i < len(rf):
            rf_i = rf[i]

        e_ser = game.kser * rf_i * rf_i * (game.bn[i] * game.dn[i])
        energy += e_tx + e_ser

    return float(energy)


def get_ablation_configs():
    return [
        {
            "name": "Full",
            "use_virtual_queue": True,
            "use_gpd_online": True,
            "use_lyapunov": True,
            "use_priority": True,
            "use_learning": True,
        },
        {
            "name": "-VQ",
            "use_virtual_queue": False,
            "use_gpd_online": True,
            "use_lyapunov": True,
            "use_priority": True,
            "use_learning": True,
        },
        {
            "name": "-GPD",
            "use_virtual_queue": True,
            "use_gpd_online": False,
            "use_lyapunov": True,
            "use_priority": True,
            "use_learning": True,
        },
        {
            "name": "-Lyapunov",
            "use_virtual_queue": True,
            "use_gpd_online": True,
            "use_lyapunov": False,
            "use_priority": True,
            "use_learning": True,
        },
        {
            "name": "-Priority",
            "use_virtual_queue": True,
            "use_gpd_online": True,
            "use_lyapunov": True,
            "use_priority": False,
            "use_learning": True,
        },
        {
            "name": "Random",
            "use_virtual_queue": True,
            "use_gpd_online": True,
            "use_lyapunov": True,
            "use_priority": True,
            "use_learning": False,
        },
    ]


def run_one(cfg, nb_episode, warmup_ratio, user_num, load_scale, bw_scale, f_scale, tmax_scale,
            seed, gpd_update_interval, gpd_min_episode_for_update, gpd_estimator,
            lyap_constraint_weight, lyap_energy_weight, lyap_constraint_scale, lyap_energy_scale,
            wolf_epsilon):
    np.random.seed(seed)
    random.seed(seed)

    actions = np.arange(8)
    actions_set = [
        [0, 5 * pow(10, 5), 0],
        [0, 10 * pow(10, 5), 0],
        [0, 20 * pow(10, 5), 0],
        [0, 30 * pow(10, 5), 0],
        [1, 0, 0.1],
        [1, 0, 0.5],
        [1, 0, 1],
        [1, 0, 2],
    ]

    lambda_n = build_lambda(user_num, load_scale=load_scale)
    gpd1_array = [4.0 * pow(10, 6) for _ in range(user_num)]
    gpd2_array = [0.3 for _ in range(user_num)]

    wolf_agent_array = [
        WoLFAgent(alpha=0.1, actions=actions, high_delta=0.02, low_delta=0.0035, epsilon=wolf_epsilon)
        for _ in range(user_num)
    ]
    queue_relay_array = [QueueRelay(lambda_n[i], gpd1_array[i], gpd2_array[i]) for i in range(user_num)]
    q_history = [[0.0] for _ in range(user_num)]

    warmup = int(nb_episode * warmup_ratio)
    stat_episodes = max(nb_episode - warmup, 1)

    queue_viol_count = np.zeros(user_num)
    tail_excess_sum = np.zeros(user_num)
    tail_excess_sq_sum = np.zeros(user_num)
    total_energy = 0.0
    offload_ratio_sum = 0.0
    action_count = np.zeros(len(actions), dtype=float)

    base_bw = 10 * pow(10, 6)
    base_f = 10 * pow(10, 9)

    for episode in range(nb_episode):
        q_array = [qr.Q for qr in queue_relay_array]
        qx_array = [qr.Qx for qr in queue_relay_array]
        qy_array = [qr.Qy for qr in queue_relay_array]
        qz_array = [qr.Qz for qr in queue_relay_array]
        m1_array = [qr.M1 for qr in queue_relay_array]
        m2_array = [qr.M2 for qr in queue_relay_array]

        if cfg["use_learning"]:
            iteration_actions = [agent.act() for agent in wolf_agent_array]
        else:
            iteration_actions = [int(np.random.choice(actions)) for _ in range(user_num)]

        qx_input = qx_array if cfg["use_virtual_queue"] else [0.0] * user_num
        qy_input = qy_array if cfg["use_virtual_queue"] else [0.0] * user_num
        qz_input = qz_array if cfg["use_virtual_queue"] else [0.0] * user_num
        m1_input = m1_array if cfg["use_virtual_queue"] else [0.0] * user_num
        m2_input = m2_array if cfg["use_virtual_queue"] else [0.0] * user_num

        reward_mode = "lyapunov" if cfg["use_lyapunov"] else "energy"
        game = MatrixGame(
            actions=iteration_actions,
            Q=q_array,
            Qx=qx_input,
            Qy=qy_input,
            Qz=qz_input,
            M1=m1_input,
            M2=m2_input,
            BW=base_bw,
            reward_mode=reward_mode,
            lyap_constraint_weight=lyap_constraint_weight,
            lyap_energy_weight=lyap_energy_weight,
            lyap_constraint_scale=lyap_constraint_scale,
            lyap_energy_scale=lyap_energy_scale,
        )

        if not cfg["use_priority"]:
            game.pr_n = np.ones(user_num)

        game.BW = int(base_bw * bw_scale)
        game.F = int(base_f * f_scale)
        game.bn = game.bn * load_scale
        game.lambda_n = game.lambda_n * load_scale
        game.t_max = game.t_max * tmax_scale

        reward, _, bn, lumbda, rff = game.step(actions=iteration_actions)
        reward = np.nan_to_num(reward, nan=-1e6, posinf=-1e6, neginf=-1e6)

        energy_episode = compute_energy(game, iteration_actions, rff)

        if (
            gpd_estimator is not None
            and cfg["use_gpd_online"]
            and episode % gpd_update_interval == 0
            and episode >= gpd_min_episode_for_update
        ):
            for i in range(user_num):
                if len(q_history[i]) < gpd_min_episode_for_update:
                    continue
                res = gpd_estimator.gpd(q_history[i], queue_relay_array[i].q0, i)
                if res:
                    new_gpd1 = float(res[0][0])
                    new_gpd2 = float(res[0][1])
                    if valid_gpd_params(new_gpd1, new_gpd2):
                        queue_relay_array[i].GPD1 = new_gpd1
                        queue_relay_array[i].GPD2 = new_gpd2
                        queue_relay_array[i].updateM1()
                        queue_relay_array[i].updateM2()

        for i in range(user_num):
            queue_relay_array[i].lumbda = lumbda[i]
            queue_relay_array[i].updateQ(bn[i], actions_set[iteration_actions[i]][0], rff[i])
            queue_relay_array[i].updateQx()
            queue_relay_array[i].updateQy()
            queue_relay_array[i].updateQz()
            # Use post-update queue samples for GPD estimation.
            q_history[i].append(queue_relay_array[i].Q)

        if episode >= warmup:
            total_energy += energy_episode
            offload_ratio_sum += float(np.mean([1 if a >= 4 else 0 for a in iteration_actions]))
            for a in iteration_actions:
                action_count[int(a)] += 1.0
            for i in range(user_num):
                q_cur = queue_relay_array[i].Q
                q0_cur = queue_relay_array[i].q0
                excess = max(q_cur - q0_cur, 0.0)
                queue_viol_count[i] += float(q_cur > q0_cur)
                tail_excess_sum[i] += excess
                tail_excess_sq_sum[i] += excess * excess

        if cfg["use_learning"]:
            for i in range(user_num):
                wolf_agent_array[i].observe(reward=reward[i])

    p_queue_user = queue_viol_count / float(stat_episodes)
    m1_user = tail_excess_sum / float(stat_episodes)
    m2_user = tail_excess_sq_sum / float(stat_episodes)
    # C7/C8 bounds come from excess distribution moments, so we need conditional moments.
    m1_excess_cond_user = np.divide(
        tail_excess_sum,
        np.maximum(queue_viol_count, 1.0),
        dtype=float,
    )
    m2_excess_cond_user = np.divide(
        tail_excess_sq_sum,
        np.maximum(queue_viol_count, 1.0),
        dtype=float,
    )

    lambda_user = np.array([qr.lumbda for qr in queue_relay_array], dtype=float)
    gpd1_user = np.array([qr.GPD1 for qr in queue_relay_array], dtype=float)
    gpd2_user = np.array([qr.GPD2 for qr in queue_relay_array], dtype=float)

    p_c6_viol = float(np.mean(p_queue_user > lambda_user))
    c7_bound = gpd1_user / np.maximum(1 - gpd2_user, 1e-6)
    c8_den = (1 - gpd2_user) * (1 - 2 * gpd2_user)
    c8_bound = 2 * gpd1_user * gpd1_user / np.maximum(c8_den, 1e-6)
    p_c7_viol = float(np.mean(m1_excess_cond_user > c7_bound))
    p_c8_viol = float(np.mean(m2_excess_cond_user > c8_bound))

    mean_overflow_prob = float(np.mean(p_queue_user))
    mean_m1_excess_cond = float(np.mean(m1_excess_cond_user))
    mean_m2_excess_cond = float(np.mean(m2_excess_cond_user))
    mean_offload_ratio = float(offload_ratio_sum / float(stat_episodes))
    total_action_samples = float(stat_episodes * user_num)
    action_ratio = action_count / max(total_action_samples, 1.0)

    avg_energy_per_slot_user = total_energy / float(stat_episodes * user_num)

    result = {
        "p_c6_viol": p_c6_viol,
        "p_c7_viol": p_c7_viol,
        "p_c8_viol": p_c8_viol,
        "mean_overflow_prob": mean_overflow_prob,
        "mean_m1_excess_cond": mean_m1_excess_cond,
        "mean_m2_excess_cond": mean_m2_excess_cond,
        "mean_offload_ratio": mean_offload_ratio,
        "total_energy": float(total_energy),
        "avg_energy_per_slot_user": float(avg_energy_per_slot_user),
    }
    for aidx in range(len(actions)):
        result[f"action_{aidx}_ratio"] = float(action_ratio[aidx])
    return result


def parse_seed_list(seed_text):
    parts = [x.strip() for x in seed_text.split(",") if x.strip()]
    return [int(x) for x in parts]


def mean_std(vals):
    arr = np.array(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--warmup_ratio", type=float, default=0.7)
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--gpd_update_interval", type=int, default=100)
    parser.add_argument("--gpd_min_episode_for_update", type=int, default=400)
    parser.add_argument("--disable_gpd_online_update", action="store_true")
    parser.add_argument("--lyap_constraint_weight", type=float, default=8.0)
    parser.add_argument("--lyap_energy_weight", type=float, default=1.0)
    parser.add_argument("--lyap_constraint_scale", type=float, default=1e25)
    parser.add_argument("--lyap_energy_scale", type=float, default=1e25)
    parser.add_argument("--wolf_epsilon", type=float, default=0.05)
    parser.add_argument("--out_summary", type=str, default="picture/reliability/ablation_results.csv")
    parser.add_argument("--out_raw", type=str, default="picture/reliability/ablation_results_raw.csv")
    args = parser.parse_args()

    seed_list = parse_seed_list(args.seeds)

    scenarios = [
        ("load_scale", [0.5, 1.0, 1.5, 2.0]),
        ("f_scale", [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]),
        ("user_num", [5, 10, 15, 20, 25, 30, 35, 40]),
    ]

    configs = get_ablation_configs()

    gpd_estimator = None
    if not args.disable_gpd_online_update:
        try:
            from gpd import GPD
            gpd_estimator = GPD()
        except Exception as exc:
            print("[WARN] GPD online update disabled:", exc)

    ensure_dir(os.path.dirname(args.out_summary))
    ensure_dir(os.path.dirname(args.out_raw))

    with open(args.out_raw, "w", newline="") as f_raw, open(args.out_summary, "w", newline="") as f_sum:
        raw_writer = csv.DictWriter(
            f_raw,
            fieldnames=[
                "scenario", "value", "ablation", "seed",
                "user_num", "load_scale", "bw_scale", "f_scale", "tmax_scale",
                "p_c6_viol", "p_c7_viol", "p_c8_viol",
                "mean_overflow_prob", "mean_m1_excess_cond", "mean_m2_excess_cond",
                "mean_offload_ratio",
                "action_0_ratio", "action_1_ratio", "action_2_ratio", "action_3_ratio",
                "action_4_ratio", "action_5_ratio", "action_6_ratio", "action_7_ratio",
                "total_energy", "avg_energy_per_slot_user",
                "lyap_constraint_weight", "lyap_energy_weight", "lyap_constraint_scale", "lyap_energy_scale",
                "wolf_epsilon",
                "episodes", "warmup_ratio",
            ],
        )
        sum_writer = csv.DictWriter(
            f_sum,
            fieldnames=[
                "scenario", "value", "ablation", "n_seed",
                "user_num", "load_scale", "bw_scale", "f_scale", "tmax_scale",
                "p_c6_viol_mean", "p_c6_viol_std",
                "p_c7_viol_mean", "p_c7_viol_std",
                "p_c8_viol_mean", "p_c8_viol_std",
                "mean_overflow_prob_mean", "mean_overflow_prob_std",
                "mean_m1_excess_cond_mean", "mean_m1_excess_cond_std",
                "mean_m2_excess_cond_mean", "mean_m2_excess_cond_std",
                "mean_offload_ratio_mean", "mean_offload_ratio_std",
                "action_0_ratio_mean", "action_0_ratio_std",
                "action_1_ratio_mean", "action_1_ratio_std",
                "action_2_ratio_mean", "action_2_ratio_std",
                "action_3_ratio_mean", "action_3_ratio_std",
                "action_4_ratio_mean", "action_4_ratio_std",
                "action_5_ratio_mean", "action_5_ratio_std",
                "action_6_ratio_mean", "action_6_ratio_std",
                "action_7_ratio_mean", "action_7_ratio_std",
                "total_energy_mean", "total_energy_std",
                "avg_energy_per_slot_user_mean", "avg_energy_per_slot_user_std",
                "lyap_constraint_weight", "lyap_energy_weight", "lyap_constraint_scale", "lyap_energy_scale",
                "wolf_epsilon",
                "episodes", "warmup_ratio",
            ],
        )
        raw_writer.writeheader()
        sum_writer.writeheader()

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
                elif scenario == "bw_scale":
                    bw_scale = float(v)
                elif scenario == "f_scale":
                    f_scale = float(v)
                elif scenario == "tmax_scale":
                    tmax_scale = float(v)

                for cfg in configs:
                    rows = []
                    for seed in seed_list:
                        res = run_one(
                            cfg=cfg,
                            nb_episode=args.episodes,
                            warmup_ratio=args.warmup_ratio,
                            user_num=user_num,
                            load_scale=load_scale,
                            bw_scale=bw_scale,
                            f_scale=f_scale,
                            tmax_scale=tmax_scale,
                            seed=seed,
                            gpd_update_interval=args.gpd_update_interval,
                            gpd_min_episode_for_update=args.gpd_min_episode_for_update,
                            gpd_estimator=gpd_estimator,
                            lyap_constraint_weight=args.lyap_constraint_weight,
                            lyap_energy_weight=args.lyap_energy_weight,
                            lyap_constraint_scale=args.lyap_constraint_scale,
                            lyap_energy_scale=args.lyap_energy_scale,
                            wolf_epsilon=args.wolf_epsilon,
                        )
                        row = {
                            "scenario": scenario,
                            "value": v,
                            "ablation": cfg["name"],
                            "seed": seed,
                            "user_num": user_num,
                            "load_scale": load_scale,
                            "bw_scale": bw_scale,
                            "f_scale": f_scale,
                            "tmax_scale": tmax_scale,
                            "p_c6_viol": res["p_c6_viol"],
                            "p_c7_viol": res["p_c7_viol"],
                            "p_c8_viol": res["p_c8_viol"],
                            "mean_overflow_prob": res["mean_overflow_prob"],
                            "mean_m1_excess_cond": res["mean_m1_excess_cond"],
                            "mean_m2_excess_cond": res["mean_m2_excess_cond"],
                            "mean_offload_ratio": res["mean_offload_ratio"],
                            "action_0_ratio": res["action_0_ratio"],
                            "action_1_ratio": res["action_1_ratio"],
                            "action_2_ratio": res["action_2_ratio"],
                            "action_3_ratio": res["action_3_ratio"],
                            "action_4_ratio": res["action_4_ratio"],
                            "action_5_ratio": res["action_5_ratio"],
                            "action_6_ratio": res["action_6_ratio"],
                            "action_7_ratio": res["action_7_ratio"],
                            "total_energy": res["total_energy"],
                            "avg_energy_per_slot_user": res["avg_energy_per_slot_user"],
                            "lyap_constraint_weight": args.lyap_constraint_weight,
                            "lyap_energy_weight": args.lyap_energy_weight,
                            "lyap_constraint_scale": args.lyap_constraint_scale,
                            "lyap_energy_scale": args.lyap_energy_scale,
                            "wolf_epsilon": args.wolf_epsilon,
                            "episodes": args.episodes,
                            "warmup_ratio": args.warmup_ratio,
                        }
                        raw_writer.writerow(row)
                        rows.append(row)

                    p6_m, p6_s = mean_std([r["p_c6_viol"] for r in rows])
                    p7_m, p7_s = mean_std([r["p_c7_viol"] for r in rows])
                    p8_m, p8_s = mean_std([r["p_c8_viol"] for r in rows])
                    of_m, of_s = mean_std([r["mean_overflow_prob"] for r in rows])
                    m1e_m, m1e_s = mean_std([r["mean_m1_excess_cond"] for r in rows])
                    m2e_m, m2e_s = mean_std([r["mean_m2_excess_cond"] for r in rows])
                    off_m, off_s = mean_std([r["mean_offload_ratio"] for r in rows])
                    a0_m, a0_s = mean_std([r["action_0_ratio"] for r in rows])
                    a1_m, a1_s = mean_std([r["action_1_ratio"] for r in rows])
                    a2_m, a2_s = mean_std([r["action_2_ratio"] for r in rows])
                    a3_m, a3_s = mean_std([r["action_3_ratio"] for r in rows])
                    a4_m, a4_s = mean_std([r["action_4_ratio"] for r in rows])
                    a5_m, a5_s = mean_std([r["action_5_ratio"] for r in rows])
                    a6_m, a6_s = mean_std([r["action_6_ratio"] for r in rows])
                    a7_m, a7_s = mean_std([r["action_7_ratio"] for r in rows])
                    et_m, et_s = mean_std([r["total_energy"] for r in rows])
                    eu_m, eu_s = mean_std([r["avg_energy_per_slot_user"] for r in rows])

                    sum_writer.writerow({
                        "scenario": scenario,
                        "value": v,
                        "ablation": cfg["name"],
                        "n_seed": len(seed_list),
                        "user_num": user_num,
                        "load_scale": load_scale,
                        "bw_scale": bw_scale,
                        "f_scale": f_scale,
                        "tmax_scale": tmax_scale,
                        "p_c6_viol_mean": p6_m,
                        "p_c6_viol_std": p6_s,
                        "p_c7_viol_mean": p7_m,
                        "p_c7_viol_std": p7_s,
                        "p_c8_viol_mean": p8_m,
                        "p_c8_viol_std": p8_s,
                        "mean_overflow_prob_mean": of_m,
                        "mean_overflow_prob_std": of_s,
                        "mean_m1_excess_cond_mean": m1e_m,
                        "mean_m1_excess_cond_std": m1e_s,
                        "mean_m2_excess_cond_mean": m2e_m,
                        "mean_m2_excess_cond_std": m2e_s,
                        "mean_offload_ratio_mean": off_m,
                        "mean_offload_ratio_std": off_s,
                        "action_0_ratio_mean": a0_m,
                        "action_0_ratio_std": a0_s,
                        "action_1_ratio_mean": a1_m,
                        "action_1_ratio_std": a1_s,
                        "action_2_ratio_mean": a2_m,
                        "action_2_ratio_std": a2_s,
                        "action_3_ratio_mean": a3_m,
                        "action_3_ratio_std": a3_s,
                        "action_4_ratio_mean": a4_m,
                        "action_4_ratio_std": a4_s,
                        "action_5_ratio_mean": a5_m,
                        "action_5_ratio_std": a5_s,
                        "action_6_ratio_mean": a6_m,
                        "action_6_ratio_std": a6_s,
                        "action_7_ratio_mean": a7_m,
                        "action_7_ratio_std": a7_s,
                        "total_energy_mean": et_m,
                        "total_energy_std": et_s,
                        "avg_energy_per_slot_user_mean": eu_m,
                        "avg_energy_per_slot_user_std": eu_s,
                        "lyap_constraint_weight": args.lyap_constraint_weight,
                        "lyap_energy_weight": args.lyap_energy_weight,
                        "lyap_constraint_scale": args.lyap_constraint_scale,
                        "lyap_energy_scale": args.lyap_energy_scale,
                        "wolf_epsilon": args.wolf_epsilon,
                        "episodes": args.episodes,
                        "warmup_ratio": args.warmup_ratio,
                    })


if __name__ == "__main__":
    main()
