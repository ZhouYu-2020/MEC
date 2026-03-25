import argparse
import csv
import os
import random

import numpy as np

from matrix_game import MatrixGame
from matrix_game_local_only import MatrixGame_local
from matrix_game_mec_only import MatrixGame_mec
from queue_relay import QueueRelay
from wolf_agent import WoLFAgent


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


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def valid_gpd_params(gpd1, gpd2):
    if not np.isfinite(gpd1) or not np.isfinite(gpd2):
        return False
    if gpd1 <= 0:
        return False
    if gpd2 <= -0.9 or gpd2 >= 0.49:
        return False
    return True


def compute_energy_main(game, actions, rf):
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

        if pr_q > 0:
            rf_i = game.F * (game.pr_n[i] * (1 if game.Q[i] > 0 else 0)) / pr_q
        else:
            rf_i = 0.0
        if i < len(rf):
            rf_i = rf[i]

        e_ser = game.kser * rf_i * rf_i * (game.bn[i] * game.dn[i])
        energy += e_tx + e_ser

    return float(energy)


def compute_energy_local(game, actions):
    energy = 0.0
    for i in range(game.num_ue):
        action = actions[i]
        f_local = game.action_space[action][1]
        energy += game.kmob * game.tau * (f_local ** 3)
    return float(energy)


def compute_energy_mec(game, actions, rf):
    energy = 0.0
    theta_pr = float(game.num_ue)
    pr_q = float(sum(1 if game.Q[i] > 0 else 0 for i in range(game.num_ue)))

    for i in range(game.num_ue):
        action = actions[i]
        tx_power = game.action_space[action][2]
        rw = game.BW / theta_pr if theta_pr > 0 else 0.0
        vn = 0.0
        if rw > 0:
            snr_term = 1 + (game.g0 * tx_power / (game.N0 * rw)) - pow(10, -5)
            if snr_term > 0:
                vn = rw * np.log2(snr_term)
        e_tx = tx_power * game.bn[i] / vn if vn > 0 and np.isfinite(vn) else 1e6

        rf_i = game.F * (1 if game.Q[i] > 0 else 0) / pr_q if pr_q > 0 else 0.0
        if i < len(rf):
            rf_i = rf[i]
        e_ser = game.kser * rf_i * rf_i * (game.bn[i] * game.dn[i])
        energy += e_tx + e_ser

    return float(energy)


def make_policy(policy_name):
    if policy_name == "main":
        return {
            "actions": np.arange(8),
            "actions_set": [
                [0, 5 * pow(10, 5), 0],
                [0, 10 * pow(10, 5), 0],
                [0, 20 * pow(10, 5), 0],
                [0, 30 * pow(10, 5), 0],
                [1, 0, 0.1],
                [1, 0, 0.5],
                [1, 0, 1],
                [1, 0, 2],
            ],
            "game_cls": MatrixGame,
            "reward_mode": "lyapunov",
        }
    if policy_name == "non_mec":
        return {
            "actions": np.arange(4),
            "actions_set": [
                [0, 5 * pow(10, 5), 0],
                [0, 10 * pow(10, 5), 0],
                [0, 20 * pow(10, 5), 0],
                [0, 30 * pow(10, 5), 0],
            ],
            "game_cls": MatrixGame_local,
            "reward_mode": None,
        }
    if policy_name == "mec_only":
        return {
            "actions": np.arange(4),
            "actions_set": [
                [1, 0, 0.1],
                [1, 0, 0.5],
                [1, 0, 1],
                [1, 0, 2],
            ],
            "game_cls": MatrixGame_mec,
            "reward_mode": None,
        }
    raise ValueError("unknown policy: " + policy_name)


def run_one(policy_name, nb_episode, warmup_ratio, user_num, load_scale, bw_scale, f_scale, tmax_scale,
            seed, gpd_update_interval, gpd_min_episode_for_update, gpd_estimator):
    np.random.seed(seed)
    random.seed(seed)

    policy = make_policy(policy_name)
    actions = policy["actions"]
    actions_set = policy["actions_set"]
    game_cls = policy["game_cls"]

    lambda_n = build_lambda(user_num, load_scale=load_scale)
    gpd1_array = [4.0 * pow(10, 6) for _ in range(user_num)]
    gpd2_array = [0.3 for _ in range(user_num)]

    wolf_agent_array = [WoLFAgent(alpha=0.1, actions=actions, high_delta=0.02, low_delta=0.0035)
                        for _ in range(user_num)]
    queue_relay_array = [QueueRelay(lambda_n[i], gpd1_array[i], gpd2_array[i]) for i in range(user_num)]
    q_history = [[0.0] for _ in range(user_num)]

    warmup = int(nb_episode * warmup_ratio)
    stat_episodes = max(nb_episode - warmup, 1)

    queue_viol_count = np.zeros(user_num)
    tail_excess_sum = np.zeros(user_num)
    tail_excess_sq_sum = np.zeros(user_num)
    total_energy = 0.0

    base_bw = 10 * pow(10, 6)
    base_f = 10 * pow(10, 9)

    for episode in range(nb_episode):
        q_array = [qr.Q for qr in queue_relay_array]
        qx_array = [qr.Qx for qr in queue_relay_array]
        qy_array = [qr.Qy for qr in queue_relay_array]
        qz_array = [qr.Qz for qr in queue_relay_array]
        m1_array = [qr.M1 for qr in queue_relay_array]
        m2_array = [qr.M2 for qr in queue_relay_array]

        iteration_actions = [agent.act() for agent in wolf_agent_array]

        if policy_name == "main":
            game = game_cls(
                actions=iteration_actions,
                Q=q_array,
                Qx=qx_array,
                Qy=qy_array,
                Qz=qz_array,
                M1=m1_array,
                M2=m2_array,
                BW=base_bw,
                reward_mode=policy["reward_mode"],
            )
        else:
            game = game_cls(
                actions=iteration_actions,
                Q=q_array,
                Qx=qx_array,
                Qy=qy_array,
                Qz=qz_array,
                M1=m1_array,
                M2=m2_array,
                BW=base_bw,
            )

        game.BW = int(base_bw * bw_scale)
        if hasattr(game, "F"):
            game.F = int(base_f * f_scale)
        if hasattr(game, "bn"):
            game.bn = game.bn * load_scale
        if hasattr(game, "lambda_n"):
            game.lambda_n = game.lambda_n * load_scale
        if hasattr(game, "t_max"):
            game.t_max = game.t_max * tmax_scale

        if policy_name in ["main", "non_mec"]:
            reward, _, bn, lumbda, rff = game.step(actions=iteration_actions)
        else:
            reward, bn, lumbda, rff = game.step(actions=iteration_actions)
        reward = np.nan_to_num(reward, nan=-1e6, posinf=-1e6, neginf=-1e6)

        if policy_name == "main":
            energy_episode = compute_energy_main(game, iteration_actions, rff)
        elif policy_name == "non_mec":
            energy_episode = compute_energy_local(game, iteration_actions)
        else:
            energy_episode = compute_energy_mec(game, iteration_actions, rff)

        if (
            gpd_estimator is not None
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
            q_history[i].append(queue_relay_array[i].Q)

        if episode >= warmup:
            total_energy += energy_episode
            for i in range(user_num):
                q_cur = queue_relay_array[i].Q
                q0_cur = queue_relay_array[i].q0
                excess = max(q_cur - q0_cur, 0.0)
                queue_viol_count[i] += float(q_cur > q0_cur)
                tail_excess_sum[i] += excess
                tail_excess_sq_sum[i] += excess * excess

        for i in range(user_num):
            wolf_agent_array[i].observe(reward=reward[i])

    p_queue_user = queue_viol_count / float(stat_episodes)
    m1_user = tail_excess_sum / float(stat_episodes)
    m2_user = tail_excess_sq_sum / float(stat_episodes)
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

    avg_energy_per_slot_user = total_energy / float(stat_episodes * user_num)

    return p_c6_viol, p_c7_viol, p_c8_viol, total_energy, avg_energy_per_slot_user


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--warmup_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpd_update_interval", type=int, default=50)
    parser.add_argument("--gpd_min_episode_for_update", type=int, default=300)
    parser.add_argument("--disable_gpd_online_update", action="store_true")
    parser.add_argument("--out", type=str, default="picture/reliability/strategy_compare_results.csv")
    args = parser.parse_args()

    scenarios = [
        ("load_scale", [0.5, 1.0, 1.5, 2.0]),
        ("f_scale", [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]),
        ("user_num", [5, 10, 15, 20, 25, 30, 35, 40]),
    ]
    policies = ["main", "non_mec", "mec_only"]

    gpd_estimator = None
    if not args.disable_gpd_online_update:
        try:
            from gpd import GPD
            gpd_estimator = GPD()
        except Exception as exc:
            print("[WARN] GPD online update disabled:", exc)

    ensure_dir(os.path.dirname(args.out))

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario", "value", "policy",
                "user_num", "load_scale", "bw_scale", "f_scale", "tmax_scale",
                "p_c6_viol", "p_c7_viol", "p_c8_viol",
                "total_energy", "avg_energy_per_slot_user",
                "episodes", "warmup_ratio", "seed",
            ],
        )
        writer.writeheader()

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

                for policy in policies:
                    p_c6_viol, p_c7_viol, p_c8_viol, total_energy, avg_energy_per_slot_user = run_one(
                        policy_name=policy,
                        nb_episode=args.episodes,
                        warmup_ratio=args.warmup_ratio,
                        user_num=user_num,
                        load_scale=load_scale,
                        bw_scale=bw_scale,
                        f_scale=f_scale,
                        tmax_scale=tmax_scale,
                        seed=args.seed,
                        gpd_update_interval=args.gpd_update_interval,
                        gpd_min_episode_for_update=args.gpd_min_episode_for_update,
                        gpd_estimator=gpd_estimator,
                    )

                    writer.writerow({
                        "scenario": scenario,
                        "value": v,
                        "policy": policy,
                        "user_num": user_num,
                        "load_scale": load_scale,
                        "bw_scale": bw_scale,
                        "f_scale": f_scale,
                        "tmax_scale": tmax_scale,
                        "p_c6_viol": p_c6_viol,
                        "p_c7_viol": p_c7_viol,
                        "p_c8_viol": p_c8_viol,
                        "total_energy": total_energy,
                        "avg_energy_per_slot_user": avg_energy_per_slot_user,
                        "episodes": args.episodes,
                        "warmup_ratio": args.warmup_ratio,
                        "seed": args.seed,
                    })


if __name__ == "__main__":
    main()
