import argparse
import csv
import os
import random

import numpy as np

from matrix_game import MatrixGame
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


def moving_avg(arr, window):
    if len(arr) < window:
        return np.array(arr, dtype=float)
    out = np.convolve(arr, np.ones(window) / window, mode="valid")
    return out


def compute_last_delay_fallback(game, actions, rf):
    last_delay = np.zeros(game.num_ue)

    theta_pr = 0.0
    for i in range(game.num_ue):
        theta_pr += game.action_space[actions[i]][0] * game.pr_n[i]

    for i in range(game.num_ue):
        action = actions[i]
        if action < 4:
            f_local = game.action_space[action][1]
            last_delay[i] = game.bn[i] * game.dn[i] / f_local
            continue

        tx_power = game.action_space[action][2]
        rw = 0.0
        if theta_pr > 0:
            rw = (game.BW * game.pr_n[i]) / theta_pr

        vn = 0.0
        if rw > 0:
            snr_term = 1 + (game.g0 * tx_power / (game.N0 * rw)) - pow(10, -5)
            if snr_term > 0:
                vn = rw * np.log2(snr_term)

        t_tx = game.bn[i] / vn if vn > 0 else float("inf")
        if rf[i] > 0:
            t_queue = game.Q[i] * game.Li / rf[i]
            t_exec = game.bn[i] * game.dn[i] / rf[i]
            last_delay[i] = t_tx + t_queue + t_exec
        else:
            last_delay[i] = float("inf")

    return last_delay


def valid_gpd_params(gpd1, gpd2):
    if not np.isfinite(gpd1) or not np.isfinite(gpd2):
        return False
    if gpd1 <= 0:
        return False
    if gpd2 <= -0.9 or gpd2 >= 0.49:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--warmup_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpd_update_interval", type=int, default=50)
    parser.add_argument("--gpd_min_episode_for_update", type=int, default=300)
    parser.add_argument("--out_csv", type=str, default="picture/reliability/single_case_reward.csv")
    args = parser.parse_args()

    user_num = 15
    load_scale = 1.0
    bw_scale = 1.0
    f_scale = 1.0
    tmax_scale = 1.0

    np.random.seed(args.seed)
    random.seed(args.seed)

    actions = np.arange(8)
    lambda_n = build_lambda(user_num, load_scale=load_scale)

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

    gpd1_array = [4.0 * pow(10, 6) for _ in range(user_num)]
    gpd2_array = [0.3 for _ in range(user_num)]

    wolf_agent_array = [WoLFAgent(alpha=0.1, actions=actions, high_delta=0.02, low_delta=0.0035)
                        for _ in range(user_num)]
    queue_relay_array = [QueueRelay(lambda_n[i], gpd1_array[i], gpd2_array[i])
                         for i in range(user_num)]

    q_history = [[0.0] for _ in range(user_num)]

    gpd_estimator = None
    try:
        from gpd import GPD
        gpd_estimator = GPD()
    except Exception as exc:
        print("[WARN] GPD online update disabled:", exc)

    warmup = int(args.episodes * args.warmup_ratio)
    stat_episodes = max(args.episodes - warmup, 1)

    delay_viol_count = np.zeros(user_num)
    reward_trace = []

    base_bw = 10 * pow(10, 6)
    base_f = 10 * pow(10, 9)

    for episode in range(args.episodes):
        q_array = [qr.Q for qr in queue_relay_array]
        qx_array = [qr.Qx for qr in queue_relay_array]
        qy_array = [qr.Qy for qr in queue_relay_array]
        qz_array = [qr.Qz for qr in queue_relay_array]
        m1_array = [qr.M1 for qr in queue_relay_array]
        m2_array = [qr.M2 for qr in queue_relay_array]

        iteration_actions = [agent.act() for agent in wolf_agent_array]

        game = MatrixGame(
            actions=iteration_actions,
            Q=q_array,
            Qx=qx_array,
            Qy=qy_array,
            Qz=qz_array,
            M1=m1_array,
            M2=m2_array,
            BW=base_bw,
            reward_mode="lyapunov",
        )

        game.BW = int(base_bw * bw_scale)
        game.F = int(base_f * f_scale)
        game.bn = game.bn * load_scale
        game.lambda_n = game.lambda_n * load_scale
        game.t_max = game.t_max * tmax_scale

        reward, _, bn, lumbda, rff = game.step(actions=iteration_actions)
        reward = np.nan_to_num(reward, nan=-1e6, posinf=-1e6, neginf=-1e6)
        reward_trace.append(float(np.mean(reward)))

        for i in range(user_num):
            q_history[i].append(q_array[i])

        if (
            gpd_estimator is not None
            and episode % args.gpd_update_interval == 0
            and episode >= args.gpd_min_episode_for_update
        ):
            for i in range(user_num):
                if len(q_history[i]) < args.gpd_min_episode_for_update:
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

        model_last_delay = getattr(game, "last_delay", None)
        if model_last_delay is not None:
            last_delay = np.array(model_last_delay)
        else:
            last_delay = compute_last_delay_fallback(game, iteration_actions, rff)

        if episode >= warmup:
            delay_viol_count += (last_delay > game.t_max).astype(float)

        for i in range(user_num):
            queue_relay_array[i].lumbda = lumbda[i]
            queue_relay_array[i].updateQ(bn[i], actions_set[iteration_actions[i]][0], rff[i])
            queue_relay_array[i].updateQx()
            queue_relay_array[i].updateQy()
            queue_relay_array[i].updateQz()

        for i in range(user_num):
            wolf_agent_array[i].observe(reward=reward[i])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward_mean"])
        for i, r in enumerate(reward_trace):
            writer.writerow([i, r])

    reward_ma = moving_avg(reward_trace, window=50)
    first = float(np.mean(reward_ma[: max(1, len(reward_ma)//5)]))
    mid = float(np.mean(reward_ma[len(reward_ma)//2 - max(1, len(reward_ma)//10): len(reward_ma)//2 + max(1, len(reward_ma)//10)]))
    last = float(np.mean(reward_ma[-max(1, len(reward_ma)//5):]))

    p_c5_viol = float(np.mean(delay_viol_count / float(stat_episodes)))

    print("single scenario fixed: load=1.0, f=1.0, user=15")
    print("reward_ma(first/mid/last)=", first, mid, last)
    print("reward_ma_delta_last_first=", last - first)
    print("p_c5_viol=", p_c5_viol)
    print("reward_csv=", args.out_csv)


if __name__ == "__main__":
    main()
