import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from matrix_game import MatrixGame
from queue_relay import QueueRelay
from wolf_agent import WoLFAgent


@dataclass
class SimConfig:
    episodes: int
    warmup_ratio: float
    gpd_update_interval: int
    gpd_min_episode_for_update: int
    lyap_constraint_weight: float
    lyap_energy_weight: float
    lyap_constraint_scale: float
    lyap_energy_scale: float
    wolf_epsilon: float
    c5_penalty_weight: float
    c5_penalty_scale: float
    mec_bw_base: float
    mec_f_base: float


@dataclass
class EnvConfig:
    user_num: int
    load_scale: float
    bw_scale: float
    f_scale: float
    tmax_scale: float


def build_lambda(user_num: int, load_scale: float = 1.0) -> np.ndarray:
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


def valid_gpd_params(gpd1: float, gpd2: float) -> bool:
    if not np.isfinite(gpd1) or not np.isfinite(gpd2):
        return False
    if gpd1 <= 0:
        return False
    if gpd2 <= -0.9 or gpd2 >= 0.49:
        return False
    return True


def compute_energy(game: MatrixGame, actions: List[int], rf: List[float]) -> float:
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


def _resource_view(game: MatrixGame, actions: List[int]):
    rw = np.zeros(game.num_ue)
    rf = np.zeros(game.num_ue)
    vn = np.zeros(game.num_ue)

    theta_pr = 0.0
    pr_q = 0.0
    for i in range(game.num_ue):
        theta_pr += game.action_space[actions[i]][0] * game.pr_n[i]
        pr_q += game.pr_n[i] * (1 if game.Q[i] > 0 else 0)

    for i in range(game.num_ue):
        if actions[i] < 4:
            continue
        if theta_pr > 0:
            rw[i] = (game.BW * game.pr_n[i]) / theta_pr
        if rw[i] > 0:
            tx_power = game.action_space[actions[i]][2]
            snr_term = 1 + (game.g0 * tx_power / (game.N0 * rw[i])) - pow(10, -5)
            if snr_term > 0:
                vn[i] = rw[i] * np.log2(snr_term)
        if pr_q > 0:
            rf[i] = game.F * (game.pr_n[i] * (1 if game.Q[i] > 0 else 0)) / pr_q

    return rw, rf, vn


def compute_delay_vector(game: MatrixGame, actions: List[int]) -> np.ndarray:
    _, rf, vn = _resource_view(game, actions)
    delay = np.zeros(game.num_ue, dtype=float)

    for i in range(game.num_ue):
        if actions[i] < 4:
            f_local = game.action_space[actions[i]][1]
            delay[i] = game.bn[i] * game.dn[i] / f_local if f_local > 0 else np.inf
            continue

        if vn[i] <= 0 or not np.isfinite(vn[i]) or rf[i] <= 0:
            delay[i] = np.inf
            continue

        t_up = game.bn[i] / vn[i]
        t_wait = game.Q[i] * game.Li / rf[i]
        t_exec = game.bn[i] * game.dn[i] / rf[i]
        delay[i] = t_up + t_wait + t_exec

    return delay


def _user_energy(game: MatrixGame, actions: List[int], idx: int) -> float:
    _, rf, vn = _resource_view(game, actions)
    a = actions[idx]
    if a < 4:
        f_local = game.action_space[a][1]
        return float(game.kmob * game.tau * (f_local ** 3))

    tx_power = game.action_space[a][2]
    e_tx = 1e6
    if vn[idx] > 0 and np.isfinite(vn[idx]):
        e_tx = tx_power * game.bn[idx] / vn[idx]
    e_ser = game.kser * rf[idx] * rf[idx] * (game.bn[idx] * game.dn[idx])
    return float(e_tx + e_ser)


def apply_c5_safety_shield(game: MatrixGame, actions: List[int], max_pass: int = 1) -> List[int]:
    # Fast shield: for violated users, move to strongest local/offload action once.
    safe_actions = list(actions)
    delay_now = compute_delay_vector(game, safe_actions)
    for i in range(game.num_ue):
        if delay_now[i] <= game.t_max[i]:
            continue

        a_cur = safe_actions[i]
        if a_cur < 4:
            safe_actions[i] = 3
        else:
            safe_actions[i] = 7

    return safe_actions


def greedy_delay_first_actions(game: MatrixGame) -> List[int]:
    """Fast greedy: only test key actions (local, nearby MEC) instead of all 8"""
    actions = [3 for _ in range(game.num_ue)]
    key_actions = [3, 6]  # Fast heuristic: test local (3) and pure MEC (6)
    
    for i in range(game.num_ue):
        best_a = actions[i]
        best_key = (1, float("inf"), float("inf"))
        for cand in key_actions:  # Only test key actions instead of 0-7
            trial = list(actions)
            trial[i] = cand
            d_i = float(compute_delay_vector(game, trial)[i])
            e_i = float(_user_energy(game, trial, i))
            feasible = 0 if d_i <= game.t_max[i] else 1
            key = (feasible, d_i, e_i)
            if key < best_key:
                best_key = key
                best_a = cand
        actions[i] = best_a
    return actions


def get_policy_variants() -> List[Dict]:
    return [
        {
            "name": "Full",
            "use_virtual_queue": True,
            "use_gpd_online": True,
            "use_lyapunov": True,
            "use_priority": True,
            "use_learning": True,
            "action_mode": "learning",
            "wolf_mode": "wolf",
            "use_c5_shield": True,
        },
        {
            "name": "IQL",
            "use_virtual_queue": True,
            "use_gpd_online": True,
            "use_lyapunov": True,
            "use_priority": True,
            "use_learning": True,
            "action_mode": "learning",
            "wolf_mode": "iql",
            "use_c5_shield": True,
        },
        {
            "name": "No-EVT",
            "use_virtual_queue": False,
            "use_gpd_online": False,
            "use_lyapunov": True,
            "use_priority": True,
            "use_learning": True,
            "action_mode": "learning",
            "wolf_mode": "wolf",
            "use_c5_shield": True,
        },
        {
            "name": "C6-Only",
            "use_virtual_queue": False,
            "use_qx_constraint": True,
            "use_evt_constraint": False,
            "use_gpd_online": False,
            "use_lyapunov": True,
            "use_priority": True,
            "use_learning": True,
            "action_mode": "learning",
            "wolf_mode": "wolf",
            "use_c5_shield": True,
        },
        {
            "name": "Lyapunov-Greedy",
            "use_virtual_queue": True,
            "use_gpd_online": True,
            "use_lyapunov": True,
            "use_priority": True,
            "use_learning": False,
            "action_mode": "greedy",
            "wolf_mode": "wolf",
            "use_c5_shield": True,
        },
        {
            "name": "Non-MEC",
            "use_virtual_queue": False,
            "use_gpd_online": False,
            "use_lyapunov": False,
            "use_priority": False,
            "use_learning": False,
            "action_mode": "non_mec",
            "wolf_mode": "wolf",
            "use_c5_shield": True,
        },
        {
            "name": "Stand-alone MEC",
            "use_virtual_queue": False,
            "use_gpd_online": False,
            "use_lyapunov": False,
            "use_priority": False,
            "use_learning": False,
            "action_mode": "standalone_mec",
            "wolf_mode": "wolf",
            "use_c5_shield": True,
        },
        {
            "name": "Random",
            "use_virtual_queue": False,
            "use_gpd_online": False,
            "use_lyapunov": False,
            "use_priority": False,
            "use_learning": False,
            "action_mode": "random",
            "wolf_mode": "wolf",
            "use_c5_shield": False,
        },
    ]


def run_policy_once(policy: Dict, env: EnvConfig, sim: SimConfig, seed: int, gpd_estimator=None) -> Dict:
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

    lambda_n = build_lambda(env.user_num, load_scale=env.load_scale)
    gpd1_array = [4.0 * pow(10, 6) for _ in range(env.user_num)]
    gpd2_array = [0.3 for _ in range(env.user_num)]

    agents = []
    if policy.get("use_learning", False):
        wolf_mode = policy.get("wolf_mode", "wolf")
        high_delta = 0.02
        low_delta = 0.0035
        if wolf_mode == "iql":
            high_delta = 0.006
            low_delta = 0.006
        agents = [
            WoLFAgent(alpha=0.1, actions=actions, high_delta=high_delta, low_delta=low_delta, epsilon=sim.wolf_epsilon)
            for _ in range(env.user_num)
        ]
    queues = [QueueRelay(lambda_n[i], gpd1_array[i], gpd2_array[i]) for i in range(env.user_num)]
    q_history = [[0.0] for _ in range(env.user_num)]

    warmup = int(sim.episodes * sim.warmup_ratio)
    stat_episodes = max(sim.episodes - warmup, 1)

    queue_viol_count = np.zeros(env.user_num)
    tail_excess_sum = np.zeros(env.user_num)
    tail_excess_sq_sum = np.zeros(env.user_num)
    c5_viol_count = np.zeros(env.user_num)
    c5_excess_ratio_sum = np.zeros(env.user_num)
    total_energy = 0.0
    offload_ratio_sum = 0.0
    action_count = np.zeros(len(actions), dtype=float)

    base_bw = float(sim.mec_bw_base)
    base_f = float(sim.mec_f_base)

    for episode in range(sim.episodes):
        q = [qr.Q for qr in queues]
        qx = [qr.Qx for qr in queues]
        qy = [qr.Qy for qr in queues]
        qz = [qr.Qz for qr in queues]
        m1 = [qr.M1 for qr in queues]
        m2 = [qr.M2 for qr in queues]

        use_virtual_queue = bool(policy.get("use_virtual_queue", False))
        use_qx_constraint = bool(policy.get("use_qx_constraint", use_virtual_queue))
        use_evt_constraint = bool(policy.get("use_evt_constraint", use_virtual_queue))

        qx_in = qx if use_qx_constraint else [0.0] * env.user_num
        qy_in = qy if use_evt_constraint else [0.0] * env.user_num
        qz_in = qz if use_evt_constraint else [0.0] * env.user_num
        m1_in = m1 if use_evt_constraint else [0.0] * env.user_num
        m2_in = m2 if use_evt_constraint else [0.0] * env.user_num

        reward_mode = "lyapunov" if policy["use_lyapunov"] else "energy"
        game = MatrixGame(
            actions=[0] * env.user_num,
            Q=q,
            Qx=qx_in,
            Qy=qy_in,
            Qz=qz_in,
            M1=m1_in,
            M2=m2_in,
            BW=base_bw,
            reward_mode=reward_mode,
            lyap_constraint_weight=sim.lyap_constraint_weight,
            lyap_energy_weight=sim.lyap_energy_weight,
            lyap_constraint_scale=sim.lyap_constraint_scale,
            lyap_energy_scale=sim.lyap_energy_scale,
        )

        if not policy["use_priority"]:
            game.pr_n = np.ones(env.user_num)

        game.BW = int(base_bw * env.bw_scale)
        game.F = int(base_f * env.f_scale)
        game.bn = game.bn * env.load_scale
        game.lambda_n = game.lambda_n * env.load_scale
        game.t_max = game.t_max * env.tmax_scale

        mode = policy.get("action_mode", "learning")
        if mode == "learning":
            act = [agent.act() for agent in agents]
        elif mode == "greedy":
            act = greedy_delay_first_actions(game)
        elif mode == "non_mec":
            act = [3 for _ in range(env.user_num)]
        elif mode == "standalone_mec":
            act = [6 for _ in range(env.user_num)]
        elif mode == "random":
            act = [int(np.random.choice(actions)) for _ in range(env.user_num)]
        else:
            act = [int(np.random.choice(actions)) for _ in range(env.user_num)]

        if policy.get("use_c5_shield", False):
            act = apply_c5_safety_shield(game, act)

        reward, _, bn, lumbda, rff = game.step(actions=act)
        reward = np.nan_to_num(reward, nan=-1e6, posinf=-1e6, neginf=-1e6)

        delay_vec = compute_delay_vector(game, act)
        tmax_vec = np.maximum(np.array(game.t_max, dtype=float), 1e-9)
        c5_excess_ratio = np.maximum(delay_vec - np.array(game.t_max, dtype=float), 0.0) / tmax_vec
        c5_excess_ratio = np.nan_to_num(c5_excess_ratio, nan=1e3, posinf=1e3, neginf=0.0)
        if sim.c5_penalty_weight > 0:
            reward = reward - sim.c5_penalty_weight * (c5_excess_ratio / max(sim.c5_penalty_scale, 1e-9))
            reward = np.nan_to_num(reward, nan=-1e6, posinf=-1e6, neginf=-1e6)

        energy_episode = compute_energy(game, act, rff)

        if (
            gpd_estimator is not None
            and policy["use_gpd_online"]
            and episode % sim.gpd_update_interval == 0
            and episode >= sim.gpd_min_episode_for_update
        ):
            for i in range(env.user_num):
                if len(q_history[i]) < sim.gpd_min_episode_for_update:
                    continue
                res = gpd_estimator.gpd(q_history[i], queues[i].q0, i)
                if res:
                    new_gpd1 = float(res[0][0])
                    new_gpd2 = float(res[0][1])
                    if valid_gpd_params(new_gpd1, new_gpd2):
                        queues[i].GPD1 = new_gpd1
                        queues[i].GPD2 = new_gpd2
                        queues[i].updateM1()
                        queues[i].updateM2()

        for i in range(env.user_num):
            queues[i].lumbda = lumbda[i]
            queues[i].updateQ(bn[i], actions_set[act[i]][0], rff[i])
            queues[i].updateQx()
            queues[i].updateQy()
            queues[i].updateQz()
            q_history[i].append(queues[i].Q)

        if episode >= warmup:
            total_energy += energy_episode
            offload_ratio_sum += float(np.mean([1 if a >= 4 else 0 for a in act]))
            for a in act:
                action_count[int(a)] += 1.0
            for i in range(env.user_num):
                q_cur = queues[i].Q
                q0_cur = queues[i].q0
                excess = max(q_cur - q0_cur, 0.0)
                queue_viol_count[i] += float(q_cur > q0_cur)
                tail_excess_sum[i] += excess
                tail_excess_sq_sum[i] += excess * excess
                c5_viol_count[i] += float(delay_vec[i] > game.t_max[i])
                c5_excess_ratio_sum[i] += float(c5_excess_ratio[i])

        if policy["use_learning"]:
            for i in range(env.user_num):
                agents[i].observe(reward=reward[i])

    p_queue_user = queue_viol_count / float(stat_episodes)
    p_c5_user = c5_viol_count / float(stat_episodes)
    m1_excess_cond = np.divide(tail_excess_sum, np.maximum(queue_viol_count, 1.0), dtype=float)
    m2_excess_cond = np.divide(tail_excess_sq_sum, np.maximum(queue_viol_count, 1.0), dtype=float)

    lambda_user = np.array([qr.lumbda for qr in queues], dtype=float)
    gpd1_user = np.array([qr.GPD1 for qr in queues], dtype=float)
    gpd2_user = np.array([qr.GPD2 for qr in queues], dtype=float)

    c7_bound = gpd1_user / np.maximum(1 - gpd2_user, 1e-6)
    c8_den = (1 - gpd2_user) * (1 - 2 * gpd2_user)
    c8_bound = 2 * gpd1_user * gpd1_user / np.maximum(c8_den, 1e-6)

    avg_energy_per_slot_user = total_energy / float(stat_episodes * env.user_num)
    total_action_samples = float(stat_episodes * env.user_num)
    action_ratio = action_count / max(total_action_samples, 1.0)

    c6_overflow_prob = float(np.mean(p_queue_user))
    c6_exceed_ratio = float(np.mean(p_queue_user > lambda_user))

    result = {
        "p_c5_viol": float(np.mean(p_c5_user)),
        # C6 now directly represents the average queue overflow probability.
        "p_c6_viol": c6_overflow_prob,
        # Keep legacy binary C6 view for backward-compatible analysis.
        "p_c6_exceed_ratio": c6_exceed_ratio,
        "p_c7_viol": float(np.mean(m1_excess_cond > c7_bound)),
        "p_c8_viol": float(np.mean(m2_excess_cond > c8_bound)),
        "mean_c5_excess_ratio": float(np.mean(c5_excess_ratio_sum / float(stat_episodes))),
        "mean_overflow_prob": c6_overflow_prob,
        "mean_m1_excess_cond": float(np.mean(m1_excess_cond)),
        "mean_m2_excess_cond": float(np.mean(m2_excess_cond)),
        "mean_offload_ratio": float(offload_ratio_sum / float(stat_episodes)),
        "total_energy": float(total_energy),
        "avg_energy_per_slot_user": float(avg_energy_per_slot_user),
    }
    for i in range(len(actions)):
        result[f"action_{i}_ratio"] = float(action_ratio[i])
    return result
