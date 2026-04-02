# New Simulation Suite

This folder contains a redesigned experiment suite built on the same solving logic as the paper and current MEC codebase.

## 1) Current MEC Running Logic (How the project works)

The original code follows this loop per time slot:

1. Each UE agent (`WoLFAgent`) selects one action from 8 discrete actions.
2. `MatrixGame` computes reward for each UE:
   - local/offload energy cost,
   - Lyapunov-based constraint penalty terms from real queue `Q` and virtual queues (`Qx`, `Qy`, `Qz`).
3. `QueueRelay` updates queue states (`Q`, `Qx`, `Qy`, `Qz`) using arrivals/service.
4. Optional GPD update refreshes tail parameters (`GPD1`, `GPD2`) for C7/C8 bounds.
5. Agents observe reward and update policy.

After warmup episodes, metrics are estimated:
- C6/C7/C8 violation probabilities,
- energy,
- offload ratio,
- action selection ratios.

## 2) New Experiment Design

### Experiment A: Stress Scan (`stress_scan_*`)
Scan the environment stress along three axes:
- `load_scale`
- `f_scale` (MEC CPU scale)
- `user_num`

Compare policy variants:
- `Full` (EVT + Lyapunov + WoLF-PHC)
- `IQL` (replace WoLF dual-rate update by same-rate learning)
- `No-EVT` (remove EVT constraints, keep learning)
- `Lyapunov-Greedy` (no RL, delay-first greedy scheduler)
- `Non-MEC` (all-local)
- `Stand-alone MEC` (all-offload)
- `Random`

Outputs include energy + C6/C7/C8 + offload/action behavior.

New in this version:
- C5 delay reliability is explicitly tracked as `p_c5_viol` and `mean_c5_excess_ratio`.
- A C5 safety shield is applied before execution, and C5 excess is also penalized in reward.

### Experiment B: Constraint-Weight Sweep (`weight_sweep_*`)
Fix environment and sweep Lyapunov constraint weight:
- `[1, 2, 5, 10, 20, 40]`

Observe trade-off among:
- C6/C7/C8 violation,
- total energy,
- offload ratio.

## 3) Files

- `sim_core.py`: reusable simulation core
- `run_new_experiments.py`: run experiments and save CSV results
- `plot_new_experiments.py`: draw figures from result CSVs
- `results/`: CSV outputs
- `figures/`: plotted figures

## 4) Run

From project root (`MEC`):

```powershell
python new_simulation/run_new_experiments.py --episodes 300 --warmup_ratio 0.6 --seeds 1,2
python new_simulation/plot_new_experiments.py --results_dir new_simulation/results --figures_dir new_simulation/figures
```

For a paper-level run, increase episodes/seeds, e.g.:

```powershell
python new_simulation/run_new_experiments.py --episodes 2000 --warmup_ratio 0.7 --seeds 1,2,3,4,5
```

Optional GPD online update:

```powershell
python new_simulation/run_new_experiments.py --enable_gpd

Tune C5 penalty strength:

```powershell
python new_simulation/run_new_experiments.py --c5_penalty_weight 20 --c5_penalty_scale 1
```
```

## 5) Notes

- `Full` may overlap with some ablation variants in low-stress regions, which is expected when constraints are not active.
- In plotting, `Full` is drawn last and thicker so it remains visible under overlap.
