import pandas as pd
import numpy as np

df = pd.read_csv('new_simulation/results_paper_c5_balanced_quick_c6new_20260402/stress_scan_raw.csv')

# Check configuration
print('=== EXPERIMENT CONFIGURATION ===')
print(f'Episodes per run: {df["episodes"].iloc[0]}')
print(f'Warmup ratio: {df["warmup_ratio"].iloc[0]}')
effective = int(df['episodes'].iloc[0] * (1 - df['warmup_ratio'].iloc[0]))
print(f'Effective episodes for metrics: {effective}')
print(f'Number of unique seeds: {df["seed"].nunique()}')
print(f'Seeds used: {sorted(df["seed"].unique())}')

# Check Stand-alone MEC variability
print('\n=== STAND-ALONE MEC Analysis ===')
sam = df[df['policy'] == 'Stand-alone MEC']
print(f'Total Stand-alone MEC rows: {len(sam)}')

user_num_sam = sam[sam['scenario']=='user_num'].sort_values(['value','seed'])
print('\nStand-alone MEC results for user_num scenario:')
print(user_num_sam[['value','seed','p_c6_viol','p_c7_viol','p_c8_viol']].to_string())

# Check variability across all policies
print('\n=== SEED VARIABILITY (stddev) ===')
print('Policy,Avg_STD,Min_STD,Max_STD')
for policy in ['Full', 'IQL', 'Stand-alone MEC', 'Random']:
    subset = df[df['policy']==policy]
    grouped = subset.groupby(['scenario','value'])['p_c6_viol'].std()
    avg_std = grouped.mean()
    print(f'{policy},{avg_std:.6f},{grouped.min():.6f},{grouped.max():.6f}')

# Check determinism of Stand-alone MEC
print('\n=== DETERMINISM CHECK (are seed results identical?) ===')
sam_pivot = df[df['policy']=='Stand-alone MEC'].pivot_table(
    index=['scenario','value'],
    columns='seed',
    values='p_c6_viol'
)
sam_pivot['identical'] = (sam_pivot[1] == sam_pivot[2]).astype(int)
print(f'Identical seed results: {sam_pivot["identical"].sum()} / {len(sam_pivot)}')
sam_pivot['diff'] = abs(sam_pivot[1] - sam_pivot[2])
print(f'Max difference: {sam_pivot["diff"].max():.10f}')
print(f'Avg difference: {sam_pivot["diff"].mean():.10f}')

# Compare with other policies
print('\n=== VARIABILITY COMPARISON ===')
for policy in ['Full', 'IQL', 'Stand-alone MEC', 'Random']:
    subset = df[df['policy']==policy]
    pivot = subset.pivot_table(
        index=['scenario','value'],
        columns='seed',
        values='p_c6_viol'
    )
    pivot['diff'] = abs(pivot[1] - pivot[2])
    print(f'{policy:20s}: max_diff={pivot["diff"].max():.10f}, avg_diff={pivot["diff"].mean():.10f}')
