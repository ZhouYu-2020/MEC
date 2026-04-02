import pandas as pd
import os

path = 'new_simulation/results_1000ep_seeds7_16_24/stress_scan_raw.csv'
if os.path.exists(path):
    df = pd.read_csv(path)
    print(f'✓ Total rows: {len(df)}')
    print(f'  Expected: ~432 (18 scenarios × 8 policies × 3 seeds)')
    print()
    print('Scenario coverage:')
    for scenario in ['load_scale', 'f_scale', 'user_num']:
        subset = df[df['scenario']==scenario]
        if len(subset) > 0:
            values = sorted(subset['value'].unique())
            policies = len(subset['policy'].unique())
            seeds = len(subset['seed'].unique())
            print(f'  ✓ {scenario}: {len(values)} values, {policies} policies, {seeds} seeds = {len(subset)} rows')
        else:
            print(f'  ✗ {scenario}: MISSING')
    print()
    seeds_list = sorted(df['seed'].unique())
    policies_list = sorted(df['policy'].unique())
    print(f'Seeds: {seeds_list}')
    print(f'Policies ({len(policies_list)}): {policies_list}')
else:
    print('✗ File not found')
