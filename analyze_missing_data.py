import pandas as pd

print('=== DETAILED MISSING DATA ANALYSIS ===')
print()

df = pd.read_csv('new_simulation/results_paper_c5_balanced_1000ep_3seed/stress_scan_raw.csv')

print('Completed load_scale section:')
load = df[df['scenario']=='load_scale']
for val in [0.5, 1.0, 1.5, 2.0]:
    subset = load[load['value']==val]
    print(f'  value={val}: {len(subset)} rows (should be 24 = 8 policies × 3 seeds)')
print()

print('Incomplete f_scale section:')
fscale = df[df['scenario']=='f_scale']
print(f'  0.4: {len(fscale[fscale["value"]==0.4])} rows')
print(f'  0.6: {len(fscale[fscale["value"]==0.6])} rows (INCOMPLETE - should be 24 but is only 3!)')
print()

print('Last few entries before experiment stopped:')
last_rows = fscale[['value','policy','seed']].sort_values(['value','policy','seed']).tail(15)
print(last_rows.to_string())

print()
print('=== ROOT CAUSE ===')
print('Experiment stopped after f_scale=0.6, policy=Full, seed=1 (first entry)')
print('This suggests the Full policy hit an error or infinite loop during episode execution')
print('Remaining scenarios never executed:')
print('  - f_scale: 0.8, 1.0, 1.2, 1.4')
print('  - user_num: all values (5-40)')
