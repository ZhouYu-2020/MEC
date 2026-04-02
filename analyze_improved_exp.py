import pandas as pd

df = pd.read_csv('new_simulation/results_paper_c5_balanced_1000ep_3seed/stress_scan_raw.csv')
print('=== 1000 Episodes + 3 Seeds Configuration ===')
print('Episodes per run:', df['episodes'].iloc[0])
print('Warmup ratio:', df['warmup_ratio'].iloc[0])
eff = int(df['episodes'].iloc[0] * (1 - df['warmup_ratio'].iloc[0]))
print('Effective episodes for metrics:', eff)
print('Number of unique seeds:', df['seed'].nunique())
print('Seeds used:', sorted(df['seed'].unique()))
print()
print('=== Variability Improvements (vs old 2-seed version) ===')
print('Policy,Avg_STD (new),Min,Max')
for policy in ['Full', 'IQL', 'Stand-alone MEC', 'Random', 'C6-Only']:
    subset = df[df['policy']==policy]
    if len(subset) == 0:
        continue
    grouped = subset.groupby(['scenario','value'])['p_c6_viol'].std()
    avg_std = grouped.mean()
    print('{:20s}: {:.8f}, {:.8f}, {:.8f}'.format(policy, avg_std, grouped.min(), grouped.max()))
    
# Compare with old results
print('\n=== Comparison with 2-seed version ===')
df_old = pd.read_csv('new_simulation/results_paper_c5_balanced_quick_c6new_20260402/stress_scan_raw.csv')
for policy in ['Full', 'Stand-alone MEC']:
    subset_old = df_old[df_old['policy']==policy]
    grouped_old = subset_old.groupby(['scenario','value'])['p_c6_viol'].std()
    avg_std_old = grouped_old.mean()
    
    subset_new = df[df['policy']==policy]
    grouped_new = subset_new.groupby(['scenario','value'])['p_c6_viol'].std()
    avg_std_new = grouped_new.mean()
    
    improvement = (avg_std_new - avg_std_old) / (avg_std_old + 1e-10) * 100
    print('{:20s}: old={:.8f}, new={:.8f}, change={:+.1f}%'.format(
        policy, avg_std_old, avg_std_new, improvement))
