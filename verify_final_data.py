import pandas as pd
import os

path = 'new_simulation/results_500ep_seeds7_16_24_final/stress_scan_raw.csv'
if os.path.exists(path):
    df = pd.read_csv(path)
    print(f'✓ Total rows: {len(df)}')
    print(f'  Expected: ~432 (18 scenarios × 8 policies × 3 seeds)')
    print()
    
    for scenario in ['load_scale', 'f_scale', 'user_num']:
        subset = df[df['scenario']==scenario]
        if len(subset) > 0:
            values = sorted(subset['value'].unique())
            policies = len(subset['policy'].unique())
            seeds = len(subset['seed'].unique())
            print(f'✓ {scenario}: {len(values)} 个值, {policies} 个策略, {seeds} 个 seed = {len(subset)} 行')
        else:
            print(f'✗ {scenario}: 缺失')
    print()
    print(f'✓ Seeds: {sorted(df["seed"].unique())}')
    print(f'✓ 策略数: {len(df["policy"].unique())}')
    print(f'✓ 完整覆盖: 是' if len(df) >= 400 else '✗ 数据不完整')
else:
    print('✗ 文件未找到')
