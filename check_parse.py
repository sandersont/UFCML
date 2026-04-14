# save as: /workspaces/UFCML/check_parse.py
# run with: python check_parse.py

import pandas as pd
import os

# Find the file
for path in ['./data/fighters_full.csv', './notebooks/data/fighters_full.csv']:
    if os.path.exists(path):
        print(f"Found: {path}")
        df = pd.read_csv(path)
        break
else:
    # Just search for it
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f == 'fighters_full.csv':
                path = os.path.join(root, f)
                print(f"Found: {path}")
                df = pd.read_csv(path)
                break

for col in ['slpm', 'sapm', 'td_avg', 'sub_avg', 'str_acc_pct', 'str_def_pct']:
    if col in df.columns:
        print(f"\n{col}:")
        print(f"  dtype: {df[col].dtype}")
        print(f"  sample: {df[col].dropna().head(10).tolist()}")
    else:
        print(f"\n{col}: NOT IN COLUMNS")

print(f"\nAll columns: {list(df.columns)}")