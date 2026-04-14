# save as: /workspaces/UFCML/find_data.py
# run with: python find_data.py

import os

for root, dirs, files in os.walk('.'):
    for f in files:
        if f == 'fights_clean.csv':
            print(f"Found: {os.path.join(root, f)}")

print(f"\nCWD: {os.getcwd()}")
print(f"\nContents of ./data/:")
if os.path.exists('./data'):
    for f in os.listdir('./data'):
        print(f"  {f}")

print(f"\nContents of ./notebooks/data/:")
if os.path.exists('./notebooks/data'):
    for f in os.listdir('./notebooks/data'):
        print(f"  {f}")