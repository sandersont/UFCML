# save as: setup_project.py
# run with: python setup_project.py
# Creates directories, .gitignore, requirements.txt — that's it

import os

ROOT = "."

dirs = ["data", "models", "notebooks", "src"]
for d in dirs:
    os.makedirs(f"{ROOT}/{d}", exist_ok=True)
    print(f"  ✅ {d}/")

with open(f"{ROOT}/.gitignore", "w") as f:
    f.write("""__pycache__/
*.pyc
data/*.csv
data/*.json
.ipynb_checkpoints/
*.egg-info/
venv/
""")
print("  ✅ .gitignore")

with open(f"{ROOT}/requirements.txt", "w") as f:
    f.write("""requests
beautifulsoup4
pandas
numpy
matplotlib
seaborn
scikit-learn
lxml
tqdm
ipykernel
""")
print("  ✅ requirements.txt")

print("\nDone. Now run:")
print("  pip install -r requirements.txt")
print("  python create_01_scraper.py")
print("  python create_02_cleaning.py")