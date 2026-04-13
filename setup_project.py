"""
Run from /workspaces/UFCML/:
   python setup_project.py
   pip install -r requirements.txt
"""

import os
import json

ROOT = "."

dirs = [
    f"{ROOT}/data",
    f"{ROOT}/models",
    f"{ROOT}/notebooks",
    f"{ROOT}/src",
]
for d in dirs:
    os.makedirs(d, exist_ok=True)

with open(f"{ROOT}/.gitignore", "w") as f:
    f.write("""__pycache__/
*.pyc
data/*.csv
data/*.json
.ipynb_checkpoints/
*.egg-info/
venv/
""")

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

def make_notebook(filepath, cells):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0"
            }
        },
        "cells": []
    }
    for cell in cells:
        cell_type = cell.get("type", "code")
        source = cell["source"]
        if isinstance(source, str):
            source = source.strip().split("\n")
            source = [line + "\n" for line in source[:-1]] + [source[-1]]
        nb_cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source,
        }
        if cell_type == "code":
            nb_cell["execution_count"] = None
            nb_cell["outputs"] = []
        nb["cells"].append(nb_cell)

    with open(filepath, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"  ✅ Created {filepath}")


# ═════════════════════════════════════════════
# NOTEBOOK 1: 01_scraper.ipynb
# ═════════════════════════════════════════════
scraper_cells = [
    {
        "type": "markdown",
        "source": "# UFC Stats Scraper\\nScrapes event, fight, and fighter data from ufcstats.com\\n\\nUses 10 concurrent threads for speed."
    },
    {
        "type": "code",
        "source": """# Cell 1 — Install Dependencies
!pip install requests beautifulsoup4 pandas numpy lxml tqdm"""
    },
    {
        "type": "code",
        "source": """# Cell 2 — Imports & Setup
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from tqdm import tqdm
import os
import json
import concurrent.futures
import threading

BASE_URL = "http://www.ufcstats.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
}
WORKERS = 10
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

session = requests.Session()
session.headers.update(HEADERS)

def get_soup(url):
    response = session.get(url)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")

print(f"✅ Ready — using {WORKERS} threads")"""
    },
    {
        "type": "code",
        "source": """# Cell 3 — Scrape All Events
def scrape_all_events():
    url = f"{BASE_URL}/statistics/events/completed?page=all"
    soup = get_soup(url)

    events = []
    rows = soup.select("tr.b-statistics__table-row")

    for row in rows:
        link = row.select_one("a.b-link")
        if link:
            date_cell = row.select("td")
            event_name = link.text.strip()
            event_url = link["href"]
            event_date = date_cell[-1].text.strip() if len(date_cell) > 1 else ""
            location = ""
            if len(date_cell) > 2:
                location = date_cell[1].text.strip()

            events.append({
                "event_name": event_name,
                "event_url": event_url,
                "event_date": event_date,
                "location": location
            })

    print(f"Found {len(events)} events")
    return events

events = scrape_all_events()
events_df = pd.DataFrame(events)
events_df.to_csv(f"{DATA_DIR}/events.csv", index=False)
print(f"✅ Saved {len(events_df)} events")
events_df.head(10)"""
    },
    {
        "type": "code",
        "source": """# Cell 4 — Scrape Fights From All Events (10 Workers)
def scrape_event_fights(event_row):
    event_url = event_row["event_url"]
    event_name = event_row["event_name"]
    event_date = event_row["event_date"]

    try:
        soup = get_soup(event_url)
    except Exception as e:
        return []

    fights = []
    rows = soup.select("tr.b-fight-details__table-row")[1:]

    for row in rows:
        cols = row.select("td")
        if len(cols) < 10:
            continue

        fight_url_tag = row.select_one("a.b-flag")
        fight_url = fight_url_tag["href"] if fight_url_tag else None

        fighter_links = cols[1].select("a")
        if len(fighter_links) < 2:
            continue

        fighter1 = fighter_links[0].text.strip()
        fighter2 = fighter_links[1].text.strip()

        result_icons = cols[0].select("i")
        win_loss = [icon.text.strip() for icon in result_icons]

        def get_col_values(col):
            paragraphs = col.select("p")
            return [p.text.strip() for p in paragraphs]

        kd_vals = get_col_values(cols[2])
        str_vals = get_col_values(cols[3])
        td_vals = get_col_values(cols[4])
        sub_vals = get_col_values(cols[5])

        weight_class = cols[6].text.strip()
        method = cols[7].text.strip()
        fight_round = cols[8].text.strip()
        fight_time = cols[9].text.strip()

        fight_data = {
            "event_name": event_name,
            "event_date": event_date,
            "fight_url": fight_url,
            "fighter_1": fighter1,
            "fighter_2": fighter2,
            "winner": fighter1 if (win_loss and win_loss[0].lower() == "win") else (
                fighter2 if (len(win_loss) > 1 and win_loss[1].lower() == "win") else "Draw/NC"
            ),
            "f1_kd": kd_vals[0] if len(kd_vals) > 0 else None,
            "f2_kd": kd_vals[1] if len(kd_vals) > 1 else None,
            "f1_str": str_vals[0] if len(str_vals) > 0 else None,
            "f2_str": str_vals[1] if len(str_vals) > 1 else None,
            "f1_td": td_vals[0] if len(td_vals) > 0 else None,
            "f2_td": td_vals[1] if len(td_vals) > 1 else None,
            "f1_sub": sub_vals[0] if len(sub_vals) > 0 else None,
            "f2_sub": sub_vals[1] if len(sub_vals) > 1 else None,
            "weight_class": weight_class,
            "method": method,
            "round": fight_round,
            "time": fight_time,
        }
        fights.append(fight_data)

    return fights

event_rows = events_df.to_dict("records")
all_fights = []

print(f"Scraping fights from {len(event_rows)} events with {WORKERS} threads...")

with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(scrape_event_fights, row): row for row in event_rows}
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Events"):
        result = future.result()
        if result:
            all_fights.extend(result)

fights_df = pd.DataFrame(all_fights)
fights_df.to_csv(f"{DATA_DIR}/fights.csv", index=False)
print(f"\\n✅ Saved {len(fights_df)} fights")
fights_df.head(10)"""
    },
    {
        "type": "code",
        "source": """# Cell 5 — Scrape Fighter Directory (10 Workers)
def scrape_fighters_for_letter(char):
    url = f"{BASE_URL}/statistics/fighters?char={char}&page=all"
    try:
        soup = get_soup(url)
    except Exception as e:
        return []

    fighters_list = []
    rows = soup.select("tr.b-statistics__table-row")

    for row in rows:
        cols = row.select("td")
        if len(cols) < 10:
            continue

        link = cols[0].select_one("a")
        if not link:
            continue

        fighter = {
            "fighter_url": link["href"],
            "first_name": cols[0].text.strip(),
            "last_name": cols[1].text.strip(),
            "nickname": cols[2].text.strip(),
            "height": cols[3].text.strip(),
            "weight": cols[4].text.strip(),
            "reach": cols[5].text.strip(),
            "stance": cols[6].text.strip(),
            "wins": cols[7].text.strip(),
            "losses": cols[8].text.strip(),
            "draws": cols[9].text.strip(),
        }
        fighters_list.append(fighter)

    return fighters_list

letters = list("abcdefghijklmnopqrstuvwxyz")
all_fighters = []

print(f"Scraping fighter directory with {WORKERS} threads...")

with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(scrape_fighters_for_letter, c): c for c in letters}
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Fighter pages"):
        result = future.result()
        if result:
            all_fighters.extend(result)

fighters_df = pd.DataFrame(all_fighters)
fighters_df.to_csv(f"{DATA_DIR}/fighters.csv", index=False)
print(f"\\n✅ Saved {len(fighters_df)} fighters")
fighters_df.head(10)"""
    },
    {
        "type": "code",
        "source": """# Cell 6 — Scrape Detailed Fight Stats (10 Workers)
def parse_fight_table(table):
    headers = [th.text.strip() for th in table.select("thead th")]
    rows_data = []
    for row in table.select("tbody tr"):
        cols = row.select("td")
        row_values = []
        for col in cols:
            paragraphs = col.select("p")
            if paragraphs:
                vals = [p.text.strip() for p in paragraphs]
                row_values.append(vals)
            else:
                row_values.append([col.text.strip()])
        rows_data.append(row_values)
    return {"headers": headers, "rows": rows_data}

def scrape_fight_details(fight_url):
    if not fight_url:
        return None
    soup = get_soup(fight_url)
    details = {}

    result_section = soup.select_one("div.b-fight-details__persons")
    if result_section:
        fighters = result_section.select("div.b-fight-details__person")
        for i, fighter in enumerate(fighters, 1):
            name = fighter.select_one("a.b-fight-details__person-link")
            status = fighter.select_one("i.b-fight-details__person-status")
            details[f"fighter_{i}_name"] = name.text.strip() if name else ""
            details[f"fighter_{i}_status"] = status.text.strip() if status else ""

    info_section = soup.select("div.b-fight-details__content p.b-fight-details__text")
    for p in info_section:
        text = p.get_text(separator="|", strip=True)
        items = text.split("|")
        for item in items:
            if ":" in item:
                key, val = item.split(":", 1)
                details[key.strip().lower().replace(" ", "_")] = val.strip()

    tables = soup.select("table.b-fight-details__table")
    if len(tables) >= 1:
        details["totals"] = parse_fight_table(tables[0])
    if len(tables) >= 2:
        details["significant_strikes"] = parse_fight_table(tables[1])

    details["fight_url"] = fight_url
    return details

def scrape_one(url):
    try:
        return scrape_fight_details(url)
    except Exception as e:
        return None

fight_urls = fights_df["fight_url"].dropna().unique()
fight_details_list = []
CHECKPOINT_EVERY = 500

print(f"Scraping {len(fight_urls)} fight detail pages with {WORKERS} threads...")

with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(scrape_one, url): url for url in fight_urls}
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Fight details"):
        result = future.result()
        if result:
            fight_details_list.append(result)

        if len(fight_details_list) % CHECKPOINT_EVERY == 0 and len(fight_details_list) > 0:
            with open(f"{DATA_DIR}/fight_details_checkpoint.json", "w") as f:
                json.dump(fight_details_list, f)

with open(f"{DATA_DIR}/fight_details.json", "w") as f:
    json.dump(fight_details_list, f)

print(f"\\n✅ Saved {len(fight_details_list)} detailed fight records")"""
    },
    {
        "type": "code",
        "source": """# Cell 7 — Verification
print("=" * 50)
print("SCRAPING SUMMARY")
print("=" * 50)
print(f"Events:         {len(events_df)}")
print(f"Fights:         {len(fights_df)}")
print(f"Fighters:       {len(fighters_df)}")
print(f"Fight Details:  {len(fight_details_list)}")
print(f"\\nFiles saved in: {DATA_DIR}/")
print(f"  - events.csv")
print(f"  - fights.csv")
print(f"  - fighters.csv")
print(f"  - fight_details.json")"""
    },
]

# ═════════════════════════════════════════════
# NOTEBOOK 2: 02_data_cleaning.ipynb
# ═════════════════════════════════════════════
cleaning_cells = [
    {
        "type": "markdown",
        "source": "# UFC Data Cleaning\\nProcess raw scraped data into analysis-ready DataFrames."
    },
    {
        "type": "code",
        "source": """# Cell 1 — Imports
import pandas as pd
import numpy as np
import json
import re

DATA_DIR = "./data"
print("✅ Ready")"""
    },
    {
        "type": "code",
        "source": """# Cell 2 — Load Raw Data
events = pd.read_csv(f"{DATA_DIR}/events.csv")
fights = pd.read_csv(f"{DATA_DIR}/fights.csv")
fighters = pd.read_csv(f"{DATA_DIR}/fighters.csv")

with open(f"{DATA_DIR}/fight_details.json", "r") as f:
    fight_details = json.load(f)

print(f"Events:        {events.shape}")
print(f"Fights:        {fights.shape}")
print(f"Fighters:      {fighters.shape}")
print(f"Fight details: {len(fight_details)}")
events.head()"""
    },
    {
        "type": "code",
        "source": """# Cell 3 — Clean Events
events_clean = events.copy()
events_clean["event_date"] = pd.to_datetime(events_clean["event_date"], format="mixed", errors="coerce")
events_clean = events_clean.sort_values("event_date", ascending=False).reset_index(drop=True)

print(f"Date range: {events_clean['event_date'].min()} to {events_clean['event_date'].max()}")
events_clean.head()"""
    },
    {
        "type": "code",
        "source": """# Cell 4 — Clean Fighters
def parse_height_to_inches(height_str):
    if pd.isna(height_str) or str(height_str).strip() in ("", "--"):
        return np.nan
    match = re.search(r"(\\d+)'\\s*(\\d+)\\"?", str(height_str))
    if match:
        feet, inches = int(match.group(1)), int(match.group(2))
        return feet * 12 + inches
    return np.nan

def parse_reach_to_inches(reach_str):
    if pd.isna(reach_str) or str(reach_str).strip() in ("", "--"):
        return np.nan
    match = re.search(r"([\\d.]+)", str(reach_str))
    return float(match.group(1)) if match else np.nan

def parse_weight_to_lbs(weight_str):
    if pd.isna(weight_str) or str(weight_str).strip() in ("", "--"):
        return np.nan
    match = re.search(r"([\\d.]+)", str(weight_str))
    return float(match.group(1)) if match else np.nan

fighters_clean = fighters.copy()
fighters_clean["full_name"] = (fighters_clean["first_name"].fillna("") + " " + fighters_clean["last_name"].fillna("")).str.strip()
fighters_clean["height_inches"] = fighters_clean["height"].apply(parse_height_to_inches)
fighters_clean["reach_inches"] = fighters_clean["reach"].apply(parse_reach_to_inches)
fighters_clean["weight_lbs"] = fighters_clean["weight"].apply(parse_weight_to_lbs)

for col in ["wins", "losses", "draws"]:
    fighters_clean[col] = pd.to_numeric(fighters_clean[col], errors="coerce").fillna(0).astype(int)

fighters_clean["total_fights"] = fighters_clean["wins"] + fighters_clean["losses"] + fighters_clean["draws"]
fighters_clean["win_pct"] = np.where(fighters_clean["total_fights"] > 0, fighters_clean["wins"] / fighters_clean["total_fights"], 0)

print(f"Fighters: {fighters_clean.shape}")
fighters_clean.head()"""
    },
    {
        "type": "code",
        "source": """# Cell 5 — Clean Fights
def parse_strikes(strike_str):
    if pd.isna(strike_str) or str(strike_str).strip() in ("", "--"):
        return np.nan, np.nan
    match = re.search(r"(\\d+)\\s+of\\s+(\\d+)", str(strike_str))
    if match:
        return int(match.group(1)), int(match.group(2))
    return np.nan, np.nan

fights_clean = fights.copy()

event_dates = events_clean.set_index("event_name")["event_date"]
fights_clean["event_date"] = fights_clean["event_name"].map(event_dates)
fights_clean["event_date"] = pd.to_datetime(fights_clean["event_date"], format="mixed", errors="coerce")

for prefix in ["f1", "f2"]:
    landed, attempted = zip(*fights_clean[f"{prefix}_str"].apply(parse_strikes))
    fights_clean[f"{prefix}_str_landed"] = pd.Series(landed)
    fights_clean[f"{prefix}_str_attempted"] = pd.Series(attempted)
    fights_clean[f"{prefix}_str_acc"] = np.where(
        fights_clean[f"{prefix}_str_attempted"] > 0,
        fights_clean[f"{prefix}_str_landed"] / fights_clean[f"{prefix}_str_attempted"],
        0
    )

    td_landed, td_attempted = zip(*fights_clean[f"{prefix}_td"].apply(parse_strikes))
    fights_clean[f"{prefix}_td_landed"] = pd.Series(td_landed)
    fights_clean[f"{prefix}_td_attempted"] = pd.Series(td_attempted)

    fights_clean[f"{prefix}_kd"] = pd.to_numeric(fights_clean[f"{prefix}_kd"], errors="coerce")
    fights_clean[f"{prefix}_sub"] = pd.to_numeric(fights_clean[f"{prefix}_sub"], errors="coerce")

fights_clean["round"] = pd.to_numeric(fights_clean["round"], errors="coerce")

def time_to_seconds(t):
    if pd.isna(t) or str(t).strip() in ("", "--"):
        return np.nan
    match = re.match(r"(\\d+):(\\d+)", str(t))
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    return np.nan

fights_clean["time_seconds"] = fights_clean["time"].apply(time_to_seconds)
fights_clean["total_time_seconds"] = ((fights_clean["round"] - 1) * 5 * 60) + fights_clean["time_seconds"]

fights_clean["f1_win"] = (fights_clean["winner"] == fights_clean["fighter_1"]).astype(int)

fights_clean["method_clean"] = fights_clean["method"].str.strip().str.upper()
fights_clean["finish_type"] = fights_clean["method_clean"].apply(lambda x:
    "KO/TKO" if "KO" in str(x) else
    "SUB" if "SUB" in str(x) else
    "DEC" if "DEC" in str(x) else
    "OTHER"
)

fights_clean["weight_class"] = fights_clean["weight_class"].str.strip()

print(f"Fights: {fights_clean.shape}")
fights_clean.head()"""
    },
    {
        "type": "code",
        "source": """# Cell 6 — Save Cleaned Data
events_clean.to_csv(f"{DATA_DIR}/events_clean.csv", index=False)
fighters_clean.to_csv(f"{DATA_DIR}/fighters_clean.csv", index=False)
fights_clean.to_csv(f"{DATA_DIR}/fights_clean.csv", index=False)

print("✅ Cleaned data saved:")
print(f"   Events:   {len(events_clean)} rows → {DATA_DIR}/events_clean.csv")
print(f"   Fighters: {len(fighters_clean)} rows → {DATA_DIR}/fighters_clean.csv")
print(f"   Fights:   {len(fights_clean)} rows → {DATA_DIR}/fights_clean.csv")"""
    },
]

# ═════════════════════════════════════════════
# NOTEBOOK 3: 03_eda.ipynb
# ═════════════════════════════════════════════
eda_cells = [
    {
        "type": "markdown",
        "source": "# UFC Fight Data — Exploratory Data Analysis"
    },
    {
        "type": "code",
        "source": """# Cell 1 — Imports & Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12

DATA_DIR = "./data"
print("✅ Ready")"""
    },
    {
        "type": "code",
        "source": """# Cell 2 — Load Cleaned Data
fights = pd.read_csv(f"{DATA_DIR}/fights_clean.csv", parse_dates=["event_date"])
fighters = pd.read_csv(f"{DATA_DIR}/fighters_clean.csv")

print("FIGHTS DATASET")
print("=" * 50)
print(f"Shape: {fights.shape}")
print(f"Date range: {fights['event_date'].min()} to {fights['event_date'].max()}")
print(f"\\nColumn types:\\n{fights.dtypes}")
print(f"\\nMissing values:\\n{fights.isnull().sum()}")
print(f"\\nBasic stats:\\n{fights.describe()}")"""
    },
    {
        "type": "code",
        "source": """# Cell 3 — Fight Outcome Distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

method_counts = fights["finish_type"].value_counts()
axes[0].pie(method_counts, labels=method_counts.index, autopct="%1.1f%%",
            startangle=90, colors=sns.color_palette("Set2"))
axes[0].set_title("Fight Finish Methods", fontsize=14, fontweight="bold")

f1_wr = fights["f1_win"].mean()
axes[1].bar(["Fighter 1 Wins", "Fighter 2 Wins"], [f1_wr, 1 - f1_wr],
            color=["#2ecc71", "#e74c3c"])
axes[1].set_ylabel("Proportion")
axes[1].set_title(f"Positional Bias Check\\n(F1 win rate: {f1_wr:.3f})", fontsize=14, fontweight="bold")
axes[1].set_ylim(0, 1)

round_counts = fights["round"].value_counts().sort_index()
axes[2].bar(round_counts.index, round_counts.values, color=sns.color_palette("viridis", len(round_counts)))
axes[2].set_xlabel("Round")
axes[2].set_ylabel("Number of Fights")
axes[2].set_title("Fight Ending Round Distribution", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/eda_outcome_distribution.png", dpi=150, bbox_inches="tight")
plt.show()"""
    },
    {
        "type": "code",
        "source": """# Cell 4 — Fights Over Time
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

fights["year"] = fights["event_date"].dt.year
yearly = fights.groupby("year").size()
axes[0].plot(yearly.index, yearly.values, marker="o", linewidth=2, markersize=6)
axes[0].fill_between(yearly.index, yearly.values, alpha=0.3)
axes[0].set_title("Number of UFC Fights Per Year", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Number of Fights")

finish_by_year = fights.groupby(["year", "finish_type"]).size().unstack(fill_value=0)
finish_pct = finish_by_year.div(finish_by_year.sum(axis=1), axis=0)
finish_pct.plot(kind="area", stacked=True, ax=axes[1], alpha=0.7)
axes[1].set_title("Fight Finish Methods Over Time (Proportion)", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Proportion")
axes[1].legend(title="Finish Type", loc="upper right")

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/eda_time_trends.png", dpi=150, bbox_inches="tight")
plt.show()"""
    },
    {
        "type": "code",
        "source": """# Cell 5 — Weight Class Analysis
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

wc_counts = fights["weight_class"].value_counts().head(15)
axes[0].barh(range(len(wc_counts)), wc_counts.values, color=sns.color_palette("coolwarm", len(wc_counts)))
axes[0].set_yticks(range(len(wc_counts)))
axes[0].set_yticklabels(wc_counts.index)
axes[0].set_xlabel("Number of Fights")
axes[0].set_title("Fights by Weight Class (Top 15)", fontsize=14, fontweight="bold")
axes[0].invert_yaxis()

ko_rates = fights.groupby("weight_class").apply(
    lambda x: (x["finish_type"] == "KO/TKO").mean()
).sort_values(ascending=True)
ko_rates = ko_rates[ko_rates.index.isin(wc_counts.index)]
ko_rates.plot(kind="barh", ax=axes[1], color=sns.color_palette("Reds_r", len(ko_rates)))
axes[1].set_xlabel("KO/TKO Rate")
axes[1].set_title("KO/TKO Rate by Weight Class", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/eda_weight_classes.png", dpi=150, bbox_inches="tight")
plt.show()"""
    },
    {
        "type": "code",
        "source": """# Cell 6 — Striking Analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

fights["winner_str_landed"] = np.where(
    fights["f1_win"] == 1, fights["f1_str_landed"], fights["f2_str_landed"]
)
fights["loser_str_landed"] = np.where(
    fights["f1_win"] == 1, fights["f2_str_landed"], fights["f1_str_landed"]
)

axes[0, 0].hist(fights["winner_str_landed"].dropna(), bins=50, alpha=0.6, label="Winner", color="#2ecc71")
axes[0, 0].hist(fights["loser_str_landed"].dropna(), bins=50, alpha=0.6, label="Loser", color="#e74c3c")
axes[0, 0].set_xlabel("Significant Strikes Landed")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].set_title("Strikes Landed: Winner vs Loser", fontsize=13, fontweight="bold")
axes[0, 0].legend()

fights["winner_str_acc"] = np.where(fights["f1_win"] == 1, fights["f1_str_acc"], fights["f2_str_acc"])
fights["loser_str_acc"] = np.where(fights["f1_win"] == 1, fights["f2_str_acc"], fights["f1_str_acc"])

axes[0, 1].hist(fights["winner_str_acc"].dropna(), bins=40, alpha=0.6, label="Winner", color="#2ecc71")
axes[0, 1].hist(fights["loser_str_acc"].dropna(), bins=40, alpha=0.6, label="Loser", color="#e74c3c")
axes[0, 1].set_xlabel("Striking Accuracy")
axes[0, 1].set_title("Striking Accuracy: Winner vs Loser", fontsize=13, fontweight="bold")
axes[0, 1].legend()

kd_diff = fights["f1_kd"].fillna(0) - fights["f2_kd"].fillna(0)
fights["kd_diff"] = kd_diff
kd_win_rate = fights.groupby("kd_diff")["f1_win"].mean()
kd_win_rate = kd_win_rate[(kd_win_rate.index >= -3) & (kd_win_rate.index <= 3)]
axes[1, 0].bar(kd_win_rate.index, kd_win_rate.values, color=sns.color_palette("RdYlGn", len(kd_win_rate)))
axes[1, 0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.7)
axes[1, 0].set_xlabel("Knockdown Differential (F1 - F2)")
axes[1, 0].set_ylabel("F1 Win Rate")
axes[1, 0].set_title("Knockdown Differential vs Win Rate", fontsize=13, fontweight="bold")

fights["str_diff"] = fights["f1_str_landed"].fillna(0) - fights["f2_str_landed"].fillna(0)
fights["str_diff_bin"] = pd.cut(fights["str_diff"], bins=20)
str_win = fights.groupby("str_diff_bin", observed=True)["f1_win"].agg(["mean", "count"])
str_win = str_win[str_win["count"] >= 10]

x_vals = [interval.mid for interval in str_win.index]
axes[1, 1].scatter(x_vals, str_win["mean"], s=str_win["count"], alpha=0.6, c="#3498db")
axes[1, 1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.7)
axes[1, 1].set_xlabel("Strike Differential (F1 - F2)")
axes[1, 1].set_ylabel("F1 Win Probability")
axes[1, 1].set_title("Strike Differential vs Win Probability\\n(size = sample count)", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/eda_striking.png", dpi=150, bbox_inches="tight")
plt.show()"""
    },
    {
        "type": "code",
        "source": """# Cell 7 — Fighter Physical Attributes
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(fighters["height_inches"].dropna(), bins=30, color="#3498db", edgecolor="white")
axes[0].set_xlabel("Height (inches)")
axes[0].set_title("Fighter Height Distribution", fontweight="bold")

axes[1].hist(fighters["reach_inches"].dropna(), bins=30, color="#e67e22", edgecolor="white")
axes[1].set_xlabel("Reach (inches)")
axes[1].set_title("Fighter Reach Distribution", fontweight="bold")

axes[2].hist(fighters["weight_lbs"].dropna(), bins=30, color="#2ecc71", edgecolor="white")
axes[2].set_xlabel("Weight (lbs)")
axes[2].set_title("Fighter Weight Distribution", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/eda_physical.png", dpi=150, bbox_inches="tight")
plt.show()"""
    },
    {
        "type": "code",
        "source": """# Cell 8 — Correlation Analysis
numeric_cols = [
    "f1_kd", "f2_kd", "f1_str_landed", "f2_str_landed",
    "f1_str_acc", "f2_str_acc", "f1_td_landed", "f2_td_landed",
    "f1_sub", "f2_sub", "round", "total_time_seconds", "f1_win"
]
corr_matrix = fights[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title("Feature Correlation Matrix", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/eda_correlations.png", dpi=150, bbox_inches="tight")
plt.show()"""
    },
    {
        "type": "code",
        "source": """# Cell 9 — Key Insights
print("=" * 60)
print("KEY STATISTICAL INSIGHTS")
print("=" * 60)

print(f"\\n📊 DATASET OVERVIEW:")
print(f"   Total fights: {len(fights)}")
print(f"   Total events: {fights['event_name'].nunique()}")
print(f"   Total fighters: {len(fighters)}")
print(f"   Date range: {fights['event_date'].min()} to {fights['event_date'].max()}")

print(f"\\n🥊 FIGHT OUTCOMES:")
print(f"   KO/TKO rate: {(fights['finish_type'] == 'KO/TKO').mean():.1%}")
print(f"   Submission rate: {(fights['finish_type'] == 'SUB').mean():.1%}")
print(f"   Decision rate: {(fights['finish_type'] == 'DEC').mean():.1%}")
print(f"   Average fight length: {fights['total_time_seconds'].mean()/60:.1f} minutes")

print(f"\\n📈 STRIKING STATS:")
print(f"   Avg sig. strikes landed (winner): {fights['winner_str_landed'].mean():.1f}")
print(f"   Avg sig. strikes landed (loser): {fights['loser_str_landed'].mean():.1f}")
print(f"   Avg striking accuracy (winner): {fights['winner_str_acc'].mean():.1%}")
print(f"   Avg striking accuracy (loser): {fights['loser_str_acc'].mean():.1%}")

print(f"\\n⚖️ POSITIONAL ANALYSIS (Red Corner = Favorite):")
print(f"   Fighter 1 (red corner) win rate: {fights['f1_win'].mean():.1%}")
print(f"   This suggests {'significant' if abs(fights['f1_win'].mean() - 0.5) > 0.05 else 'minimal'} positional bias")
print(f"   Red corner baseline will be our naive model to beat")"""
    },
    {
        "type": "code",
        "source": """# Cell 10 — Correlations & Data Quality
print("🔗 TOP CORRELATIONS WITH F1 WINNING:")
win_corr = corr_matrix["f1_win"].drop("f1_win").sort_values(key=abs, ascending=False)
for feat, corr in win_corr.head(8).items():
    direction = "↑" if corr > 0 else "↓"
    print(f"   {direction} {feat}: {corr:.3f}")

print(f"\\n{'=' * 60}")
print("DATA QUALITY REPORT")
print("=" * 60)

print(f"\\nMissing Value Percentages:")
for col in fights.columns:
    missing_pct = fights[col].isnull().mean() * 100
    if missing_pct > 0:
        print(f"   {col}: {missing_pct:.1f}%")

print(f"\\nDraw/NC fights: {(fights['winner'] == 'Draw/NC').sum()}")
print(f"Weight classes found: {fights['weight_class'].nunique()}")
print(f"\\nWeight class breakdown:")
print(fights["weight_class"].value_counts().to_string())"""
    },
]

# ─────────────────────────────────────────────
# CREATE EVERYTHING
# ─────────────────────────────────────────────
print("Creating UFC Predictor project...\n")
print(f"  ✅ Created directories")
print(f"  ✅ Created {ROOT}/.gitignore")
print(f"  ✅ Created {ROOT}/requirements.txt")

make_notebook(f"{ROOT}/notebooks/01_scraper.ipynb", scraper_cells)
make_notebook(f"{ROOT}/notebooks/02_data_cleaning.ipynb", cleaning_cells)
make_notebook(f"{ROOT}/notebooks/03_eda.ipynb", eda_cells)

print(f"\n🎉 Done! Now run:")
print(f"   pip install -r requirements.txt")
print(f"   Then open notebooks/01_scraper.ipynb and Run All")