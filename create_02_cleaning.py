"""
save as: /workspaces/UFCML/create_02_cleaning.py
run with: python create_02_cleaning.py
Then open notebooks/02_data_cleaning.ipynb and Run All

INPUT FILES (all in ./data/):
- events.csv              (from 01_scraper Cell 2)
- fights.csv              (from 01_scraper Cell 7)
- fighters_full.csv       (from 01_scraper Cell 5)
- fight_details.json      (from 01_scraper Cell 6)

OUTPUT FILES (all in ./data/):
- events_clean.csv        (Cell 4)
- fighters_clean.csv      (Cell 5)
- fights_clean.csv        (Cell 7)

CLEANING RULES:
- Pre-2015 dropped (corner data unreliable, pre-2010 red WR = 100%)
- Draw/NC dropped (~1.5%, not predictable)
- Draw/NC detection: direct comparison, NOT regex
  (str.contains("NC") falsely matches Francisco, Duncan, Vince)
- Height/reach "--" → NaN, Weight "-- lbs." → NaN
- Profile stats have leading "|" from scraper's get_text(separator='|')
  All parse functions strip "|" first via clean_val()
- Profile pct fields "|XX%" → 0.XX float
- Profile float fields "|X.XX" → float
- Raw duplicate columns dropped after cleaning (height/weight/reach/dob/pct fields)
- Method field: embedded newlines stripped, raw method column dropped
- Fight stats from fight_details.json (full "X of Y" format)
- Sig strike breakdown from JSON Table 1 (head/body/leg/distance/clinch/ground)
- JSON totals row: [0]=names [1]=KD [2]=sig_str [3]=sig_pct [4]=total_str
                   [5]=TD [6]=TD_pct [7]=sub_att [8]=rev [9]=ctrl
- JSON sig strikes row: [0]=names [1]=sig_str [2]=sig_pct [3]=head [4]=body
                        [5]=leg [6]=distance [7]=clinch [8]=ground
- F1 = Red corner = Favorite | F2 = Blue corner = Underdog
- Post-2015 red corner WR ≈ 57.1% — this is the baseline
"""

import json

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
            "language_info": {"name": "python", "version": "3.12.0"}
        },
        "cells": []
    }
    for cell in cells:
        cell_type = cell.get("type", "code")
        source = cell["source"]
        if isinstance(source, str):
            source = source.strip().split("\n")
            source = [line + "\n" for line in source[:-1]] + [source[-1]]
        nb_cell = {"cell_type": cell_type, "metadata": {}, "source": source}
        if cell_type == "code":
            nb_cell["execution_count"] = None
            nb_cell["outputs"] = []
        nb["cells"].append(nb_cell)
    with open(filepath, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"  Created {filepath}")


cells = [
    # ═══════════════════════════════════════
    # notebooks/02_data_cleaning.ipynb — Markdown
    # ═══════════════════════════════════════
    {
        "type": "markdown",
        "source": (
            "# UFC Data Cleaning\n"
            "\n"
            "| Cell | What | Output |\n"
            "|------|------|--------|\n"
            "| 1 | Imports & constants | — |\n"
            "| 2 | Load raw data | — |\n"
            "| 3 | Year filter + drop Draw/NC | — |\n"
            "| 4 | Clean events | `data/events_clean.csv` |\n"
            "| 5 | Clean fighters (directory + profiles) | `data/fighters_clean.csv` |\n"
            "| 6 | Clean fights (JSON stats) | — |\n"
            "| 7 | Save + verify | `data/fights_clean.csv` |\n"
            "\n"
            "**F1 = Red corner = Favorite | F2 = Blue corner = Underdog**\n"
            "**Baseline: ~57% red corner win rate**"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/02_data_cleaning.ipynb — Cell 1
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/02_data_cleaning.ipynb — Cell 1: Imports & Constants\n"
            "\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import json\n"
            "import re\n"
            "import os\n"
            "\n"
            "# Auto-detect data path\n"
            "if os.path.exists('./data/fights.csv'):\n"
            "    DATA_DIR = './data'\n"
            "elif os.path.exists('../data/fights.csv'):\n"
            "    DATA_DIR = '../data'\n"
            "else:\n"
            "    raise FileNotFoundError('Cannot find data/ directory')\n"
            "\n"
            "CUTOFF_YEAR = 2015\n"
            "print(f'DATA_DIR: {DATA_DIR}')\n"
            "print(f'Cutoff year: {CUTOFF_YEAR}')"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/02_data_cleaning.ipynb — Cell 2
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/02_data_cleaning.ipynb — Cell 2: Load Raw Data\n"
            "\n"
            "events = pd.read_csv(f'{DATA_DIR}/events.csv')\n"
            "fights = pd.read_csv(f'{DATA_DIR}/fights.csv')\n"
            "fighters = pd.read_csv(f'{DATA_DIR}/fighters_full.csv')\n"
            "\n"
            "with open(f'{DATA_DIR}/fight_details.json', 'r') as f:\n"
            "    fight_details = json.load(f)\n"
            "\n"
            "print(f'Events:        {events.shape}')\n"
            "print(f'Fights:        {fights.shape}')\n"
            "print(f'Fighters:      {fighters.shape}')\n"
            "print(f'Fight details: {len(fight_details)}')\n"
            "\n"
            "# Quick sanity check\n"
            "decided = (fights['winner']==fights['fighter_1']) | (fights['winner']==fights['fighter_2'])\n"
            "f1w = (fights['winner']==fights['fighter_1']).sum()\n"
            "f2w = (fights['winner']==fights['fighter_2']).sum()\n"
            "dnc = (~decided).sum()\n"
            "print(f'\\nAll-time: Red {f1w} | Blue {f2w} | Draw/NC {dnc}')\n"
            "print(f'Red WR (decided): {f1w/(f1w+f2w):.1%}')\n"
            "\n"
            "# Verify profile columns exist\n"
            "for col in ['slpm','sapm','str_acc_pct','str_def_pct',\n"
            "            'td_avg','td_acc_pct','td_def_pct','sub_avg','dob']:\n"
            "    status = '✅' if col in fighters.columns else '❌ MISSING'\n"
            "    if col in fighters.columns:\n"
            "        sample = fighters[col].dropna().head(3).tolist()\n"
            "        print(f'  {col:15s} {status}  sample: {sample}')\n"
            "    else:\n"
            "        print(f'  {col:15s} {status}')"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/02_data_cleaning.ipynb — Cell 3
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/02_data_cleaning.ipynb — Cell 3: Year Filter + Drop Draw/NC\n"
            "\n"
            "# Parse event dates and map to fights\n"
            "events['event_date_parsed'] = pd.to_datetime(\n"
            "    events['event_date'], format='mixed', errors='coerce')\n"
            "date_map = events.set_index('event_name')['event_date_parsed']\n"
            "fights['event_date_parsed'] = fights['event_name'].map(date_map)\n"
            "fights['year'] = fights['event_date_parsed'].dt.year\n"
            "\n"
            "# Show red WR by era to justify cutoff\n"
            "print('RED CORNER WIN RATE BY ERA:')\n"
            "decided = fights[\n"
            "    (fights['winner']==fights['fighter_1']) |\n"
            "    (fights['winner']==fights['fighter_2'])\n"
            "].copy()\n"
            "decided['f1_win'] = (decided['winner']==decided['fighter_1']).astype(int)\n"
            "for label, lo, hi in [('Pre-2010',0,2010),('2010-2014',2010,2015),\n"
            "                      ('2015-2019',2015,2020),('2020+',2020,2100)]:\n"
            "    s = decided[(decided['year']>=lo)&(decided['year']<hi)]\n"
            "    if len(s)>0:\n"
            "        print(f'  {label:12s} {len(s):>5d} fights  RedWR={s[\"f1_win\"].mean():.1%}')\n"
            "\n"
            "# Apply year filter\n"
            "pre = len(fights)\n"
            "fights = fights[fights['year'] >= CUTOFF_YEAR].reset_index(drop=True)\n"
            "print(f'\\nYear filter: {pre} -> {len(fights)}')\n"
            "\n"
            "# Drop Draw/NC — direct comparison, NOT regex\n"
            "# str.contains('NC') falsely matches Francisco, Duncan, Vince\n"
            "decided_mask = (fights['winner']==fights['fighter_1']) | \\\n"
            "               (fights['winner']==fights['fighter_2'])\n"
            "dnc = (~decided_mask).sum()\n"
            "if dnc > 0:\n"
            "    print(f'\\nDropping {dnc} Draw/NC:')\n"
            "    print(fights[~decided_mask][['fighter_1','fighter_2','winner']].to_string())\n"
            "fights = fights[decided_mask].reset_index(drop=True)\n"
            "print(f'After cleanup: {len(fights)} decided fights')\n"
            "print(f'Red WR: {(fights[\"winner\"]==fights[\"fighter_1\"]).mean():.1%}')"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/02_data_cleaning.ipynb — Cell 4
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/02_data_cleaning.ipynb — Cell 4: Clean Events\n"
            "# Output: data/events_clean.csv\n"
            "\n"
            "events_clean = events.copy()\n"
            "events_clean['event_date'] = pd.to_datetime(\n"
            "    events_clean['event_date'], format='mixed', errors='coerce')\n"
            "events_clean = events_clean.sort_values('event_date', ascending=False).reset_index(drop=True)\n"
            "events_clean = events_clean[\n"
            "    events_clean['event_date'].dt.year >= CUTOFF_YEAR\n"
            "].reset_index(drop=True)\n"
            "\n"
            "events_clean.to_csv(f'{DATA_DIR}/events_clean.csv', index=False)\n"
            "print(f'Saved {len(events_clean)} events to data/events_clean.csv')\n"
            "print(f'Range: {events_clean[\"event_date\"].min()} to {events_clean[\"event_date\"].max()}')"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/02_data_cleaning.ipynb — Cell 5
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/02_data_cleaning.ipynb — Cell 5: Clean Fighters\n"
            "# Parses directory info (height/reach/weight) + profile stats (SLpM, etc.)\n"
            "# Height/reach '--' → NaN, weight '-- lbs.' → NaN\n"
            "# BUG FIX: scraper used get_text(separator='|') which left leading '|'\n"
            "#   on all profile values: '|3.29', '|38%' etc.\n"
            "#   All parse functions now strip '|' first via clean_val().\n"
            "# Raw columns dropped after cleaning — only clean versions kept.\n"
            "# Output: data/fighters_clean.csv\n"
            "\n"
            "def clean_val(v):\n"
            "    '''Strip leading | left by scraper get_text(separator=\"|\")'''\n"
            "    if pd.isna(v):\n"
            "        return ''\n"
            "    return str(v).replace('|','').strip()\n"
            "\n"
            "def parse_height(h):\n"
            "    h = clean_val(h)\n"
            "    if not h or h == '--':\n"
            "        return np.nan\n"
            "    m = re.search(r\"(\\d+)'\\s*(\\d+)\\\"\", h)\n"
            "    return int(m.group(1))*12 + int(m.group(2)) if m else np.nan\n"
            "\n"
            "def parse_reach(r):\n"
            "    r = clean_val(r)\n"
            "    if not r or r == '--':\n"
            "        return np.nan\n"
            "    m = re.search(r'([\\d.]+)', r)\n"
            "    return float(m.group(1)) if m else np.nan\n"
            "\n"
            "def parse_weight(w):\n"
            "    w = clean_val(w)\n"
            "    if not w or w == '--':\n"
            "        return np.nan\n"
            "    m = re.search(r'([\\d.]+)', w)\n"
            "    return float(m.group(1)) if m else np.nan\n"
            "\n"
            "def parse_pct(p):\n"
            "    p = clean_val(p)\n"
            "    if not p or p in ('--','0%'):\n"
            "        return np.nan\n"
            "    m = re.search(r'([\\d.]+)', p)\n"
            "    return float(m.group(1))/100.0 if m else np.nan\n"
            "\n"
            "def parse_float(v):\n"
            "    v = clean_val(v)\n"
            "    if not v or v == '--':\n"
            "        return np.nan\n"
            "    try:\n"
            "        return float(v)\n"
            "    except ValueError:\n"
            "        return np.nan\n"
            "\n"
            "fc = fighters.copy()\n"
            "fc['full_name'] = (fc['first_name'].fillna('')+' '+fc['last_name'].fillna('')).str.strip()\n"
            "\n"
            "# Physical\n"
            "fc['height_inches'] = fc['height'].apply(parse_height)\n"
            "fc['reach_inches'] = fc['reach'].apply(parse_reach)\n"
            "fc['weight_lbs'] = fc['weight'].apply(parse_weight)\n"
            "\n"
            "# Record\n"
            "for col in ['wins','losses','draws']:\n"
            "    fc[col] = pd.to_numeric(fc[col], errors='coerce').fillna(0).astype(int)\n"
            "fc['total_fights'] = fc['wins'] + fc['losses'] + fc['draws']\n"
            "fc['win_pct'] = np.where(fc['total_fights']>0, fc['wins']/fc['total_fights'], 0)\n"
            "\n"
            "# Profile career rate stats — strip | first\n"
            "fc['slpm'] = fc['slpm'].apply(parse_float)\n"
            "fc['sapm'] = fc['sapm'].apply(parse_float)\n"
            "fc['str_acc_career'] = fc['str_acc_pct'].apply(parse_pct)\n"
            "fc['str_def_career'] = fc['str_def_pct'].apply(parse_pct)\n"
            "fc['td_avg'] = fc['td_avg'].apply(parse_float)\n"
            "fc['td_acc_career'] = fc['td_acc_pct'].apply(parse_pct)\n"
            "fc['td_def_career'] = fc['td_def_pct'].apply(parse_pct)\n"
            "fc['sub_avg'] = fc['sub_avg'].apply(parse_float)\n"
            "\n"
            "# DOB\n"
            "fc['dob_parsed'] = pd.to_datetime(\n"
            "    fc['dob'].apply(clean_val), format='mixed', errors='coerce')\n"
            "\n"
            "# Drop raw duplicate columns — clean versions are kept\n"
            "raw_dupes = ['height', 'weight', 'reach', 'dob',\n"
            "             'str_acc_pct', 'str_def_pct', 'td_acc_pct', 'td_def_pct']\n"
            "fc = fc.drop(columns=raw_dupes)\n"
            "\n"
            "fighters_clean = fc\n"
            "fighters_clean.to_csv(f'{DATA_DIR}/fighters_clean.csv', index=False)\n"
            "\n"
            "print(f'Saved {len(fighters_clean)} fighters to data/fighters_clean.csv')\n"
            "print(f'Columns: {list(fighters_clean.columns)}')\n"
            "print(f'\\nPhysical:')\n"
            "for col in ['height_inches','reach_inches','weight_lbs','stance']:\n"
            "    miss = fighters_clean[col].isnull().sum()\n"
            "    print(f'  {col:15s} {miss} missing ({miss/len(fighters_clean)*100:.1f}%)')\n"
            "print(f'\\nProfile career stats:')\n"
            "for col in ['slpm','sapm','str_acc_career','str_def_career',\n"
            "            'td_avg','td_acc_career','td_def_career','sub_avg']:\n"
            "    nn = fighters_clean[col].notna().sum()\n"
            "    mn = fighters_clean[col].mean()\n"
            "    print(f'  {col:18s} {nn:>5d} non-null  mean={mn:.3f}')\n"
            "print(f'\\nDOB: {fighters_clean[\"dob_parsed\"].notna().sum()} parsed')\n"
            "\n"
            "# Verify no raw pipe values remain\n"
            "print(f'\\nVerification — sample row:')\n"
            "print(fighters_clean.iloc[1].to_string())"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/02_data_cleaning.ipynb — Cell 6
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/02_data_cleaning.ipynb — Cell 6: Clean Fights\n"
            "# Pull stats from fight_details.json (full X-of-Y format)\n"
            "# JSON totals row: [0]=names [1]=KD [2]=sig_str [3]=sig_pct [4]=total_str\n"
            "#                  [5]=TD [6]=TD_pct [7]=sub_att [8]=rev [9]=ctrl\n"
            "# JSON sig strikes: [0]=names [1]=sig_str [2]=sig_pct [3]=head [4]=body\n"
            "#                   [5]=leg [6]=distance [7]=clinch [8]=ground\n"
            "# Raw method column dropped — only method_clean kept.\n"
            "\n"
            "def parse_xofy(s):\n"
            "    '''Parse \"X of Y\" → (landed, attempted)'''\n"
            "    if pd.isna(s) or str(s).strip() in ('','--','---'):\n"
            "        return np.nan, np.nan\n"
            "    m = re.search(r'(\\d+)\\s+of\\s+(\\d+)', str(s))\n"
            "    return (int(m.group(1)), int(m.group(2))) if m else (np.nan, np.nan)\n"
            "\n"
            "def parse_ctrl(c):\n"
            "    '''Parse \"M:SS\" → seconds'''\n"
            "    if pd.isna(c) or str(c).strip() in ('','--','---'):\n"
            "        return np.nan\n"
            "    m = re.match(r'(\\d+):(\\d+)', str(c).strip())\n"
            "    return int(m.group(1))*60 + int(m.group(2)) if m else np.nan\n"
            "\n"
            "# Build lookup from JSON — totals + sig strike breakdown\n"
            "detail_stats = {}\n"
            "for fd in fight_details:\n"
            "    url = fd.get('fight_url','')\n"
            "    if not url:\n"
            "        continue\n"
            "    entry = {}\n"
            "\n"
            "    # Totals table (Table 0)\n"
            "    t_rows = fd.get('totals',{}).get('rows',[])\n"
            "    if t_rows:\n"
            "        r = t_rows[0]\n"
            "        def sg(ci, fi):\n"
            "            try: return r[ci][fi]\n"
            "            except: return None\n"
            "        entry.update({\n"
            "            'f1_kd': sg(1,0), 'f2_kd': sg(1,1),\n"
            "            'f1_sig_str': sg(2,0), 'f2_sig_str': sg(2,1),\n"
            "            'f1_total_str': sg(4,0), 'f2_total_str': sg(4,1),\n"
            "            'f1_td': sg(5,0), 'f2_td': sg(5,1),\n"
            "            'f1_sub': sg(7,0), 'f2_sub': sg(7,1),\n"
            "            'f1_rev': sg(8,0), 'f2_rev': sg(8,1),\n"
            "            'f1_ctrl': sg(9,0), 'f2_ctrl': sg(9,1),\n"
            "        })\n"
            "\n"
            "    # Sig strikes breakdown (Table 1)\n"
            "    s_rows = fd.get('significant_strikes',{}).get('rows',[])\n"
            "    if s_rows:\n"
            "        r2 = s_rows[0]\n"
            "        def sg2(ci, fi):\n"
            "            try: return r2[ci][fi]\n"
            "            except: return None\n"
            "        entry.update({\n"
            "            'f1_head': sg2(3,0), 'f2_head': sg2(3,1),\n"
            "            'f1_body': sg2(4,0), 'f2_body': sg2(4,1),\n"
            "            'f1_leg': sg2(5,0), 'f2_leg': sg2(5,1),\n"
            "            'f1_distance': sg2(6,0), 'f2_distance': sg2(6,1),\n"
            "            'f1_clinch': sg2(7,0), 'f2_clinch': sg2(7,1),\n"
            "            'f1_ground': sg2(8,0), 'f2_ground': sg2(8,1),\n"
            "        })\n"
            "\n"
            "    detail_stats[url] = entry\n"
            "\n"
            "match = sum(1 for u in fights['fight_url'] if u in detail_stats)\n"
            "print(f'JSON lookup: {len(detail_stats)} fights')\n"
            "print(f'Match rate: {match}/{len(fights)} ({match/len(fights):.1%})')\n"
            "\n"
            "# Build clean fights\n"
            "fc = fights.copy()\n"
            "\n"
            "# Map clean event dates\n"
            "ed = events_clean.set_index('event_name')['event_date']\n"
            "fc['event_date'] = fc['event_name'].map(ed)\n"
            "fc['event_date'] = pd.to_datetime(fc['event_date'], format='mixed', errors='coerce')\n"
            "\n"
            "# Pull all JSON stats\n"
            "all_json_cols = [\n"
            "    'f1_kd','f2_kd','f1_sig_str','f2_sig_str','f1_total_str','f2_total_str',\n"
            "    'f1_td','f2_td','f1_sub','f2_sub','f1_rev','f2_rev','f1_ctrl','f2_ctrl',\n"
            "    'f1_head','f2_head','f1_body','f2_body','f1_leg','f2_leg',\n"
            "    'f1_distance','f2_distance','f1_clinch','f2_clinch','f1_ground','f2_ground',\n"
            "]\n"
            "for col in all_json_cols:\n"
            "    fc[col] = fc['fight_url'].map(\n"
            "        lambda u, c=col: detail_stats.get(u, {}).get(c))\n"
            "\n"
            "# Parse sig strikes: X of Y → landed, attempted, accuracy\n"
            "for p in ['f1','f2']:\n"
            "    l, a = zip(*fc[f'{p}_sig_str'].apply(parse_xofy))\n"
            "    fc[f'{p}_str_landed'] = pd.Series(l, dtype='float64')\n"
            "    fc[f'{p}_str_attempted'] = pd.Series(a, dtype='float64')\n"
            "    fc[f'{p}_str_acc'] = np.where(\n"
            "        fc[f'{p}_str_attempted']>0,\n"
            "        fc[f'{p}_str_landed']/fc[f'{p}_str_attempted'], 0)\n"
            "\n"
            "# Parse total strikes\n"
            "for p in ['f1','f2']:\n"
            "    l, a = zip(*fc[f'{p}_total_str'].apply(parse_xofy))\n"
            "    fc[f'{p}_total_str_landed'] = pd.Series(l, dtype='float64')\n"
            "    fc[f'{p}_total_str_attempted'] = pd.Series(a, dtype='float64')\n"
            "\n"
            "# Parse takedowns\n"
            "for p in ['f1','f2']:\n"
            "    l, a = zip(*fc[f'{p}_td'].apply(parse_xofy))\n"
            "    fc[f'{p}_td_landed'] = pd.Series(l, dtype='float64')\n"
            "    fc[f'{p}_td_attempted'] = pd.Series(a, dtype='float64')\n"
            "    fc[f'{p}_td_acc'] = np.where(\n"
            "        fc[f'{p}_td_attempted']>0,\n"
            "        fc[f'{p}_td_landed']/fc[f'{p}_td_attempted'], 0)\n"
            "\n"
            "# Parse sig strike location: head/body/leg/distance/clinch/ground\n"
            "for p in ['f1','f2']:\n"
            "    for loc in ['head','body','leg','distance','clinch','ground']:\n"
            "        col = f'{p}_{loc}'\n"
            "        l, a = zip(*fc[col].apply(parse_xofy))\n"
            "        fc[f'{p}_{loc}_landed'] = pd.Series(l, dtype='float64')\n"
            "        fc[f'{p}_{loc}_attempted'] = pd.Series(a, dtype='float64')\n"
            "\n"
            "# Parse remaining numeric\n"
            "for p in ['f1','f2']:\n"
            "    fc[f'{p}_kd'] = pd.to_numeric(fc[f'{p}_kd'], errors='coerce')\n"
            "    fc[f'{p}_sub'] = pd.to_numeric(fc[f'{p}_sub'], errors='coerce')\n"
            "    fc[f'{p}_rev'] = pd.to_numeric(fc[f'{p}_rev'], errors='coerce')\n"
            "    fc[f'{p}_ctrl_seconds'] = fc[f'{p}_ctrl'].apply(parse_ctrl)\n"
            "\n"
            "# Round & time\n"
            "fc['round'] = pd.to_numeric(fc['round'], errors='coerce')\n"
            "def time_to_sec(t):\n"
            "    if pd.isna(t) or str(t).strip() in ('','--'):\n"
            "        return np.nan\n"
            "    m = re.match(r'(\\d+):(\\d+)', str(t))\n"
            "    return int(m.group(1))*60+int(m.group(2)) if m else np.nan\n"
            "fc['time_seconds'] = fc['time'].apply(time_to_sec)\n"
            "fc['total_time_seconds'] = ((fc['round']-1)*5*60) + fc['time_seconds']\n"
            "\n"
            "# Target\n"
            "fc['f1_win'] = (fc['winner']==fc['fighter_1']).astype(int)\n"
            "\n"
            "# Clean method — strip embedded newlines from scraping\n"
            "fc['method_clean'] = (fc['method']\n"
            "    .str.replace(r'\\s*\\n+\\s*', ' ', regex=True)\n"
            "    .str.strip())\n"
            "fc['finish_type'] = fc['method_clean'].str.upper().apply(lambda x:\n"
            "    'KO/TKO' if 'KO' in str(x) else\n"
            "    'SUB' if 'SUB' in str(x) else\n"
            "    'DEC' if 'DEC' in str(x) else 'OTHER')\n"
            "fc['weight_class'] = fc['weight_class'].str.strip()\n"
            "\n"
            "# Drop raw/temp columns — clean versions are kept\n"
            "drop = ['event_date_parsed','year','method',\n"
            "        'f1_str','f2_str',\n"
            "        'f1_sig_str','f2_sig_str','f1_total_str','f2_total_str',\n"
            "        'f1_td','f2_td','f1_ctrl','f2_ctrl',\n"
            "        'f1_head','f2_head','f1_body','f2_body','f1_leg','f2_leg',\n"
            "        'f1_distance','f2_distance','f1_clinch','f2_clinch',\n"
            "        'f1_ground','f2_ground']\n"
            "fc = fc.drop(columns=[c for c in drop if c in fc.columns])\n"
            "\n"
            "fights_clean = fc\n"
            "print(f'Fights: {fights_clean.shape}')\n"
            "print(f'Red WR: {fights_clean[\"f1_win\"].mean():.1%}')\n"
            "print(f'\\nFinish types:')\n"
            "print(fights_clean['finish_type'].value_counts().to_string())\n"
            "print(f'\\nColumns ({len(fights_clean.columns)}):')\n"
            "print(list(fights_clean.columns))"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/02_data_cleaning.ipynb — Cell 7
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/02_data_cleaning.ipynb — Cell 7: Save + Verify\n"
            "# Output: data/fights_clean.csv\n"
            "\n"
            "fights_clean.to_csv(f'{DATA_DIR}/fights_clean.csv', index=False)\n"
            "\n"
            "print('=' * 60)\n"
            "print('CLEANING COMPLETE')\n"
            "print('=' * 60)\n"
            "print(f'\\nFiles saved to {DATA_DIR}/:')\n"
            "print(f'  events_clean.csv    {len(events_clean):>5d} rows')\n"
            "print(f'  fighters_clean.csv  {len(fighters_clean):>5d} rows')\n"
            "print(f'  fights_clean.csv    {len(fights_clean):>5d} rows')\n"
            "\n"
            "print(f'\\nBaseline: Red WR = {fights_clean[\"f1_win\"].mean():.1%}')\n"
            "\n"
            "# Fight stat coverage\n"
            "print(f'\\nFight stat coverage:')\n"
            "stat_groups = {\n"
            "    'Striking': ['f1_str_landed','f1_str_attempted','f1_str_acc',\n"
            "                 'f1_total_str_landed','f1_total_str_attempted'],\n"
            "    'Grappling': ['f1_td_landed','f1_td_attempted','f1_td_acc',\n"
            "                  'f1_kd','f1_sub','f1_rev','f1_ctrl_seconds'],\n"
            "    'Location': ['f1_head_landed','f1_head_attempted',\n"
            "                 'f1_body_landed','f1_body_attempted',\n"
            "                 'f1_leg_landed','f1_leg_attempted'],\n"
            "    'Position': ['f1_distance_landed','f1_distance_attempted',\n"
            "                 'f1_clinch_landed','f1_clinch_attempted',\n"
            "                 'f1_ground_landed','f1_ground_attempted'],\n"
            "}\n"
            "for group, cols in stat_groups.items():\n"
            "    print(f'\\n  {group}:')\n"
            "    for col in cols:\n"
            "        nn = fights_clean[col].notna().sum()\n"
            "        mn = fights_clean[col].mean()\n"
            "        print(f'    {col:30s} {nn:>5d}/{len(fights_clean)}  mean={mn:.2f}')\n"
            "\n"
            "# Fighter career stat coverage\n"
            "print(f'\\nFighter career stat coverage:')\n"
            "for col in ['slpm','sapm','str_acc_career','str_def_career',\n"
            "            'td_avg','td_acc_career','td_def_career','sub_avg']:\n"
            "    nn = fighters_clean[col].notna().sum()\n"
            "    mn = fighters_clean[col].mean()\n"
            "    print(f'  {col:18s} {nn:>5d}/{len(fighters_clean)}  mean={mn:.3f}')\n"
            "\n"
            "print(f'\\nDOB parsed: {fighters_clean[\"dob_parsed\"].notna().sum()}/{len(fighters_clean)}')\n"
            "\n"
            "# Verify no raw columns remain\n"
            "print(f'\\nFighters columns: {list(fighters_clean.columns)}')\n"
            "print(f'Fights columns: {list(fights_clean.columns)}')\n"
            "\n"
            "print(f'\\nSample fights:')\n"
            "print(fights_clean[['fighter_1','fighter_2','winner','f1_win',\n"
            "    'f1_str_landed','f2_str_landed','f1_head_landed',\n"
            "    'f1_body_landed','f1_leg_landed','finish_type']].head(10).to_string())\n"
            "\n"
            "print(f'\\nSample fighters:')\n"
            "print(fighters_clean[['full_name','height_inches','reach_inches',\n"
            "    'slpm','sapm','str_acc_career','str_def_career',\n"
            "    'td_avg','td_acc_career','td_def_career','sub_avg']].head(5).to_string())\n"
            "\n"
            "print(f'\\nNext: open 03_eda.ipynb')"
        )
    },
]

make_notebook("./notebooks/02_data_cleaning.ipynb", cells)
print("\nDone. Open notebooks/02_data_cleaning.ipynb and Run All")