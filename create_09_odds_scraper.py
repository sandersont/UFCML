#!/usr/bin/env python
"""
create_09_odds_scraper.py
Generates notebooks/09_odds_scraper.ipynb
Scrapes moneyline odds from 10 BestFightOdds event pages.
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
    # ===================================================
    # Markdown
    # ===================================================
    {
        "type": "markdown",
        "source": (
            "# 09 -- Historical Odds Scraper\n"
            "\n"
            "Scrapes moneyline odds from BestFightOdds for 10 numbered UFC events.\n"
            "Averages odds across all available sportsbooks per fighter.\n"
            "\n"
            "| Cell | Stage | Output |\n"
            "|------|-------|--------|\n"
            "| 1 | Setup | session, helpers |\n"
            "| 2 | Event URLs | hardcoded list |\n"
            "| 3 | Scrape + parse | all 10 events |\n"
            "| 4 | Average + clean | compute avg odds, add vig-free probs |\n"
            "| 5 | Save + summary | `data/odds_historical.csv` |\n"
            "\n"
            "**BFO page structure:**\n"
            "- Two tables: Table 0 = left headers (names), Table 1 = odds data\n"
            "- Fighter names in `<span class='t-b-fcc'>`\n"
            "- Moneyline odds in `<td class='but-sg'>` (skip `but-sgp` = props)\n"
            "- Prop rows have `class='pr'` -- skip\n"
            "- Fights come in row pairs: fighter 1 then fighter 2\n"
            "- Odds text has arrows to strip: `+207\\u25b2`, `-255\\u25bc`"
        )
    },

    # ===================================================
    # Cell 1 - setup
    # ===================================================
    {
        "type": "code",
        "source": (
            "# Cell 1: Imports & Setup\n"
            "\n"
            "import requests\n"
            "from bs4 import BeautifulSoup\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import re\n"
            "import os\n"
            "import time\n"
            "from tqdm import tqdm\n"
            "\n"
            "BASE = 'https://www.bestfightodds.com'\n"
            "HEADERS = {\n"
            "    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '\n"
            "                  'AppleWebKit/537.36 (KHTML, like Gecko) '\n"
            "                  'Chrome/124.0.0.0 Safari/537.36',\n"
            "    'Referer': 'https://www.bestfightodds.com/',\n"
            "}\n"
            "\n"
            "DELAY = 2.0\n"
            "DATA_DIR = './data' if os.path.exists('./data/model_data.csv') else '../data'\n"
            "os.makedirs(DATA_DIR, exist_ok=True)\n"
            "\n"
            "session = requests.Session()\n"
            "session.headers.update(HEADERS)\n"
            "\n"
            "def get_soup(url):\n"
            "    resp = session.get(url, timeout=30)\n"
            "    resp.raise_for_status()\n"
            "    time.sleep(DELAY)\n"
            "    return BeautifulSoup(resp.text, 'lxml')\n"
            "\n"
            "print(f'Ready  |  delay={DELAY}s  |  data_dir={DATA_DIR}')"
        )
    },

    # ===================================================
    # Cell 2 - event URLs
    # ===================================================
    {
        "type": "code",
        "source": (
            "# Cell 2: Event URLs (hardcoded)\n"
            "\n"
            "EVENT_URLS = [\n"
            "    'https://www.bestfightodds.com/events/ufc-315-3708',\n"
            "    'https://www.bestfightodds.com/events/ufc-316-3702',\n"
            "    'https://www.bestfightodds.com/events/ufc-319-3800',\n"
            "    'https://www.bestfightodds.com/events/ufc-320-3853',\n"
            "    'https://www.bestfightodds.com/events/ufc-321-odds-3780',\n"
            "    'https://www.bestfightodds.com/events/ufc-322-3830',\n"
            "    'https://www.bestfightodds.com/events/ufc-323-3951',\n"
            "    'https://www.bestfightodds.com/events/ufc-324-3973',\n"
            "    'https://www.bestfightodds.com/events/ufc-326-4065',\n"
            "    'https://www.bestfightodds.com/events/ufc-327-4074',\n"
            "]\n"
            "\n"
            "print(f'{len(EVENT_URLS)} events to scrape')\n"
            "for url in EVENT_URLS:\n"
            "    print(f'  {url}')"
        )
    },

    # ===================================================
    # Cell 3 - scrape and parse
    # ===================================================
    {
        "type": "code",
        "source": (
            "# Cell 3: Scrape odds from each event page\n"
            "#\n"
            "# Page structure:\n"
            "#   Table 0 (odds-table-responsive-header): left column with fighter names\n"
            "#   Table 1 (odds-table): odds data with sportsbook columns\n"
            "#\n"
            "#   Table 0 rows:\n"
            "#     - Fighter 1 row: has id='mu-XXXXX', name in <span class='t-b-fcc'>\n"
            "#     - Fighter 2 row: name in <span class='t-b-fcc'>, no id\n"
            "#     - Prop rows: class='pr' -- Over/Under, method, round lines\n"
            "#\n"
            "#   Table 1 rows (same order):\n"
            "#     - Fighter 1: odds in <td class='but-sg'> cells\n"
            "#     - Fighter 2: odds in <td class='but-sg'> cells\n"
            "#     - Prop rows: <td class='but-sgp'> -- skip\n"
            "#\n"
            "#   Odds text: '+207\\u25b2', '-255\\u25bc' -- strip arrows\n"
            "\n"
            "def parse_odds_text(text):\n"
            "    # Strip arrows and unicode, extract American odds integer\n"
            "    text = text.strip()\n"
            "    # remove common arrow chars\n"
            "    text = text.replace('\\u25b2', '').replace('\\u25bc', '')\n"
            "    text = text.replace('\\u2191', '').replace('\\u2193', '')\n"
            "    text = text.replace('\\u2212', '-')  # unicode minus\n"
            "    text = text.strip()\n"
            "    if not text or text in ('', '-', 'N/A'):\n"
            "        return np.nan\n"
            "    m = re.search(r'([+-]?\\d+)', text)\n"
            "    if m:\n"
            "        return int(m.group(1))\n"
            "    return np.nan\n"
            "\n"
            "\n"
            "def scrape_event(url):\n"
            "    soup = get_soup(url)\n"
            "\n"
            "    # event name from <h1>\n"
            "    h1 = soup.select_one('h1')\n"
            "    event_name = h1.get_text(strip=True) if h1 else ''\n"
            "\n"
            "    # date from <span class='table-header-date'>\n"
            "    date_el = soup.select_one('span.table-header-date')\n"
            "    date_text = date_el.get_text(strip=True) if date_el else ''\n"
            "\n"
            "    # year from <title>: 'UFC 300 Odds: ... for April 14 | Best Fight Odds'\n"
            "    # or 'UFC 327 Odds: ... | Best Fight Odds'\n"
            "    title_el = soup.select_one('title')\n"
            "    title_text = title_el.get_text(strip=True) if title_el else ''\n"
            "\n"
            "    # get both tables\n"
            "    tables = soup.select('table.odds-table')\n"
            "    if len(tables) < 2:\n"
            "        print(f'  WARNING: {event_name} -- less than 2 tables found')\n"
            "        return []\n"
            "\n"
            "    # Table 0 = names, Table 1 = odds\n"
            "    name_rows = tables[0].select('tr')\n"
            "    odds_rows = tables[1].select('tr')\n"
            "\n"
            "    # first row of table 1 is the header (sportsbook names) -- skip it\n"
            "    # pair up name_rows and odds_rows (they should align after header)\n"
            "\n"
            "    fights = []\n"
            "    i = 0  # index into name_rows (no header row in table 0)\n"
            "    j = 1  # index into odds_rows (skip header row in table 1)\n"
            "\n"
            "    while i < len(name_rows) - 1 and j < len(odds_rows) - 1:\n"
            "        nrow1 = name_rows[i]\n"
            "        nrow2 = name_rows[i + 1]\n"
            "\n"
            "        # skip prop rows\n"
            "        if 'pr' in nrow1.get('class', []):\n"
            "            i += 1\n"
            "            j += 1\n"
            "            continue\n"
            "\n"
            "        # fighter 1 name\n"
            "        name1_el = nrow1.select_one('span.t-b-fcc')\n"
            "        if not name1_el:\n"
            "            i += 1\n"
            "            j += 1\n"
            "            continue\n"
            "        name1 = name1_el.get_text(strip=True)\n"
            "\n"
            "        # fighter 2 -- skip any prop rows between them\n"
            "        # find next non-prop row with a fighter name\n"
            "        k = i + 1\n"
            "        name2 = None\n"
            "        while k < len(name_rows):\n"
            "            if 'pr' in name_rows[k].get('class', []):\n"
            "                k += 1\n"
            "                continue\n"
            "            name2_el = name_rows[k].select_one('span.t-b-fcc')\n"
            "            if name2_el:\n"
            "                name2 = name2_el.get_text(strip=True)\n"
            "                break\n"
            "            k += 1\n"
            "\n"
            "        if not name2:\n"
            "            i = k + 1\n"
            "            j = k + 1\n"
            "            continue\n"
            "\n"
            "        # count prop rows between fighter 1 and fighter 2 in name_rows\n"
            "        n_props_between = k - i - 1\n"
            "\n"
            "        # get odds from table 1\n"
            "        # fighter 1 odds row = j\n"
            "        # fighter 2 odds row = j + 1 + n_props_between (skip same props)\n"
            "        orow1 = odds_rows[j] if j < len(odds_rows) else None\n"
            "        orow2_idx = j + 1 + n_props_between\n"
            "\n"
            "        # actually simpler: odds_rows align with name_rows\n"
            "        # fighter 1 = j, props = j+1 through j+n_props_between, fighter 2 = j+1+n_props_between\n"
            "        # BUT there might also be prop rows AFTER fighter 2\n"
            "        # safest: just use j for fighter 1, find fighter 2 at orow2_idx\n"
            "\n"
            "        orow2 = odds_rows[orow2_idx] if orow2_idx < len(odds_rows) else None\n"
            "\n"
            "        odds1_list = []\n"
            "        odds2_list = []\n"
            "\n"
            "        if orow1:\n"
            "            for td in orow1.select('td'):\n"
            "                classes = td.get('class', [])\n"
            "                # only moneyline cells, not props\n"
            "                if 'but-sg' in classes and 'but-sgp' not in ' '.join(classes):\n"
            "                    val = parse_odds_text(td.get_text(strip=True))\n"
            "                    if not np.isnan(val):\n"
            "                        odds1_list.append(int(val))\n"
            "\n"
            "        if orow2:\n"
            "            for td in orow2.select('td'):\n"
            "                classes = td.get('class', [])\n"
            "                if 'but-sg' in classes and 'but-sgp' not in ' '.join(classes):\n"
            "                    val = parse_odds_text(td.get_text(strip=True))\n"
            "                    if not np.isnan(val):\n"
            "                        odds2_list.append(int(val))\n"
            "\n"
            "        if odds1_list or odds2_list:\n"
            "            fights.append({\n"
            "                'event_name': event_name,\n"
            "                'event_date': date_text,\n"
            "                'event_url': url,\n"
            "                'fighter_1': name1,\n"
            "                'fighter_2': name2,\n"
            "                'odds_1_all': odds1_list,\n"
            "                'odds_2_all': odds2_list,\n"
            "                'n_books_1': len(odds1_list),\n"
            "                'n_books_2': len(odds2_list),\n"
            "            })\n"
            "\n"
            "        # advance past fighter 2 and its prop rows\n"
            "        # find next fighter 1 (next row with id starting with 'mu-')\n"
            "        i = k + 1\n"
            "        j = orow2_idx + 1\n"
            "\n"
            "        # skip trailing prop rows\n"
            "        while i < len(name_rows) and 'pr' in name_rows[i].get('class', []):\n"
            "            i += 1\n"
            "            j += 1\n"
            "\n"
            "    return fights\n"
            "\n"
            "\n"
            "# -- scrape all events --\n"
            "all_fights = []\n"
            "for url in tqdm(EVENT_URLS, desc='Events'):\n"
            "    try:\n"
            "        fights = scrape_event(url)\n"
            "        all_fights.extend(fights)\n"
            "        print(f'  {fights[0][\"event_name\"] if fights else \"?\"}: {len(fights)} fights')\n"
            "    except Exception as e:\n"
            "        print(f'  ERROR {url}: {e}')\n"
            "\n"
            "print(f'\\nTotal: {len(all_fights)} fights across {len(EVENT_URLS)} events')"
        )
    },

    # ===================================================
    # Cell 4 - average and clean
    # ===================================================
    {
        "type": "code",
        "source": (
            "# Cell 4: Compute average and best odds, add derived columns\n"
            "\n"
            "odds_df = pd.DataFrame(all_fights)\n"
            "print(f'Raw records: {len(odds_df)}')\n"
            "\n"
            "# average odds across books\n"
            "odds_df['odds_1_avg'] = odds_df['odds_1_all'].apply(\n"
            "    lambda x: int(round(np.mean(x))) if len(x) > 0 else np.nan)\n"
            "odds_df['odds_2_avg'] = odds_df['odds_2_all'].apply(\n"
            "    lambda x: int(round(np.mean(x))) if len(x) > 0 else np.nan)\n"
            "\n"
            "# best odds (most favorable to bettor)\n"
            "odds_df['odds_1_best'] = odds_df['odds_1_all'].apply(\n"
            "    lambda x: int(max(x)) if len(x) > 0 else np.nan)\n"
            "odds_df['odds_2_best'] = odds_df['odds_2_all'].apply(\n"
            "    lambda x: int(max(x)) if len(x) > 0 else np.nan)\n"
            "\n"
            "# implied probabilities (from average odds)\n"
            "def american_to_implied(am):\n"
            "    if pd.isna(am):\n"
            "        return np.nan\n"
            "    am = int(am)\n"
            "    return 100 / (am + 100) if am > 0 else abs(am) / (abs(am) + 100)\n"
            "\n"
            "odds_df['imp_1'] = odds_df['odds_1_avg'].apply(american_to_implied)\n"
            "odds_df['imp_2'] = odds_df['odds_2_avg'].apply(american_to_implied)\n"
            "odds_df['overround'] = odds_df['imp_1'] + odds_df['imp_2']\n"
            "\n"
            "# vig-free probabilities\n"
            "odds_df['fair_1'] = odds_df['imp_1'] / odds_df['overround']\n"
            "odds_df['fair_2'] = odds_df['imp_2'] / odds_df['overround']\n"
            "\n"
            "# drop rows with no odds on either side\n"
            "has_both = odds_df['odds_1_avg'].notna() & odds_df['odds_2_avg'].notna()\n"
            "dropped = (~has_both).sum()\n"
            "odds_df = odds_df[has_both].reset_index(drop=True)\n"
            "print(f'With both sides: {len(odds_df)} (dropped {dropped})')\n"
            "\n"
            "# summary\n"
            "print(f'\\nEvents: {odds_df[\"event_name\"].nunique()}')\n"
            "print(f'Avg books per fighter: {odds_df[\"n_books_1\"].mean():.1f}')\n"
            "print(f'Avg overround: {odds_df[\"overround\"].mean():.3f}')\n"
            "\n"
            "print()\n"
            "for ev in odds_df['event_name'].unique():\n"
            "    sub = odds_df[odds_df['event_name'] == ev]\n"
            "    print(f'  {ev:30s}  {len(sub):2d} fights  avg vig {sub[\"overround\"].mean():.3f}')\n"
            "\n"
            "print()\n"
            "print(odds_df[['event_name','fighter_1','fighter_2',\n"
            "               'odds_1_avg','odds_2_avg','n_books_1','fair_1','fair_2']]\n"
            "      .head(15).to_string(index=False))"
        )
    },

    # ===================================================
    # Cell 5 - save
    # ===================================================
    {
        "type": "code",
        "source": (
            "# Cell 5: Save odds_historical.csv\n"
            "\n"
            "save_cols = ['event_name', 'event_date', 'event_url',\n"
            "             'fighter_1', 'fighter_2',\n"
            "             'odds_1_avg', 'odds_2_avg',\n"
            "             'odds_1_best', 'odds_2_best',\n"
            "             'n_books_1', 'n_books_2',\n"
            "             'imp_1', 'imp_2', 'overround',\n"
            "             'fair_1', 'fair_2']\n"
            "\n"
            "out = f'{DATA_DIR}/odds_historical.csv'\n"
            "odds_df[save_cols].to_csv(out, index=False)\n"
            "\n"
            "print(f'Saved {len(odds_df)} fights to {out}')\n"
            "print(f'  Events: {odds_df[\"event_name\"].nunique()}')\n"
            "print(f'  Fights with both sides: {len(odds_df)}')\n"
            "print(f'  File size: {os.path.getsize(out)/1024:.1f} KB')\n"
            "print(f'\\nNext: notebook 10 (backtest) merges these with model predictions')"
        )
    },
]

make_notebook("./notebooks/09_odds_scraper.ipynb", cells)
print("\nDone. Open notebooks/09_odds_scraper.ipynb and Run All")
print("Expected runtime: ~20 seconds (10 events at 2s delay)")