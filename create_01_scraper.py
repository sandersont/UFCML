"""
save as: /workspaces/UFCML/create_01_scraper.py
run with: python create_01_scraper.py
Then open notebooks/01_scraper.ipynb and Run All

SCRAPING STAGES (5 stages, 10 threads each):
1. Event list — names, dates, locations
2. Event fights — fight rows (winner listed first, NOT red corner)
3. Fighter directory — name, height, weight, reach, stance, W-L-D, fighter_url
4. Fighter profiles — FOLLOW each fighter_url for SLpM, SApM, Str.Acc, Str.Def,
                      TD Avg, TD Acc, TD Def, Sub Avg, DOB
5. Fight details — red/blue corner order, W/L status, full X-of-Y stats

OUTPUT FILES (all in ./data/):
- events.csv              (Cell 2)
- fights_raw.csv          (Cell 3)
- fighters.csv            (Cell 4)
- fighters_full.csv       (Cell 5)
- fight_details.json      (Cell 6)
- fights.csv              (Cell 7)

LESSONS LEARNED:
- Event page date is inside TD[0] alongside event name, TD[1] is location
- Event page ALWAYS lists winner first (not red corner)
- Fight detail page lists red corner first with W/L status
- Fighter directory only has basic info — must FOLLOW URLs for career rate stats
- Profile stat keys vary: "Str. Def" vs "Str. Def." — handle both
- JSON totals row: [0]=names [1]=KD [2]=sig_str [3]=sig_pct [4]=total_str
                   [5]=TD [6]=TD_pct [7]=sub_att [8]=rev [9]=ctrl
- Uses tqdm (not tqdm.notebook), 10 concurrent threads
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
    # notebooks/01_scraper.ipynb — Markdown
    # ═══════════════════════════════════════
    {
        "type": "markdown",
        "source": (
            "# UFC Stats Scraper\n"
            "\n"
            "| Cell | Stage | Output |\n"
            "|------|-------|--------|\n"
            "| 1 | Setup | session, helpers |\n"
            "| 2 | Event list | `data/events.csv` |\n"
            "| 3 | Event fights | `data/fights_raw.csv` |\n"
            "| 4 | Fighter directory | `data/fighters.csv` |\n"
            "| 5 | Fighter profiles | `data/fighters_full.csv` |\n"
            "| 6 | Fight details | `data/fight_details.json` |\n"
            "| 7 | Rebuild corners | `data/fights.csv` |\n"
            "| 8 | Summary | verification |\n"
            "\n"
            "**F1 = Red corner = Favorite | F2 = Blue corner = Underdog**"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/01_scraper.ipynb — Cell 1
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/01_scraper.ipynb — Cell 1: Imports & Setup\n"
            "\n"
            "import requests\n"
            "from bs4 import BeautifulSoup\n"
            "import pandas as pd\n"
            "import re\n"
            "from tqdm import tqdm\n"
            "import os\n"
            "import json\n"
            "import concurrent.futures\n"
            "\n"
            "BASE_URL = 'http://www.ufcstats.com'\n"
            "HEADERS = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'}\n"
            "WORKERS = 10\n"
            "DATA_DIR = './data'\n"
            "os.makedirs(DATA_DIR, exist_ok=True)\n"
            "\n"
            "session = requests.Session()\n"
            "session.headers.update(HEADERS)\n"
            "\n"
            "def get_soup(url):\n"
            "    resp = session.get(url)\n"
            "    resp.raise_for_status()\n"
            "    return BeautifulSoup(resp.text, 'lxml')\n"
            "\n"
            "print(f'Ready — {WORKERS} threads')"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/01_scraper.ipynb — Cell 2
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/01_scraper.ipynb — Cell 2: Scrape Event List\n"
            "# Date is inside TD[0] alongside event name. TD[1] is location.\n"
            "# Output: data/events.csv\n"
            "\n"
            "def scrape_all_events():\n"
            "    url = f'{BASE_URL}/statistics/events/completed?page=all'\n"
            "    soup = get_soup(url)\n"
            "    events = []\n"
            "    for row in soup.select('tr.b-statistics__table-row'):\n"
            "        link = row.select_one('a.b-link')\n"
            "        if not link:\n"
            "            continue\n"
            "        cells = row.select('td')\n"
            "        td0 = cells[0].get_text(separator='|', strip=True).split('|')\n"
            "        events.append({\n"
            "            'event_name': td0[0].strip(),\n"
            "            'event_url': link['href'],\n"
            "            'event_date': td0[1].strip() if len(td0) > 1 else '',\n"
            "            'location': cells[1].text.strip() if len(cells) > 1 else ''\n"
            "        })\n"
            "    return events\n"
            "\n"
            "events = scrape_all_events()\n"
            "events_df = pd.DataFrame(events)\n"
            "events_df.to_csv(f'{DATA_DIR}/events.csv', index=False)\n"
            "print(f'Saved {len(events_df)} events to data/events.csv')\n"
            "events_df[['event_name','event_date','location']].head()"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/01_scraper.ipynb — Cell 3
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/01_scraper.ipynb — Cell 3: Scrape Event Fights\n"
            "# Event page always lists WINNER first — NOT red corner.\n"
            "# We fix corner order in Cell 7 using fight detail pages.\n"
            "# Output: data/fights_raw.csv\n"
            "\n"
            "def scrape_event_fights(event_row):\n"
            "    try:\n"
            "        soup = get_soup(event_row['event_url'])\n"
            "    except:\n"
            "        return []\n"
            "    fights = []\n"
            "    for row in soup.select('tr.b-fight-details__table-row')[1:]:\n"
            "        cols = row.select('td')\n"
            "        if len(cols) < 10:\n"
            "            continue\n"
            "        links = cols[1].select('a')\n"
            "        if len(links) < 2:\n"
            "            continue\n"
            "        flag = row.select_one('a.b-flag')\n"
            "        gv = lambda c: [p.text.strip() for p in c.select('p')]\n"
            "        kd, st, td, sb = gv(cols[2]), gv(cols[3]), gv(cols[4]), gv(cols[5])\n"
            "        fights.append({\n"
            "            'event_name': event_row['event_name'],\n"
            "            'event_date': event_row['event_date'],\n"
            "            'fight_url': flag['href'] if flag else None,\n"
            "            'fighter_1_winner': links[0].text.strip(),\n"
            "            'fighter_2_loser': links[1].text.strip(),\n"
            "            'winner_kd': kd[0] if kd else None,\n"
            "            'loser_kd': kd[1] if len(kd)>1 else None,\n"
            "            'winner_str': st[0] if st else None,\n"
            "            'loser_str': st[1] if len(st)>1 else None,\n"
            "            'winner_td': td[0] if td else None,\n"
            "            'loser_td': td[1] if len(td)>1 else None,\n"
            "            'winner_sub': sb[0] if sb else None,\n"
            "            'loser_sub': sb[1] if len(sb)>1 else None,\n"
            "            'weight_class': cols[6].text.strip(),\n"
            "            'method': cols[7].text.strip(),\n"
            "            'round': cols[8].text.strip(),\n"
            "            'time': cols[9].text.strip(),\n"
            "        })\n"
            "    return fights\n"
            "\n"
            "all_fights = []\n"
            "print(f'Scraping fights from {len(events_df)} events...')\n"
            "with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:\n"
            "    futs = {ex.submit(scrape_event_fights, r): r\n"
            "            for r in events_df.to_dict('records')}\n"
            "    for fut in tqdm(concurrent.futures.as_completed(futs),\n"
            "                    total=len(futs), desc='Events'):\n"
            "        res = fut.result()\n"
            "        if res:\n"
            "            all_fights.extend(res)\n"
            "\n"
            "fights_raw_df = pd.DataFrame(all_fights)\n"
            "fights_raw_df.to_csv(f'{DATA_DIR}/fights_raw.csv', index=False)\n"
            "print(f'\\nSaved {len(fights_raw_df)} fights to data/fights_raw.csv')"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/01_scraper.ipynb — Cell 4
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/01_scraper.ipynb — Cell 4: Scrape Fighter Directory\n"
            "# Gets basic info + fighter_url. We FOLLOW URLs in Cell 5.\n"
            "# Output: data/fighters.csv\n"
            "\n"
            "def scrape_fighters_for_letter(char):\n"
            "    try:\n"
            "        soup = get_soup(f'{BASE_URL}/statistics/fighters?char={char}&page=all')\n"
            "    except:\n"
            "        return []\n"
            "    out = []\n"
            "    for row in soup.select('tr.b-statistics__table-row'):\n"
            "        cols = row.select('td')\n"
            "        if len(cols) < 10:\n"
            "            continue\n"
            "        link = cols[0].select_one('a')\n"
            "        if not link:\n"
            "            continue\n"
            "        out.append({\n"
            "            'fighter_url': link['href'],\n"
            "            'first_name': cols[0].text.strip(),\n"
            "            'last_name': cols[1].text.strip(),\n"
            "            'nickname': cols[2].text.strip(),\n"
            "            'height': cols[3].text.strip(),\n"
            "            'weight': cols[4].text.strip(),\n"
            "            'reach': cols[5].text.strip(),\n"
            "            'stance': cols[6].text.strip(),\n"
            "            'wins': cols[7].text.strip(),\n"
            "            'losses': cols[8].text.strip(),\n"
            "            'draws': cols[9].text.strip(),\n"
            "        })\n"
            "    return out\n"
            "\n"
            "all_fighters = []\n"
            "print('Scraping fighter directory...')\n"
            "with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:\n"
            "    futs = {ex.submit(scrape_fighters_for_letter, c): c\n"
            "            for c in 'abcdefghijklmnopqrstuvwxyz'}\n"
            "    for fut in tqdm(concurrent.futures.as_completed(futs),\n"
            "                    total=len(futs), desc='Letters'):\n"
            "        res = fut.result()\n"
            "        if res:\n"
            "            all_fighters.extend(res)\n"
            "\n"
            "fighters_df = pd.DataFrame(all_fighters)\n"
            "fighters_df.to_csv(f'{DATA_DIR}/fighters.csv', index=False)\n"
            "print(f'\\nSaved {len(fighters_df)} fighters to data/fighters.csv')\n"
            "fighters_df.head()"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/01_scraper.ipynb — Cell 5
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/01_scraper.ipynb — Cell 5: Scrape Fighter Profiles\n"
            "# THIS WAS MISSING BEFORE — follows each fighter_url to get:\n"
            "# SLpM, SApM, Str.Acc, Str.Def, TD Avg, TD Acc, TD Def, Sub Avg, DOB\n"
            "# Output: data/fighters_full.csv\n"
            "\n"
            "def scrape_fighter_profile(url):\n"
            "    if not url or pd.isna(url):\n"
            "        return None\n"
            "    try:\n"
            "        soup = get_soup(url)\n"
            "        stats = {'fighter_url': url}\n"
            "        key_map = {\n"
            "            'SLpM': 'slpm',\n"
            "            'Str. Acc.': 'str_acc_pct',\n"
            "            'SApM': 'sapm',\n"
            "            'Str. Def': 'str_def_pct',\n"
            "            'Str. Def.': 'str_def_pct',\n"
            "            'TD Avg.': 'td_avg',\n"
            "            'TD Acc.': 'td_acc_pct',\n"
            "            'TD Def.': 'td_def_pct',\n"
            "            'Sub. Avg.': 'sub_avg',\n"
            "        }\n"
            "        for li in soup.select('li.b-list__box-list-item'):\n"
            "            text = li.get_text(separator='|', strip=True)\n"
            "            if ':' not in text:\n"
            "                continue\n"
            "            key, val = text.split(':', 1)\n"
            "            key, val = key.strip(), val.strip()\n"
            "            if key in key_map:\n"
            "                stats[key_map[key]] = val\n"
            "            elif key == 'DOB':\n"
            "                stats['dob'] = val\n"
            "        return stats\n"
            "    except Exception as e:\n"
            "        return {'fighter_url': url, 'error': str(e)}\n"
            "\n"
            "urls = fighters_df['fighter_url'].dropna().unique()\n"
            "results = []\n"
            "print(f'Scraping {len(urls)} fighter profiles...')\n"
            "print('Getting SLpM, SApM, Str.Acc, Str.Def, TD Avg, TD Acc, TD Def, Sub Avg, DOB')\n"
            "\n"
            "with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:\n"
            "    futs = {ex.submit(scrape_fighter_profile, u): u for u in urls}\n"
            "    for fut in tqdm(concurrent.futures.as_completed(futs),\n"
            "                    total=len(futs), desc='Profiles'):\n"
            "        res = fut.result()\n"
            "        if res:\n"
            "            results.append(res)\n"
            "\n"
            "profiles_df = pd.DataFrame(results)\n"
            "if 'error' in profiles_df.columns:\n"
            "    errs = profiles_df['error'].notna().sum()\n"
            "    print(f'Errors: {errs}')\n"
            "    profiles_df = profiles_df[profiles_df['error'].isna()].drop(columns=['error'])\n"
            "\n"
            "fighters_full_df = fighters_df.merge(profiles_df, on='fighter_url', how='left')\n"
            "fighters_full_df.to_csv(f'{DATA_DIR}/fighters_full.csv', index=False)\n"
            "\n"
            "print(f'\\nSaved {len(fighters_full_df)} fighters to data/fighters_full.csv')\n"
            "print(f'\\nProfile stat coverage:')\n"
            "for col in ['slpm','sapm','str_acc_pct','str_def_pct',\n"
            "            'td_avg','td_acc_pct','td_def_pct','sub_avg','dob']:\n"
            "    if col in fighters_full_df.columns:\n"
            "        nn = fighters_full_df[col].notna().sum()\n"
            "        print(f'  {col:15s} {nn:>5d} / {len(fighters_full_df)}')\n"
            "    else:\n"
            "        print(f'  {col:15s} MISSING')\n"
            "fighters_full_df.head()"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/01_scraper.ipynb — Cell 6
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/01_scraper.ipynb — Cell 6: Scrape Fight Details\n"
            "# Gives TRUE red/blue corner order + W/L status + full X-of-Y stats\n"
            "# Output: data/fight_details.json\n"
            "\n"
            "def parse_fight_table(table):\n"
            "    headers = [th.text.strip() for th in table.select('thead th')]\n"
            "    rows_data = []\n"
            "    for row in table.select('tbody tr'):\n"
            "        cols = row.select('td')\n"
            "        row_vals = []\n"
            "        for col in cols:\n"
            "            ps = col.select('p')\n"
            "            if ps:\n"
            "                row_vals.append([p.text.strip() for p in ps])\n"
            "            else:\n"
            "                row_vals.append([col.text.strip()])\n"
            "        rows_data.append(row_vals)\n"
            "    return {'headers': headers, 'rows': rows_data}\n"
            "\n"
            "def scrape_fight_details(url):\n"
            "    if not url:\n"
            "        return None\n"
            "    try:\n"
            "        soup = get_soup(url)\n"
            "        d = {'fight_url': url}\n"
            "        # Fighter names + W/L (red corner first)\n"
            "        persons = soup.select_one('div.b-fight-details__persons')\n"
            "        if persons:\n"
            "            for i, fp in enumerate(persons.select('div.b-fight-details__person'), 1):\n"
            "                name = fp.select_one('a.b-fight-details__person-link')\n"
            "                status = fp.select_one('i.b-fight-details__person-status')\n"
            "                d[f'fighter_{i}_name'] = name.text.strip() if name else ''\n"
            "                d[f'fighter_{i}_status'] = status.text.strip() if status else ''\n"
            "        # Method, round, time\n"
            "        for p in soup.select('div.b-fight-details__content p.b-fight-details__text'):\n"
            "            text = p.get_text(separator='|', strip=True)\n"
            "            for item in text.split('|'):\n"
            "                if ':' in item:\n"
            "                    k, v = item.split(':', 1)\n"
            "                    d[k.strip().lower().replace(' ', '_')] = v.strip()\n"
            "        # Stats tables\n"
            "        tables = soup.select('table.b-fight-details__table')\n"
            "        if len(tables) >= 1:\n"
            "            d['totals'] = parse_fight_table(tables[0])\n"
            "        if len(tables) >= 2:\n"
            "            d['significant_strikes'] = parse_fight_table(tables[1])\n"
            "        return d\n"
            "    except:\n"
            "        return None\n"
            "\n"
            "fight_urls = fights_raw_df['fight_url'].dropna().unique()\n"
            "fight_details_list = []\n"
            "CHECKPOINT = 500\n"
            "\n"
            "print(f'Scraping {len(fight_urls)} fight detail pages...')\n"
            "with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:\n"
            "    futs = {ex.submit(scrape_fight_details, u): u for u in fight_urls}\n"
            "    for fut in tqdm(concurrent.futures.as_completed(futs),\n"
            "                    total=len(futs), desc='Fight details'):\n"
            "        res = fut.result()\n"
            "        if res:\n"
            "            fight_details_list.append(res)\n"
            "        if len(fight_details_list) % CHECKPOINT == 0 and fight_details_list:\n"
            "            with open(f'{DATA_DIR}/fight_details_checkpoint.json', 'w') as f:\n"
            "                json.dump(fight_details_list, f)\n"
            "\n"
            "with open(f'{DATA_DIR}/fight_details.json', 'w') as f:\n"
            "    json.dump(fight_details_list, f)\n"
            "print(f'\\nSaved {len(fight_details_list)} records to data/fight_details.json')"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/01_scraper.ipynb — Cell 7
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/01_scraper.ipynb — Cell 7: Rebuild fights.csv\n"
            "# Event page: winner first. Detail page: red corner first.\n"
            "# Use detail page for corner assignment, event page for stat mapping.\n"
            "# Draw/NC detection: direct comparison, NOT regex.\n"
            "# Output: data/fights.csv\n"
            "\n"
            "detail_lookup = {}\n"
            "for fd in fight_details_list:\n"
            "    url = fd.get('fight_url', '')\n"
            "    f1 = fd.get('fighter_1_name', '').strip()\n"
            "    f2 = fd.get('fighter_2_name', '').strip()\n"
            "    s1 = fd.get('fighter_1_status', '').strip()\n"
            "    s2 = fd.get('fighter_2_status', '').strip()\n"
            "    if url and f1:\n"
            "        detail_lookup[url] = {\n"
            "            'red': f1, 'blue': f2, 'rs': s1, 'bs': s2\n"
            "        }\n"
            "\n"
            "rebuilt = []\n"
            "skipped = 0\n"
            "for _, row in fights_raw_df.iterrows():\n"
            "    url = row['fight_url']\n"
            "    if url not in detail_lookup:\n"
            "        skipped += 1\n"
            "        continue\n"
            "    dl = detail_lookup[url]\n"
            "    red, blue = dl['red'], dl['blue']\n"
            "    winner = red if dl['rs']=='W' else (blue if dl['bs']=='W' else 'Draw/NC')\n"
            "    ew = row['fighter_1_winner']  # event page winner\n"
            "    if red == ew:\n"
            "        rk,bk = row['winner_kd'],row['loser_kd']\n"
            "        rs,bs = row['winner_str'],row['loser_str']\n"
            "        rt,bt = row['winner_td'],row['loser_td']\n"
            "        ru,bu = row['winner_sub'],row['loser_sub']\n"
            "    else:\n"
            "        rk,bk = row['loser_kd'],row['winner_kd']\n"
            "        rs,bs = row['loser_str'],row['winner_str']\n"
            "        rt,bt = row['loser_td'],row['winner_td']\n"
            "        ru,bu = row['loser_sub'],row['winner_sub']\n"
            "    rebuilt.append({\n"
            "        'event_name': row['event_name'], 'event_date': row['event_date'],\n"
            "        'fight_url': url, 'fighter_1': red, 'fighter_2': blue,\n"
            "        'winner': winner,\n"
            "        'f1_kd': rk, 'f2_kd': bk, 'f1_str': rs, 'f2_str': bs,\n"
            "        'f1_td': rt, 'f2_td': bt, 'f1_sub': ru, 'f2_sub': bu,\n"
            "        'weight_class': row['weight_class'], 'method': row['method'],\n"
            "        'round': row['round'], 'time': row['time'],\n"
            "    })\n"
            "\n"
            "fights_df = pd.DataFrame(rebuilt)\n"
            "decided = (fights_df['winner']==fights_df['fighter_1']) | \\\n"
            "          (fights_df['winner']==fights_df['fighter_2'])\n"
            "f1w = (fights_df['winner']==fights_df['fighter_1']).sum()\n"
            "f2w = (fights_df['winner']==fights_df['fighter_2']).sum()\n"
            "dnc = (~decided).sum()\n"
            "\n"
            "fights_df.to_csv(f'{DATA_DIR}/fights.csv', index=False)\n"
            "print(f'Saved {len(fights_df)} fights to data/fights.csv')\n"
            "print(f'Skipped (no detail): {skipped}')\n"
            "print(f'Red wins:  {f1w} ({f1w/len(fights_df):.1%})')\n"
            "print(f'Blue wins: {f2w} ({f2w/len(fights_df):.1%})')\n"
            "print(f'Draw/NC:   {dnc} ({dnc/len(fights_df):.1%})')\n"
            "print(f'Red WR (decided): {f1w/(f1w+f2w):.1%}')\n"
            "fights_df.head()"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/01_scraper.ipynb — Cell 8
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/01_scraper.ipynb — Cell 8: Summary\n"
            "\n"
            "print('=' * 60)\n"
            "print('SCRAPING COMPLETE')\n"
            "print('=' * 60)\n"
            "\n"
            "files = {\n"
            "    'events.csv': len(events_df),\n"
            "    'fights_raw.csv': len(fights_raw_df),\n"
            "    'fights.csv': len(fights_df),\n"
            "    'fighters.csv': len(fighters_df),\n"
            "    'fighters_full.csv': len(fighters_full_df),\n"
            "    'fight_details.json': len(fight_details_list),\n"
            "}\n"
            "for fname, count in files.items():\n"
            "    path = f'{DATA_DIR}/{fname}'\n"
            "    size = os.path.getsize(path) / 1024 if os.path.exists(path) else 0\n"
            "    print(f'  {fname:30s} {count:>6d} rows  {size:>8.1f} KB')\n"
            "\n"
            "print(f'\\nFighter profile stats:')\n"
            "for col in ['slpm','sapm','str_acc_pct','str_def_pct',\n"
            "            'td_avg','td_acc_pct','td_def_pct','sub_avg','dob']:\n"
            "    if col in fighters_full_df.columns:\n"
            "        nn = fighters_full_df[col].notna().sum()\n"
            "        print(f'  {col:15s} {nn:>5d} / {len(fighters_full_df)}')\n"
            "\n"
            "print(f'\\nNext: open 02_data_cleaning.ipynb')"
        )
    },
]

make_notebook("./notebooks/01_scraper.ipynb", cells)
print("\nDone. Open notebooks/01_scraper.ipynb and Run All")
print("Expected runtime: ~20-30 min (fighter profiles are the slowest stage)")