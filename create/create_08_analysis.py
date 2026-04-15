#!/usr/bin/env python
"""
create_08_betting.py
Generates notebooks/08_betting.ipynb
Betting value finder: paste American odds, get +EV picks with half-Kelly sizing
"""

import json, pathlib

def mk_nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": cells
    }

def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.strip().splitlines(True)
    }

def md(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.strip().splitlines(True)
    }

cells = []

# =====================================================================
# CELL 1 - intro
# =====================================================================
cells.append(md(
"# 08 -- Betting Value Finder\n"
"\n"
"**What this does:**\n"
"1. You paste an event URL and American odds for each fight\n"
"2. Production models predict win probabilities\n"
"3. Converts American odds to implied probability (vig removed)\n"
"4. Calculates edge = model probability - fair implied probability\n"
"5. Filters: all 3 models unanimous **and** confidence >= 55%\n"
"6. Sizes bets with **half-Kelly criterion**\n"
"7. Outputs a bet card with stakes, edge, and risk summary\n"
"\n"
"**Betting rules (hardcoded):**\n"
"- Only bet when all 3 models are **unanimous**\n"
"- Only bet when ensemble confidence **>= 55%** (MEDIUM tier+)\n"
"- Only bet when **positive edge** exists\n"
"- Stake = half Kelly = (edge / (decimal_odds - 1)) / 2\n"
))

# =====================================================================
# CELL 2 - config instructions
# =====================================================================
cells.append(md(
"## Edit the next cell\n"
"\n"
"| Field | What to enter |\n"
"|-------|---------------|\n"
"| `EVENT_URL` | UFCStats event URL, or `None` to auto-detect latest |\n"
"| `BANKROLL` | Total dollars you are willing to risk on this card |\n"
"| `ODDS` | One dict per fight: `fighter_1` (red), `fighter_2` (blue), American odds |\n"
"\n"
"Fighter names must match the UFCStats card closely (fuzzy matching handles small typos).\n"
))

# =====================================================================
# CELL 3 - user config
# =====================================================================
cells.append(code(
"# ==============================================================\n"
"# EDIT THIS CELL\n"
"# ==============================================================\n"
"\n"
"EVENT_URL = None   # or paste full UFCStats event URL\n"
"\n"
"BANKROLL = 1000    # dollars\n"
"\n"
"ODDS = [\n"
"    # fighter_1 = red corner (listed first on UFCStats)\n"
"    # fighter_2 = blue corner\n"
"    # odds_1 / odds_2 = American moneyline from your book\n"
'    {"fighter_1": "Fighter A", "fighter_2": "Fighter B", "odds_1": -200, "odds_2": +170},\n'
'    {"fighter_1": "Fighter C", "fighter_2": "Fighter D", "odds_1": +110, "odds_2": -130},\n'
"    # add or remove rows as needed\n"
"]\n"
))

# =====================================================================
# CELL 4 - imports and load
# =====================================================================
cells.append(code(
"import pandas as pd, numpy as np, json, warnings, time, re\n"
"from pathlib import Path\n"
"from datetime import datetime\n"
"from difflib import SequenceMatcher\n"
"\n"
"import requests\n"
"from bs4 import BeautifulSoup\n"
"\n"
"import matplotlib.pyplot as plt\n"
"import matplotlib.ticker as mticker\n"
"import seaborn as sns\n"
"from matplotlib.patches import Patch\n"
"\n"
"from xgboost import XGBClassifier\n"
"import lightgbm as lgbm\n"
"from catboost import CatBoostClassifier\n"
"\n"
"warnings.filterwarnings('ignore')\n"
"sns.set_theme(style='whitegrid', palette='colorblind')\n"
"\n"
"# -- paths --\n"
"DATA = Path('./data') if Path('./data/model_data.csv').exists() else Path('../data')\n"
"MODEL_DIR = Path('../models') if Path('../models/xgb_prod.json').exists() else Path('./models')\n"
"\n"
"# -- data --\n"
"model_data = pd.read_csv(DATA / 'model_data.csv', parse_dates=['event_date'])\n"
"model_data = model_data.sort_values('event_date').reset_index(drop=True)\n"
"\n"
"fighters_clean = pd.read_csv(DATA / 'fighters_clean.csv')\n"
"\n"
"with open(DATA / 'feature_list.txt') as f:\n"
"    FEATURES = [l.strip() for l in f if l.strip()]\n"
"\n"
"# -- models --\n"
"xgb_mod = XGBClassifier()\n"
"xgb_mod.load_model(str(MODEL_DIR / 'xgb_prod.json'))\n"
"\n"
"lgb_mod = lgbm.Booster(model_file=str(MODEL_DIR / 'lgb_prod.txt'))\n"
"\n"
"cat_mod = CatBoostClassifier()\n"
"cat_mod.load_model(str(MODEL_DIR / 'cat_prod.cbm'))\n"
"\n"
"# -- ensemble weights --\n"
"with open(DATA / 'best_params.json') as f:\n"
"    bp = json.load(f)\n"
"ew = bp.get('ensemble_weights', {'xgb': 1/3, 'lgb': 1/3, 'cat': 1/3})\n"
"W = {k: ew.get(k, 1/3) for k in ('xgb', 'lgb', 'cat')}\n"
"\n"
"print(f'Data         : {len(model_data):,} fights')\n"
"print(f'Features     : {len(FEATURES)}')\n"
"print(f\"Weights      : XGB={W['xgb']:.3f}  LGB={W['lgb']:.3f}  CAT={W['cat']:.3f}\")\n"
))

# =====================================================================
# CELL 5 - odds utilities
# =====================================================================
cells.append(code(
"# -- odds conversion --\n"
"\n"
"def american_to_decimal(am):\n"
"    # American to decimal odds\n"
"    return (1 + am / 100) if am > 0 else (1 + 100 / abs(am))\n"
"\n"
"def american_to_implied(am):\n"
"    # American to raw implied prob (includes vig)\n"
"    return 100 / (am + 100) if am > 0 else abs(am) / (abs(am) + 100)\n"
"\n"
"def remove_vig(imp1, imp2):\n"
"    # Strip overround to get true probabilities\n"
"    t = imp1 + imp2\n"
"    return imp1 / t, imp2 / t\n"
"\n"
"def half_kelly(edge, dec_odds):\n"
"    # Half-Kelly fraction; returns 0 when no edge\n"
"    if edge <= 0 or dec_odds <= 1:\n"
"        return 0.0\n"
"    return (edge / (dec_odds - 1)) / 2\n"
"\n"
"# -- process user odds --\n"
"odds_df = pd.DataFrame(ODDS)\n"
"assert len(odds_df) > 0, 'No odds entered'\n"
"for c in ('fighter_1', 'fighter_2', 'odds_1', 'odds_2'):\n"
"    assert c in odds_df.columns, f'Missing column: {c}'\n"
"\n"
"odds_df['dec_1']     = odds_df['odds_1'].apply(american_to_decimal)\n"
"odds_df['dec_2']     = odds_df['odds_2'].apply(american_to_decimal)\n"
"odds_df['imp_raw_1'] = odds_df['odds_1'].apply(american_to_implied)\n"
"odds_df['imp_raw_2'] = odds_df['odds_2'].apply(american_to_implied)\n"
"odds_df['overround'] = odds_df['imp_raw_1'] + odds_df['imp_raw_2']\n"
"\n"
"fair = odds_df.apply(lambda r: remove_vig(r['imp_raw_1'], r['imp_raw_2']),\n"
"                     axis=1, result_type='expand')\n"
"odds_df['fair_1'], odds_df['fair_2'] = fair[0], fair[1]\n"
"\n"
"print(f\"{len(odds_df)} fights with odds  |  avg vig {odds_df['overround'].mean():.1%}\")\n"
"print()\n"
"print(odds_df[['fighter_1','fighter_2','odds_1','odds_2',\n"
"               'dec_1','dec_2','fair_1','fair_2','overround']]\n"
"      .to_string(index=False, float_format='%.3f'))\n"
))

# =====================================================================
# CELL 6 - scrape event
# =====================================================================
cells.append(code(
"HEADERS = {'User-Agent': 'Mozilla/5.0'}\n"
"\n"
"def get_soup(url):\n"
"    r = requests.get(url, headers=HEADERS)\n"
"    r.raise_for_status()\n"
"    return BeautifulSoup(r.text, 'lxml')\n"
"\n"
"# auto-detect if needed\n"
"if EVENT_URL is None:\n"
"    soup = get_soup('http://www.ufcstats.com/statistics/events/completed')\n"
"    link = soup.select_one('table.b-statistics__table-events tbody tr td a')\n"
"    EVENT_URL = link['href']\n"
"    print(f'Auto-detected: {EVENT_URL}')\n"
"\n"
"soup = get_soup(EVENT_URL)\n"
"event_name = soup.select_one('h2.b-content__title span').get_text(strip=True)\n"
"print(f'Event: {event_name}')\n"
"print()\n"
"\n"
"fight_urls = []\n"
"for r in soup.select('table.b-fight-details__table tbody tr.b-fight-details__table-row'):\n"
"    link = r.get('data-link', '')\n"
"    if link:\n"
"        fight_urls.append(link)\n"
"\n"
"card = []\n"
"for furl in fight_urls:\n"
"    fs = get_soup(furl)\n"
"    names = [a.get_text(strip=True) for a in\n"
"             fs.select('div.b-fight-details__person a.b-fight-details__person-link')]\n"
"    if len(names) < 2:\n"
"        continue\n"
"    statuses = fs.select('div.b-fight-details__person i.b-fight-details__person-status')\n"
"    winner = ''\n"
"    for s, n in zip(statuses, names):\n"
"        if 'W' in s.get_text(strip=True):\n"
"            winner = n\n"
"    card.append({'fighter_1': names[0], 'fighter_2': names[1],\n"
"                 'winner': winner, 'fight_url': furl})\n"
"\n"
"card_df = pd.DataFrame(card)\n"
"for _, r in card_df.iterrows():\n"
"    tag = 'done' if r['winner'] else 'upcoming'\n"
"    print(f\"  [{tag:8s}] {r['fighter_1']:25s} vs {r['fighter_2']}\")\n"
))

# =====================================================================
# CELL 7 - match fighters and build features
# =====================================================================
cells.append(code(
"def fuzzy(name, pool, thresh=0.75):\n"
"    best, sc = None, 0\n"
"    nl = name.lower().strip()\n"
"    for c in pool:\n"
"        s = SequenceMatcher(None, nl, c.lower().strip()).ratio()\n"
"        if s > sc:\n"
"            best, sc = c, s\n"
"    return best if sc >= thresh else None\n"
"\n"
"def latest_snapshot(fighter, data):\n"
"    # Most recent feature row for a fighter (check f1 then f2)\n"
"    for prefix in ('fighter_1', 'fighter_2'):\n"
"        rows = data[data[prefix] == fighter].sort_values('event_date')\n"
"        if len(rows):\n"
"            tag = 'f1' if prefix == 'fighter_1' else 'f2'\n"
"            row = rows.iloc[-1]\n"
"            return {c: row[c] for c in data.columns if c.startswith(tag + '_')}, tag\n"
"    return None, None\n"
"\n"
"def rename_prefix(snap, old, new):\n"
"    return {new + k[len(old):]: v for k, v in snap.items() if k.startswith(old)}\n"
"\n"
"def profile_fallback(name, fdf):\n"
"    row = fdf[fdf['full_name'] == name]\n"
"    if len(row) == 0:\n"
"        m = fuzzy(name, fdf['full_name'].tolist())\n"
"        if m:\n"
"            row = fdf[fdf['full_name'] == m]\n"
"            print(f\"    fuzzy: '{name}' -> '{m}'\")\n"
"    if len(row) == 0:\n"
"        return None\n"
"    r = row.iloc[0]\n"
"    return {\n"
"        'profile_win_pct':        r.get('win_pct',        np.nan),\n"
"        'profile_slpm':           r.get('slpm',           np.nan),\n"
"        'profile_sapm':           r.get('sapm',           np.nan),\n"
"        'profile_str_acc_career': r.get('str_acc_career', np.nan),\n"
"        'profile_str_def_career': r.get('str_def_career', np.nan),\n"
"        'profile_td_avg':         r.get('td_avg',         np.nan),\n"
"        'profile_td_acc_career':  r.get('td_acc_career',  np.nan),\n"
"        'profile_td_def_career':  r.get('td_def_career',  np.nan),\n"
"        'profile_sub_avg':        r.get('sub_avg',        np.nan),\n"
"        'profile_total_fights':   r.get('total_fights',   np.nan),\n"
"        'height_inches':          r.get('height_inches',  np.nan),\n"
"        'reach_inches':           r.get('reach_inches',   np.nan),\n"
"        'weight_lbs':             r.get('weight_lbs',     np.nan),\n"
"        'age':                    np.nan,\n"
"    }\n"
"\n"
"# -- map odds names to card names --\n"
"card_names = set(card_df['fighter_1'].tolist() + card_df['fighter_2'].tolist())\n"
"nmap = {}\n"
"for on in set(odds_df['fighter_1'].tolist() + odds_df['fighter_2'].tolist()):\n"
"    if on in card_names:\n"
"        nmap[on] = on\n"
"    else:\n"
"        m = fuzzy(on, list(card_names))\n"
"        if m:\n"
"            nmap[on] = m\n"
"            print(f\"  odds->card: '{on}' -> '{m}'\")\n"
"        else:\n"
"            print(f\"  WARNING: no card match for '{on}'\")\n"
"\n"
"# -- assemble feature rows --\n"
"pred_rows = []\n"
"for _, o in odds_df.iterrows():\n"
"    f1c = nmap.get(o['fighter_1'], o['fighter_1'])\n"
"    f2c = nmap.get(o['fighter_2'], o['fighter_2'])\n"
"\n"
"    row, cov = {}, 'full'\n"
"\n"
"    for side, cname, prefix in [('f1', f1c, 'f1'), ('f2', f2c, 'f2')]:\n"
"        snap, src = latest_snapshot(cname, model_data)\n"
"        if snap:\n"
"            row.update(rename_prefix(snap, src + '_', prefix + '_'))\n"
"        else:\n"
"            fb = profile_fallback(cname, fighters_clean)\n"
"            if fb:\n"
"                for k, v in fb.items():\n"
"                    row[f'{prefix}_{k}'] = v\n"
"                cov = 'profile_only'\n"
"                print(f'  WARNING: {cname}: profile-only')\n"
"            else:\n"
"                cov = 'missing'\n"
"                print(f'  ERROR: {cname}: not found')\n"
"\n"
"    # differentials\n"
"    for feat in FEATURES:\n"
"        if feat.startswith('diff_'):\n"
"            base = feat[5:]\n"
"            v1 = row.get(f'f1_{base}', np.nan)\n"
"            v2 = row.get(f'f2_{base}', np.nan)\n"
"            try:\n"
"                row[feat] = float(v1) - float(v2)\n"
"            except (TypeError, ValueError):\n"
"                row[feat] = np.nan\n"
"\n"
"    row.update({'fighter_1': o['fighter_1'], 'fighter_2': o['fighter_2'],\n"
"                'coverage': cov, 'odds_1': o['odds_1'], 'odds_2': o['odds_2'],\n"
"                'dec_1': o['dec_1'], 'dec_2': o['dec_2'],\n"
"                'fair_1': o['fair_1'], 'fair_2': o['fair_2']})\n"
"    pred_rows.append(row)\n"
"\n"
"pred_df = pd.DataFrame(pred_rows)\n"
"n_full = (pred_df['coverage']=='full').sum()\n"
"n_prof = (pred_df['coverage']=='profile_only').sum()\n"
"n_miss = (pred_df['coverage']=='missing').sum()\n"
"print(f'\\nBuilt {len(pred_df)} fights  |  full={n_full}  profile_only={n_prof}  missing={n_miss}')\n"
))

# =====================================================================
# CELL 8 - predictions
# =====================================================================
cells.append(code(
"X = pred_df.reindex(columns=FEATURES)\n"
"\n"
"p_xgb = xgb_mod.predict_proba(X)[:, 1]\n"
"p_lgb = lgb_mod.predict(X)\n"
"p_cat = cat_mod.predict_proba(X)[:, 1]\n"
"p_ens = W['xgb']*p_xgb + W['lgb']*p_lgb + W['cat']*p_cat\n"
"\n"
"pred_df['p_xgb']     = p_xgb\n"
"pred_df['p_lgb']     = p_lgb\n"
"pred_df['p_cat']     = p_cat\n"
"pred_df['p_ensemble'] = p_ens\n"
"pred_df['pick_red']   = (p_ens >= 0.5).astype(int)\n"
"pred_df['confidence'] = pred_df['p_ensemble'].apply(lambda p: max(p, 1-p))\n"
"pred_df['predicted_winner'] = pred_df.apply(\n"
"    lambda r: r['fighter_1'] if r['pick_red'] else r['fighter_2'], axis=1)\n"
"\n"
"pred_df['votes_red'] = ((p_xgb >= .5).astype(int)\n"
"                       + (p_lgb >= .5).astype(int)\n"
"                       + (p_cat >= .5).astype(int))\n"
"pred_df['unanimous'] = pred_df['votes_red'].isin([0, 3])\n"
"\n"
"def assign_tier(r):\n"
"    if not r['unanimous']:      return 'NO_CONF'\n"
"    if r['confidence'] >= 0.80: return 'VERY_HIGH'\n"
"    if r['confidence'] >= 0.65: return 'HIGH'\n"
"    if r['confidence'] >= 0.55: return 'MEDIUM'\n"
"    return 'LOW'\n"
"\n"
"pred_df['tier'] = pred_df.apply(assign_tier, axis=1)\n"
"\n"
"tier_sym = {'VERY_HIGH': '[VH]', 'HIGH': '[HI]', 'MEDIUM': '[MD]',\n"
"            'LOW': '[LO]', 'NO_CONF': '[NC]'}\n"
"print('Predictions\\n')\n"
"for _, r in pred_df.iterrows():\n"
"    sym = tier_sym.get(r['tier'], '')\n"
"    print(f\"  {sym:6s} {r['tier']:10s}  {r['predicted_winner']:25s} \"\n"
"          f\"({r['confidence']:.1%})  \"\n"
"          f\"XGB={r['p_xgb']:.3f}  LGB={r['p_lgb']:.3f}  CAT={r['p_cat']:.3f}\")\n"
))

# =====================================================================
# CELL 9 - edge and kelly
# =====================================================================
cells.append(code(
"rows = []\n"
"for _, r in pred_df.iterrows():\n"
"    red = r['pick_red']\n"
"    if red:\n"
"        mp   = r['p_ensemble']\n"
"        fair = r['fair_1']\n"
"        am   = r['odds_1']\n"
"        dec  = r['dec_1']\n"
"        who  = r['fighter_1']\n"
"    else:\n"
"        mp   = 1 - r['p_ensemble']\n"
"        fair = r['fair_2']\n"
"        am   = r['odds_2']\n"
"        dec  = r['dec_2']\n"
"        who  = r['fighter_2']\n"
"\n"
"    edge  = mp - fair\n"
"    kf    = half_kelly(edge, dec)\n"
"    stake = round(kf * BANKROLL, 2)\n"
"    to_win = round(stake * (dec - 1), 2)\n"
"\n"
"    rows.append({\n"
"        'fighter_1': r['fighter_1'], 'fighter_2': r['fighter_2'],\n"
"        'bet_on': who, 'american': int(am), 'decimal': round(dec, 3),\n"
"        'model_prob': round(mp, 4), 'implied_fair': round(fair, 4),\n"
"        'edge': round(edge, 4), 'tier': r['tier'],\n"
"        'unanimous': r['unanimous'], 'confidence': round(r['confidence'], 4),\n"
"        'kelly_frac': round(kf, 4), 'stake': stake, 'to_win': to_win,\n"
"        'coverage': r['coverage'],\n"
"    })\n"
"\n"
"bets = pd.DataFrame(rows)\n"
"\n"
"# -- filters --\n"
"qualified = (bets[(bets['unanimous'] == True)\n"
"                & (bets['confidence'] >= 0.55)\n"
"                & (bets['edge'] > 0)]\n"
"             .sort_values('edge', ascending=False)\n"
"             .reset_index(drop=True))\n"
"\n"
"filt_disagree = (~bets['unanimous']).sum()\n"
"filt_lowconf  = ((bets['unanimous']) & (bets['confidence'] < 0.55)).sum()\n"
"filt_noedge   = ((bets['unanimous']) & (bets['confidence'] >= 0.55) & (bets['edge'] <= 0)).sum()\n"
"\n"
"print(f'Fights analyzed : {len(bets)}')\n"
"print(f'Filtered out    : {filt_disagree} disagree  |  {filt_lowconf} low conf  |  {filt_noedge} no edge')\n"
"print(f'Qualified bets  : {len(qualified)}')\n"
"if len(qualified):\n"
"    print(f\"Total stake     : ${qualified['stake'].sum():,.2f}  \"\n"
"          f\"({qualified['stake'].sum()/BANKROLL:.1%} of bankroll)\")\n"
))

# =====================================================================
# CELL 10 - bet card
# =====================================================================
cells.append(code(
"if len(qualified) == 0:\n"
"    print('No qualified bets on this card.\\n')\n"
"    print(bets[['fighter_1','fighter_2','bet_on','tier','edge','model_prob','implied_fair']]\n"
"          .to_string(index=False))\n"
"else:\n"
"    ts = qualified['stake'].sum()\n"
"    print('=' * 85)\n"
"    print(f'  BET CARD  --  {event_name}')\n"
"    print(f'  Bankroll ${BANKROLL:,.0f}  |  Action ${ts:,.2f} ({ts/BANKROLL:.1%})')\n"
"    print('=' * 85)\n"
"    print()\n"
"\n"
"    for i, (_, b) in enumerate(qualified.iterrows(), 1):\n"
"        tier_tag = b['tier']\n"
"        print(f'  Bet #{i}  [{tier_tag}]')\n"
"        print(f\"  |-- Fight:   {b['fighter_1']} vs {b['fighter_2']}\")\n"
"        print(f\"  |-- Pick:    {b['bet_on']}  ({b['american']:+d})\")\n"
"        print(f\"  |-- Model:   {b['model_prob']:.1%}  vs  Market: {b['implied_fair']:.1%}\"\n"
"              f\"  ->  Edge: {b['edge']:+.1%}\")\n"
"        print(f\"  |-- Stake:   ${b['stake']:,.2f}  (half-Kelly {b['kelly_frac']:.2%})\")\n"
"        print(f\"  |-- To win:  ${b['to_win']:,.2f}\")\n"
"        if b['coverage'] == 'profile_only':\n"
"            print(f'      WARNING: profile-only features')\n"
"        print()\n"
"\n"
"    print('-' * 85)\n"
"    print(f\"  Total risked  : ${ts:,.2f}\")\n"
"    print(f\"  Max profit    : ${qualified['to_win'].sum():,.2f}\")\n"
"    print(f\"  Avg edge      : {qualified['edge'].mean():+.1%}\")\n"
"    print(f\"  Bets          : {len(qualified)} / {len(bets)}\")\n"
"    print('-' * 85)\n"
))

# =====================================================================
# CELL 11 - full table
# =====================================================================
cells.append(code(
"print('FULL CARD ANALYSIS\\n')\n"
"\n"
"show = bets[['fighter_1','fighter_2','bet_on','american',\n"
"             'model_prob','implied_fair','edge',\n"
"             'tier','confidence','kelly_frac','stake','coverage']].copy()\n"
"\n"
"def action_label(r):\n"
"    if r['unanimous'] and r['confidence'] >= 0.55 and r['edge'] > 0:\n"
"        return 'BET'\n"
"    if r['unanimous'] and r['confidence'] >= 0.55:\n"
"        return 'NO EDGE'\n"
"    if r['unanimous']:\n"
"        return 'LOW CONF'\n"
"    return 'DISAGREE'\n"
"\n"
"show['action'] = bets.apply(action_label, axis=1)\n"
"\n"
"fmt = {'model_prob':   '{:.1%}'.format,\n"
"       'implied_fair': '{:.1%}'.format,\n"
"       'edge':         '{:+.1%}'.format,\n"
"       'confidence':   '{:.1%}'.format,\n"
"       'kelly_frac':   '{:.2%}'.format,\n"
"       'stake':        '${:,.0f}'.format}\n"
"for c, fn in fmt.items():\n"
"    show[c] = show[c].apply(fn)\n"
"\n"
"print(show.to_string(index=False))\n"
))

# =====================================================================
# CELL 12 - edge visualization
# =====================================================================
cells.append(code(
"fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(bets) * 0.45)))\n"
"\n"
"# -- panel 1: edge bars --\n"
"ax = axes[0]\n"
"colors = []\n"
"for _, r in bets.iterrows():\n"
"    if r['edge'] > 0 and r['unanimous'] and r['confidence'] >= 0.55:\n"
"        colors.append('#2ecc71')\n"
"    elif r['edge'] > 0:\n"
"        colors.append('#f39c12')\n"
"    else:\n"
"        colors.append('#e74c3c')\n"
"\n"
"ax.barh(range(len(bets)), bets['edge'] * 100, color=colors)\n"
"ax.set_yticks(range(len(bets)))\n"
"ax.set_yticklabels(bets['bet_on'], fontsize=9)\n"
"ax.axvline(0, color='black', lw=0.8)\n"
"ax.set_xlabel('Edge (%)')\n"
"ax.set_title('Model Edge vs Market')\n"
"ax.invert_yaxis()\n"
"for i, e in enumerate(bets['edge']):\n"
"    ax.text(e * 100 + 0.3, i, f'{e:+.1%}', va='center', fontsize=8)\n"
"ax.legend(handles=[Patch(color='#2ecc71', label='Qualified bet'),\n"
"                   Patch(color='#f39c12', label='Edge but filtered'),\n"
"                   Patch(color='#e74c3c', label='No edge')],\n"
"          loc='lower right', fontsize=8)\n"
"\n"
"# -- panel 2: model vs market scatter --\n"
"ax = axes[1]\n"
"ax.scatter(bets['implied_fair'], bets['model_prob'],\n"
"           c=colors, s=80, edgecolors='black', lw=0.5, zorder=3)\n"
"ax.plot([0.15, 0.95], [0.15, 0.95], 'k--', alpha=0.3, label='No edge')\n"
"for _, r in bets.iterrows():\n"
"    nm = r['bet_on'][:15]\n"
"    off = 0.015 if r['model_prob'] > r['implied_fair'] else -0.015\n"
"    ax.annotate(nm, (r['implied_fair'], r['model_prob'] + off),\n"
"                fontsize=7, ha='center')\n"
"ax.set_xlabel('Market Implied Prob (no-vig)')\n"
"ax.set_ylabel('Model Probability')\n"
"ax.set_title('Model vs Market')\n"
"ax.legend(fontsize=8)\n"
"ax.set_xlim(0.15, 0.95)\n"
"ax.set_ylim(0.15, 0.95)\n"
"\n"
"plt.tight_layout()\n"
"plt.savefig(DATA / 'betting_edge_chart.png', dpi=150, bbox_inches='tight')\n"
"plt.show()\n"
))

# =====================================================================
# CELL 13 - risk summary
# =====================================================================
cells.append(code(
"print('=' * 60)\n"
"print('  RISK SUMMARY')\n"
"print('=' * 60)\n"
"\n"
"if len(qualified) == 0:\n"
"    print('  No bets placed.')\n"
"else:\n"
"    ts = qualified['stake'].sum()\n"
"    n  = len(qualified)\n"
"\n"
"    print(f'  Bets          : {n}')\n"
"    print(f'  Total staked  : ${ts:,.2f}  ({ts/BANKROLL:.1%} of roll)')\n"
"    print(f\"  Largest bet   : ${qualified['stake'].max():,.2f}\")\n"
"    print(f'  Avg bet       : ${ts/n:,.2f}')\n"
"    print()\n"
"\n"
"    # scenario table\n"
"    print('  Scenario analysis:')\n"
"    sq = qualified.sort_values('stake', ascending=False).reset_index(drop=True)\n"
"    for w in range(n + 1):\n"
"        pnl = 0\n"
"        for j in range(n):\n"
"            if j < w:\n"
"                pnl += sq.loc[j, 'to_win']\n"
"            else:\n"
"                pnl -= sq.loc[j, 'stake']\n"
"        pct = pnl / BANKROLL\n"
"        bar = '#' * max(0, int((pct + 0.5) * 30))\n"
"        print(f'    {w}W-{n-w}L : {pnl:>+10,.2f}  ({pct:>+6.1%})  {bar}')\n"
"\n"
"    print()\n"
"    print('  Half-Kelly limits exposure but does not eliminate risk.')\n"
"    print('  Never bet money you cannot afford to lose.')\n"
"print('=' * 60)\n"
))

# =====================================================================
# CELL 14 - results instructions
# =====================================================================
cells.append(md(
"## Results -- run after the event\n"
"\n"
"Re-run **Cell 6** (scrape) to pick up winners, then run the next cell.\n"
))

# =====================================================================
# CELL 15 - score results
# =====================================================================
cells.append(code(
"# -- build winner lookup from card_df --\n"
"wmap = {}\n"
"for _, r in card_df.iterrows():\n"
"    wmap[(r['fighter_1'], r['fighter_2'])] = r['winner']\n"
"\n"
"bets['actual_winner'] = bets.apply(\n"
"    lambda r: wmap.get((r['fighter_1'], r['fighter_2']), ''), axis=1)\n"
"\n"
"has = bets['actual_winner'].str.len() > 0\n"
"\n"
"if has.sum() == 0:\n"
"    print('Event not complete yet -- re-run Cell 6 + this cell after it finishes.')\n"
"else:\n"
"    bets.loc[has, 'correct'] = bets.loc[has, 'bet_on'] == bets.loc[has, 'actual_winner']\n"
"    bets.loc[has, 'pnl'] = bets.loc[has].apply(\n"
"        lambda r: r['to_win'] if r['correct'] else (-r['stake'] if r['stake'] > 0 else 0),\n"
"        axis=1)\n"
"\n"
"    sc = bets.loc[has & (bets['unanimous'] == True)\n"
"                      & (bets['confidence'] >= 0.55)\n"
"                      & (bets['edge'] > 0)].copy()\n"
"\n"
"    print('=' * 80)\n"
"    print(f'  RESULTS -- {event_name}')\n"
"    print('=' * 80)\n"
"    print()\n"
"\n"
"    if len(sc) == 0:\n"
"        print('  No qualified bets were placed.')\n"
"    else:\n"
"        for _, b in sc.iterrows():\n"
"            tag = 'WIN' if b['correct'] else 'LOSS'\n"
"            print(f\"  [{tag:4s}] {b['bet_on']:25s} ({b['american']:+d})  \"\n"
"                  f\"${b['stake']:.0f} -> P&L ${b['pnl']:+,.2f}\")\n"
"\n"
"        tw = int(sc['correct'].sum())\n"
"        tl = len(sc) - tw\n"
"        tp = sc['pnl'].sum()\n"
"        tsk = sc['stake'].sum()\n"
"        print(f'\\n  Record  : {tw}W-{tl}L ({tw/len(sc):.0%})')\n"
"        print(f'  Staked  : ${tsk:,.2f}')\n"
"        print(f'  P&L     : ${tp:+,.2f}')\n"
"        if tsk > 0:\n"
"            print(f'  ROI     : {tp/tsk:+.1%}')\n"
"        print(f'  Bankroll: ${BANKROLL:,.0f} -> ${BANKROLL + tp:,.2f}')\n"
"        print('=' * 80)\n"
"\n"
"    # all picks accuracy\n"
"    ac = bets.loc[has]\n"
"    nc = (ac['bet_on'] == ac['actual_winner']).sum()\n"
"    print(f'\\nAll picks: {nc}/{len(ac)} ({nc/len(ac):.1%})')\n"
"\n"
"    print('\\nBy tier:')\n"
"    for t in ['VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW', 'NO_CONF']:\n"
"        sub = ac[ac['tier'] == t]\n"
"        if len(sub):\n"
"            c = (sub['bet_on'] == sub['actual_winner']).sum()\n"
"            print(f'   {t:10s} {c}/{len(sub)} ({c/len(sub):.0%})')\n"
))

# =====================================================================
# CELL 16 - save
# =====================================================================
cells.append(code(
"slug = re.sub(r'[^A-Za-z0-9]+', '_', event_name)[:50]\n"
"out = DATA / f'bets_{slug}.csv'\n"
"\n"
"bets[['fighter_1','fighter_2','bet_on','american','decimal',\n"
"      'model_prob','implied_fair','edge','tier','confidence',\n"
"      'unanimous','kelly_frac','stake','to_win','coverage']].to_csv(out, index=False)\n"
"\n"
"print(f'Saved {out}')\n"
"print(f'  {len(qualified)} qualified bets  |  {len(bets)} total fights')\n"
))

# =================================================================
# write notebook
# =================================================================
nb = mk_nb(cells)
out_dir = pathlib.Path("notebooks")
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "08_betting.ipynb"

with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Created {out_path}  ({len(cells)} cells)")