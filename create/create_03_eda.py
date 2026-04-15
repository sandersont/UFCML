"""
save as: /workspaces/UFCML/create_03_eda.py
run with: python create_03_eda.py
Then open notebooks/03_eda.ipynb and Run All

INPUTS (all in ./data/):
- fights_clean.csv      (from 02_cleaning Cell 7)
- fighters_clean.csv    (from 02_cleaning Cell 5)

CELLS:
1  — Load data
2  — Outcome overview (finish types, corner WR, rounds)
3  — Fight length analysis
4  — Red WR over time (yearly + 95% CI)
5  — Red WR by finish type over time
6  — Weight class deep dive
7  — Winner vs loser stats
8  — Stat differentials vs win probability
9  — Strike location breakdown (NEW: head/body/leg)
10 — Strike position breakdown (NEW: distance/clinch/ground)
11 — Fighter career rate stats (NEW: SLpM, SApM, etc.)
12 — Career stat differentials vs win rate (NEW)
13 — Physical attributes
14 — Physical differentials vs win rate
15 — Correlation matrix
16 — Top correlations bar chart
17 — Stance matchups
18 — Key insights + next steps
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
    # Markdown intro
    # ═══════════════════════════════════════
    {
        "type": "markdown",
        "source": (
            "# UFC Fight Data — Exploratory Data Analysis\n"
            "\n"
            "| Cell | Section |\n"
            "|------|---------|\n"
            "| 1 | Load data |\n"
            "| 2 | Outcome overview |\n"
            "| 3 | Fight length |\n"
            "| 4 | Red WR over time |\n"
            "| 5 | Red WR by finish type |\n"
            "| 6 | Weight classes |\n"
            "| 7 | Winner vs loser stats |\n"
            "| 8 | Stat differentials |\n"
            "| 9 | Strike location (head/body/leg) |\n"
            "| 10 | Strike position (distance/clinch/ground) |\n"
            "| 11 | Fighter career rate stats |\n"
            "| 12 | Career stat diffs vs win rate |\n"
            "| 13 | Physical attributes |\n"
            "| 14 | Physical diffs vs win rate |\n"
            "| 15 | Correlation matrix |\n"
            "| 16 | Top correlations |\n"
            "| 17 | Stance matchups |\n"
            "| 18 | Key insights |\n"
            "\n"
            "**F1 = Red corner (favorite) | F2 = Blue corner (underdog)**\n"
            "**Baseline: 57.1%**"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/03_eda.ipynb — Cell 1
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/03_eda.ipynb — Cell 1: Load Data\n"
            "\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import warnings\n"
            "import os\n"
            "warnings.filterwarnings('ignore')\n"
            "\n"
            "plt.style.use('seaborn-v0_8-darkgrid')\n"
            "sns.set_palette('husl')\n"
            "plt.rcParams['figure.figsize'] = (14, 6)\n"
            "plt.rcParams['font.size'] = 12\n"
            "plt.rcParams['axes.titlesize'] = 14\n"
            "plt.rcParams['axes.titleweight'] = 'bold'\n"
            "\n"
            "if os.path.exists('./data/fights_clean.csv'):\n"
            "    DATA_DIR = './data'\n"
            "elif os.path.exists('../data/fights_clean.csv'):\n"
            "    DATA_DIR = '../data'\n"
            "else:\n"
            "    raise FileNotFoundError('Cannot find fights_clean.csv')\n"
            "\n"
            "fights = pd.read_csv(f'{DATA_DIR}/fights_clean.csv')\n"
            "fighters = pd.read_csv(f'{DATA_DIR}/fighters_clean.csv')\n"
            "fights['event_date'] = pd.to_datetime(fights['event_date'], errors='coerce')\n"
            "fights['year'] = fights['event_date'].dt.year\n"
            "\n"
            "n_fighters = len(set(fights['fighter_1']) | set(fights['fighter_2']))\n"
            "print(f'DATA_DIR: {DATA_DIR}')\n"
            "print(f'Fights: {len(fights)} | Fighters: {n_fighters} | '\n"
            "      f'Events: {fights[\"event_name\"].nunique()} | '\n"
            "      f'Red WR: {fights[\"f1_win\"].mean():.1%}')\n"
            "print(f'Columns: {len(fights.columns)}')\n"
            "print(f'Finish types: {dict(fights[\"finish_type\"].value_counts())}')"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/03_eda.ipynb — Cell 2
    # ═══════════════════════════════════════
    {"type": "markdown", "source": "## 1. Outcome Overview"},
    {
        "type": "code",
        "source": (
            "# notebooks/03_eda.ipynb — Cell 2: Outcome Overview\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(20, 6))\n"
            "\n"
            "fc = fights['finish_type'].value_counts()\n"
            "axes[0].pie(fc, labels=fc.index, autopct='%1.1f%%', startangle=90,\n"
            "            colors=['#e74c3c','#f39c12','#3498db','#95a5a6'][:len(fc)])\n"
            "axes[0].set_title('Finish Methods')\n"
            "\n"
            "wr = fights['f1_win'].mean()\n"
            "axes[1].bar(['Red (Fav)', 'Blue (Dog)'], [wr, 1-wr],\n"
            "            color=['#e74c3c','#3498db'], width=0.5)\n"
            "axes[1].set_ylabel('Win Rate')\n"
            "axes[1].set_title(f'Corner Win Rates — Red:{wr:.1%} Blue:{1-wr:.1%}')\n"
            "axes[1].set_ylim(0, 0.75)\n"
            "axes[1].axhline(y=0.5, color='gray', ls='--', alpha=0.5)\n"
            "\n"
            "rc = fights['round'].value_counts().sort_index()\n"
            "axes[2].bar(rc.index, rc.values,\n"
            "            color=['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db'][:len(rc)])\n"
            "axes[2].set_xlabel('Round'); axes[2].set_ylabel('Count')\n"
            "axes[2].set_title('Fight Ending Round')\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.savefig(f'{DATA_DIR}/eda_01_outcomes.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()\n"
            "\n"
            "for ft, cnt in fc.items():\n"
            "    print(f'  {ft:10s} {cnt:>5d} ({cnt/len(fights)*100:.1f}%)')\n"
            "print(f'Avg fight length: {fights[\"total_time_seconds\"].mean()/60:.1f} min')"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/03_eda.ipynb — Cell 3
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/03_eda.ipynb — Cell 3: Fight Length Analysis\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n"
            "\n"
            "for ft in ['KO/TKO','SUB','DEC']:\n"
            "    t = fights[fights['finish_type']==ft]['total_time_seconds']/60\n"
            "    axes[0].hist(t.dropna(), bins=40, alpha=0.5, label=ft, density=True)\n"
            "axes[0].set_xlabel('Fight Length (min)')\n"
            "axes[0].set_title('Fight Length by Finish Type')\n"
            "axes[0].legend()\n"
            "\n"
            "fights['time_bin'] = pd.cut(fights['total_time_seconds']/60,\n"
            "    bins=[0,5,10,15,20,25,30], labels=['0-5','5-10','10-15','15-20','20-25','25+'])\n"
            "tw = fights.groupby('time_bin', observed=True)['f1_win'].agg(['mean','count'])\n"
            "colors = ['#e74c3c' if m>0.5 else '#3498db' for m in tw['mean']]\n"
            "axes[1].bar(range(len(tw)), tw['mean'], color=colors)\n"
            "axes[1].set_xticks(range(len(tw))); axes[1].set_xticklabels(tw.index)\n"
            "axes[1].set_ylabel('Red WR'); axes[1].set_title('Red WR by Fight Length')\n"
            "axes[1].axhline(y=0.5, color='gray', ls='--', alpha=0.7)\n"
            "axes[1].set_ylim(0.3, 0.8)\n"
            "for i,(_, r) in enumerate(tw.iterrows()):\n"
            "    axes[1].text(i, r['mean']+0.01, f'{r[\"mean\"]:.0%}\\nn={int(r[\"count\"])}',\n"
            "                ha='center', va='bottom', fontsize=9)\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.savefig(f'{DATA_DIR}/eda_02_fight_length.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/03_eda.ipynb — Cell 4
    # ═══════════════════════════════════════
    {"type": "markdown", "source": "## 2. Red Corner Advantage Over Time"},
    {
        "type": "code",
        "source": (
            "# notebooks/03_eda.ipynb — Cell 4: Red WR Over Time\n"
            "\n"
            "fig, axes = plt.subplots(2, 1, figsize=(16, 10))\n"
            "\n"
            "yearly = fights.groupby('year').size()\n"
            "axes[0].bar(yearly.index, yearly.values, color='#3498db', edgecolor='white')\n"
            "axes[0].set_title('Fights Per Year'); axes[0].set_ylabel('Count')\n"
            "for x,y in zip(yearly.index, yearly.values):\n"
            "    axes[0].text(x, y+5, str(y), ha='center', fontsize=9)\n"
            "\n"
            "ys = fights.groupby('year')['f1_win'].agg(['mean','count','std'])\n"
            "ys['se'] = ys['std']/np.sqrt(ys['count'])\n"
            "ys['lo'] = ys['mean'] - 1.96*ys['se']\n"
            "ys['hi'] = ys['mean'] + 1.96*ys['se']\n"
            "axes[1].plot(ys.index, ys['mean'], 'o-', color='#e74c3c', lw=2, ms=8)\n"
            "axes[1].fill_between(ys.index, ys['lo'], ys['hi'], alpha=0.2, color='#e74c3c')\n"
            "axes[1].axhline(y=0.5, color='gray', ls='--', alpha=0.7)\n"
            "axes[1].set_title('Red Corner Win Rate (95% CI)')\n"
            "axes[1].set_ylabel('Win Rate'); axes[1].set_ylim(0.40, 0.75)\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.savefig(f'{DATA_DIR}/eda_03_time_trends.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()\n"
            "\n"
            "for yr, r in ys.iterrows():\n"
            "    print(f'  {int(yr)}: {r[\"mean\"]:.3f} ± {r[\"se\"]:.3f} (n={int(r[\"count\"])})')"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/03_eda.ipynb — Cell 5
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/03_eda.ipynb — Cell 5: Red WR by Finish Type Over Time\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(14, 6))\n"
            "cmap = {'KO/TKO':'#e74c3c', 'SUB':'#f39c12', 'DEC':'#3498db'}\n"
            "for ft in ['KO/TKO','SUB','DEC']:\n"
            "    s = fights[fights['finish_type']==ft].groupby('year')['f1_win'].mean()\n"
            "    ax.plot(s.index, s.values, 'o-', label=ft, color=cmap[ft], lw=2, ms=6)\n"
            "ax.axhline(y=0.5, color='gray', ls='--', alpha=0.7)\n"
            "ax.set_title('Red Corner WR by Finish Type Over Time')\n"
            "ax.set_ylabel('Win Rate'); ax.set_ylim(0.30, 0.85); ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.savefig(f'{DATA_DIR}/eda_04_wr_by_finish.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/03_eda.ipynb — Cell 6
    # ═══════════════════════════════════════
    {"type": "markdown", "source": "## 3. Weight Classes"},
    {
        "type": "code",
        "source": (
            "# notebooks/03_eda.ipynb — Cell 6: Weight Class Deep Dive\n"
            "\n"
            "fig, axes = plt.subplots(2, 2, figsize=(20, 14))\n"
            "wcc = fights['weight_class'].value_counts()\n"
            "top = wcc.index.tolist()\n"
            "\n"
            "axes[0,0].barh(range(len(wcc)), wcc.values,\n"
            "    color=sns.color_palette('viridis', len(wcc)))\n"
            "axes[0,0].set_yticks(range(len(wcc))); axes[0,0].set_yticklabels(wcc.index)\n"
            "axes[0,0].set_xlabel('Fights'); axes[0,0].set_title('Fights by Weight Class')\n"
            "axes[0,0].invert_yaxis()\n"
            "\n"
            "wwr = fights.groupby('weight_class')['f1_win'].mean().reindex(top)\n"
            "axes[0,1].barh(range(len(wwr)), wwr.values,\n"
            "    color=['#e74c3c' if v>0.5 else '#3498db' for v in wwr.values])\n"
            "axes[0,1].set_yticks(range(len(wwr))); axes[0,1].set_yticklabels(wwr.index)\n"
            "axes[0,1].set_xlabel('Red WR'); axes[0,1].set_title('Red WR by Weight Class')\n"
            "axes[0,1].axvline(x=0.5, color='gray', ls='--'); axes[0,1].invert_yaxis()\n"
            "\n"
            "wko = fights.groupby('weight_class').apply(\n"
            "    lambda x: (x['finish_type']=='KO/TKO').mean()).reindex(top)\n"
            "axes[1,0].barh(range(len(wko)), wko.values,\n"
            "    color=sns.color_palette('Reds_r', len(wko)))\n"
            "axes[1,0].set_yticks(range(len(wko))); axes[1,0].set_yticklabels(wko.index)\n"
            "axes[1,0].set_xlabel('KO Rate'); axes[1,0].set_title('KO Rate by Weight Class')\n"
            "axes[1,0].invert_yaxis()\n"
            "\n"
            "ft_wc = fights.groupby(['weight_class','finish_type']).size().unstack(fill_value=0).reindex(top)\n"
            "ft_pct = ft_wc.div(ft_wc.sum(axis=1), axis=0)\n"
            "cols = [c for c in ['KO/TKO','SUB','DEC'] if c in ft_pct.columns]\n"
            "ft_pct[cols].plot(kind='barh', stacked=True, ax=axes[1,1],\n"
            "    color=['#e74c3c','#f39c12','#3498db'][:len(cols)])\n"
            "axes[1,1].set_title('Finish Type by Weight Class'); axes[1,1].invert_yaxis()\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.savefig(f'{DATA_DIR}/eda_05_weight_classes.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()\n"
            "\n"
            "for wc in top:\n"
            "    s = fights[fights['weight_class']==wc]\n"
            "    print(f'  {wc:40s} n={len(s):>4d}  RedWR={s[\"f1_win\"].mean():.0%}  '\n"
            "          f'KO={(s[\"finish_type\"]==\"KO/TKO\").mean():.0%}')"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/03_eda.ipynb — Cell 7
    # ═══════════════════════════════════════
    {"type": "markdown", "source": "## 4. Winner vs Loser Stats"},
    {
        "type": "code",
        "source": (
            "# notebooks/03_eda.ipynb — Cell 7: Winner vs Loser Stats\n"
            "\n"
            "stat_list = ['str_landed','str_attempted','str_acc','td_landed','td_attempted',\n"
            "             'td_acc','kd','sub','rev','ctrl_seconds',\n"
            "             'head_landed','body_landed','leg_landed',\n"
            "             'distance_landed','clinch_landed','ground_landed']\n"
            "for stat in stat_list:\n"
            "    c1, c2 = f'f1_{stat}', f'f2_{stat}'\n"
            "    if c1 in fights.columns and c2 in fights.columns:\n"
            "        fights[f'winner_{stat}'] = np.where(fights['f1_win']==1, fights[c1], fights[c2])\n"
            "        fights[f'loser_{stat}'] = np.where(fights['f1_win']==1, fights[c2], fights[c1])\n"
            "\n"
            "fig, axes = plt.subplots(2, 3, figsize=(20, 12))\n"
            "pairs = [\n"
            "    (axes[0,0], 'str_landed', 'Sig Strikes Landed', 50),\n"
            "    (axes[0,1], 'str_acc', 'Strike Accuracy', 40),\n"
            "    (axes[0,2], 'kd', 'Knockdowns', None),\n"
            "    (axes[1,0], 'td_landed', 'Takedowns Landed', 30),\n"
            "    (axes[1,1], 'ctrl_seconds', 'Control Time (sec)', 40),\n"
            "    (axes[1,2], 'sub', 'Sub Attempts', None),\n"
            "]\n"
            "for ax, stat, title, nbins in pairs:\n"
            "    w = fights[f'winner_{stat}'].dropna()\n"
            "    l = fights[f'loser_{stat}'].dropna()\n"
            "    if nbins:\n"
            "        ax.hist(w, bins=nbins, alpha=0.6, label='Winner', color='#2ecc71', density=True)\n"
            "        ax.hist(l, bins=nbins, alpha=0.6, label='Loser', color='#e74c3c', density=True)\n"
            "    else:\n"
            "        for lbl, vals, clr in [('Winner',w,'#2ecc71'),('Loser',l,'#e74c3c')]:\n"
            "            vc = vals.value_counts().sort_index().head(8)\n"
            "            off = 0.15 if lbl=='Winner' else -0.15\n"
            "            ax.bar(vc.index+off, vc.values/len(vals), width=0.3, alpha=0.7,\n"
            "                   label=lbl, color=clr)\n"
            "    ax.set_title(title); ax.legend()\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.savefig(f'{DATA_DIR}/eda_06_winner_vs_loser.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()\n"
            "\n"
            "print('Winner vs Loser means:')\n"
            "for s in stat_list:\n"
            "    if f'winner_{s}' in fights.columns:\n"
            "        wm = fights[f'winner_{s}'].mean()\n"
            "        lm = fights[f'loser_{s}'].mean()\n"
            "        print(f'  {s:20s} W={wm:8.2f}  L={lm:8.2f}  diff={wm-lm:+.2f}')"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/03_eda.ipynb — Cell 8
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/03_eda.ipynb — Cell 8: Stat Differentials vs Win Probability\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(20, 6))\n"
            "\n"
            "fights['str_diff'] = fights['f1_str_landed'] - fights['f2_str_landed']\n"
            "sb = pd.cut(fights['str_diff'], bins=25)\n"
            "sw = fights.groupby(sb, observed=True)['f1_win'].agg(['mean','count'])\n"
            "sw = sw[sw['count']>=10]\n"
            "xv = [i.mid for i in sw.index]\n"
            "axes[0].scatter(xv, sw['mean'], s=sw['count']*0.8, alpha=0.6, c='#e74c3c')\n"
            "axes[0].axhline(y=0.5, color='gray', ls='--'); axes[0].axvline(x=0, color='gray', ls='--', alpha=0.3)\n"
            "axes[0].set_xlabel('Sig Strike Diff (R-B)'); axes[0].set_ylabel('Red WR')\n"
            "axes[0].set_title('Strike Diff vs Win Prob')\n"
            "\n"
            "fights['kd_diff'] = fights['f1_kd'] - fights['f2_kd']\n"
            "kw = fights.groupby('kd_diff')['f1_win'].agg(['mean','count'])\n"
            "kw = kw[(kw.index>=-4)&(kw.index<=4)]\n"
            "kc = ['#3498db' if i<0 else '#e74c3c' if i>0 else '#95a5a6' for i in kw.index]\n"
            "axes[1].bar(kw.index, kw['mean'], color=kc)\n"
            "axes[1].axhline(y=0.5, color='gray', ls='--')\n"
            "axes[1].set_xlabel('KD Diff (R-B)'); axes[1].set_ylabel('Red WR')\n"
            "axes[1].set_title('Knockdown Diff vs Win Rate')\n"
            "for k, r in kw.iterrows():\n"
            "    axes[1].text(k, r['mean']+0.02, f'n={int(r[\"count\"])}', ha='center', fontsize=8)\n"
            "\n"
            "fights['td_diff'] = fights['f1_td_landed'] - fights['f2_td_landed']\n"
            "tb = pd.cut(fights['td_diff'], bins=15)\n"
            "tw2 = fights.groupby(tb, observed=True)['f1_win'].agg(['mean','count'])\n"
            "tw2 = tw2[tw2['count']>=10]\n"
            "xt = [i.mid for i in tw2.index]\n"
            "axes[2].scatter(xt, tw2['mean'], s=tw2['count']*1.5, alpha=0.6, c='#2ecc71')\n"
            "axes[2].axhline(y=0.5, color='gray', ls='--')\n"
            "axes[2].set_xlabel('TD Diff (R-B)'); axes[2].set_ylabel('Red WR')\n"
            "axes[2].set_title('Takedown Diff vs Win Prob')\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.savefig(f'{DATA_DIR}/eda_07_differentials.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/03_eda.ipynb — Cell 9
    # ═══════════════════════════════════════
    {"type": "markdown", "source": "## 5. Strike Location & Position Breakdown (NEW)"},
    {
        "type": "code",
        "source": (
            "# notebooks/03_eda.ipynb — Cell 9: Strike Location (head/body/leg)\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(20, 6))\n"
            "\n"
            "# Overall distribution of where strikes land\n"
            "for p, ax, title, color in [\n"
            "    ('winner', axes[0], 'Winner Strike Targets', '#2ecc71'),\n"
            "    ('loser', axes[1], 'Loser Strike Targets', '#e74c3c'),\n"
            "]:\n"
            "    head = fights[f'{p}_head_landed'].mean()\n"
            "    body = fights[f'{p}_body_landed'].mean()\n"
            "    leg = fights[f'{p}_leg_landed'].mean()\n"
            "    total = head + body + leg\n"
            "    vals = [head/total, body/total, leg/total]\n"
            "    ax.bar(['Head','Body','Leg'], vals, color=[color,'#f39c12','#9b59b6'])\n"
            "    ax.set_title(title); ax.set_ylabel('Proportion')\n"
            "    for i, v in enumerate(vals):\n"
            "        ax.text(i, v+0.01, f'{v:.0%}\\n({[head,body,leg][i]:.1f}/fight)',\n"
            "                ha='center', fontsize=10)\n"
            "\n"
            "# Head strike diff vs win rate\n"
            "fights['head_diff'] = fights['f1_head_landed'] - fights['f2_head_landed']\n"
            "hb = pd.cut(fights['head_diff'], bins=20)\n"
            "hw = fights.groupby(hb, observed=True)['f1_win'].agg(['mean','count'])\n"
            "hw = hw[hw['count']>=10]\n"
            "xh = [i.mid for i in hw.index]\n"
            "axes[2].scatter(xh, hw['mean'], s=hw['count'], alpha=0.6, c='#e74c3c')\n"
            "axes[2].axhline(y=0.5, color='gray', ls='--')\n"
            "axes[2].axvline(x=0, color='gray', ls='--', alpha=0.3)\n"
            "axes[2].set_xlabel('Head Strike Diff (R-B)')\n"
            "axes[2].set_ylabel('Red WR')\n"
            "axes[2].set_title('Head Strike Diff vs Win Prob')\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.savefig(f'{DATA_DIR}/eda_08_strike_location.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()\n"
            "\n"
            "print('Strike location means (per fight):')\n"
            "for loc in ['head','body','leg']:\n"
            "    wm = fights[f'winner_{loc}_landed'].mean()\n"
            "    lm = fights[f'loser_{loc}_landed'].mean()\n"
            "    print(f'  {loc:6s}  W={wm:.2f}  L={lm:.2f}  diff={wm-lm:+.2f}')"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/03_eda.ipynb — Cell 10
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/03_eda.ipynb — Cell 10: Strike Position (distance/clinch/ground)\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(20, 6))\n"
            "\n"
            "for p, ax, title, color in [\n"
            "    ('winner', axes[0], 'Winner Strike Position', '#2ecc71'),\n"
            "    ('loser', axes[1], 'Loser Strike Position', '#e74c3c'),\n"
            "]:\n"
            "    dist = fights[f'{p}_distance_landed'].mean()\n"
            "    clin = fights[f'{p}_clinch_landed'].mean()\n"
            "    gnd = fights[f'{p}_ground_landed'].mean()\n"
            "    total = dist + clin + gnd\n"
            "    vals = [dist/total, clin/total, gnd/total]\n"
            "    ax.bar(['Distance','Clinch','Ground'], vals,\n"
            "           color=['#3498db','#e67e22','#1abc9c'])\n"
            "    ax.set_title(title); ax.set_ylabel('Proportion')\n"
            "    for i, v in enumerate(vals):\n"
            "        ax.text(i, v+0.01, f'{v:.0%}\\n({[dist,clin,gnd][i]:.1f}/fight)',\n"
            "                ha='center', fontsize=10)\n"
            "\n"
            "# Ground strike diff vs win rate\n"
            "fights['ground_diff'] = fights['f1_ground_landed'] - fights['f2_ground_landed']\n"
            "gb = pd.cut(fights['ground_diff'], bins=15)\n"
            "gw = fights.groupby(gb, observed=True)['f1_win'].agg(['mean','count'])\n"
            "gw = gw[gw['count']>=10]\n"
            "xg = [i.mid for i in gw.index]\n"
            "axes[2].scatter(xg, gw['mean'], s=gw['count']*1.5, alpha=0.6, c='#1abc9c')\n"
            "axes[2].axhline(y=0.5, color='gray', ls='--')\n"
            "axes[2].axvline(x=0, color='gray', ls='--', alpha=0.3)\n"
            "axes[2].set_xlabel('Ground Strike Diff (R-B)')\n"
            "axes[2].set_ylabel('Red WR')\n"
            "axes[2].set_title('Ground Strike Diff vs Win Prob')\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.savefig(f'{DATA_DIR}/eda_09_strike_position.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()\n"
            "\n"
            "print('Strike position means (per fight):')\n"
            "for pos in ['distance','clinch','ground']:\n"
            "    wm = fights[f'winner_{pos}_landed'].mean()\n"
            "    lm = fights[f'loser_{pos}_landed'].mean()\n"
            "    print(f'  {pos:10s}  W={wm:.2f}  L={lm:.2f}  diff={wm-lm:+.2f}')"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/03_eda.ipynb — Cell 11
    # ═══════════════════════════════════════
    {"type": "markdown", "source": "## 6. Fighter Career Rate Stats (NEW)"},
    {
        "type": "code",
        "source": (
            "# notebooks/03_eda.ipynb — Cell 11: Fighter Career Rate Stats\n"
            "\n"
            "rate_cols = ['slpm','sapm','str_acc_career','str_def_career',\n"
            "             'td_avg','td_acc_career','td_def_career','sub_avg']\n"
            "labels = ['SLpM','SApM','Str Acc','Str Def',\n"
            "          'TD Avg','TD Acc','TD Def','Sub Avg']\n"
            "\n"
            "fig, axes = plt.subplots(2, 4, figsize=(20, 10))\n"
            "axes = axes.flatten()\n"
            "\n"
            "colors = sns.color_palette('Set2', 8)\n"
            "for i, (col, label) in enumerate(zip(rate_cols, labels)):\n"
            "    data = fighters[col].dropna()\n"
            "    if len(data) > 0:\n"
            "        axes[i].hist(data, bins=40, color=colors[i], edgecolor='white')\n"
            "        axes[i].axvline(x=data.mean(), color='red', ls='--', lw=2)\n"
            "        axes[i].set_title(f'{label}\\nmean={data.mean():.3f}')\n"
            "    else:\n"
            "        axes[i].set_title(f'{label} (no data)')\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.savefig(f'{DATA_DIR}/eda_10_career_stats.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()\n"
            "\n"
            "print('Fighter career stat summary:')\n"
            "for col, label in zip(rate_cols, labels):\n"
            "    d = fighters[col].dropna()\n"
            "    print(f'  {label:10s} n={len(d):>5d}  '\n"
            "          f'mean={d.mean():.3f}  std={d.std():.3f}  '\n"
            "          f'min={d.min():.3f}  max={d.max():.3f}')"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/03_eda.ipynb — Cell 12
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/03_eda.ipynb — Cell 12: Career Stat Diffs vs Win Rate\n"
            "\n"
            "fp = fighters[['full_name','slpm','sapm','str_acc_career','str_def_career',\n"
            "               'td_avg','td_acc_career','td_def_career','sub_avg','win_pct']].copy()\n"
            "\n"
            "# Merge fighter career stats into fights\n"
            "fm = fights.merge(\n"
            "    fp.rename(columns=lambda c: f'f1_{c}' if c!='full_name' else 'fighter_1'),\n"
            "    on='fighter_1', how='left')\n"
            "fm = fm.merge(\n"
            "    fp.rename(columns=lambda c: f'f2_{c}' if c!='full_name' else 'fighter_2'),\n"
            "    on='fighter_2', how='left')\n"
            "\n"
            "diff_stats = [\n"
            "    ('slpm', 'SLpM Diff'),\n"
            "    ('sapm', 'SApM Diff (lower=better)'),\n"
            "    ('str_acc_career', 'Str Acc Diff'),\n"
            "    ('str_def_career', 'Str Def Diff'),\n"
            "    ('td_avg', 'TD Avg Diff'),\n"
            "    ('win_pct', 'Win% Diff'),\n"
            "]\n"
            "\n"
            "fig, axes = plt.subplots(2, 3, figsize=(20, 12))\n"
            "axes = axes.flatten()\n"
            "\n"
            "for i, (stat, title) in enumerate(diff_stats):\n"
            "    col = f'diff_{stat}'\n"
            "    fm[col] = fm[f'f1_{stat}'] - fm[f'f2_{stat}']\n"
            "    valid = fm[col].dropna()\n"
            "    if len(valid) < 50:\n"
            "        axes[i].set_title(f'{title} (insufficient data)')\n"
            "        continue\n"
            "    bns = pd.cut(fm[col], bins=15)\n"
            "    grp = fm.groupby(bns, observed=True)['f1_win'].agg(['mean','count'])\n"
            "    grp = grp[grp['count']>=15]\n"
            "    xv = [interval.mid for interval in grp.index]\n"
            "    axes[i].scatter(xv, grp['mean'], s=grp['count']*2, alpha=0.6, c='#e74c3c')\n"
            "    axes[i].axhline(y=0.5, color='gray', ls='--')\n"
            "    axes[i].axvline(x=0, color='gray', ls='--', alpha=0.3)\n"
            "    axes[i].set_ylabel('Red WR'); axes[i].set_title(title)\n"
            "    corr = fm[col].corr(fm['f1_win'])\n"
            "    axes[i].text(0.05, 0.95, f'r={corr:.3f}', transform=axes[i].transAxes,\n"
            "                fontsize=11, va='top', fontweight='bold')\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.savefig(f'{DATA_DIR}/eda_11_career_diffs.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()\n"
            "\n"
            "print('Career stat diff correlations with f1_win:')\n"
            "for stat, title in diff_stats:\n"
            "    col = f'diff_{stat}'\n"
            "    valid = fm[[col,'f1_win']].dropna()\n"
            "    if len(valid) > 0:\n"
            "        corr = valid[col].corr(valid['f1_win'])\n"
            "        print(f'  {title:30s} r={corr:+.4f}  (n={len(valid)})')"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/03_eda.ipynb — Cell 13
    # ═══════════════════════════════════════
    {"type": "markdown", "source": "## 7. Physical Attributes"},
    {
        "type": "code",
        "source": (
            "# notebooks/03_eda.ipynb — Cell 13: Physical Attributes\n"
            "\n"
            "fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n"
            "\n"
            "axes[0,0].hist(fighters['height_inches'].dropna(), bins=30, color='#3498db', edgecolor='white')\n"
            "axes[0,0].set_xlabel('Height (in)'); axes[0,0].set_title('Height')\n"
            "\n"
            "axes[0,1].hist(fighters['reach_inches'].dropna(), bins=30, color='#e67e22', edgecolor='white')\n"
            "axes[0,1].set_xlabel('Reach (in)'); axes[0,1].set_title('Reach')\n"
            "\n"
            "axes[0,2].hist(fighters['weight_lbs'].dropna(), bins=30, color='#2ecc71', edgecolor='white')\n"
            "axes[0,2].set_xlabel('Weight (lbs)'); axes[0,2].set_title('Weight')\n"
            "\n"
            "v = fighters.dropna(subset=['height_inches','reach_inches'])\n"
            "axes[1,0].scatter(v['height_inches'], v['reach_inches'], alpha=0.3, s=10, color='#9b59b6')\n"
            "cr = v['height_inches'].corr(v['reach_inches'])\n"
            "axes[1,0].set_xlabel('Height'); axes[1,0].set_ylabel('Reach')\n"
            "axes[1,0].set_title(f'Height vs Reach (r={cr:.2f})')\n"
            "\n"
            "v2 = v.copy()\n"
            "v2['ape'] = v2['reach_inches'] - v2['height_inches']\n"
            "axes[1,1].hist(v2['ape'], bins=30, color='#1abc9c', edgecolor='white')\n"
            "axes[1,1].set_xlabel('Ape Index'); axes[1,1].set_title('Ape Index (reach-height)')\n"
            "axes[1,1].axvline(x=0, color='red', ls='--')\n"
            "\n"
            "# DOB / Age distribution\n"
            "dob_valid = fighters['dob_parsed'].dropna()\n"
            "ages = (pd.Timestamp.now() - pd.to_datetime(dob_valid)).dt.days / 365.25\n"
            "axes[1,2].hist(ages, bins=30, color='#e74c3c', edgecolor='white')\n"
            "axes[1,2].set_xlabel('Age'); axes[1,2].set_title(f'Fighter Age (n={len(ages)})')\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.savefig(f'{DATA_DIR}/eda_12_physical.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/03_eda.ipynb — Cell 14
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/03_eda.ipynb — Cell 14: Physical Diffs vs Win Rate\n"
            "\n"
            "fp2 = fighters[['full_name','height_inches','reach_inches','win_pct']].copy()\n"
            "fm2 = fights.merge(\n"
            "    fp2.rename(columns={'full_name':'fighter_1','height_inches':'f1_ht',\n"
            "                       'reach_inches':'f1_rc','win_pct':'f1_wp'}),\n"
            "    on='fighter_1', how='left')\n"
            "fm2 = fm2.merge(\n"
            "    fp2.rename(columns={'full_name':'fighter_2','height_inches':'f2_ht',\n"
            "                       'reach_inches':'f2_rc','win_pct':'f2_wp'}),\n"
            "    on='fighter_2', how='left')\n"
            "fm2['ht_diff'] = fm2['f1_ht'] - fm2['f2_ht']\n"
            "fm2['rc_diff'] = fm2['f1_rc'] - fm2['f2_rc']\n"
            "fm2['wp_diff'] = fm2['f1_wp'] - fm2['f2_wp']\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(20, 6))\n"
            "for ax, col, clr, title in [\n"
            "    (axes[0], 'ht_diff', '#3498db', 'Height Diff vs WR'),\n"
            "    (axes[1], 'rc_diff', '#e67e22', 'Reach Diff vs WR'),\n"
            "    (axes[2], 'wp_diff', '#2ecc71', 'Win% Diff vs WR'),\n"
            "]:\n"
            "    valid = fm2[col].dropna()\n"
            "    if len(valid) < 50:\n"
            "        ax.set_title(f'{title} (insufficient data)'); continue\n"
            "    bns = pd.cut(fm2[col], bins=15)\n"
            "    grp = fm2.groupby(bns, observed=True)['f1_win'].agg(['mean','count'])\n"
            "    grp = grp[grp['count']>=15]\n"
            "    xv = [i.mid for i in grp.index]\n"
            "    ax.scatter(xv, grp['mean'], s=grp['count']*2, alpha=0.6, c=clr)\n"
            "    ax.axhline(y=0.5, color='gray', ls='--')\n"
            "    ax.axvline(x=0, color='gray', ls='--', alpha=0.3)\n"
            "    ax.set_ylabel('Red WR'); ax.set_title(title)\n"
            "    corr = fm2[col].corr(fm2['f1_win'])\n"
            "    ax.text(0.05, 0.95, f'r={corr:.3f}', transform=ax.transAxes,\n"
            "            fontsize=11, va='top', fontweight='bold')\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.savefig(f'{DATA_DIR}/eda_13_physical_diffs.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()\n"
            "\n"
            "for col, name in [('ht_diff','Height'),('rc_diff','Reach'),('wp_diff','Win%')]:\n"
            "    m = fm2[col].notna()\n"
            "    if m.sum() > 0:\n"
            "        print(f'  {name} diff corr with f1_win: '\n"
            "              f'{fm2.loc[m,col].corr(fm2.loc[m,\"f1_win\"]):.4f}')"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/03_eda.ipynb — Cell 15
    # ═══════════════════════════════════════
    {"type": "markdown", "source": "## 8. Correlations"},
    {
        "type": "code",
        "source": (
            "# notebooks/03_eda.ipynb — Cell 15: Correlation Matrix\n"
            "\n"
            "num_cols = [\n"
            "    'f1_kd','f2_kd','f1_str_landed','f2_str_landed',\n"
            "    'f1_str_acc','f2_str_acc','f1_td_landed','f2_td_landed',\n"
            "    'f1_td_acc','f2_td_acc','f1_sub','f2_sub',\n"
            "    'f1_ctrl_seconds','f2_ctrl_seconds',\n"
            "    'f1_head_landed','f2_head_landed',\n"
            "    'f1_body_landed','f2_body_landed',\n"
            "    'f1_leg_landed','f2_leg_landed',\n"
            "    'f1_distance_landed','f2_distance_landed',\n"
            "    'f1_ground_landed','f2_ground_landed',\n"
            "    'round','total_time_seconds','f1_win'\n"
            "]\n"
            "cols = [c for c in num_cols if c in fights.columns]\n"
            "cm = fights[cols].corr()\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(20, 16))\n"
            "mask = np.triu(np.ones_like(cm, dtype=bool))\n"
            "sns.heatmap(cm, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',\n"
            "            center=0, vmin=-1, vmax=1, square=True, ax=ax,\n"
            "            annot_kws={'size':7}, linewidths=0.5)\n"
            "ax.set_title('Correlation Matrix (F1=Red, F2=Blue)', fontsize=16, fontweight='bold')\n"
            "plt.tight_layout()\n"
            "plt.savefig(f'{DATA_DIR}/eda_14_correlations.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/03_eda.ipynb — Cell 16
    # ═══════════════════════════════════════
    {
        "type": "code",
        "source": (
            "# notebooks/03_eda.ipynb — Cell 16: Top Correlations Bar Chart\n"
            "\n"
            "wc_corr = cm['f1_win'].drop('f1_win').sort_values(key=abs, ascending=False)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(12, 10))\n"
            "ax.barh(range(len(wc_corr)), wc_corr.values,\n"
            "        color=['#e74c3c' if c>0 else '#3498db' for c in wc_corr.values])\n"
            "ax.set_yticks(range(len(wc_corr))); ax.set_yticklabels(wc_corr.index)\n"
            "ax.set_xlabel('Correlation with f1_win')\n"
            "ax.set_title('Feature Correlations with Red Corner Win')\n"
            "ax.axvline(x=0, color='gray', lw=1); ax.invert_yaxis()\n"
            "for i, (f, c) in enumerate(wc_corr.items()):\n"
            "    ha = 'left' if c>0 else 'right'\n"
            "    ax.text(c+(0.005 if c>0 else -0.005), i, f'{c:.3f}', va='center', ha=ha, fontsize=9)\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.savefig(f'{DATA_DIR}/eda_15_top_correlations.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()\n"
            "\n"
            "print('All correlations with f1_win:')\n"
            "for f, c in wc_corr.items():\n"
            "    print(f'  {\"+\" if c>0 else \"-\"} {f:30s} {c:+.4f}')"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/03_eda.ipynb — Cell 17
    # ═══════════════════════════════════════
    {"type": "markdown", "source": "## 9. Stance Matchups"},
    {
        "type": "code",
        "source": (
            "# notebooks/03_eda.ipynb — Cell 17: Stance Matchups\n"
            "\n"
            "fs = fighters[['full_name','stance']].copy()\n"
            "fs = fs[fs['stance'].isin(['Orthodox','Southpaw','Switch'])]\n"
            "\n"
            "sm = fights.merge(\n"
            "    fs.rename(columns={'full_name':'fighter_1','stance':'f1_stance'}),\n"
            "    on='fighter_1', how='left')\n"
            "sm = sm.merge(\n"
            "    fs.rename(columns={'full_name':'fighter_2','stance':'f2_stance'}),\n"
            "    on='fighter_2', how='left')\n"
            "sm = sm.dropna(subset=['f1_stance','f2_stance']).copy()\n"
            "sm['matchup'] = sm['f1_stance'] + ' vs ' + sm['f2_stance']\n"
            "\n"
            "ms = sm.groupby('matchup')['f1_win'].agg(['mean','count'])\n"
            "ms = ms[ms['count']>=20].sort_values('mean', ascending=False)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(12, 6))\n"
            "mcolors = ['#e74c3c' if m>0.5 else '#3498db' for m in ms['mean']]\n"
            "ax.barh(range(len(ms)), ms['mean'], color=mcolors)\n"
            "ax.set_yticks(range(len(ms))); ax.set_yticklabels(ms.index)\n"
            "ax.axvline(x=0.5, color='gray', ls='--')\n"
            "ax.set_xlabel('Red Corner Win Rate')\n"
            "ax.set_title('Red Corner WR by Stance Matchup (n≥20)')\n"
            "ax.invert_yaxis()\n"
            "for i, (_, r) in enumerate(ms.iterrows()):\n"
            "    ax.text(r['mean']+0.005, i,\n"
            "            f'{r[\"mean\"]:.0%} (n={int(r[\"count\"])})', va='center', fontsize=10)\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.savefig(f'{DATA_DIR}/eda_16_stances.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()\n"
            "\n"
            "print(f'Fights with stance data: {len(sm)}/{len(fights)}')\n"
            "print(ms.to_string())"
        )
    },

    # ═══════════════════════════════════════
    # notebooks/03_eda.ipynb — Cell 18
    # ═══════════════════════════════════════
    {"type": "markdown", "source": "## 10. Key Insights & Next Steps"},
    {
        "type": "code",
        "source": (
            "# notebooks/03_eda.ipynb — Cell 18: Key Insights\n"
            "\n"
            "print('=' * 70)\n"
            "print('EDA KEY INSIGHTS')\n"
            "print('=' * 70)\n"
            "\n"
            "print(f'\\n1. DATASET: {len(fights)} decided fights, 2015-2026')\n"
            "print(f'   Baseline (always pick red): {fights[\"f1_win\"].mean():.1%}')\n"
            "\n"
            "print(f'\\n2. FINISH TYPES:')\n"
            "for ft in ['DEC','KO/TKO','SUB','OTHER']:\n"
            "    n = (fights['finish_type']==ft).sum()\n"
            "    if n > 0:\n"
            "        print(f'   {ft:8s} {n/len(fights):.1%} ({n})')\n"
            "\n"
            "print(f'\\n3. IN-FIGHT CORRELATIONS WITH f1_win:')\n"
            "top_c = cm['f1_win'].drop('f1_win').sort_values(key=abs, ascending=False)\n"
            "for f, c in top_c.head(8).items():\n"
            "    print(f'   {f:30s} {c:+.3f}')\n"
            "\n"
            "print(f'\\n4. CAREER STAT DIFF CORRELATIONS (pre-fight usable):')\n"
            "for stat, title in diff_stats:\n"
            "    col = f'diff_{stat}'\n"
            "    valid = fm[[col,'f1_win']].dropna()\n"
            "    if len(valid) > 0:\n"
            "        corr = valid[col].corr(valid['f1_win'])\n"
            "        print(f'   {title:30s} {corr:+.4f}')\n"
            "\n"
            "print(f'\\n5. STRIKE LOCATION (winner vs loser):')\n"
            "for loc in ['head','body','leg']:\n"
            "    wm = fights[f'winner_{loc}_landed'].mean()\n"
            "    lm = fights[f'loser_{loc}_landed'].mean()\n"
            "    print(f'   {loc:6s}  W={wm:.2f}  L={lm:.2f}  diff={wm-lm:+.2f}')\n"
            "\n"
            "print(f'\\n6. STRIKE POSITION (winner vs loser per fight):')\n"
            "for pos in ['distance','clinch','ground']:\n"
            "    wm = fights[f'winner_{pos}_landed'].mean()\n"
            "    lm = fights[f'loser_{pos}_landed'].mean()\n"
            "    print(f'   {pos:10s}  W={wm:.2f}  L={lm:.2f}  diff={wm-lm:+.2f}')\n"
            "\n"
            "print(f'\\n7. PHYSICAL DIFFS:')\n"
            "for col, name in [('ht_diff','Height'),('rc_diff','Reach'),('wp_diff','Win%')]:\n"
            "    if col in fm2.columns:\n"
            "        m = fm2[col].notna()\n"
            "        if m.sum() > 0:\n"
            "            corr = fm2.loc[m,col].corr(fm2.loc[m,'f1_win'])\n"
            "            print(f'   {name:10s} diff corr: {corr:+.4f}')\n"
            "\n"
            "print(f'\\n8. FIGHTER CAREER RATE STATS:')\n"
            "for col, label in zip(\n"
            "    ['slpm','sapm','str_acc_career','str_def_career',\n"
            "     'td_avg','td_acc_career','td_def_career','sub_avg'],\n"
            "    ['SLpM','SApM','Str Acc','Str Def',\n"
            "     'TD Avg','TD Acc','TD Def','Sub Avg']):\n"
            "    d = fighters[col].dropna()\n"
            "    print(f'   {label:10s} n={len(d):>5d}  mean={d.mean():.3f}  std={d.std():.3f}')\n"
            "\n"
            "print(f'\\n9. DATA AVAILABLE FOR FEATURE ENGINEERING:')\n"
            "print(f'   Fight stats: {len(fights.columns)} columns, 0 nulls in core stats')\n"
            "print(f'   Fighter career: SLpM, SApM, Str Acc/Def, TD Avg/Acc/Def, Sub Avg')\n"
            "print(f'   Strike location: head/body/leg landed + attempted')\n"
            "print(f'   Strike position: distance/clinch/ground landed + attempted')\n"
            "print(f'   Physical: height, reach, weight, stance, DOB/age')\n"
            "print(f'   Context: weight class, method, round, time')\n"
            "\n"
            "print(f'\\n10. FEATURE ENGINEERING PRIORITIES:')\n"
            "print(f'   - Current in-fight stats must become PRE-FIGHT rolling averages')\n"
            "print(f'   - Career rate stats (SLpM etc.) are already pre-fight usable')\n"
            "print(f'   - Compute differentials: F1_stat - F2_stat for all features')\n"
            "print(f'   - Rolling windows: last 3 and 5 fights')\n"
            "print(f'   - Win streak, days since last fight, activity rate')\n"
            "print(f'   - Opponent quality (avg opp win rate)')\n"
            "print(f'   - CRITICAL: only use data from BEFORE each fight (anti-leakage)')\n"
            "\n"
            "print(f'\\n   BASELINE TO BEAT: {fights[\"f1_win\"].mean():.1%}')\n"
            "print('=' * 70)"
        )
    },
]

make_notebook("./notebooks/03_eda.ipynb", cells)
print("\nDone. Open notebooks/03_eda.ipynb and Run All")