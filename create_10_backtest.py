#!/usr/bin/env python3
"""create_10_backtest.py — Generates notebooks/10_backtest.ipynb"""

import json, os

cells = []
def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)})
def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "source": source.strip().splitlines(True),
                   "execution_count": None, "outputs": []})

# ============================================================
# Cell 1: Intro
# ============================================================
md("""
# 10 — Backtest: Model vs Market

**Goal:** Merge historical BFO odds with model predictions to answer:
1. Does the model beat the closing line?
2. Would the betting strategy have been profitable?
3. How do confidence tiers perform against the market?

**Inputs:**
- `odds_historical.csv` — BFO odds (from NB09)
- `model_data.csv` — features for all fights
- `fights_clean.csv` — results (winner)
- `fighters_clean.csv` — profile fallback
- Production models from NB06
- `best_params.json` — ensemble weights
- `feature_list.txt` — feature columns
""")

# ============================================================
# Cell 2: Imports & load
# ============================================================
code("""
import pandas as pd
import numpy as np
import json, os, re, warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)
sns.set_style('whitegrid')

# Auto-detect data path
DATA = "./data/" if os.path.exists("./data/odds_historical.csv") else "../data/"
MODEL = "./models/" if os.path.exists("./models/") else "../models/"
if not os.path.exists(MODEL + "xgb_prod.json"):
    MODEL = "./models/" if os.path.exists("./models/xgb_prod.json") else "../models/"

print(f"Data path: {DATA}")
print(f"Model path: {MODEL}")

# Load odds
odds = pd.read_csv(f"{DATA}odds_historical.csv")
print(f"\\nOdds: {len(odds)} fights across {odds['event_name'].nunique()} events")

# Load model data + fights + fighters
model_data = pd.read_csv(f"{DATA}model_data.csv")
fights = pd.read_csv(f"{DATA}fights_clean.csv")
fighters = pd.read_csv(f"{DATA}fighters_clean.csv")

# Load features
with open(f"{DATA}feature_list.txt") as f:
    FEATURES = [line.strip() for line in f if line.strip()]
print(f"Features: {len(FEATURES)}")

# Load ensemble weights
with open(f"{DATA}best_params.json") as f:
    bp = json.load(f)
W = bp.get('ensemble_weights', {'xgb': 1/3, 'lgb': 1/3, 'cat': 1/3})
print(f"Ensemble weights: { {k: round(v,3) for k,v in W.items()} }")

# Load production models
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(f"{MODEL}xgb_prod.json")

lgb_model = lgb.Booster(model_file=f"{MODEL}lgb_prod.txt")

cat_model = cb.CatBoostClassifier()
cat_model.load_model(f"{MODEL}cat_prod.cbm")

print("\\nAll models loaded.")
""")

# ============================================================
# Cell 3: Deduplicate odds + clean event names
# ============================================================
code("""
# Clean event names — strip "Odds" suffix and normalize
odds['event_clean'] = odds['event_name'].str.replace(r'\\s*Odds.*', '', regex=True).str.strip()
odds['ufc_num'] = odds['event_clean'].str.extract(r'UFC\\s*(\\d+)').astype(float)

print(f"Before dedup: {len(odds)} fights")

# Deduplicate: same event + overlapping fighter names (e.g. Lipski / da Silva + Wang Cong / Cong Wang)
# Strategy: within each event, flag fights where fighter names are substrings of another fight's names
dupes_to_drop = []
for evt, grp in odds.groupby('event_clean'):
    rows = grp.reset_index()
    for i in range(len(rows)):
        for j in range(i+1, len(rows)):
            names_i = set(rows.loc[i, 'fighter_1'].lower().split() + rows.loc[i, 'fighter_2'].lower().split())
            names_j = set(rows.loc[j, 'fighter_1'].lower().split() + rows.loc[j, 'fighter_2'].lower().split())
            overlap = len(names_i & names_j)
            if overlap >= 2:  # at least 2 name tokens in common
                # drop the one with fewer books
                if rows.loc[i, 'n_books_1'] >= rows.loc[j, 'n_books_1']:
                    dupes_to_drop.append(rows.loc[j, 'index'])
                else:
                    dupes_to_drop.append(rows.loc[i, 'index'])

if dupes_to_drop:
    print(f"Dropping {len(dupes_to_drop)} duplicate(s):")
    for idx in dupes_to_drop:
        row = odds.loc[idx]
        print(f"  {row['event_clean']}: {row['fighter_1']} vs {row['fighter_2']}")
    odds = odds.drop(index=dupes_to_drop).reset_index(drop=True)

print(f"After dedup: {len(odds)} fights across {odds['event_clean'].nunique()} events")
print()

# Coverage per event
cov = odds.groupby('event_clean').agg(
    fights=('fighter_1', 'size'),
    avg_overround=('overround', 'mean')
).reset_index()
print(cov.to_string(index=False))
""")

# ============================================================
# Cell 4: Match BFO fighters to UFCStats fighters
# ============================================================
code("""
# We need to match each BFO fight to a row in model_data
# BFO names may differ from UFCStats names (e.g. "Du Plessis" vs "du Plessis")

# Build lookup from model_data: last fight per fighter pair
# model_data has fighter_1, fighter_2, event_date, f1_win
model_data['event_date'] = pd.to_datetime(model_data['event_date'])
fights['event_date'] = pd.to_datetime(fights['event_date'])

# Normalize names for matching
def normalize_name(name):
    if pd.isna(name):
        return ''
    name = str(name).strip().lower()
    name = re.sub(r'[^a-z\\s]', '', name)  # remove non-alpha
    name = re.sub(r'\\s+', ' ', name)       # collapse spaces
    return name

# Build a mapping: for each fight in model_data, create normalized name keys
model_data['f1_norm'] = model_data['fighter_1'].apply(normalize_name)
model_data['f2_norm'] = model_data['fighter_2'].apply(normalize_name)

# Also get results from fights_clean
fights['f1_norm'] = fights['fighter_1'].apply(normalize_name)
fights['f2_norm'] = fights['fighter_2'].apply(normalize_name)

# For each odds row, try to find matching model_data row
# Match by: event_name contains UFC number AND normalized fighter names match
# Fighter order may differ between BFO and UFCStats

odds['f1_norm'] = odds['fighter_1'].apply(normalize_name)
odds['f2_norm'] = odds['fighter_2'].apply(normalize_name)

def fuzzy_name_match(bfo_name, ufcstats_name):
    \"\"\"Check if names match — exact or subset of tokens.\"\"\"
    if bfo_name == ufcstats_name:
        return True
    bfo_tokens = set(bfo_name.split())
    ufc_tokens = set(ufcstats_name.split())
    if not bfo_tokens or not ufc_tokens:
        return False
    # last name match + at least one other token
    if len(bfo_tokens & ufc_tokens) >= max(1, min(len(bfo_tokens), len(ufc_tokens)) - 1):
        return True
    return False

matched = []
unmatched = []

for idx, orow in odds.iterrows():
    ufc_num = orow['ufc_num']
    bfo_f1 = orow['f1_norm']
    bfo_f2 = orow['f2_norm']
    
    # Find matching fights in model_data by UFC number in event_name
    candidates = model_data[
        model_data['event_name'].str.contains(f'UFC {int(ufc_num)}', case=False, na=False)
    ] if not np.isnan(ufc_num) else pd.DataFrame()
    
    found = False
    for _, crow in candidates.iterrows():
        ufc_f1 = crow['f1_norm']
        ufc_f2 = crow['f2_norm']
        
        # Check both orderings
        if (fuzzy_name_match(bfo_f1, ufc_f1) and fuzzy_name_match(bfo_f2, ufc_f2)):
            matched.append({**orow.to_dict(), 'md_index': crow.name, 'swapped': False})
            found = True
            break
        elif (fuzzy_name_match(bfo_f1, ufc_f2) and fuzzy_name_match(bfo_f2, ufc_f1)):
            matched.append({**orow.to_dict(), 'md_index': crow.name, 'swapped': True})
            found = True
            break
    
    if not found:
        unmatched.append(orow.to_dict())

print(f"Matched: {len(matched)} / {len(odds)}")
if unmatched:
    print(f"\\nUnmatched ({len(unmatched)}):")
    for u in unmatched:
        print(f"  {u['event_clean']}: {u['fighter_1']} vs {u['fighter_2']}")

matched_df = pd.DataFrame(matched)
print(f"\\nMatched fights per event:")
print(matched_df.groupby('event_clean').size().to_string())
""")

# ============================================================
# Cell 5: Build features & predict
# ============================================================
code("""
# For each matched fight, pull features from model_data and run predictions

results = []

for _, mrow in matched_df.iterrows():
    md_idx = int(mrow['md_index'])
    swapped = mrow['swapped']
    md_row = model_data.loc[md_idx]
    
    # Get features
    X = md_row[FEATURES].values.reshape(1, -1)
    X_df = pd.DataFrame(X, columns=FEATURES)
    
    # Predict P(f1_win)
    p_xgb = xgb_model.predict_proba(X_df)[:, 1][0]
    p_lgb = lgb_model.predict(X_df)[0]  # Booster returns probability directly
    p_cat = cat_model.predict_proba(X_df)[:, 1][0]
    
    p_ens = W['xgb'] * p_xgb + W['lgb'] * p_lgb + W['cat'] * p_cat
    
    # If BFO order was swapped relative to UFCStats, flip probabilities
    # p_ens is P(ufcstats_f1 wins)
    # If swapped: BFO_f1 = UFCStats_f2, so P(BFO_f1 wins) = 1 - p_ens
    if swapped:
        p_bfo_f1 = 1 - p_ens
        p_xgb_bfo = 1 - p_xgb
        p_lgb_bfo = 1 - p_lgb
        p_cat_bfo = 1 - p_cat
    else:
        p_bfo_f1 = p_ens
        p_xgb_bfo = p_xgb
        p_lgb_bfo = p_lgb
        p_cat_bfo = p_cat
    
    p_bfo_f2 = 1 - p_bfo_f1
    
    # Model picks
    pick_xgb = mrow['fighter_1'] if p_xgb_bfo > 0.5 else mrow['fighter_2']
    pick_lgb = mrow['fighter_1'] if p_lgb_bfo > 0.5 else mrow['fighter_2']
    pick_cat = mrow['fighter_1'] if p_cat_bfo > 0.5 else mrow['fighter_2']
    
    picks = [pick_xgb, pick_lgb, pick_cat]
    unanimous = (len(set(picks)) == 1)
    
    model_pick = mrow['fighter_1'] if p_bfo_f1 > 0.5 else mrow['fighter_2']
    model_conf = max(p_bfo_f1, p_bfo_f2)
    
    # Confidence tier
    if not unanimous:
        tier = 'NO_CONF'
    elif model_conf >= 0.80:
        tier = 'VERY_HIGH'
    elif model_conf >= 0.65:
        tier = 'HIGH'
    elif model_conf >= 0.55:
        tier = 'MEDIUM'
    else:
        tier = 'LOW'
    
    # Actual result from fights_clean
    actual_winner = md_row.get('winner', None)
    f1_win = md_row.get('f1_win', None)
    
    # Map result to BFO fighter order
    if swapped:
        bfo_f1_won = (f1_win == 0)  # BFO_f1 = UFCStats_f2
    else:
        bfo_f1_won = (f1_win == 1)
    
    actual_winner_bfo = mrow['fighter_1'] if bfo_f1_won else mrow['fighter_2']
    model_correct = (model_pick == actual_winner_bfo)
    
    results.append({
        'event': mrow['event_clean'],
        'ufc_num': mrow['ufc_num'],
        'fighter_1': mrow['fighter_1'],
        'fighter_2': mrow['fighter_2'],
        'odds_1': mrow['odds_1_avg'],
        'odds_2': mrow['odds_2_avg'],
        'fair_1': mrow['fair_1'],
        'fair_2': mrow['fair_2'],
        'model_p1': round(p_bfo_f1, 4),
        'model_p2': round(p_bfo_f2, 4),
        'p_xgb': round(p_xgb_bfo, 4),
        'p_lgb': round(p_lgb_bfo, 4),
        'p_cat': round(p_cat_bfo, 4),
        'model_pick': model_pick,
        'unanimous': unanimous,
        'confidence': round(model_conf, 4),
        'tier': tier,
        'actual_winner': actual_winner_bfo,
        'correct': model_correct,
        'swapped': swapped,
    })

res = pd.DataFrame(results)
print(f"Predictions generated: {len(res)} fights")
print(f"\\nOverall accuracy: {res['correct'].mean():.1%} ({res['correct'].sum()}/{len(res)})")
print(f"\\nPer event:")
evt_acc = res.groupby('event').agg(
    fights=('correct', 'size'),
    correct=('correct', 'sum'),
    accuracy=('correct', 'mean')
).reset_index()
evt_acc['accuracy'] = evt_acc['accuracy'].apply(lambda x: f"{x:.1%}")
print(evt_acc.to_string(index=False))
""")

# ============================================================
# Cell 6: Accuracy by confidence tier
# ============================================================
code("""
print("=" * 70)
print("MODEL ACCURACY BY CONFIDENCE TIER")
print("=" * 70)

tier_order = ['VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW', 'NO_CONF']
tier_stats = res.groupby('tier').agg(
    fights=('correct', 'size'),
    correct=('correct', 'sum'),
    accuracy=('correct', 'mean'),
    avg_conf=('confidence', 'mean')
).reindex(tier_order).dropna(how='all')

tier_stats['accuracy_str'] = tier_stats['accuracy'].apply(lambda x: f"{x:.1%}")
tier_stats['avg_conf_str'] = tier_stats['avg_conf'].apply(lambda x: f"{x:.1%}")
print(tier_stats[['fights', 'correct', 'accuracy_str', 'avg_conf_str']].to_string())

print(f"\\n--- Unanimous only ---")
unan = res[res['unanimous']]
print(f"Fights: {len(unan)}  Correct: {unan['correct'].sum()}  Accuracy: {unan['correct'].mean():.1%}")

print(f"\\n--- Split (non-unanimous) ---")
split = res[~res['unanimous']]
if len(split) > 0:
    print(f"Fights: {len(split)}  Correct: {split['correct'].sum()}  Accuracy: {split['correct'].mean():.1%}")
else:
    print("No split decisions in sample")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: accuracy by tier
tier_plot = tier_stats.reset_index()
colors = {'VERY_HIGH': '#2ecc71', 'HIGH': '#27ae60', 'MEDIUM': '#f39c12', 'LOW': '#e67e22', 'NO_CONF': '#e74c3c'}
bars = axes[0].bar(tier_plot['tier'], tier_plot['accuracy'], 
                    color=[colors.get(t, '#95a5a6') for t in tier_plot['tier']])
for bar, (_, row) in zip(bars, tier_plot.iterrows()):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{row['accuracy']:.0%}\\n(n={int(row['fights'])})", ha='center', fontsize=9)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy by Confidence Tier')
axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Coin flip')
axes[0].set_ylim(0, 1.05)
axes[0].legend()

# Right: accuracy by event
evt_plot = res.groupby('event').agg(accuracy=('correct', 'mean'), n=('correct', 'size')).reset_index()
evt_plot = evt_plot.sort_values('event')
bars2 = axes[1].bar(range(len(evt_plot)), evt_plot['accuracy'], color='steelblue')
for i, (_, row) in enumerate(evt_plot.iterrows()):
    axes[1].text(i, row['accuracy'] + 0.01, f"{row['accuracy']:.0%}\\n(n={int(row['n'])})", 
                 ha='center', fontsize=8)
axes[1].set_xticks(range(len(evt_plot)))
axes[1].set_xticklabels(evt_plot['event'], rotation=45, ha='right', fontsize=8)
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Model Accuracy by Event')
axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
axes[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(f"{DATA}backtest_accuracy.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {DATA}backtest_accuracy.png")
""")

# ============================================================
# Cell 7: Betting simulation
# ============================================================
code("""
print("=" * 70)
print("BETTING SIMULATION")
print("=" * 70)

def american_to_decimal(odds):
    if odds > 0:
        return 1 + odds / 100
    else:
        return 1 + 100 / abs(odds)

# For each fight, compute edge and Kelly stake
bets = []

for _, row in res.iterrows():
    # Model pick and probability
    if row['model_p1'] >= row['model_p2']:
        pick = row['fighter_1']
        model_prob = row['model_p1']
        market_fair = row['fair_1']
        american_odds = row['odds_1']
    else:
        pick = row['fighter_2']
        model_prob = row['model_p2']
        market_fair = row['fair_2']
        american_odds = row['odds_2']
    
    decimal_odds = american_to_decimal(american_odds)
    edge = model_prob - market_fair
    
    # Half-Kelly
    if edge > 0 and decimal_odds > 1:
        kelly_frac = (edge / (decimal_odds - 1)) / 2
    else:
        kelly_frac = 0
    
    stake = kelly_frac * BANKROLL
    won = (pick == row['actual_winner'])
    pnl = stake * (decimal_odds - 1) if won else -stake
    
    # Qualification
    qualifies = (row['unanimous'] and row['confidence'] >= 0.55 and edge > 0)
    
    bets.append({
        **row.to_dict(),
        'pick': pick,
        'model_prob': model_prob,
        'market_fair': market_fair,
        'american_odds': american_odds,
        'decimal_odds': round(decimal_odds, 4),
        'edge': round(edge, 4),
        'kelly_frac': round(kelly_frac, 4),
        'stake': round(stake, 2),
        'won': won,
        'pnl': round(pnl, 2),
        'qualifies': qualifies,
    })

BANKROLL = 1000  # reset for simulation

bets_df = pd.DataFrame(bets)
qualified = bets_df[bets_df['qualifies']].copy()

print(f"\\nAll fights: {len(bets_df)}")
print(f"Qualified bets: {len(qualified)} ({len(qualified)/len(bets_df)*100:.0f}%)")
print(f"  Unanimous: {bets_df['unanimous'].sum()}")
print(f"  Conf >= 55%: {(bets_df['confidence'] >= 0.55).sum()}")
print(f"  +EV: {(bets_df['edge'] > 0).sum()}")

if len(qualified) > 0:
    print(f"\\n{'='*70}")
    print(f"QUALIFIED BETS SUMMARY")
    print(f"{'='*70}")
    print(f"Record: {qualified['won'].sum()}W - {(~qualified['won']).sum()}L ({qualified['won'].mean():.1%})")
    print(f"Total staked: ${qualified['stake'].sum():.2f}")
    print(f"Total P&L: ${qualified['pnl'].sum():.2f}")
    roi = qualified['pnl'].sum() / qualified['stake'].sum() * 100 if qualified['stake'].sum() > 0 else 0
    print(f"ROI: {roi:+.1f}%")
    print(f"Avg edge: {qualified['edge'].mean():.1%}")
    print(f"Avg stake: ${qualified['stake'].mean():.2f}")
    print(f"Avg odds: {qualified['american_odds'].mean():+.0f}")
""")

# ============================================================
# Cell 8: Qualified bet details
# ============================================================
code("""
if len(qualified) > 0:
    print(f"{'='*120}")
    print(f"{'Event':<12} {'Pick':<22} {'Odds':>6} {'Model':>7} {'Market':>7} {'Edge':>7} {'Stake':>8} {'Result':>7} {'P&L':>9}")
    print(f"{'='*120}")
    
    for evt in sorted(qualified['event'].unique()):
        evt_bets = qualified[qualified['event'] == evt].sort_values('edge', ascending=False)
        for _, b in evt_bets.iterrows():
            result_str = 'WIN ✅' if b['won'] else 'LOSS ❌'
            print(f"{b['event']:<12} {b['pick']:<22} {b['american_odds']:>+6.0f} "
                  f"{b['model_prob']:>6.1%} {b['market_fair']:>6.1%} {b['edge']:>+6.1%} "
                  f"${b['stake']:>7.2f} {result_str:>7} ${b['pnl']:>+8.2f}")
        evt_pnl = evt_bets['pnl'].sum()
        print(f"{'':>12} {'':>22} {'':>6} {'':>7} {'':>7} {'':>7} {'':>8} {'':>7} ${evt_pnl:>+8.2f}  ← event total")
        print()
    
    print(f"{'='*120}")
    total_pnl = qualified['pnl'].sum()
    total_staked = qualified['stake'].sum()
    print(f"{'TOTAL':<12} {'':<22} {'':>6} {'':>7} {'':>7} {'':>7} "
          f"${total_staked:>7.2f} {qualified['won'].sum()}W-{(~qualified['won']).sum()}L ${total_pnl:>+8.2f}")
""")

# ============================================================
# Cell 9: Edge analysis — model vs market
# ============================================================
code("""
print("=" * 70)
print("MODEL vs MARKET — CLOSING LINE VALUE")
print("=" * 70)

# For every fight (not just qualified), compare model prob to market prob
# If model consistently assigns higher prob to the winner, it has CLV

# Did the model assign higher probability to the actual winner than the market?
clv_rows = []
for _, row in bets_df.iterrows():
    if row['actual_winner'] == row['fighter_1']:
        model_winner_prob = row['model_p1']
        market_winner_prob = row['fair_1']
    else:
        model_winner_prob = row['model_p2']
        market_winner_prob = row['fair_2']
    
    clv_rows.append({
        'event': row['event'],
        'fight': f"{row['fighter_1']} vs {row['fighter_2']}",
        'winner': row['actual_winner'],
        'model_prob': model_winner_prob,
        'market_prob': market_winner_prob,
        'clv': model_winner_prob - market_winner_prob,
        'tier': row['tier'],
    })

clv_df = pd.DataFrame(clv_rows)

print(f"\\nAvg model prob assigned to winner: {clv_df['model_prob'].mean():.1%}")
print(f"Avg market prob assigned to winner: {clv_df['market_prob'].mean():.1%}")
print(f"Avg CLV (model - market): {clv_df['clv'].mean():+.1%}")
print(f"CLV positive in {(clv_df['clv'] > 0).sum()}/{len(clv_df)} fights ({(clv_df['clv'] > 0).mean():.1%})")

print(f"\\nCLV by tier:")
clv_tier = clv_df.groupby('tier').agg(
    fights=('clv', 'size'),
    avg_clv=('clv', 'mean'),
    pct_positive=('clv', lambda x: (x > 0).mean())
).reindex(['VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW', 'NO_CONF']).dropna(how='all')
clv_tier['avg_clv'] = clv_tier['avg_clv'].apply(lambda x: f"{x:+.1%}")
clv_tier['pct_positive'] = clv_tier['pct_positive'].apply(lambda x: f"{x:.0%}")
print(clv_tier.to_string())

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Left: CLV distribution
axes[0].hist(clv_df['clv'], bins=25, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0].axvline(x=clv_df['clv'].mean(), color='green', linestyle='-', linewidth=2, label=f"Mean: {clv_df['clv'].mean():+.1%}")
axes[0].set_xlabel('CLV (Model - Market)')
axes[0].set_ylabel('Count')
axes[0].set_title('Closing Line Value Distribution')
axes[0].legend()

# Middle: model prob vs market prob scatter
ax1 = axes[1]
correct = clv_df[clv_df['model_prob'] > 0.5]  # model "picked" the winner
wrong = clv_df[clv_df['model_prob'] <= 0.5]
ax1.scatter(correct['market_prob'], correct['model_prob'], alpha=0.5, c='green', s=30, label='Model correct')
ax1.scatter(wrong['market_prob'], wrong['model_prob'], alpha=0.5, c='red', s=30, label='Model wrong')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax1.set_xlabel('Market Prob (Winner)')
ax1.set_ylabel('Model Prob (Winner)')
ax1.set_title('Model vs Market — Prob Assigned to Winner')
ax1.legend(fontsize=8)

# Right: cumulative CLV over fights
clv_df_sorted = clv_df.sort_values('event').reset_index(drop=True)
clv_df_sorted['cum_clv'] = clv_df_sorted['clv'].cumsum()
axes[2].plot(clv_df_sorted.index, clv_df_sorted['cum_clv'], 'b-', linewidth=2)
axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[2].set_xlabel('Fight #')
axes[2].set_ylabel('Cumulative CLV')
axes[2].set_title('Cumulative CLV Over Time')
axes[2].fill_between(clv_df_sorted.index, 0, clv_df_sorted['cum_clv'], 
                     where=clv_df_sorted['cum_clv'] > 0, alpha=0.15, color='green')
axes[2].fill_between(clv_df_sorted.index, 0, clv_df_sorted['cum_clv'], 
                     where=clv_df_sorted['cum_clv'] < 0, alpha=0.15, color='red')

plt.tight_layout()
plt.savefig(f"{DATA}backtest_clv.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {DATA}backtest_clv.png")
""")

# ============================================================
# Cell 10: Bankroll simulation
# ============================================================
code("""
print("=" * 70)
print("BANKROLL SIMULATION")
print("=" * 70)

# Simulate chronological betting with rolling bankroll
# Sort qualified bets by event (chronological)
if len(qualified) > 0:
    sim = qualified.sort_values(['ufc_num']).reset_index(drop=True)
    
    starting_bankroll = BANKROLL
    bankroll = starting_bankroll
    history = [{'fight': 0, 'bankroll': bankroll, 'event': 'Start'}]
    
    for i, row in sim.iterrows():
        # Recalculate stake based on current bankroll
        if row['edge'] > 0 and row['decimal_odds'] > 1:
            kelly_frac = (row['edge'] / (row['decimal_odds'] - 1)) / 2
            kelly_frac = min(kelly_frac, 0.10)  # cap at 10% per bet
        else:
            kelly_frac = 0
        
        stake = kelly_frac * bankroll
        
        if row['won']:
            pnl = stake * (row['decimal_odds'] - 1)
        else:
            pnl = -stake
        
        bankroll += pnl
        history.append({
            'fight': i + 1,
            'bankroll': bankroll,
            'event': row['event'],
            'pick': row['pick'],
            'result': 'W' if row['won'] else 'L',
            'stake': round(stake, 2),
            'pnl': round(pnl, 2),
        })
    
    hist_df = pd.DataFrame(history)
    
    print(f"Starting bankroll: ${starting_bankroll:,.2f}")
    print(f"Final bankroll:    ${bankroll:,.2f}")
    print(f"Net P&L:           ${bankroll - starting_bankroll:+,.2f}")
    print(f"Return:            {(bankroll / starting_bankroll - 1) * 100:+.1f}%")
    print(f"Max bankroll:      ${hist_df['bankroll'].max():,.2f}")
    print(f"Min bankroll:      ${hist_df['bankroll'].min():,.2f}")
    print(f"Max drawdown:      ${hist_df['bankroll'].max() - hist_df['bankroll'].min():,.2f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(hist_df['fight'], hist_df['bankroll'], 'b-o', markersize=5, linewidth=2)
    ax.axhline(y=starting_bankroll, color='gray', linestyle='--', alpha=0.5, label=f'Start: ${starting_bankroll}')
    ax.fill_between(hist_df['fight'], starting_bankroll, hist_df['bankroll'],
                    where=hist_df['bankroll'] >= starting_bankroll, alpha=0.15, color='green')
    ax.fill_between(hist_df['fight'], starting_bankroll, hist_df['bankroll'],
                    where=hist_df['bankroll'] < starting_bankroll, alpha=0.15, color='red')
    
    # Annotate event boundaries
    prev_event = None
    for _, h in hist_df.iterrows():
        if h.get('event') != prev_event and h['fight'] > 0:
            ax.axvline(x=h['fight'], color='gray', linestyle=':', alpha=0.3)
            ax.text(h['fight'], ax.get_ylim()[1], h.get('event', ''), rotation=90, 
                    fontsize=7, va='top', ha='right', alpha=0.5)
        prev_event = h.get('event')
    
    ax.set_xlabel('Bet #')
    ax.set_ylabel('Bankroll ($)')
    ax.set_title('Bankroll Simulation — Qualified Bets Only (Half-Kelly, 10% Cap)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{DATA}backtest_bankroll.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {DATA}backtest_bankroll.png")
else:
    print("No qualified bets to simulate.")
""")

# ============================================================
# Cell 11: Market comparison — favorites vs model
# ============================================================
code("""
print("=" * 70)
print("STRATEGY COMPARISON")
print("=" * 70)

# Compare strategies on the full matched sample
strategies = {}

# 1. Always pick favorite (lower implied odds = higher market probability)
fav_correct = 0
for _, row in res.iterrows():
    fav = row['fighter_1'] if row['fair_1'] > row['fair_2'] else row['fighter_2']
    if fav == row['actual_winner']:
        fav_correct += 1
strategies['Always Favorite'] = fav_correct / len(res)

# 2. Always pick f1 (red corner)
strategies['Always Red'] = res.apply(
    lambda r: r['fighter_1'] == r['actual_winner'], axis=1
).mean()

# 3. Model — all fights
strategies['Model (all)'] = res['correct'].mean()

# 4. Model — unanimous only
if len(res[res['unanimous']]) > 0:
    strategies['Model (unanimous)'] = res[res['unanimous']]['correct'].mean()

# 5. Model — HIGH+ tier
high_plus = res[res['tier'].isin(['VERY_HIGH', 'HIGH'])]
if len(high_plus) > 0:
    strategies['Model (HIGH+)'] = high_plus['correct'].mean()

# 6. Qualified bets only
if len(qualified) > 0:
    strategies['Qualified bets'] = qualified['won'].mean()

print(f"\\n{'Strategy':<25} {'Accuracy':>10} {'N':>6}")
print("-" * 45)
for name, acc in sorted(strategies.items(), key=lambda x: -x[1]):
    if name == 'Model (all)':
        n = len(res)
    elif name == 'Model (unanimous)':
        n = len(res[res['unanimous']])
    elif name == 'Model (HIGH+)':
        n = len(high_plus)
    elif name == 'Qualified bets':
        n = len(qualified)
    else:
        n = len(res)
    print(f"{name:<25} {acc:>9.1%} {n:>6}")

# Bar chart
fig, ax = plt.subplots(figsize=(10, 5))
names = list(strategies.keys())
accs = list(strategies.values())
colors = ['#e74c3c' if a < 0.55 else '#f39c12' if a < 0.65 else '#27ae60' if a < 0.75 else '#2ecc71' for a in accs]
bars = ax.barh(names, accs, color=colors, edgecolor='white')
ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50%')
for bar, acc in zip(bars, accs):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f"{acc:.1%}", va='center', fontsize=10)
ax.set_xlabel('Accuracy')
ax.set_title('Strategy Comparison — Backtest')
ax.set_xlim(0, 1.05)
plt.tight_layout()
plt.savefig(f"{DATA}backtest_strategies.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {DATA}backtest_strategies.png")
""")

# ============================================================
# Cell 12: Save & summary
# ============================================================
code("""
print("=" * 70)
print("SAVE & SUMMARY")
print("=" * 70)

# Save full results
bets_df.to_csv(f"{DATA}backtest_full.csv", index=False)
print(f"Saved: {DATA}backtest_full.csv ({len(bets_df)} fights)")

if len(qualified) > 0:
    qualified.to_csv(f"{DATA}backtest_qualified.csv", index=False)
    print(f"Saved: {DATA}backtest_qualified.csv ({len(qualified)} bets)")

# Summary stats
print(f"\\n{'='*70}")
print(f"BACKTEST SUMMARY")
print(f"{'='*70}")
print(f"Period:              {res['event'].iloc[0]} → {res['event'].iloc[-1]}")
print(f"Events:              {res['event'].nunique()}")
print(f"Total fights:        {len(res)}")
print(f"Model accuracy:      {res['correct'].mean():.1%}")
if len(unan) > 0:
    print(f"Unanimous accuracy:  {unan['correct'].mean():.1%} ({len(unan)} fights)")
print(f"Avg CLV:             {clv_df['clv'].mean():+.1%}")

if len(qualified) > 0:
    total_staked = qualified['stake'].sum()
    total_pnl = qualified['pnl'].sum()
    roi = total_pnl / total_staked * 100 if total_staked > 0 else 0
    print(f"\\nBetting:")
    print(f"  Qualified bets:    {len(qualified)}")
    print(f"  Record:            {qualified['won'].sum()}W - {(~qualified['won']).sum()}L ({qualified['won'].mean():.1%})")
    print(f"  Total staked:      ${total_staked:,.2f}")
    print(f"  Total P&L:         ${total_pnl:+,.2f}")
    print(f"  ROI:               {roi:+.1f}%")

print(f"\\n⚠️  SAMPLE SIZE WARNING:")
print(f"  {len(res)} fights is too small for statistical significance.")
print(f"  A model with true 79% accuracy could show 71-87% on n={len(res)} (95% CI).")
if len(qualified) > 0:
    print(f"  {len(qualified)} bets is insufficient to confirm betting edge.")
    print(f"  Continue tracking live results over 10-15+ events for validation.")

print(f"\\nFiles saved:")
print(f"  {DATA}backtest_full.csv")
if len(qualified) > 0:
    print(f"  {DATA}backtest_qualified.csv")
print(f"  {DATA}backtest_accuracy.png")
print(f"  {DATA}backtest_clv.png")
print(f"  {DATA}backtest_bankroll.png")
print(f"  {DATA}backtest_strategies.png")
""")

# ============================================================
# Write notebook
# ============================================================
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

os.makedirs("notebooks", exist_ok=True)
path = "notebooks/10_backtest.ipynb"
with open(path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Created {path} — {len(cells)} cells")