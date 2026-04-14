#!/usr/bin/env python3
"""create_07_predict.py — generates notebooks/07_predict.ipynb"""

import json, pathlib

cells = []
def code(source, **kw):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()]
    })

def md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()]
    })

# ── Cell 1 ── Imports & Load ─────────────────────────────────────────
md("""# 07 — Fight Predictor
Input two fighters → get win probability from tuned XGBoost & LightGBM ensemble.  
Uses the same feature engineering pipeline as training.""")

code("""
import pandas as pd
import numpy as np
import joblib
import warnings, os
from datetime import datetime
warnings.filterwarnings('ignore')

DATA = './data/' if os.path.exists('./data/model_data.csv') else '../data/'
MODELS = '../models/' if os.path.exists('../models/xgb_tuned.joblib') else './models/'

# Load models
xgb_model = joblib.load(f'{MODELS}xgb_tuned.joblib')
lgb_model = joblib.load(f'{MODELS}lgb_tuned.joblib')
print("✅ Models loaded")

# Load data
model_data = pd.read_csv(f'{DATA}model_data.csv', parse_dates=['event_date'])
model_data = model_data.sort_values('event_date').reset_index(drop=True)
fighters_clean = pd.read_csv(f'{DATA}fighters_clean.csv')
fights_clean = pd.read_csv(f'{DATA}fights_clean.csv', parse_dates=['event_date'])
fights_clean = fights_clean.sort_values('event_date').reset_index(drop=True)

# Load feature list
with open(f'{DATA}feature_list.txt', 'r') as f:
    all_features = [line.strip() for line in f.readlines()]

print(f"✅ Data loaded: {len(model_data)} fights, {len(fighters_clean)} fighters")
print(f"✅ Features: {len(all_features)}")
print(f"✅ Latest fight in data: {model_data.event_date.max().date()}")
""")

# ── Cell 2 ── Fighter Lookup ─────────────────────────────────────────
md("""## Fighter Lookup
Search for fighters by name to get the exact spelling used in the dataset.""")

code("""
def search_fighter(query):
    \"\"\"Search fighters by partial name match.\"\"\"
    query = query.lower().strip()
    mask = fighters_clean['full_name'].str.lower().str.contains(query, na=False)
    results = fighters_clean[mask][['full_name', 'stance', 'wins', 'losses',
                                     'height_inches', 'reach_inches', 'weight_lbs']].copy()
    if len(results) == 0:
        print(f"No fighters found matching '{query}'")
        # Try first/last name separately
        words = query.split()
        for w in words:
            mask = fighters_clean['full_name'].str.lower().str.contains(w, na=False)
            partial = fighters_clean[mask]['full_name'].tolist()
            if partial:
                print(f"  Did you mean: {partial[:10]}")
    else:
        print(f"Found {len(results)} match(es):")
        print(results.to_string(index=False))
    return results

# Example searches — change these to find your fighters
search_fighter("islam")
print()
search_fighter("volkanovski")
""")

# ── Cell 3 ── Build Fighter Features ─────────────────────────────────
md("""## Feature Builder
Extracts the most recent feature snapshot for any fighter from model_data.  
This represents their stats heading into their next fight.""")

code("""
def get_fighter_latest(fighter_name, model_data, fighters_clean):
    \"\"\"
    Get the most recent feature set for a fighter.
    Returns a dict of all f1_* features (as if they're fighter_1).
    \"\"\"
    # Find their most recent fight as either f1 or f2
    as_f1 = model_data[model_data['fighter_1'] == fighter_name]
    as_f2 = model_data[model_data['fighter_2'] == fighter_name]

    latest_f1 = as_f1.iloc[-1] if len(as_f1) > 0 else None
    latest_f2 = as_f2.iloc[-1] if len(as_f2) > 0 else None

    # Pick the more recent appearance
    if latest_f1 is not None and latest_f2 is not None:
        if latest_f1['event_date'] >= latest_f2['event_date']:
            latest = latest_f1
            side = 'f1'
        else:
            latest = latest_f2
            side = 'f2'
    elif latest_f1 is not None:
        latest = latest_f1
        side = 'f1'
    elif latest_f2 is not None:
        latest = latest_f2
        side = 'f2'
    else:
        print(f"⚠️ {fighter_name} not found in fight data!")
        return None, None

    last_date = latest['event_date']
    last_result = None

    # Extract features — need to map f2_* to f1_* if they were fighter_2
    features = {}

    if side == 'f1':
        # Already in f1 position — grab f1_* columns directly
        for col in latest.index:
            if col.startswith('f1_'):
                features[col] = latest[col]
        last_result = 'W' if latest['f1_win'] == 1 else 'L'
    else:
        # They were f2 — need to rename f2_* → f1_*
        for col in latest.index:
            if col.startswith('f2_'):
                new_col = 'f1_' + col[3:]
                features[new_col] = latest[col]
        last_result = 'L' if latest['f1_win'] == 1 else 'W'

    print(f"  {fighter_name}: last fight {last_date.date()} ({last_result}), "
          f"found as {side}")

    return features, last_date


def update_career_after_last_fight(features, fighter_name, model_data):
    \"\"\"
    The features from model_data reflect stats BEFORE that fight.
    After the fight happened, career stats should be updated.
    This is an approximation — we update win rate and fight count
    based on the result of their last fight.
    \"\"\"
    # Find their last fight result
    as_f1 = model_data[model_data['fighter_1'] == fighter_name]
    as_f2 = model_data[model_data['fighter_2'] == fighter_name]

    all_fights = []
    for _, row in as_f1.iterrows():
        all_fights.append({'date': row['event_date'], 'won': row['f1_win'] == 1})
    for _, row in as_f2.iterrows():
        all_fights.append({'date': row['event_date'], 'won': row['f1_win'] == 0})

    all_fights = sorted(all_fights, key=lambda x: x['date'])

    if not all_fights:
        return features

    last_fight = all_fights[-1]

    # Update career stats to include the last fight's result
    if 'f1_career_fights' in features and not pd.isna(features.get('f1_career_fights')):
        old_fights = features['f1_career_fights']
        old_wr = features.get('f1_career_win_rate', 0.5)
        if not pd.isna(old_wr):
            old_wins = old_wr * old_fights
            new_wins = old_wins + (1 if last_fight['won'] else 0)
            new_fights = old_fights + 1
            features['f1_career_fights'] = new_fights
            features['f1_career_win_rate'] = new_wins / new_fights

    # Update streak
    if last_fight['won']:
        features['f1_win_streak'] = features.get('f1_win_streak', 0) + 1
        features['f1_loss_streak'] = 0
    else:
        features['f1_loss_streak'] = features.get('f1_loss_streak', 0) + 1
        features['f1_win_streak'] = 0

    return features


print("✅ Feature builder ready")
""")

# ── Cell 4 ── Prediction Function ────────────────────────────────────
md("""## Prediction Engine
Assembles features for both fighters, computes differentials, and runs both models.""")

code("""
def predict_fight(fighter_1, fighter_2, weight_class=None, event_date=None):
    \"\"\"
    Predict fight outcome.
    fighter_1 = red corner (favorite)
    fighter_2 = blue corner (underdog)
    \"\"\"
    print(f"\\n{'='*60}")
    print(f"  {fighter_1}  vs  {fighter_2}")
    print(f"{'='*60}")

    if event_date is None:
        event_date = datetime.now()

    # Get latest features for each fighter
    print(f"\\nLooking up fighters...")
    f1_features, f1_last = get_fighter_latest(fighter_1, model_data, fighters_clean)
    f2_features, f2_last = get_fighter_latest(fighter_2, model_data, fighters_clean)

    if f1_features is None or f2_features is None:
        print("❌ Cannot predict — fighter not found")
        return None

    # Update career stats to include their last fight
    f1_features = update_career_after_last_fight(f1_features, fighter_1, model_data)
    f2_features_renamed = {}
    for k, v in f2_features.items():
        f2_features_renamed['f2_' + k[3:]] = v  # f1_* → f2_*

    # Combine into single row
    row = {}
    row.update(f1_features)
    row.update(f2_features_renamed)

    # Compute differentials
    f1_cols = [c for c in row.keys() if c.startswith('f1_')]
    for f1_col in f1_cols:
        suffix = f1_col[3:]
        f2_col = f2_prefix = 'f2_' + suffix
        diff_col = 'diff_' + suffix
        if f2_col in row:
            f1_val = row[f1_col]
            f2_val = row[f2_col]
            if isinstance(f1_val, (int, float)) and isinstance(f2_val, (int, float)):
                row[diff_col] = f1_val - f2_val

    # Weight class encoding
    weight_order = {
        "Women's Strawweight": 1, "Women's Flyweight": 2,
        "Women's Bantamweight": 3, "Women's Featherweight": 4,
        "Flyweight": 5, "Bantamweight": 6, "Featherweight": 7,
        "Lightweight": 8, "Welterweight": 9, "Middleweight": 10,
        "Light Heavyweight": 11, "Heavyweight": 12, "Catch Weight": 6,
    }
    if weight_class:
        row['weight_class_ord'] = weight_order.get(weight_class, 6)

    # Stance encoding
    stance_map = {'Orthodox': 0, 'Southpaw': 1, 'Switch': 2}
    f1_stance = row.get('f1_stance', None)
    f2_stance = row.get('f2_stance', None)
    if isinstance(f1_stance, str):
        row['f1_stance_enc'] = stance_map.get(f1_stance, -1)
    if isinstance(f2_stance, str):
        row['f2_stance_enc'] = stance_map.get(f2_stance, -1)

    # Stance matchup features
    f1_s = row.get('f1_stance_enc', -1)
    f2_s = row.get('f2_stance_enc', -1)
    row['ortho_vs_south'] = int(
        (f1_s == 0 and f2_s == 1) or (f1_s == 1 and f2_s == 0))
    row['has_switch'] = int(f1_s == 2 or f2_s == 2)

    # Build feature vector
    feature_row = pd.DataFrame([row])

    # Ensure all features exist, fill missing with NaN
    for feat in all_features:
        if feat not in feature_row.columns:
            feature_row[feat] = np.nan

    X = feature_row[all_features]

    # Predict
    xgb_prob = xgb_model.predict_proba(X)[:, 1][0]
    lgb_prob = lgb_model.predict_proba(X)[:, 1][0]
    ens_prob = (xgb_prob + lgb_prob) / 2

    # Display results
    print(f"\\n{'─'*60}")
    print(f"  {'Model':<12} {'P(Red wins)':>12} {'P(Blue wins)':>13} {'Pick':>8}")
    print(f"{'─'*60}")

    for name, prob in [('XGBoost', xgb_prob), ('LightGBM', lgb_prob), ('Ensemble', ens_prob)]:
        pick = fighter_1 if prob >= 0.5 else fighter_2
        conf = max(prob, 1 - prob)
        print(f"  {name:<12} {prob:>11.1%} {1-prob:>12.1%} {pick:>8}")

    print(f"{'─'*60}")

    # Confidence assessment
    conf = abs(ens_prob - 0.5)
    if conf > 0.25:
        conf_label = "HIGH"
    elif conf > 0.15:
        conf_label = "MEDIUM"
    elif conf > 0.05:
        conf_label = "LOW"
    else:
        conf_label = "TOSS-UP"

    winner = fighter_1 if ens_prob >= 0.5 else fighter_2
    loser = fighter_2 if ens_prob >= 0.5 else fighter_1
    win_prob = max(ens_prob, 1 - ens_prob)

    print(f"\\n  🏆 PREDICTION: {winner} defeats {loser}")
    print(f"  📊 Confidence: {conf_label} ({win_prob:.1%})")

    # Key differentials
    print(f"\\n  Key differentials:")
    key_diffs = [
        ('diff_profile_win_pct', 'Win %'),
        ('diff_age', 'Age'),
        ('diff_profile_slpm', 'Strikes/min'),
        ('diff_profile_str_acc_career', 'Strike accuracy'),
        ('diff_profile_str_def_career', 'Strike defense'),
        ('diff_career_win_rate', 'Career WR (rolling)'),
        ('diff_win_streak', 'Win streak'),
        ('diff_reach_inches', 'Reach (inches)'),
    ]
    for feat, label in key_diffs:
        val = row.get(feat, np.nan)
        if not pd.isna(val):
            direction = f"→ favors {fighter_1}" if val > 0 else f"→ favors {fighter_2}" if val < 0 else "→ even"
            print(f"    {label:<22} {val:>+8.3f}  {direction}")

    return {
        'fighter_1': fighter_1, 'fighter_2': fighter_2,
        'xgb_prob': xgb_prob, 'lgb_prob': lgb_prob, 'ens_prob': ens_prob,
        'pick': winner, 'confidence': conf_label,
    }

print("✅ Prediction engine ready")
""")

# ── Cell 5 ── Single Fight Prediction ─────────────────────────────────
md("""## Predict a Fight
Change the fighter names below and run the cell.""")

code("""
# ═══════════════════════════════════════════
#   ENTER FIGHTERS HERE
# ═══════════════════════════════════════════

result = predict_fight(
    fighter_1 = "Islam Makhachev",     # Red corner (favorite)
    fighter_2 = "Alexander Volkanovski", # Blue corner (underdog)
    weight_class = "Lightweight",
)
""")

# ── Cell 6 ── Predict Full Card ──────────────────────────────────────
md("""## Predict a Full Card
Enter all fights on the card and get predictions for every bout.""")

code("""
def predict_card(card_name, fights):
    \"\"\"
    Predict an entire fight card.
    fights = list of (fighter_1, fighter_2, weight_class) tuples
    \"\"\"
    print(f"\\n{'#'*60}")
    print(f"  {card_name}")
    print(f"{'#'*60}")

    results = []
    for f1, f2, wc in fights:
        r = predict_fight(f1, f2, weight_class=wc)
        if r:
            results.append(r)

    # Summary table
    if results:
        print(f"\\n\\n{'#'*60}")
        print(f"  CARD SUMMARY: {card_name}")
        print(f"{'#'*60}")
        print(f"\\n  {'Fight':<40} {'Pick':<20} {'Prob':>6} {'Conf':<8}")
        print(f"  {'─'*76}")
        for r in results:
            fight = f"{r['fighter_1']} vs {r['fighter_2']}"
            prob = max(r['ens_prob'], 1 - r['ens_prob'])
            print(f"  {fight:<40} {r['pick']:<20} {prob:>5.1%} {r['confidence']:<8}")

        # Card stats
        high_conf = sum(1 for r in results if r['confidence'] == 'HIGH')
        med_conf = sum(1 for r in results if r['confidence'] == 'MEDIUM')
        low_conf = sum(1 for r in results if r['confidence'] == 'LOW')
        toss = sum(1 for r in results if r['confidence'] == 'TOSS-UP')
        print(f"\\n  Confidence breakdown: {high_conf} HIGH | {med_conf} MEDIUM | {low_conf} LOW | {toss} TOSS-UP")

    return results


# ═══════════════════════════════════════════
#   ENTER CARD HERE
# ═══════════════════════════════════════════

card = predict_card("UFC Fight Night — Example", [
    ("Islam Makhachev", "Alexander Volkanovski", "Lightweight"),
    ("Jon Jones", "Stipe Miocic", "Heavyweight"),
    # Add more fights...
])
""")

# ── Cell 7 ── Fighter Comparison ─────────────────────────────────────
md("""## Fighter Comparison
Side-by-side stats for any two fighters.""")

code("""
def compare_fighters(fighter_1, fighter_2):
    \"\"\"Side-by-side comparison of two fighters' current stats.\"\"\"

    f1_feat, f1_date = get_fighter_latest(fighter_1, model_data, fighters_clean)
    f2_feat, f2_date = get_fighter_latest(fighter_2, model_data, fighters_clean)

    if f1_feat is None or f2_feat is None:
        return

    # Profile stats from fighters_clean
    f1_profile = fighters_clean[fighters_clean['full_name'] == fighter_1].iloc[0] if len(fighters_clean[fighters_clean['full_name'] == fighter_1]) > 0 else None
    f2_profile = fighters_clean[fighters_clean['full_name'] == fighter_2].iloc[0] if len(fighters_clean[fighters_clean['full_name'] == fighter_2]) > 0 else None

    print(f"\\n{'='*60}")
    print(f"  {fighter_1}  vs  {fighter_2}")
    print(f"{'='*60}")

    stats = [
        ('Record', 'wins', 'losses'),
        ('Win %', 'win_pct', None),
        ('Height (in)', 'height_inches', None),
        ('Reach (in)', 'reach_inches', None),
        ('Weight (lbs)', 'weight_lbs', None),
        ('Stance', 'stance', None),
        ('SLpM', 'slpm', None),
        ('SApM', 'sapm', None),
        ('Str Acc', 'str_acc_career', None),
        ('Str Def', 'str_def_career', None),
        ('TD Avg', 'td_avg', None),
        ('TD Acc', 'td_acc_career', None),
        ('TD Def', 'td_def_career', None),
        ('Sub Avg', 'sub_avg', None),
    ]

    print(f"\\n  {'Stat':<18} {fighter_1:>18} {fighter_2:>18}  {'Δ':>8}")
    print(f"  {'─'*66}")

    for label, key1, key2 in stats:
        if f1_profile is not None and f2_profile is not None:
            if label == 'Record':
                f1_val = f"{int(f1_profile['wins'])}-{int(f1_profile['losses'])}"
                f2_val = f"{int(f2_profile['wins'])}-{int(f2_profile['losses'])}"
                print(f"  {label:<18} {f1_val:>18} {f2_val:>18}")
                continue

            f1_val = f1_profile.get(key1, np.nan)
            f2_val = f2_profile.get(key1, np.nan)

            if isinstance(f1_val, str) or isinstance(f2_val, str):
                print(f"  {label:<18} {str(f1_val):>18} {str(f2_val):>18}")
            elif not pd.isna(f1_val) and not pd.isna(f2_val):
                diff = f1_val - f2_val
                better = '←' if diff > 0 else '→' if diff < 0 else '='
                print(f"  {label:<18} {f1_val:>18.3f} {f2_val:>18.3f}  {diff:>+7.3f} {better}")
            else:
                f1_str = f"{f1_val:.3f}" if not pd.isna(f1_val) else "N/A"
                f2_str = f"{f2_val:.3f}" if not pd.isna(f2_val) else "N/A"
                print(f"  {label:<18} {f1_str:>18} {f2_str:>18}")

    # Rolling stats
    print(f"\\n  {'Rolling Stats':<18} {fighter_1:>18} {fighter_2:>18}  {'Δ':>8}")
    print(f"  {'─'*66}")

    rolling_stats = [
        ('Career WR', 'career_win_rate'),
        ('Career fights', 'career_fights'),
        ('Win streak', 'win_streak'),
        ('Loss streak', 'loss_streak'),
        ('Str acc (true)', 'career_str_acc_true'),
        ('Str def (true)', 'career_str_def_true'),
        ('TD acc (true)', 'career_td_acc_true'),
        ('Last 3 wins', 'last3_won'),
        ('Last 5 wins', 'last5_won'),
    ]

    for label, suffix in rolling_stats:
        f1_key = f'f1_{suffix}'
        f2_key = f'f1_{suffix}'  # f2 features stored as f1_ in their dict
        f1_val = f1_feat.get(f1_key, np.nan)
        f2_val = f2_feat.get(f2_key, np.nan)

        if not pd.isna(f1_val) and not pd.isna(f2_val):
            diff = f1_val - f2_val
            better = '←' if diff > 0 else '→' if diff < 0 else '='
            print(f"  {label:<18} {f1_val:>18.3f} {f2_val:>18.3f}  {diff:>+7.3f} {better}")
        else:
            f1_str = f"{f1_val:.3f}" if not pd.isna(f1_val) else "N/A"
            f2_str = f"{f2_val:.3f}" if not pd.isna(f2_val) else "N/A"
            print(f"  {label:<18} {f1_str:>18} {f2_str:>18}")


# ═══════════════════════════════════════════
compare_fighters("Islam Makhachev", "Alexander Volkanovski")
""")

# ── Write notebook ───────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {"name": "python", "version": "3.10.12"}
    },
    "cells": cells,
}

out = pathlib.Path("notebooks/07_predict.ipynb")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(f"Created {out}  ({len(cells)} cells)")