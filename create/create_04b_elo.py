#!/usr/bin/env python3
"""create_04b_elo.py – Generates notebooks/04b_elo.ipynb
Custom MMA Elo rating system with K-factor optimization,
analytics, and feature generation for the ML pipeline.

Uses ALL historical fights (pre-2015 included) for rating warm-up,
then generates features only for the 2015+ model_data window.
"""

import json, pathlib

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"}}, "cells": []}

def md(src):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": src.strip().splitlines(True)})

def code(src):
    nb["cells"].append({"cell_type": "code", "metadata": {}, "source": src.strip().splitlines(True),
                         "execution_count": None, "outputs": []})

# ============================================================
# CELL 1 — Intro
# ============================================================
md("""
# 04b — Custom MMA Elo Rating System

## Purpose
Build a dynamic Elo rating system purpose-built for MMA that:
1. Assigns every fighter a skill rating that evolves fight-by-fight
2. Accounts for MMA-specific factors (finishes, round, inactivity, experience)
3. Optimizes its own hyperparameters via grid search
4. Generates powerful new features for the ML pipeline (NB05/06/07/08)

## Why Elo?
Our current features are strong but have blind spots:
- **Profile stats** are static career totals (UFC's current numbers)
- **Rolling averages** weight all opponents equally
- **Win streaks** don't distinguish beating cans vs killers

Elo solves all three: it's **dynamic**, **opponent-adjusted**, and **transitive**.
Beating a 1900-rated fighter moves your rating far more than beating a 1300-rated fighter.
This is the missing signal.

## Architecture
- Warm-up on ALL fights (including pre-2015) so fighters entering the 2015+ window
  already have informed ratings instead of cold-starting at 1500
- Generate features only for 2015+ fights (matching model_data.csv)
- Grid search K-factor, finish bonus, decay rate to maximize predictive accuracy
- Output: updated model_data.csv with ~15 new Elo features

## New Features Generated
| Feature | Description |
|---------|-------------|
| `f1_elo`, `f2_elo` | Pre-fight Elo ratings |
| `diff_elo` | Rating differential (f1 - f2) |
| `elo_expected` | Classic Elo win probability for f1 |
| `f1_elo_momentum`, `f2_elo_momentum` | Rating change over last 3 fights |
| `diff_elo_momentum` | Momentum differential |
| `f1_elo_peak`, `f2_elo_peak` | Career peak rating |
| `diff_elo_peak` | Peak differential |
| `f1_elo_vs_peak`, `f2_elo_vs_peak` | Current / peak ratio (form indicator) |
| `diff_elo_vs_peak` | Form differential |
| `f1_elo_volatility`, `f2_elo_volatility` | Std dev of last 5 rating changes |
| `diff_elo_volatility` | Volatility differential |
| `f1_elo_fights`, `f2_elo_fights` | Total Elo-tracked fights (confidence) |
| `diff_elo_fights` | Experience differential |
""")

# ============================================================
# CELL 2 — Imports & Load Data
# ============================================================
code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from itertools import product
from tqdm.notebook import tqdm
import warnings, json, time
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.figsize': (14, 6), 'figure.dpi': 110,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 11
})

# ---------- auto-detect data path ----------
DATA = Path('./data') if Path('./data/fights_clean.csv').exists() else Path('../data')
assert (DATA / 'fights_clean.csv').exists(), f"Cannot find data at {DATA}"
print(f"Data path: {DATA}")

# ---------- load 2015+ cleaned fights (for feature generation) ----------
fights = pd.read_csv(DATA / 'fights_clean.csv', parse_dates=['event_date'])
fights = fights.sort_values('event_date').reset_index(drop=True)
print(f"Fights (2015+, clean): {len(fights):,} rows  |  {fights.event_date.min().date()} → {fights.event_date.max().date()}")

# ---------- load ALL fights (including pre-2015 for warm-up) ----------
# We need the raw fights + fight details to get pre-2015 data
# fights_raw.csv has all 8,637 fights but without detailed stats
# We'll rebuild a minimal warm-up dataset from fights_raw + fights_clean

fights_raw = pd.read_csv(DATA / 'fights_raw.csv')
print(f"Fights (raw, all years): {len(fights_raw):,} rows")

# fights_raw has: event_name, event_date, event_url, fight_url, fighter_1, fighter_2, winner
# We need: fighter_1, fighter_2, winner, event_date, finish_type, round

# For pre-2015 fights we won't have clean finish_type/round — we'll use defaults
# For 2015+ fights we use the clean data which has everything

# Load events to get dates for raw fights
events = pd.read_csv(DATA / 'events.csv', parse_dates=['event_date'])
event_dates = events.set_index('event_name')['event_date'].to_dict()

# Build warm-up dataset from raw fights that are NOT in our clean set
if 'event_date' not in fights_raw.columns:
    fights_raw['event_date'] = fights_raw['event_name'].map(event_dates)

fights_raw['event_date'] = pd.to_datetime(fights_raw['event_date'])
fights_raw = fights_raw.dropna(subset=['event_date']).sort_values('event_date').reset_index(drop=True)

# Pre-2015 fights for warm-up
warmup = fights_raw[fights_raw['event_date'] < '2015-01-01'].copy()
warmup['finish_type'] = 'UNK'  # We don't have clean finish type for pre-2015
warmup['round'] = 3            # Default assumption
warmup['total_rounds'] = 3
warmup['source'] = 'warmup'

# 2015+ fights from clean data
main = fights.copy()
main['source'] = 'main'
if 'total_rounds' not in main.columns:
    # Infer total rounds from weight class / title bout
    main['total_rounds'] = 3  # Default; title bouts are 5 but we approximate

# Combine: warmup + main, sorted chronologically
# Standardize columns
warmup_cols = warmup[['event_date', 'fighter_1', 'fighter_2', 'winner', 
                       'finish_type', 'round', 'total_rounds', 'source']].copy()

main_cols = main[['event_date', 'fighter_1', 'fighter_2', 'winner',
                   'finish_type', 'round', 'total_rounds', 'source']].copy()

# Map finish_type for main data
if 'finish_type' in main.columns:
    main_cols['finish_type'] = main['finish_type']
else:
    main_cols['finish_type'] = 'DEC'  # fallback

all_fights = pd.concat([warmup_cols, main_cols], ignore_index=True)
all_fights = all_fights.sort_values('event_date').reset_index(drop=True)

# Remove draws/NC (winner must be one of the two fighters)
all_fights = all_fights[
    (all_fights['winner'] == all_fights['fighter_1']) | 
    (all_fights['winner'] == all_fights['fighter_2'])
].reset_index(drop=True)

print(f"\\nAll fights for Elo (warm-up + main): {len(all_fights):,}")
print(f"  Warm-up (pre-2015): {(all_fights.source == 'warmup').sum():,}")
print(f"  Main (2015+):       {(all_fights.source == 'main').sum():,}")
print(f"  Date range: {all_fights.event_date.min().date()} → {all_fights.event_date.max().date()}")

# Quick sanity check
print(f"\\nUnique fighters: {pd.concat([all_fights.fighter_1, all_fights.fighter_2]).nunique():,}")
""")

# ============================================================
# CELL 3 — Core Elo Engine
# ============================================================
code("""
class MMAElo:
    \"\"\"
    Custom Elo rating system designed for MMA.
    
    MMA-specific features:
    - Dynamic K-factor: new fighters adapt fast, veterans stabilize
    - Finish bonus: KO/SUB wins move ratings more than decisions
    - Round scaling: earlier finishes = bigger rating swings
    - Inactivity decay: ratings drift toward mean during long layoffs
    - Rating floor: prevents ratings from going unrealistically low
    
    Parameters
    ----------
    k_base : float
        Base K-factor for experienced fighters.
    k_new : float  
        K-factor for brand new fighters (decays toward k_base with experience).
    k_exp_decay : float
        Controls how fast K decays from k_new to k_base. 
        K = k_base + (k_new - k_base) * exp(-k_exp_decay * n_fights)
    finish_mult : float
        Multiplier applied to K for KO/TKO/SUB finishes (1.0 = no bonus).
    round_scale : bool
        If True, earlier-round finishes get a bigger multiplier.
    decay_months : float
        Months of inactivity before decay kicks in.
    decay_rate : float
        Per-month decay rate toward starting rating (0 = no decay).
    starting : float
        Starting rating for new fighters.
    floor : float
        Minimum rating (prevents unrealistic lows).
    \"\"\"
    
    def __init__(self, k_base=32, k_new=80, k_exp_decay=0.1,
                 finish_mult=1.5, round_scale=True,
                 decay_months=18, decay_rate=0.03,
                 starting=1500, floor=1100):
        self.k_base = k_base
        self.k_new = k_new
        self.k_exp_decay = k_exp_decay
        self.finish_mult = finish_mult
        self.round_scale = round_scale
        self.decay_months = decay_months
        self.decay_rate = decay_rate
        self.starting = starting
        self.floor = floor
        
        # State
        self.ratings = {}          # fighter → current rating
        self.fight_counts = {}     # fighter → total fights processed
        self.last_fight_date = {}  # fighter → date of last fight
        self.history = []          # list of dicts (full audit trail)
        self.rating_changes = {}   # fighter → list of recent deltas
    
    def reset(self):
        \"\"\"Clear all state for a fresh run.\"\"\"
        self.ratings.clear()
        self.fight_counts.clear()
        self.last_fight_date.clear()
        self.history.clear()
        self.rating_changes.clear()
    
    def get_rating(self, fighter):
        \"\"\"Get current rating (or starting rating for new fighters).\"\"\"
        return self.ratings.get(fighter, self.starting)
    
    def get_k(self, fighter):
        \"\"\"Dynamic K-factor based on experience.\"\"\"
        n = self.fight_counts.get(fighter, 0)
        return self.k_base + (self.k_new - self.k_base) * np.exp(-self.k_exp_decay * n)
    
    def expected_score(self, ra, rb):
        \"\"\"Classic Elo expected score: P(A beats B).\"\"\"
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
    
    def _apply_decay(self, fighter, fight_date):
        \"\"\"
        Decay rating toward starting if fighter has been inactive.
        Returns decayed rating.
        \"\"\"
        if self.decay_rate <= 0 or fighter not in self.last_fight_date:
            return self.get_rating(fighter)
        
        last = self.last_fight_date[fighter]
        months_inactive = (fight_date - last).days / 30.44
        
        if months_inactive <= self.decay_months:
            return self.get_rating(fighter)
        
        excess_months = months_inactive - self.decay_months
        current = self.get_rating(fighter)
        decay_amount = self.decay_rate * excess_months * (current - self.starting)
        decayed = current - decay_amount
        
        # Don't decay past starting rating (or below floor)
        if current > self.starting:
            decayed = max(decayed, self.starting)
        elif current < self.starting:
            decayed = min(decayed, self.starting)
        
        return max(decayed, self.floor)
    
    def _get_multiplier(self, finish_type, round_num, total_rounds):
        \"\"\"
        Compute the K-factor multiplier based on how the fight ended.
        
        Logic:
        - Decisions: multiplier = 1.0 (base)
        - Finishes (KO/SUB): multiplier = finish_mult
        - Round scaling: earlier finish → bigger multiplier
          Round 1 finish gets full bonus, later rounds get less
        \"\"\"
        mult = 1.0
        
        # Finish bonus
        is_finish = finish_type in ('KO/TKO', 'SUB', 'KO', 'TKO', 'SUBMISSION')
        if is_finish and self.finish_mult > 1.0:
            mult = self.finish_mult
            
            # Round scaling: linear decay from full bonus in R1 to 1.0 in final round
            if self.round_scale and total_rounds > 1:
                try:
                    r = float(round_num)
                    tr = float(total_rounds)
                    # R1 → full mult, last round → 1.0
                    round_factor = 1.0 - (r - 1) / tr
                    # Interpolate between 1.0 and finish_mult
                    mult = 1.0 + (self.finish_mult - 1.0) * round_factor
                except (ValueError, TypeError):
                    pass  # Keep base finish_mult if round parsing fails
        
        return mult
    
    def update(self, fighter_1, fighter_2, winner, fight_date,
               finish_type='DEC', round_num=3, total_rounds=3,
               record=True):
        \"\"\"
        Process one fight and update ratings.
        
        Parameters
        ----------
        fighter_1, fighter_2 : str
            Fighter names (f1 = red corner, f2 = blue corner).
        winner : str
            Name of the winner (must be fighter_1 or fighter_2).
        fight_date : datetime
            Date of the fight.
        finish_type : str
            How the fight ended (KO/TKO, SUB, DEC, etc.).
        round_num : int
            Round the fight ended.
        total_rounds : int
            Total scheduled rounds.
        record : bool
            Whether to log this fight in history.
            
        Returns
        -------
        dict with pre-fight ratings, expected scores, and rating changes.
        \"\"\"
        # Apply inactivity decay before computing ratings
        ra_pre = self._apply_decay(fighter_1, fight_date)
        rb_pre = self._apply_decay(fighter_2, fight_date)
        
        # Store decayed ratings
        self.ratings[fighter_1] = ra_pre
        self.ratings[fighter_2] = rb_pre
        
        # Expected scores
        ea = self.expected_score(ra_pre, rb_pre)
        eb = 1.0 - ea
        
        # Actual scores
        if winner == fighter_1:
            sa, sb = 1.0, 0.0
        elif winner == fighter_2:
            sa, sb = 0.0, 1.0
        else:
            # Shouldn't happen (draws/NC filtered out)
            return None
        
        # K-factors (dynamic per fighter)
        ka = self.get_k(fighter_1)
        kb = self.get_k(fighter_2)
        
        # Finish multiplier
        mult = self._get_multiplier(finish_type, round_num, total_rounds)
        
        # Elo update
        delta_a = ka * mult * (sa - ea)
        delta_b = kb * mult * (sb - eb)
        
        ra_post = max(self.floor, ra_pre + delta_a)
        rb_post = max(self.floor, rb_pre + delta_b)
        
        # Update state
        self.ratings[fighter_1] = ra_post
        self.ratings[fighter_2] = rb_post
        self.fight_counts[fighter_1] = self.fight_counts.get(fighter_1, 0) + 1
        self.fight_counts[fighter_2] = self.fight_counts.get(fighter_2, 0) + 1
        self.last_fight_date[fighter_1] = fight_date
        self.last_fight_date[fighter_2] = fight_date
        
        # Track rating changes for volatility
        if fighter_1 not in self.rating_changes:
            self.rating_changes[fighter_1] = []
        if fighter_2 not in self.rating_changes:
            self.rating_changes[fighter_2] = []
        self.rating_changes[fighter_1].append(delta_a)
        self.rating_changes[fighter_2].append(delta_b)
        
        result = {
            'fighter_1': fighter_1,
            'fighter_2': fighter_2,
            'winner': winner,
            'fight_date': fight_date,
            'finish_type': finish_type,
            'round': round_num,
            'f1_elo_pre': ra_pre,
            'f2_elo_pre': rb_pre,
            'f1_elo_post': ra_post,
            'f2_elo_post': rb_post,
            'f1_delta': delta_a,
            'f2_delta': delta_b,
            'f1_expected': ea,
            'f2_expected': eb,
            'f1_k': ka,
            'f2_k': kb,
            'multiplier': mult,
            'f1_fights': self.fight_counts[fighter_1],
            'f2_fights': self.fight_counts[fighter_2],
            'higher_rated_won': (ra_pre > rb_pre and winner == fighter_1) or 
                                 (rb_pre > ra_pre and winner == fighter_2),
            'correct_prediction': (ea > 0.5 and winner == fighter_1) or
                                   (ea < 0.5 and winner == fighter_2) or
                                   (ea == 0.5)  # tie → count as correct
        }
        
        if record:
            self.history.append(result)
        
        return result
    
    def run(self, fights_df, progress=True):
        \"\"\"
        Process an entire dataframe of fights chronologically.
        
        Expects columns: fighter_1, fighter_2, winner, event_date,
                         finish_type, round, total_rounds
        \"\"\"
        self.reset()
        
        iterator = fights_df.iterrows()
        if progress:
            iterator = tqdm(list(iterator), desc='Processing fights', leave=True)
        
        for _, row in iterator:
            self.update(
                fighter_1=row['fighter_1'],
                fighter_2=row['fighter_2'],
                winner=row['winner'],
                fight_date=row['event_date'],
                finish_type=row.get('finish_type', 'DEC'),
                round_num=row.get('round', 3),
                total_rounds=row.get('total_rounds', 3)
            )
        
        return pd.DataFrame(self.history)
    
    def get_snapshot(self):
        \"\"\"Return current ratings for all fighters as a DataFrame.\"\"\"
        data = []
        for fighter, rating in self.ratings.items():
            data.append({
                'fighter': fighter,
                'elo': rating,
                'fights': self.fight_counts.get(fighter, 0),
                'last_fight': self.last_fight_date.get(fighter, None)
            })
        return pd.DataFrame(data).sort_values('elo', ascending=False).reset_index(drop=True)
    
    def predict(self, fighter_1, fighter_2, fight_date=None):
        \"\"\"Get pre-fight Elo prediction without updating ratings.\"\"\"
        ra = self.get_rating(fighter_1)
        rb = self.get_rating(fighter_2)
        
        # Optionally apply decay
        if fight_date is not None:
            ra = self._apply_decay(fighter_1, fight_date)
            rb = self._apply_decay(fighter_2, fight_date)
        
        ea = self.expected_score(ra, rb)
        return {
            'f1_elo': ra, 'f2_elo': rb,
            'f1_win_prob': ea, 'f2_win_prob': 1 - ea,
            'elo_diff': ra - rb,
            'pick': fighter_1 if ea >= 0.5 else fighter_2,
            'confidence': max(ea, 1 - ea)
        }


print("MMAElo class defined.")
print("Key methods: .run(df), .predict(f1, f2), .get_snapshot()")
""")

# ============================================================
# CELL 4 — K-Factor Optimization (Grid Search)
# ============================================================
code("""
# ============================================================
# Grid search for optimal Elo hyperparameters
# Metric: log loss of Elo-predicted probabilities on 2015+ fights
# We use ALL fights for warm-up but only score on 2015+ (main) fights
# ============================================================
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss

# Define search grid
param_grid = {
    'k_base':       [20, 28, 36, 44],
    'k_new':        [50, 70, 90, 120],
    'k_exp_decay':  [0.05, 0.10, 0.15],
    'finish_mult':  [1.0, 1.25, 1.5, 1.75],
    'decay_rate':   [0.0, 0.02, 0.04],
}

# All combinations
keys = list(param_grid.keys())
combos = list(product(*param_grid.values()))
print(f"Grid search: {len(combos):,} parameter combinations")

# Pre-compute indices for 2015+ scoring
main_start_idx = all_fights[all_fights.source == 'main'].index[0]

results = []
best_ll = 999
best_params = None

t0 = time.time()
for i, vals in enumerate(tqdm(combos, desc='Grid search')):
    params = dict(zip(keys, vals))
    
    elo = MMAElo(
        k_base=params['k_base'],
        k_new=params['k_new'],
        k_exp_decay=params['k_exp_decay'],
        finish_mult=params['finish_mult'],
        round_scale=True,
        decay_months=18,
        decay_rate=params['decay_rate'],
        starting=1500,
        floor=1100
    )
    
    history_df = elo.run(all_fights, progress=False)
    
    # Score only on 2015+ fights
    main_hist = history_df[history_df['fight_date'] >= '2015-01-01'].copy()
    
    if len(main_hist) < 100:
        continue
    
    # Elo's prediction: f1_expected is P(f1 wins)
    y_true = (main_hist['winner'] == main_hist['fighter_1']).astype(int)
    y_prob = main_hist['f1_expected'].values
    
    ll = log_loss(y_true, y_prob)
    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    brier = brier_score_loss(y_true, y_prob)
    higher_rated_wr = main_hist['higher_rated_won'].mean()
    
    results.append({**params, 'log_loss': ll, 'accuracy': acc, 
                    'brier': brier, 'higher_rated_wr': higher_rated_wr,
                    'n_fights': len(main_hist)})
    
    if ll < best_ll:
        best_ll = ll
        best_params = params.copy()

elapsed = time.time() - t0
results_df = pd.DataFrame(results).sort_values('log_loss')

print(f"\\nGrid search complete in {elapsed:.1f}s")
print(f"\\n{'='*60}")
print(f"BEST PARAMETERS (by log loss)")
print(f"{'='*60}")
for k, v in best_params.items():
    print(f"  {k:20s}: {v}")
best_row = results_df.iloc[0]
print(f"\\n  Log Loss:          {best_row['log_loss']:.4f}")
print(f"  Accuracy:          {best_row['accuracy']:.4f}")
print(f"  Brier Score:       {best_row['brier']:.4f}")
print(f"  Higher-Rated WR:   {best_row['higher_rated_wr']:.4f}")
print(f"  Scored on:         {best_row['n_fights']:.0f} fights (2015+)")

# Show top 10 combos
print(f"\\nTop 10 parameter combinations:")
print(results_df.head(10).to_string(index=False))

# Show worst for contrast
print(f"\\nBottom 3 (worst):")
print(results_df.tail(3).to_string(index=False))
""")

# ============================================================
# CELL 5 — Parameter Sensitivity Analysis
# ============================================================
code("""
# ============================================================
# Visualize how each parameter affects performance
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Elo Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')

param_names = ['k_base', 'k_new', 'k_exp_decay', 'finish_mult', 'decay_rate']
metrics = ['log_loss', 'accuracy']

for idx, param in enumerate(param_names):
    ax = axes.flat[idx]
    
    # Group by this param, average over other params
    grouped = results_df.groupby(param).agg({
        'log_loss': ['mean', 'min', 'std'],
        'accuracy': ['mean', 'max', 'std']
    }).reset_index()
    
    grouped.columns = [param, 'll_mean', 'll_min', 'll_std', 
                        'acc_mean', 'acc_max', 'acc_std']
    
    # Plot log loss (lower = better)
    color1 = '#e74c3c'
    color2 = '#2ecc71'
    
    ax2 = ax.twinx()
    
    bars = ax.bar(range(len(grouped)), grouped['ll_mean'], 
                   alpha=0.6, color=color1, label='Avg Log Loss', width=0.4,
                   yerr=grouped['ll_std'], capsize=3)
    
    ax2.plot(range(len(grouped)), grouped['acc_mean'], 
             color=color2, marker='o', linewidth=2, markersize=8, label='Avg Accuracy')
    
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped[param].values)
    ax.set_xlabel(param, fontweight='bold')
    ax.set_ylabel('Log Loss', color=color1)
    ax2.set_ylabel('Accuracy', color=color2)
    
    # Mark best value
    best_val = best_params[param]
    best_idx = list(grouped[param].values).index(best_val)
    ax.get_children()[best_idx].set_edgecolor('black')
    ax.get_children()[best_idx].set_linewidth(2)

# Use last subplot for a summary
ax = axes.flat[5]
ax.axis('off')
summary_text = "OPTIMAL PARAMETERS\\n" + "="*30 + "\\n"
for k, v in best_params.items():
    summary_text += f"{k}: {v}\\n"
summary_text += f"\\nLog Loss: {best_row['log_loss']:.4f}"
summary_text += f"\\nAccuracy: {best_row['accuracy']:.4f}"
summary_text += f"\\nBrier:    {best_row['brier']:.4f}"
ax.text(0.1, 0.5, summary_text, fontsize=13, fontfamily='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow'))

plt.tight_layout()
plt.savefig(DATA / 'elo_parameter_sensitivity.png', bbox_inches='tight')
plt.show()
print("Saved: elo_parameter_sensitivity.png")
""")

# ============================================================
# CELL 6 — Run Optimal Elo & Full History
# ============================================================
code("""
# ============================================================
# Run the optimized Elo system on all fights
# ============================================================
elo = MMAElo(
    k_base=best_params['k_base'],
    k_new=best_params['k_new'],
    k_exp_decay=best_params['k_exp_decay'],
    finish_mult=best_params['finish_mult'],
    round_scale=True,
    decay_months=18,
    decay_rate=best_params['decay_rate'],
    starting=1500,
    floor=1100
)

elo_history = elo.run(all_fights, progress=True)

# Split into warm-up and main
warmup_hist = elo_history[elo_history['fight_date'] < '2015-01-01']
main_hist = elo_history[elo_history['fight_date'] >= '2015-01-01']

print(f"\\nElo history: {len(elo_history):,} total fights")
print(f"  Warm-up (pre-2015): {len(warmup_hist):,}")
print(f"  Main (2015+):       {len(main_hist):,}")

# Performance on 2015+ fights
y_true = (main_hist['winner'] == main_hist['fighter_1']).astype(int)
y_prob = main_hist['f1_expected'].values
y_pred = (y_prob >= 0.5).astype(int)

from sklearn.metrics import classification_report
print(f"\\n{'='*60}")
print(f"OPTIMIZED ELO PERFORMANCE (2015+ fights)")
print(f"{'='*60}")
print(f"  Accuracy:        {accuracy_score(y_true, y_pred):.4f}")
print(f"  Log Loss:        {log_loss(y_true, y_prob):.4f}")
print(f"  Brier Score:     {brier_score_loss(y_true, y_prob):.4f}")
print(f"  Higher-Rated WR: {main_hist['higher_rated_won'].mean():.4f}")
print(f"  Avg |Elo Diff|:  {abs(main_hist['f1_elo_pre'] - main_hist['f2_elo_pre']).mean():.1f}")
print(f"  Avg Multiplier:  {main_hist['multiplier'].mean():.3f}")

# Elo vs always-pick-red baseline
baseline_acc = y_true.mean()
elo_acc = accuracy_score(y_true, y_pred)
print(f"\\n  Always-Red Baseline: {baseline_acc:.4f}")
print(f"  Elo Accuracy:        {elo_acc:.4f}")
print(f"  Elo Lift:            +{elo_acc - baseline_acc:.4f}")

# Rating distribution
snapshot = elo.get_snapshot()
print(f"\\n{'='*60}")
print(f"RATING DISTRIBUTION (all {len(snapshot):,} fighters)")
print(f"{'='*60}")
print(snapshot['elo'].describe().to_string())
""")

# ============================================================
# CELL 7 — Elo Analytics & Visualizations
# ============================================================
code("""
# ============================================================
# Comprehensive Elo analytics
# ============================================================
fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.suptitle('MMA Elo System Analytics', fontsize=18, fontweight='bold', y=0.98)

# --- 1. Rating distribution ---
ax = axes[0, 0]
snapshot['elo'].hist(bins=50, ax=ax, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(1500, color='red', linestyle='--', linewidth=1.5, label='Starting (1500)')
ax.axvline(snapshot['elo'].median(), color='orange', linestyle='--', linewidth=1.5, label=f"Median ({snapshot['elo'].median():.0f})")
ax.set_title('Current Rating Distribution', fontweight='bold')
ax.set_xlabel('Elo Rating')
ax.legend()

# --- 2. Top 20 fighters ---
ax = axes[0, 1]
top20 = snapshot.head(20).sort_values('elo')
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, 20))
ax.barh(range(20), top20['elo'], color=colors, edgecolor='white')
ax.set_yticks(range(20))
ax.set_yticklabels(top20['fighter'].values, fontsize=9)
ax.set_xlabel('Elo Rating')
ax.set_title('Top 20 Fighters (Current)', fontweight='bold')
for i, (_, row) in enumerate(top20.iterrows()):
    ax.text(row['elo'] + 5, i, f"{row['elo']:.0f}", va='center', fontsize=8)

# --- 3. Elo accuracy over time ---
ax = axes[0, 2]
main_hist_copy = main_hist.copy()
main_hist_copy['year'] = main_hist_copy['fight_date'].dt.year
yearly = main_hist_copy.groupby('year').agg(
    accuracy=('correct_prediction', 'mean'),
    n_fights=('correct_prediction', 'count'),
    higher_wr=('higher_rated_won', 'mean')
).reset_index()

ax.plot(yearly['year'], yearly['accuracy'], 'o-', color='steelblue', 
        linewidth=2, markersize=8, label='Elo Accuracy')
ax.plot(yearly['year'], yearly['higher_wr'], 's--', color='coral',
        linewidth=1.5, markersize=6, label='Higher-Rated WR')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Accuracy')
ax.set_title('Elo Accuracy by Year', fontweight='bold')
ax.legend()
ax.set_ylim(0.45, 0.75)

# --- 4. Elo diff distribution (winners vs losers) ---
ax = axes[1, 0]
main_hist_copy['winner_elo'] = np.where(
    main_hist_copy['winner'] == main_hist_copy['fighter_1'],
    main_hist_copy['f1_elo_pre'], main_hist_copy['f2_elo_pre'])
main_hist_copy['loser_elo'] = np.where(
    main_hist_copy['winner'] == main_hist_copy['fighter_1'],
    main_hist_copy['f2_elo_pre'], main_hist_copy['f1_elo_pre'])
main_hist_copy['elo_diff_winner'] = main_hist_copy['winner_elo'] - main_hist_copy['loser_elo']

ax.hist(main_hist_copy['elo_diff_winner'], bins=60, color='steelblue', 
        edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
median_diff = main_hist_copy['elo_diff_winner'].median()
ax.axvline(median_diff, color='orange', linestyle='--', linewidth=1.5,
           label=f'Median: {median_diff:+.0f}')
ax.set_title('Winner Elo Advantage (Pre-Fight)', fontweight='bold')
ax.set_xlabel('Winner Elo - Loser Elo')
ax.legend()

# --- 5. Accuracy by Elo diff bucket ---
ax = axes[1, 1]
main_hist_copy['elo_diff_abs'] = abs(main_hist_copy['f1_elo_pre'] - main_hist_copy['f2_elo_pre'])
bins = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000]
labels = ['0-25', '25-50', '50-75', '75-100', '100-150', '150-200', '200-300', '300-500', '500+']
main_hist_copy['diff_bucket'] = pd.cut(main_hist_copy['elo_diff_abs'], bins=bins, labels=labels)

bucket_acc = main_hist_copy.groupby('diff_bucket', observed=True).agg(
    accuracy=('higher_rated_won', 'mean'),
    n_fights=('higher_rated_won', 'count')
).reset_index()

bars = ax.bar(range(len(bucket_acc)), bucket_acc['accuracy'], 
              color='steelblue', edgecolor='white', alpha=0.8)
ax.set_xticks(range(len(bucket_acc)))
ax.set_xticklabels(bucket_acc['diff_bucket'], rotation=45, fontsize=9)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('|Elo Difference|')
ax.set_ylabel('Higher-Rated Win Rate')
ax.set_title('Accuracy by Rating Gap', fontweight='bold')

# Add count labels
for i, (_, row) in enumerate(bucket_acc.iterrows()):
    ax.text(i, row['accuracy'] + 0.01, f"n={row['n_fights']}", 
            ha='center', fontsize=8, color='gray')

# --- 6. Rating trajectories of famous fighters ---
ax = axes[1, 2]

# Find fighters with most fights in main period
fighter_fight_counts = pd.concat([
    main_hist['fighter_1'], main_hist['fighter_2']
]).value_counts()

# Pick some interesting fighters (high-rated with enough fights)
active = snapshot[(snapshot['fights'] >= 15) & (snapshot['elo'] > 1600)].head(6)['fighter'].tolist()

if len(active) < 4:
    active = fighter_fight_counts.head(6).index.tolist()

for fighter in active[:6]:
    # Get this fighter's history
    f_hist = elo_history[
        (elo_history['fighter_1'] == fighter) | (elo_history['fighter_2'] == fighter)
    ].copy()
    
    f_hist['rating'] = np.where(
        f_hist['fighter_1'] == fighter, f_hist['f1_elo_post'], f_hist['f2_elo_post'])
    
    f_hist = f_hist[f_hist['fight_date'] >= '2015-01-01']
    
    if len(f_hist) >= 5:
        ax.plot(f_hist['fight_date'], f_hist['rating'], '-o', 
                markersize=3, linewidth=1.5, label=fighter, alpha=0.8)

ax.axhline(1500, color='gray', linestyle=':', alpha=0.5)
ax.set_title('Rating Trajectories (Top Fighters)', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Elo Rating')
ax.legend(fontsize=8, loc='best')

# --- 7. K-factor distribution ---
ax = axes[2, 0]
ax.hist(main_hist['f1_k'], bins=40, alpha=0.6, color='steelblue', 
        label='F1 K-factor', edgecolor='white')
ax.hist(main_hist['f2_k'], bins=40, alpha=0.6, color='coral',
        label='F2 K-factor', edgecolor='white')
ax.set_title('K-Factor Distribution', fontweight='bold')
ax.set_xlabel('K-Factor')
ax.legend()

# --- 8. Multiplier distribution ---
ax = axes[2, 1]
mult_counts = main_hist['multiplier'].value_counts().sort_index()
ax.bar(range(len(mult_counts)), mult_counts.values, color='steelblue', edgecolor='white')
ax.set_xticks(range(len(mult_counts)))
ax.set_xticklabels([f"{x:.2f}" for x in mult_counts.index], rotation=45, fontsize=9)
ax.set_title('Finish Multiplier Distribution', fontweight='bold')
ax.set_xlabel('Multiplier')
ax.set_ylabel('Count')

# --- 9. Upset rate by Elo confidence ---
ax = axes[2, 2]
main_hist_copy['elo_conf'] = main_hist_copy['f1_expected'].apply(lambda x: max(x, 1-x))
conf_bins = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0]
conf_labels = ['50-55%', '55-60%', '60-65%', '65-70%', '70-75%', '75-80%', '80-85%', '85-90%', '90%+']
main_hist_copy['conf_bucket'] = pd.cut(main_hist_copy['elo_conf'], bins=conf_bins, labels=conf_labels)

conf_perf = main_hist_copy.groupby('conf_bucket', observed=True).agg(
    accuracy=('correct_prediction', 'mean'),
    n_fights=('correct_prediction', 'count')
).reset_index()

colors_conf = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(conf_perf)))
ax.bar(range(len(conf_perf)), conf_perf['accuracy'], color=colors_conf, edgecolor='white')
ax.plot(range(len(conf_perf)), [float(l.split('-')[0].replace('%',''))/100 
        if '-' in l else 0.9 for l in conf_perf['conf_bucket']], 
        'k--', linewidth=1, alpha=0.5, label='Perfect calibration')
ax.set_xticks(range(len(conf_perf)))
ax.set_xticklabels(conf_perf['conf_bucket'], rotation=45, fontsize=9)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('Elo Confidence')
ax.set_ylabel('Actual Accuracy')
ax.set_title('Calibration: Confidence vs Accuracy', fontweight='bold')
ax.legend(fontsize=9)

for i, (_, row) in enumerate(conf_perf.iterrows()):
    ax.text(i, row['accuracy'] + 0.01, f"n={row['n_fights']}", 
            ha='center', fontsize=7, color='gray')

plt.tight_layout()
plt.savefig(DATA / 'elo_analytics.png', bbox_inches='tight', dpi=120)
plt.show()
print("Saved: elo_analytics.png")
""")

# ============================================================
# CELL 8 — Generate Elo Features for model_data
# ============================================================
code("""
# ============================================================
# Generate Elo features and merge into model_data
# ============================================================
model_data = pd.read_csv(DATA / 'model_data.csv', parse_dates=['event_date'])
print(f"model_data loaded: {model_data.shape}")

# Filter Elo history to 2015+ only (matching model_data)
elo_main = elo_history[elo_history['fight_date'] >= '2015-01-01'].copy()
print(f"Elo history (2015+): {len(elo_main):,} fights")

# ============================================================
# Build per-fighter Elo feature snapshots from history
# ============================================================
# For each fight, we need the PRE-FIGHT snapshot of each fighter

def compute_elo_features(elo_hist_df):
    \"\"\"
    From the full Elo history, compute derived features for each fight.
    All features are PRE-FIGHT (no leakage).
    \"\"\"
    rows = []
    
    # Track per-fighter rolling data
    fighter_elo_history = {}  # fighter → list of (date, pre_rating, post_rating, delta)
    fighter_peaks = {}        # fighter → peak rating ever
    
    for _, fight in elo_hist_df.iterrows():
        f1 = fight['fighter_1']
        f2 = fight['fighter_2']
        date = fight['fight_date']
        
        # --- F1 features (pre-fight) ---
        f1_hist = fighter_elo_history.get(f1, [])
        
        f1_elo = fight['f1_elo_pre']
        f1_peak = max([h[2] for h in f1_hist] + [fight['f1_elo_pre']]) if f1_hist else fight['f1_elo_pre']
        f1_vs_peak = f1_elo / f1_peak if f1_peak > 0 else 1.0
        
        # Momentum: sum of last 3 deltas
        f1_deltas = [h[3] for h in f1_hist]
        f1_momentum = sum(f1_deltas[-3:]) if len(f1_deltas) >= 3 else (
            sum(f1_deltas) if f1_deltas else 0)
        
        # Volatility: std of last 5 deltas
        f1_vol = np.std(f1_deltas[-5:]) if len(f1_deltas) >= 3 else np.nan
        
        f1_elo_fights = len(f1_hist)
        
        # --- F2 features (pre-fight) ---
        f2_hist = fighter_elo_history.get(f2, [])
        
        f2_elo = fight['f2_elo_pre']
        f2_peak = max([h[2] for h in f2_hist] + [fight['f2_elo_pre']]) if f2_hist else fight['f2_elo_pre']
        f2_vs_peak = f2_elo / f2_peak if f2_peak > 0 else 1.0
        
        f2_deltas = [h[3] for h in f2_hist]
        f2_momentum = sum(f2_deltas[-3:]) if len(f2_deltas) >= 3 else (
            sum(f2_deltas) if f2_deltas else 0)
        
        f2_vol = np.std(f2_deltas[-5:]) if len(f2_deltas) >= 3 else np.nan
        
        f2_elo_fights = len(f2_hist)
        
        # --- Fight-level features ---
        elo_expected = fight['f1_expected']
        
        row = {
            'fighter_1': f1,
            'fighter_2': f2,
            'fight_date': date,
            'f1_elo': f1_elo,
            'f2_elo': f2_elo,
            'diff_elo': f1_elo - f2_elo,
            'elo_expected': elo_expected,
            'f1_elo_momentum': f1_momentum,
            'f2_elo_momentum': f2_momentum,
            'diff_elo_momentum': f1_momentum - f2_momentum,
            'f1_elo_peak': f1_peak,
            'f2_elo_peak': f2_peak,
            'diff_elo_peak': f1_peak - f2_peak,
            'f1_elo_vs_peak': f1_vs_peak,
            'f2_elo_vs_peak': f2_vs_peak,
            'diff_elo_vs_peak': f1_vs_peak - f2_vs_peak,
            'f1_elo_volatility': f1_vol,
            'f2_elo_volatility': f2_vol,
            'diff_elo_volatility': f1_vol - f2_vol if pd.notna(f1_vol) and pd.notna(f2_vol) else np.nan,
            'f1_elo_fights': f1_elo_fights,
            'f2_elo_fights': f2_elo_fights,
            'diff_elo_fights': f1_elo_fights - f2_elo_fights,
        }
        rows.append(row)
        
        # Update running history (AFTER recording pre-fight features)
        if f1 not in fighter_elo_history:
            fighter_elo_history[f1] = []
        fighter_elo_history[f1].append((date, fight['f1_elo_pre'], fight['f1_elo_post'], fight['f1_delta']))
        
        if f2 not in fighter_elo_history:
            fighter_elo_history[f2] = []
        fighter_elo_history[f2].append((date, fight['f2_elo_pre'], fight['f2_elo_post'], fight['f2_delta']))
    
    return pd.DataFrame(rows)

print("Computing Elo features...")
elo_features = compute_elo_features(elo_main)
print(f"Elo features computed: {elo_features.shape}")
print(f"\\nNew features ({elo_features.shape[1] - 3} feature columns):")
feat_cols = [c for c in elo_features.columns if c not in ['fighter_1', 'fighter_2', 'fight_date']]
for c in feat_cols:
    print(f"  {c:30s}  NaN: {elo_features[c].isna().mean():.1%}  Mean: {elo_features[c].mean():.2f}")
""")

# ============================================================
# CELL 9 — Merge with model_data
# ============================================================
code("""
# ============================================================
# Merge Elo features into model_data
# ============================================================
# Match on fighter_1, fighter_2, event_date
# Both datasets are sorted chronologically so we can merge on all three

# Ensure date types match
elo_features['fight_date'] = pd.to_datetime(elo_features['fight_date'])
model_data['event_date'] = pd.to_datetime(model_data['event_date'])

# Merge keys
merge_keys = ['fighter_1', 'fighter_2', 'fight_date']
elo_features = elo_features.rename(columns={'fight_date': 'event_date'})

# Drop any existing Elo columns from model_data (in case of re-run)
existing_elo_cols = [c for c in model_data.columns if 'elo' in c.lower()]
if existing_elo_cols:
    print(f"Dropping {len(existing_elo_cols)} existing Elo columns from model_data")
    model_data = model_data.drop(columns=existing_elo_cols)

# Merge
merge_cols = ['fighter_1', 'fighter_2', 'event_date'] + feat_cols
model_data_elo = model_data.merge(
    elo_features[merge_cols],
    on=['fighter_1', 'fighter_2', 'event_date'],
    how='left'
)

# Check merge success
n_matched = model_data_elo['f1_elo'].notna().sum()
n_total = len(model_data_elo)
print(f"Merge results:")
print(f"  model_data rows:    {n_total:,}")
print(f"  Elo features found: {n_matched:,} ({n_matched/n_total:.1%})")
print(f"  Missing:            {n_total - n_matched:,}")

if n_matched < n_total * 0.95:
    print("\\n⚠️  WARNING: More than 5% of fights missing Elo features!")
    print("  This might indicate a merge key mismatch.")
    print("  Checking for name mismatches...")
    
    # Show some unmatched fights
    unmatched = model_data_elo[model_data_elo['f1_elo'].isna()]
    print(f"\\n  Sample unmatched fights:")
    print(unmatched[['event_date', 'fighter_1', 'fighter_2']].head(10).to_string(index=False))

# Verify no shape change
assert len(model_data_elo) == len(model_data), \\
    f"Row count changed! {len(model_data)} → {len(model_data_elo)}"

print(f"\\nFinal shape: {model_data_elo.shape} (was {model_data.shape})")
print(f"New columns added: {model_data_elo.shape[1] - model_data.shape[1]}")
""")

# ============================================================
# CELL 10 — Correlation Analysis
# ============================================================
code("""
# ============================================================
# How do Elo features correlate with f1_win?
# ============================================================
target = 'f1_win'
elo_feat_cols = [c for c in feat_cols if c in model_data_elo.columns]

# Correlations with target
corrs = model_data_elo[elo_feat_cols + [target]].corr()[target].drop(target).sort_values(ascending=False)

print("Elo Feature Correlations with f1_win:")
print("="*55)
for feat, corr in corrs.items():
    bar = '█' * int(abs(corr) * 100)
    sign = '+' if corr > 0 else '-'
    print(f"  {feat:30s}  {sign}{abs(corr):.4f}  {bar}")

# Compare with top existing features
print(f"\\n{'='*55}")
print("Context: Top 5 existing features for comparison")
print("="*55)
existing_feats = [c for c in model_data.columns if c not in ['f1_win', 'event_date', 'fighter_1', 'fighter_2', 'event_name', 'fight_url', 'weight_class', 'stance_matchup']]
existing_corrs = model_data[existing_feats + [target]].corr()[target].drop(target).abs().sort_values(ascending=False)
for feat, corr in existing_corrs.head(5).items():
    print(f"  {feat:30s}  {corr:.4f}")

# Visual
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Bar chart of Elo correlations
ax = axes[0]
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in corrs.values]
ax.barh(range(len(corrs)), corrs.values, color=colors, edgecolor='white')
ax.set_yticks(range(len(corrs)))
ax.set_yticklabels(corrs.index, fontsize=9)
ax.set_xlabel('Correlation with f1_win')
ax.set_title('Elo Feature Correlations', fontweight='bold')
ax.axvline(0, color='gray', linewidth=0.5)
ax.invert_yaxis()

# Scatter: diff_elo vs f1_win probability (binned)
ax = axes[1]
valid = model_data_elo.dropna(subset=['diff_elo', target])
valid['elo_bin'] = pd.cut(valid['diff_elo'], bins=20)
binned = valid.groupby('elo_bin', observed=True).agg(
    elo_mid=('diff_elo', 'mean'),
    win_rate=(target, 'mean'),
    count=(target, 'count')
).reset_index()

# Size by count
sizes = (binned['count'] / binned['count'].max() * 200) + 20
ax.scatter(binned['elo_mid'], binned['win_rate'], s=sizes, 
           c=binned['win_rate'], cmap='RdYlGn', edgecolors='black', linewidth=0.5)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Elo Differential (f1 - f2)')
ax.set_ylabel('F1 Win Rate')
ax.set_title('Elo Diff vs Win Probability', fontweight='bold')

# Fit a logistic-style curve for reference
from numpy.polynomial.polynomial import polyfit
z = np.polyfit(binned['elo_mid'], binned['win_rate'], 3)
x_smooth = np.linspace(binned['elo_mid'].min(), binned['elo_mid'].max(), 100)
y_smooth = np.polyval(z, x_smooth)
ax.plot(x_smooth, np.clip(y_smooth, 0, 1), 'k--', linewidth=1.5, alpha=0.5, label='Trend')
ax.legend()

plt.tight_layout()
plt.savefig(DATA / 'elo_correlations.png', bbox_inches='tight')
plt.show()
print("Saved: elo_correlations.png")
""")

# ============================================================
# CELL 11 — Quick Model Validation (with vs without Elo)
# ============================================================
code("""
# ============================================================
# Quick validation: XGBoost with vs without Elo features
# This tells us if Elo actually helps the tree models
# ============================================================
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss

# Load original feature list
with open(DATA / 'feature_list.txt', 'r') as f:
    original_features = [line.strip() for line in f if line.strip()]

print(f"Original features: {len(original_features)}")

elo_feature_names = [c for c in elo_feat_cols if c in model_data_elo.columns]
all_features = original_features + elo_feature_names
print(f"With Elo features: {len(all_features)} (+{len(elo_feature_names)})")

# Same temporal split as NB05/NB06
split_date = pd.Timestamp('2025-07-01')
train_mask = model_data_elo['event_date'] < split_date
test_mask = model_data_elo['event_date'] >= split_date

original_valid = [f for f in original_features if f in model_data_elo.columns]
all_valid = [f for f in all_features if f in model_data_elo.columns]

X_train_orig = model_data_elo.loc[train_mask, original_valid]
X_test_orig = model_data_elo.loc[test_mask, original_valid]
X_train_elo = model_data_elo.loc[train_mask, all_valid]
X_test_elo = model_data_elo.loc[test_mask, all_valid]
y_train = model_data_elo.loc[train_mask, 'f1_win']
y_test = model_data_elo.loc[test_mask, 'f1_win']

print(f"\\nTrain: {len(X_train_orig):,} | Test: {len(X_test_orig):,}")

# Load best params from NB06 if available
try:
    with open(DATA / 'best_params.json', 'r') as f:
        bp = json.load(f)
    xgb_params = bp.get('xgb', {})
    for drop_key in ['n_estimators', 'early_stopping_rounds']:
        xgb_params.pop(drop_key, None)
    print(f"Loaded tuned XGBoost params from NB06")
except Exception:
    xgb_params = {}
    print("Using default XGBoost params")

common = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'random_state': 42,
    'n_estimators': 1000,
    'early_stopping_rounds': 50,
    'verbosity': 0,
    **xgb_params
}

def train_and_eval(X_tr, X_te, y_tr, y_te, label):
    model = XGBClassifier(**common)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    pred = model.predict(X_te)
    prob = model.predict_proba(X_te)[:, 1]
    trees = model.best_iteration + 1 if hasattr(model, 'best_iteration') and model.best_iteration is not None else common['n_estimators']
    metrics = {
        'label': label,
        'n_features': X_tr.shape[1],
        'accuracy': accuracy_score(y_te, pred),
        'log_loss': log_loss(y_te, prob),
        'auc': roc_auc_score(y_te, prob),
        'brier': brier_score_loss(y_te, prob),
        'trees': trees
    }
    return model, prob, metrics

# --- Model A: Without Elo ---
print("\\nTraining XGBoost WITHOUT Elo features...")
model_a, prob_a, m_a = train_and_eval(X_train_orig, X_test_orig, y_train, y_test, 'Without Elo')
print(f"  Acc: {m_a['accuracy']:.4f}  LL: {m_a['log_loss']:.4f}  AUC: {m_a['auc']:.4f}  Brier: {m_a['brier']:.4f}")

# --- Model B: With Elo ---
print("\\nTraining XGBoost WITH Elo features...")
model_b, prob_b, m_b = train_and_eval(X_train_elo, X_test_elo, y_train, y_test, 'With Elo')
print(f"  Acc: {m_b['accuracy']:.4f}  LL: {m_b['log_loss']:.4f}  AUC: {m_b['auc']:.4f}  Brier: {m_b['brier']:.4f}")

# --- Model C: Elo-only ---
print("\\nTraining XGBoost with ONLY Elo features...")
model_c, prob_c, m_c = train_and_eval(
    model_data_elo.loc[train_mask, elo_feature_names],
    model_data_elo.loc[test_mask, elo_feature_names],
    y_train, y_test, 'Elo Only')
print(f"  Acc: {m_c['accuracy']:.4f}  LL: {m_c['log_loss']:.4f}  AUC: {m_c['auc']:.4f}  Brier: {m_c['brier']:.4f}")

# --- Model D: Elo expected score alone (logistic baseline) ---
from sklearn.metrics import accuracy_score as acc_s
elo_test = model_data_elo.loc[test_mask, 'elo_expected'].values
elo_test_clean = np.where(np.isnan(elo_test), 0.5, elo_test)
pred_d = (elo_test_clean >= 0.5).astype(int)
m_d = {
    'label': 'Elo Expected (raw)',
    'n_features': 1,
    'accuracy': acc_s(y_test, pred_d),
    'log_loss': log_loss(y_test, np.clip(elo_test_clean, 1e-7, 1-1e-7)),
    'auc': roc_auc_score(y_test, elo_test_clean),
    'brier': brier_score_loss(y_test, elo_test_clean),
    'trees': 0
}
print(f"\\nElo Expected (raw probability, no ML):")
print(f"  Acc: {m_d['accuracy']:.4f}  LL: {m_d['log_loss']:.4f}  AUC: {m_d['auc']:.4f}  Brier: {m_d['brier']:.4f}")

# --- Comparison Table ---
print(f"\\n{'='*78}")
print(f"MODEL COMPARISON — Does Elo Help?")
print(f"{'='*78}")
print(f"{'Model':<25s} {'Feats':>6s} {'Acc':>8s} {'LL':>8s} {'AUC':>8s} {'Brier':>8s} {'Trees':>6s}")
print(f"{'-'*78}")
for m in [m_a, m_b, m_c, m_d]:
    print(f"{m['label']:<25s} {m['n_features']:>6d} {m['accuracy']:>8.4f} {m['log_loss']:>8.4f} {m['auc']:>8.4f} {m['brier']:>8.4f} {m['trees']:>6d}")
print(f"{'-'*78}")

d_acc = m_b['accuracy'] - m_a['accuracy']
d_ll  = m_b['log_loss'] - m_a['log_loss']
d_auc = m_b['auc'] - m_a['auc']
d_bri = m_b['brier'] - m_a['brier']
print(f"{'Delta (With - Without)':<25s} {'':>6s} {d_acc:>+8.4f} {d_ll:>+8.4f} {d_auc:>+8.4f} {d_bri:>+8.4f}")

verdict_parts = []
if d_acc > 0:
    verdict_parts.append(f"Accuracy +{d_acc:.4f}")
if d_auc > 0:
    verdict_parts.append(f"AUC +{d_auc:.4f}")
if d_ll < 0:
    verdict_parts.append(f"LogLoss {d_ll:.4f}")
if d_bri < 0:
    verdict_parts.append(f"Brier {d_bri:.4f}")

if d_acc > 0 or d_auc > 0.003:
    print(f"\\n✅  Elo features IMPROVE the model!")
    for v in verdict_parts:
        print(f"    {v}")
elif d_acc == 0 and abs(d_auc) < 0.003:
    print(f"\\n➡️  Elo features are NEUTRAL — no harm, slight signal overlap with existing features.")
    print(f"    Still worth including: Elo provides cold-start coverage and interpretability.")
else:
    print(f"\\n⚠️  Elo features show MIXED results on this test set.")
    print(f"    Consider: test set is small ({len(y_test)} fights). Elo signal may help on future data.")
    print(f"    Elo-only model shows the raw signal strength independent of other features.")

# Feature importance for the Elo features in Model B
elo_importances = {}
for feat, imp in zip(all_valid, model_b.feature_importances_):
    if feat in elo_feature_names:
        elo_importances[feat] = imp

if elo_importances:
    print(f"\\n{'='*78}")
    print(f"Elo Feature Importance in Combined Model (XGBoost gain)")
    print(f"{'='*78}")
    sorted_imp = sorted(elo_importances.items(), key=lambda x: x[1], reverse=True)
    total_imp = sum(model_b.feature_importances_)
    for feat, imp in sorted_imp:
        pct = imp / total_imp * 100
        bar = '█' * int(pct * 2)
        print(f"  {feat:30s}  {pct:5.2f}%  {bar}")
    elo_total_pct = sum(v for v in elo_importances.values()) / total_imp * 100
    print(f"\\n  Total Elo importance: {elo_total_pct:.1f}% of all feature importance")
""")

# ============================================================
# CELL 12 — Elo Standalone Deep Dive
# ============================================================
code("""
# ============================================================
# Elo as a standalone predictor — deep dive
# How does pure Elo compare to our full ML pipeline?
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Elo Standalone Analysis', fontsize=16, fontweight='bold')

# --- 1. ROC curves ---
from sklearn.metrics import roc_curve

ax = axes[0, 0]
for prob, label, color in [(prob_a, 'Without Elo', '#3498db'),
                            (prob_b, 'With Elo', '#2ecc71'),
                            (prob_c, 'Elo-Only XGB', '#e67e22'),
                            (elo_test_clean, 'Elo Raw Prob', '#e74c3c')]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_test, prob):.3f})",
            color=color, linewidth=2)

ax.plot([0,1], [0,1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves', fontweight='bold')
ax.legend(fontsize=9)

# --- 2. Where Elo flips predictions ---
ax = axes[0, 1]
pred_orig = (prob_a >= 0.5).astype(int)
pred_with = (prob_b >= 0.5).astype(int)
flipped = pred_orig != pred_with
n_flipped = flipped.sum()

if n_flipped > 0:
    flip_correct_new = accuracy_score(y_test[flipped], pred_with[flipped])
    flip_correct_old = accuracy_score(y_test[flipped], pred_orig[flipped])
    
    categories = ['Elo Flipped\\n(Correct)', 'Elo Flipped\\n(Wrong)', 
                   'Unchanged\\n(Correct)', 'Unchanged\\n(Wrong)']
    flip_correct = (pred_with[flipped] == y_test.values[flipped]).sum()
    flip_wrong = n_flipped - flip_correct
    unchanged_correct = (pred_with[~flipped] == y_test.values[~flipped]).sum()
    unchanged_wrong = (~flipped).sum() - unchanged_correct
    
    values = [flip_correct, flip_wrong, unchanged_correct, unchanged_wrong]
    colors_pie = ['#2ecc71', '#e74c3c', '#3498db', '#95a5a6']
    
    ax.bar(categories, values, color=colors_pie, edgecolor='white')
    ax.set_title(f'Prediction Changes ({n_flipped} flipped)', fontweight='bold')
    ax.set_ylabel('Count')
    
    for i, v in enumerate(values):
        ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
else:
    ax.text(0.5, 0.5, f'No predictions flipped\\n(models agree on all {len(y_test)} fights)',
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title('Prediction Changes', fontweight='bold')

# --- 3. Elo expected vs actual (calibration) ---
ax = axes[1, 0]
from sklearn.calibration import calibration_curve

for prob, label, color in [(prob_a, 'Without Elo', '#3498db'),
                            (prob_b, 'With Elo', '#2ecc71'),
                            (elo_test_clean, 'Elo Raw', '#e74c3c')]:
    try:
        fraction_pos, mean_pred = calibration_curve(y_test, prob, n_bins=8, strategy='uniform')
        ax.plot(mean_pred, fraction_pos, 'o-', label=label, color=color, linewidth=2, markersize=6)
    except Exception:
        pass

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Actual Win Rate')
ax.set_title('Calibration Curves', fontweight='bold')
ax.legend(fontsize=9)

# --- 4. Probability distribution shift ---
ax = axes[1, 1]
ax.hist(prob_a, bins=30, alpha=0.5, color='#3498db', label='Without Elo', density=True)
ax.hist(prob_b, bins=30, alpha=0.5, color='#2ecc71', label='With Elo', density=True)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Predicted P(F1 wins)')
ax.set_ylabel('Density')
ax.set_title('Probability Distribution Shift', fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig(DATA / 'elo_validation.png', bbox_inches='tight', dpi=120)
plt.show()
print("Saved: elo_validation.png")

print(f"\\nPrediction flip summary:")
print(f"  Total test fights:      {len(y_test)}")
print(f"  Predictions flipped:    {n_flipped} ({n_flipped/len(y_test)*100:.1f}%)")
if n_flipped > 0:
    print(f"  Flipped → correct:      {flip_correct}/{n_flipped} ({flip_correct/n_flipped*100:.1f}%)")
    print(f"  Flipped → wrong:        {flip_wrong}/{n_flipped} ({flip_wrong/n_flipped*100:.1f}%)")
    net = flip_correct - flip_wrong
    print(f"  Net impact:             {'+' if net >= 0 else ''}{net} fights")
""")

# ============================================================
# CELL 13 — Save Everything
# ============================================================
code("""
# ============================================================
# Save updated model_data and Elo artifacts
# ============================================================

# 1. Save updated model_data with Elo features
model_data_elo.to_csv(DATA / 'model_data.csv', index=False)
print(f"✅ Saved: model_data.csv ({model_data_elo.shape[0]:,} rows × {model_data_elo.shape[1]:,} cols)")

# 2. Save updated feature list
updated_features = original_features + elo_feature_names
with open(DATA / 'feature_list.txt', 'w') as f:
    for feat in updated_features:
        f.write(feat + '\\n')
print(f"✅ Saved: feature_list.txt ({len(updated_features)} features, was {len(original_features)})")

# 3. Save Elo ratings snapshot
snapshot = elo.get_snapshot()
snapshot.to_csv(DATA / 'elo_ratings.csv', index=False)
print(f"✅ Saved: elo_ratings.csv ({len(snapshot):,} fighters)")

# 4. Save full Elo history
elo_history.to_csv(DATA / 'elo_history.csv', index=False)
print(f"✅ Saved: elo_history.csv ({len(elo_history):,} fights)")

# 5. Save Elo parameters
elo_config = {
    'best_params': best_params,
    'fixed_params': {
        'round_scale': True,
        'decay_months': 18,
        'starting': 1500,
        'floor': 1100,
    },
    'performance': {
        'accuracy': float(best_row['accuracy']),
        'log_loss': float(best_row['log_loss']),
        'brier': float(best_row['brier']),
        'higher_rated_wr': float(best_row['higher_rated_wr']),
        'n_fights_scored': int(best_row['n_fights']),
    },
    'grid_search': {
        'n_combinations': len(combos),
        'param_grid': {k: [float(x) if isinstance(x, (int, float)) else x for x in v] 
                       for k, v in param_grid.items()},
    },
    'validation': {
        'without_elo_acc': float(m_a['accuracy']),
        'with_elo_acc': float(m_b['accuracy']),
        'elo_only_acc': float(m_c['accuracy']),
        'delta_accuracy': float(d_acc),
        'delta_auc': float(d_auc),
        'delta_log_loss': float(d_ll),
        'n_elo_features': len(elo_feature_names),
        'elo_features': elo_feature_names,
    }
}

with open(DATA / 'elo_config.json', 'w') as f:
    json.dump(elo_config, f, indent=2, default=str)
print(f"✅ Saved: elo_config.json")

# 6. Save grid search results
results_df.to_csv(DATA / 'elo_grid_search.csv', index=False)
print(f"✅ Saved: elo_grid_search.csv ({len(results_df)} combinations)")

print(f"\\n{'='*60}")
print(f"ALL FILES SAVED")
print(f"{'='*60}")
""")

# ============================================================
# CELL 14 — Summary & Next Steps
# ============================================================
md("""
# Summary

## What We Built
- **MMAElo**: A custom Elo rating system with dynamic K-factors, finish bonuses, 
  round scaling, and inactivity decay — purpose-built for MMA
- **Grid-searched** across 576+ hyperparameter combinations to minimize log loss
- **15 new features** added to model_data.csv for the ML pipeline

## Elo Feature Groups

| Group | Features | What It Captures |
|-------|----------|-----------------|
| **Rating** | f1_elo, f2_elo, diff_elo | Raw skill estimate |
| **Expected** | elo_expected | Classic Elo win probability |
| **Momentum** | f1/f2/diff_elo_momentum | Recent trajectory (last 3 deltas) |
| **Peak** | f1/f2/diff_elo_peak | Career ceiling |
| **Form** | f1/f2/diff_elo_vs_peak | Current vs peak (decline detection) |
| **Volatility** | f1/f2/diff_elo_volatility | Rating stability |
| **Experience** | f1/f2/diff_elo_fights | Elo-tracked fight count |

## Key Properties
- **No leakage**: All features use PRE-FIGHT ratings only
- **Warm-up**: Pre-2015 fights used for rating initialization, not scored
- **Opponent-adjusted**: Unlike raw win%, Elo accounts for opponent strength
- **Dynamic**: Ratings evolve — captures improvement, decline, and ring rust
- **Cold-start friendly**: New fighters start at 1500 with high K (fast adaptation)

## What Changed
- `model_data.csv`: Now has 15 additional Elo columns
- `feature_list.txt`: Updated with Elo feature names
- Downstream notebooks (05, 06, 07, 08) will automatically pick up Elo features 
  on next run since they read `feature_list.txt`

## Next Steps
1. **Re-run NB05** (modeling) — see if Elo features improve baseline models
2. **Re-run NB06** (tuning) — re-tune with expanded feature set
3. **Re-run NB07/08** (predict/bet) — production models now include Elo signal
4. **Monitor**: Track if diff_elo appears in top feature importance rankings

## Files Saved

| File | Description |
|------|-------------|
| `model_data.csv` | Updated with 15 Elo features |
| `feature_list.txt` | Updated feature list |
| `elo_ratings.csv` | Current ratings for all fighters |
| `elo_history.csv` | Full fight-by-fight Elo history |
| `elo_config.json` | Best params + validation results |
| `elo_grid_search.csv` | All grid search results |
| `elo_parameter_sensitivity.png` | Parameter sensitivity plots |
| `elo_analytics.png` | 9-panel Elo analytics dashboard |
| `elo_correlations.png` | Correlation analysis |
| `elo_validation.png` | With/without Elo comparison |
""")

# ============================================================
# Write notebook
# ============================================================
out = pathlib.Path("notebooks/04b_elo.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(f"✅  Wrote {out}  ({len(nb['cells'])} cells)")



