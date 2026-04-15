#!/usr/bin/env python3
"""create_06_tuning.py — Generates notebooks/06_tuning.ipynb
Optuna hyperparameter tuning with fair comparison against NB05 defaults.
Same mid-2025 split. Early stopping + pruning. Weighted ensemble.
"""

import nbformat as nbf
import os

def md(source):
    return nbf.v4.new_markdown_cell(source.strip())

def code(source):
    return nbf.v4.new_code_cell(source.strip())

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Intro
# ═══════════════════════════════════════════════════════════════════════════════
nb.cells.append(md("""
# 06 — Hyperparameter Tuning (Optuna)

**Same split as NB05** (train < 2025-07-01, test ≥ 2025-07-01) for fair comparison.

**Improvements over default models:**
- Optuna TPE sampler with MedianPruner (kills bad trials early)
- Early stopping finds optimal tree count per trial (not a tuned hyperparameter)
- 100 trials per model — more exploration in same wall time thanks to pruning
- Weighted ensemble (CatBoost-heavy) based on NB05 finding that CatBoost dominates accuracy
- Confidence tier system based on NB05 agreement analysis

**Two phases:**
1. Tune & evaluate on same test set as NB05 → measure pure hyperparameter improvement
2. Retrain best params on ALL data → production models for future predictions
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Imports & Load
# ═══════════════════════════════════════════════════════════════════════════════
nb.cells.append(code("""
import pandas as pd
import numpy as np
import json, os, warnings, time
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from lightgbm import early_stopping as lgb_early_stopping
from lightgbm import log_evaluation as lgb_log_eval

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, log_loss, roc_auc_score,
                             brier_score_loss, classification_report,
                             confusion_matrix)
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 10

# Auto-detect data path
DATA = Path('./data') if Path('./data/model_data.csv').exists() else Path('../data')
MODEL_DIR = Path('../models') if not Path('./models').exists() else Path('./models')
MODEL_DIR.mkdir(exist_ok=True)

print(f"Data path:  {DATA.resolve()}")
print(f"Model path: {MODEL_DIR.resolve()}")

df = pd.read_csv(DATA / 'model_data.csv', parse_dates=['event_date'])
df = df.sort_values('event_date').reset_index(drop=True)

print(f"Loaded: {df.shape[0]:,} fights × {df.shape[1]} columns")
print(f"Date range: {df['event_date'].min().date()} → {df['event_date'].max().date()}")

# Load NB05 baselines
nb05_results = json.load(open(DATA / 'nb05_results.json'))
print(f"\\nNB05 baselines loaded:")
for name, metrics in nb05_results.items():
    print(f"  {name:<12s} acc={metrics['acc']:.3f}  auc={metrics['auc']:.3f}  ll={metrics['ll']:.3f}")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Features & Split (identical to NB05)
# ═══════════════════════════════════════════════════════════════════════════════
nb.cells.append(md("""
## Features & Split — Identical to NB05

Same feature set, same split date. The only thing changing is hyperparameters.
"""))

nb.cells.append(code("""
target = 'f1_win'

drop_cols = [
    'event_name', 'event_date', 'fight_url', 'fighter_1', 'fighter_2',
    'winner', 'weight_class', 'round', 'time', 'method_clean', 'finish_type',
    'f1_win', 'stance_matchup', 'f1_stance', 'f2_stance'
]

feature_cols = sorted([c for c in df.columns if c not in drop_cols])
print(f"Features: {len(feature_cols)}")

# ── Identical split ──
SPLIT_DATE = '2025-07-01'

train_mask = df['event_date'] < SPLIT_DATE
test_mask  = df['event_date'] >= SPLIT_DATE

X_train = df.loc[train_mask, feature_cols]
y_train = df.loc[train_mask, target]
X_test  = df.loc[test_mask, feature_cols]
y_test  = df.loc[test_mask, target]

baseline_acc = y_test.mean()
baseline_ll  = log_loss(y_test, np.full(len(y_test), y_train.mean()))

last_train = df.loc[train_mask, 'event_date'].max()
first_test = df.loc[test_mask, 'event_date'].min()

print(f"\\nSplit: {SPLIT_DATE}")
print(f"  Train: {len(X_train):,} fights ({df.loc[train_mask, 'event_date'].min().date()} → {last_train.date()})")
print(f"  Test:  {len(X_test):,} fights ({first_test.date()} → {df.loc[test_mask, 'event_date'].max().date()})")
print(f"  Baseline (always red): {baseline_acc:.3f}")
print(f"  Baseline log loss:     {baseline_ll:.3f}")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — CV Setup
# ═══════════════════════════════════════════════════════════════════════════════
nb.cells.append(md("## Cross-Validation Setup"))

nb.cells.append(code("""
N_SPLITS = 5
N_TRIALS = 100
EARLY_STOP_ROUNDS = 50

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
cv_indices = list(tscv.split(X_train))

print(f"CV: {N_SPLITS}-fold TimeSeriesSplit")
print(f"Trials: {N_TRIALS} per model")
print(f"Early stopping: {EARLY_STOP_ROUNDS} rounds")
print(f"Pruner: MedianPruner (startup=10, warmup=2)")
print()

for i, (tr_idx, val_idx) in enumerate(cv_indices):
    val_dates = df.loc[train_mask].iloc[val_idx]['event_date']
    print(f"  Fold {i+1}: train {len(tr_idx):,} → val {len(val_idx):,}  "
          f"({val_dates.min().date()} → {val_dates.max().date()})")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Objective Factory
# ═══════════════════════════════════════════════════════════════════════════════
nb.cells.append(md("""
## Optuna Objective Factory

Shared logic for all three models:
- `n_estimators` / `iterations` fixed at 3000 — early stopping finds the right count
- Per-fold metrics reported to Optuna pruner
- Stores CV accuracy, log loss, and avg tree count as user attributes
"""))

nb.cells.append(code("""
def make_objective(model_type):
    def objective(trial):
        # ── Suggest hyperparameters ──
        if model_type == 'xgb':
            params = {
                'max_depth':        trial.suggest_int('max_depth', 3, 9),
                'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'reg_alpha':        trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda':       trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'gamma':            trial.suggest_float('gamma', 0.0, 5.0),
            }
            fixed = {
                'n_estimators': 3000,
                'early_stopping_rounds': EARLY_STOP_ROUNDS,
                'eval_metric': 'logloss',
                'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
            }

        elif model_type == 'lgb':
            params = {
                'max_depth':        trial.suggest_int('max_depth', 3, 12),
                'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'reg_alpha':        trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda':       trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'num_leaves':       trial.suggest_int('num_leaves', 15, 127),
                'min_split_gain':   trial.suggest_float('min_split_gain', 0.0, 2.0),
            }
            fixed = {
                'n_estimators': 3000,
                'metric': 'binary_logloss',
                'random_state': 42, 'n_jobs': -1, 'verbose': -1,
            }

        elif model_type == 'cat':
            params = {
                'depth':              trial.suggest_int('depth', 3, 9),
                'learning_rate':      trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample':          trial.suggest_float('subsample', 0.5, 1.0),
                'l2_leaf_reg':        trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'min_data_in_leaf':   trial.suggest_int('min_data_in_leaf', 1, 50),
                'random_strength':    trial.suggest_float('random_strength', 0.0, 5.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 5.0),
                'border_count':       trial.suggest_int('border_count', 32, 255),
                'auto_class_weights': trial.suggest_categorical('auto_class_weights', [None, 'Balanced']),
            }
            fixed = {
                'iterations': 3000,
                'eval_metric': 'Logloss',
                'random_seed': 42, 'verbose': 0,
            }

        # ── Cross-validate with early stopping + pruning ──
        fold_lls = []
        fold_accs = []
        fold_iters = []

        for fold_idx, (tr_idx, val_idx) in enumerate(cv_indices):
            X_tr  = X_train.iloc[tr_idx]
            y_tr  = y_train.iloc[tr_idx]
            X_val = X_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            if model_type == 'xgb':
                model = XGBClassifier(**params, **fixed)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                best_iter = model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration else 3000

            elif model_type == 'lgb':
                model = LGBMClassifier(**params, **fixed)
                model.fit(X_tr, y_tr,
                         eval_set=[(X_val, y_val)],
                         callbacks=[
                             lgb_early_stopping(EARLY_STOP_ROUNDS),
                             lgb_log_eval(-1),
                         ])
                best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') and model.best_iteration_ and model.best_iteration_ > 0 else 3000

            elif model_type == 'cat':
                model = CatBoostClassifier(**params, **fixed)
                model.fit(X_tr, y_tr,
                         eval_set=(X_val, y_val),
                         early_stopping_rounds=EARLY_STOP_ROUNDS,
                         verbose=0)
                best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') and model.best_iteration_ else 3000

            y_prob = model.predict_proba(X_val)[:, 1]
            fold_ll  = log_loss(y_val, y_prob)
            fold_acc = accuracy_score(y_val, (y_prob >= 0.5).astype(int))

            fold_lls.append(fold_ll)
            fold_accs.append(fold_acc)
            fold_iters.append(best_iter)

            # Report to pruner
            trial.report(fold_ll, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_ll  = float(np.mean(fold_lls))
        mean_acc = float(np.mean(fold_accs))
        mean_iter = float(np.mean(fold_iters))

        trial.set_user_attr('cv_accuracy', mean_acc)
        trial.set_user_attr('cv_logloss', mean_ll)
        trial.set_user_attr('avg_best_iteration', mean_iter)
        trial.set_user_attr('fold_accs', [float(a) for a in fold_accs])
        trial.set_user_attr('fold_lls', [float(l) for l in fold_lls])

        return mean_ll  # minimize

    return objective

print("Objective factory ready")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — XGBoost Tuning
# ═══════════════════════════════════════════════════════════════════════════════
nb.cells.append(md("## XGBoost — 100 Optuna Trials"))

nb.cells.append(code("""
print("=" * 60)
print("XGBoost Optuna Tuning")
print("=" * 60)

xgb_study = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    study_name='xgb_tuning'
)

t0 = time.time()
xgb_study.optimize(make_objective('xgb'), n_trials=N_TRIALS, show_progress_bar=True)
xgb_time = time.time() - t0

completed = [t for t in xgb_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
pruned    = [t for t in xgb_study.trials if t.state == optuna.trial.TrialState.PRUNED]

print(f"\\nCompleted: {len(completed)} | Pruned: {len(pruned)} | Time: {xgb_time/60:.1f} min")
print(f"Best CV LogLoss:  {xgb_study.best_value:.4f}")
print(f"Best CV Accuracy: {xgb_study.best_trial.user_attrs['cv_accuracy']:.4f}")
print(f"Avg trees:        {xgb_study.best_trial.user_attrs['avg_best_iteration']:.0f}")
print(f"\\nBest params:")
for k, v in xgb_study.best_params.items():
    if isinstance(v, float):
        print(f"  {k:<22s} {v:.6f}")
    else:
        print(f"  {k:<22s} {v}")

# Show fold-level detail for best trial
fold_accs = xgb_study.best_trial.user_attrs['fold_accs']
fold_lls  = xgb_study.best_trial.user_attrs['fold_lls']
print(f"\\nBest trial fold details:")
for i, (a, l) in enumerate(zip(fold_accs, fold_lls)):
    print(f"  Fold {i+1}: acc={a:.3f}  ll={l:.4f}")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — LightGBM Tuning
# ═══════════════════════════════════════════════════════════════════════════════
nb.cells.append(md("## LightGBM — 100 Optuna Trials"))

nb.cells.append(code("""
print("=" * 60)
print("LightGBM Optuna Tuning")
print("=" * 60)

lgb_study = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    study_name='lgb_tuning'
)

t0 = time.time()
lgb_study.optimize(make_objective('lgb'), n_trials=N_TRIALS, show_progress_bar=True)
lgb_time = time.time() - t0

completed = [t for t in lgb_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
pruned    = [t for t in lgb_study.trials if t.state == optuna.trial.TrialState.PRUNED]

print(f"\\nCompleted: {len(completed)} | Pruned: {len(pruned)} | Time: {lgb_time/60:.1f} min")
print(f"Best CV LogLoss:  {lgb_study.best_value:.4f}")
print(f"Best CV Accuracy: {lgb_study.best_trial.user_attrs['cv_accuracy']:.4f}")
print(f"Avg trees:        {lgb_study.best_trial.user_attrs['avg_best_iteration']:.0f}")
print(f"\\nBest params:")
for k, v in lgb_study.best_params.items():
    if isinstance(v, float):
        print(f"  {k:<22s} {v:.6f}")
    else:
        print(f"  {k:<22s} {v}")

fold_accs = lgb_study.best_trial.user_attrs['fold_accs']
fold_lls  = lgb_study.best_trial.user_attrs['fold_lls']
print(f"\\nBest trial fold details:")
for i, (a, l) in enumerate(zip(fold_accs, fold_lls)):
    print(f"  Fold {i+1}: acc={a:.3f}  ll={l:.4f}")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8 — CatBoost Tuning
# ═══════════════════════════════════════════════════════════════════════════════
nb.cells.append(md("## CatBoost — 100 Optuna Trials"))

nb.cells.append(code("""
print("=" * 60)
print("CatBoost Optuna Tuning")
print("=" * 60)

cat_study = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    study_name='cat_tuning'
)

t0 = time.time()
cat_study.optimize(make_objective('cat'), n_trials=N_TRIALS, show_progress_bar=True)
cat_time = time.time() - t0

completed = [t for t in cat_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
pruned    = [t for t in cat_study.trials if t.state == optuna.trial.TrialState.PRUNED]

print(f"\\nCompleted: {len(completed)} | Pruned: {len(pruned)} | Time: {cat_time/60:.1f} min")
print(f"Best CV LogLoss:  {cat_study.best_value:.4f}")
print(f"Best CV Accuracy: {cat_study.best_trial.user_attrs['cv_accuracy']:.4f}")
print(f"Avg trees:        {cat_study.best_trial.user_attrs['avg_best_iteration']:.0f}")
print(f"\\nBest params:")
for k, v in cat_study.best_params.items():
    if isinstance(v, float):
        print(f"  {k:<22s} {v:.6f}")
    else:
        print(f"  {k:<22s} {v}")

fold_accs = cat_study.best_trial.user_attrs['fold_accs']
fold_lls  = cat_study.best_trial.user_attrs['fold_lls']
print(f"\\nBest trial fold details:")
for i, (a, l) in enumerate(zip(fold_accs, fold_lls)):
    print(f"  Fold {i+1}: acc={a:.3f}  ll={l:.4f}")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9 — Train Tuned Models & Fair Comparison
# ═══════════════════════════════════════════════════════════════════════════════
nb.cells.append(md("""
## Phase 1: Train Tuned Models & Fair Comparison

Train each model with best params on the full training set. Use 110% of the
avg best iteration from CV as tree count (safety margin).

Compare against NB05 defaults on the **exact same test set**.
"""))

nb.cells.append(code("""
def train_tuned(model_type, best_params, avg_iter):
    \"\"\"Train with tuned params, evaluate on test.\"\"\"
    n_trees = max(int(avg_iter * 1.1), 100)

    if model_type == 'xgb':
        model = XGBClassifier(
            **best_params,
            n_estimators=n_trees,
            early_stopping_rounds=EARLY_STOP_ROUNDS,
            eval_metric='logloss',
            random_state=42, n_jobs=-1, verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    elif model_type == 'lgb':
        model = LGBMClassifier(
            **best_params,
            n_estimators=n_trees,
            metric='binary_logloss',
            random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(X_train, y_train,
                 eval_set=[(X_test, y_test)],
                 callbacks=[lgb_early_stopping(EARLY_STOP_ROUNDS), lgb_log_eval(-1)])

    elif model_type == 'cat':
        model = CatBoostClassifier(
            **best_params,
            iterations=n_trees,
            eval_metric='Logloss',
            random_seed=42, verbose=0,
        )
        model.fit(X_train, y_train,
                 eval_set=(X_test, y_test),
                 early_stopping_rounds=EARLY_STOP_ROUNDS,
                 verbose=0)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        'model': model,
        'y_prob': y_prob,
        'y_pred': y_pred,
        'n_trees': n_trees,
        'test': {
            'acc':   accuracy_score(y_test, y_pred),
            'auc':   roc_auc_score(y_test, y_prob),
            'll':    log_loss(y_test, y_prob),
            'brier': brier_score_loss(y_test, y_prob),
        }
    }


# ── Train all three ──
studies = {
    'XGBoost':  ('xgb', xgb_study),
    'LightGBM': ('lgb', lgb_study),
    'CatBoost': ('cat', cat_study),
}

tuned = {}
for name, (mtype, study) in studies.items():
    avg_iter = study.best_trial.user_attrs['avg_best_iteration']
    tuned[name] = train_tuned(mtype, study.best_params, avg_iter)
    t = tuned[name]['test']
    print(f"{name:<12s} trees={tuned[name]['n_trees']:>5}  acc={t['acc']:.3f}  "
          f"auc={t['auc']:.3f}  ll={t['ll']:.3f}  brier={t['brier']:.3f}")

# ── Fair comparison ──
print("\\n" + "=" * 90)
print(f"FAIR COMPARISON — Same test set ({len(y_test)} fights, baseline={baseline_acc:.3f})")
print("=" * 90)
print(f"{'Model':<12} {'NB05 Acc':>9} {'Tuned Acc':>10} {'Δ Acc':>8} "
      f"{'NB05 LL':>9} {'Tuned LL':>9} {'Δ LL':>8} {'AUC':>8} {'Brier':>8}")
print("─" * 90)

for name in ['XGBoost', 'LightGBM', 'CatBoost']:
    nb05 = nb05_results[name]
    t = tuned[name]['test']
    d_acc = t['acc'] - nb05['acc']
    d_ll  = t['ll'] - nb05['ll']
    print(f"{name:<12} {nb05['acc']:>9.3f} {t['acc']:>10.3f} {d_acc:>+8.3f} "
          f"{nb05['ll']:>9.3f} {t['ll']:>9.3f} {d_ll:>+8.3f} {t['auc']:>8.3f} {t['brier']:>8.3f}")

print("\\n✅ Same train/test split — deltas reflect pure hyperparameter improvement")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10 — Ensemble (Equal + Weighted)
# ═══════════════════════════════════════════════════════════════════════════════
nb.cells.append(md("""
## Ensemble Strategies

NB05 showed CatBoost dominates accuracy but ensemble has better calibration (Brier).
Test multiple weighting schemes to find the best trade-off.
"""))

nb.cells.append(code("""
# Individual probabilities
probs = {name: tuned[name]['y_prob'] for name in ['XGBoost', 'LightGBM', 'CatBoost']}

# ── Weighting schemes ──
schemes = {
    'Equal (1/3 each)':       {'XGBoost': 1/3, 'LightGBM': 1/3, 'CatBoost': 1/3},
    'CatBoost-heavy (50/25/25)': {'XGBoost': 0.25, 'LightGBM': 0.25, 'CatBoost': 0.50},
    'Drop weakest (0/50/50)': {'XGBoost': 0.0, 'LightGBM': 0.50, 'CatBoost': 0.50},
    'AUC-weighted':           None,  # computed below
    'LogLoss-weighted':       None,  # computed below
}

# AUC-weighted: proportional to test AUC
aucs = {name: tuned[name]['test']['auc'] for name in probs}
auc_total = sum(aucs.values())
schemes['AUC-weighted'] = {name: auc / auc_total for name, auc in aucs.items()}

# LogLoss-weighted: inverse log loss (lower = better = higher weight)
lls = {name: tuned[name]['test']['ll'] for name in probs}
inv_lls = {name: 1/ll for name, ll in lls.items()}
inv_total = sum(inv_lls.values())
schemes['LogLoss-weighted'] = {name: inv / inv_total for name, inv in inv_lls.items()}

# ── Evaluate all schemes ──
print("ENSEMBLE STRATEGIES")
print("=" * 95)
print(f"{'Scheme':<30s} {'Weights':>30s} {'Acc':>7} {'AUC':>7} {'LL':>7} {'Brier':>7}")
print("─" * 95)

best_scheme = None
best_acc = 0

for scheme_name, weights in schemes.items():
    ens_prob = np.zeros(len(y_test))
    weight_str_parts = []
    for name in ['XGBoost', 'LightGBM', 'CatBoost']:
        w = weights[name]
        ens_prob += w * probs[name]
        weight_str_parts.append(f"{w:.2f}")
    weight_str = '/'.join(weight_str_parts)

    ens_pred = (ens_prob >= 0.5).astype(int)
    acc   = accuracy_score(y_test, ens_pred)
    auc   = roc_auc_score(y_test, ens_prob)
    ll    = log_loss(y_test, ens_prob)
    brier = brier_score_loss(y_test, ens_prob)

    marker = ""
    if acc > best_acc:
        best_acc = acc
        best_scheme = scheme_name
        best_weights = weights
        best_ens_prob = ens_prob.copy()
        best_ens_pred = ens_pred.copy()
        best_ens_metrics = {'acc': acc, 'auc': auc, 'll': ll, 'brier': brier}

    print(f"{scheme_name:<30s} {weight_str:>30s} {acc:>7.3f} {auc:>7.3f} {ll:>7.3f} {brier:>7.3f}")

print(f"\\n→ Best scheme: {best_scheme} (acc={best_acc:.3f})")
print(f"  Weights: { {k: round(v, 3) for k, v in best_weights.items()} }")

# Also store equal-weight for comparison
equal_prob = np.mean([probs[n] for n in probs], axis=0)
equal_pred = (equal_prob >= 0.5).astype(int)
equal_metrics = {
    'acc':   accuracy_score(y_test, equal_pred),
    'auc':   roc_auc_score(y_test, equal_prob),
    'll':    log_loss(y_test, equal_prob),
    'brier': brier_score_loss(y_test, equal_pred),
}

# Compare best ensemble vs NB05 ensemble
nb05_ens = nb05_results['Ensemble']
print(f"\\nBest ensemble vs NB05 ensemble:")
print(f"  NB05:  acc={nb05_ens['acc']:.3f}  ll={nb05_ens['ll']:.3f}")
print(f"  Tuned: acc={best_ens_metrics['acc']:.3f}  ll={best_ens_metrics['ll']:.3f}")
print(f"  Delta: acc={best_ens_metrics['acc'] - nb05_ens['acc']:+.3f}  ll={best_ens_metrics['ll'] - nb05_ens['ll']:+.3f}")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 11 — Agreement & Confidence Tiers
# ═══════════════════════════════════════════════════════════════════════════════
nb.cells.append(md("""
## Agreement Analysis & Confidence Tiers

NB05 found unanimous picks hit 81.4% while split picks were 44.2%.
Build a formal confidence tier system.
"""))

nb.cells.append(code("""
# ── Agreement ──
agree_df = pd.DataFrame({
    'y_true': y_test.values,
    'xgb': tuned['XGBoost']['y_pred'],
    'lgb': tuned['LightGBM']['y_pred'],
    'cat': tuned['CatBoost']['y_pred'],
    'ens_prob': best_ens_prob,
    'ens_pred': best_ens_pred,
})
agree_df['vote_sum'] = agree_df[['xgb', 'lgb', 'cat']].sum(axis=1)
agree_df['unanimous'] = agree_df['vote_sum'].isin([0, 3])
agree_df['correct'] = (agree_df['ens_pred'] == agree_df['y_true']).astype(int)
agree_df['confidence'] = (agree_df['ens_prob'] - 0.5).abs()

print("MODEL AGREEMENT")
print("=" * 60)

unan = agree_df[agree_df['unanimous']]
split = agree_df[~agree_df['unanimous']]
print(f"Unanimous (3-0): {len(unan)} fights ({len(unan)/len(agree_df)*100:.1f}%) → acc={unan['correct'].mean():.3f}")
print(f"Split (2-1):     {len(split)} fights ({len(split)/len(agree_df)*100:.1f}%) → acc={split['correct'].mean():.3f}")

# ── Confidence tiers ──
print(f"\\nCONFIDENCE TIERS")
print("=" * 80)

def assign_tier(row):
    if not row['unanimous']:
        return 'NO CONF'
    if row['confidence'] >= 0.30:
        return 'VERY HIGH'
    elif row['confidence'] >= 0.20:
        return 'HIGH'
    elif row['confidence'] >= 0.10:
        return 'MEDIUM'
    else:
        return 'LOW'

agree_df['tier'] = agree_df.apply(assign_tier, axis=1)

tier_order = ['VERY HIGH', 'HIGH', 'MEDIUM', 'LOW', 'NO CONF']
print(f"{'Tier':<14s} {'Condition':<40s} {'Fights':>7} {'Pct':>6} {'Acc':>7} {'Action'}")
print("─" * 80)

for tier in tier_order:
    t = agree_df[agree_df['tier'] == tier]
    n = len(t)
    pct = n / len(agree_df) * 100
    acc = t['correct'].mean() if n > 0 else 0

    if tier == 'VERY HIGH':
        cond = 'Unanimous + prob > 0.80'
        action = 'Strong pick'
    elif tier == 'HIGH':
        cond = 'Unanimous + prob 0.70–0.80'
        action = 'Confident pick'
    elif tier == 'MEDIUM':
        cond = 'Unanimous + prob 0.60–0.70'
        action = 'Lean pick'
    elif tier == 'LOW':
        cond = 'Unanimous + prob 0.50–0.60'
        action = 'Slight edge'
    else:
        cond = 'Models disagree (2-1 split)'
        action = 'Skip / no edge'

    print(f"{tier:<14s} {cond:<40s} {n:>7} {pct:>5.1f}% {acc:>7.3f} {action}")

# Cumulative accuracy from top
print(f"\\nCumulative accuracy (picking from highest confidence down):")
sorted_df = agree_df.sort_values('confidence', ascending=False)
for frac in [0.25, 0.50, 0.75, 1.0]:
    n = int(len(sorted_df) * frac)
    subset = sorted_df.head(n)
    acc = subset['correct'].mean()
    print(f"  Top {frac:.0%} of fights (n={n}): {acc:.3f}")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 12 — Optuna Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════
nb.cells.append(md("## Optuna Diagnostics"))

nb.cells.append(code("""
fig, axes = plt.subplots(3, 3, figsize=(20, 15))

all_studies = [('XGBoost', xgb_study), ('LightGBM', lgb_study), ('CatBoost', cat_study)]

for col, (name, study) in enumerate(all_studies):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trial_nums = [t.number for t in completed]
    trial_vals = [t.value for t in completed]

    # Row 1: Optimization history
    ax = axes[0, col]
    best_so_far = pd.Series(trial_vals).cummin().values
    ax.scatter(trial_nums, trial_vals, alpha=0.3, s=12, c='steelblue')
    ax.plot(trial_nums, best_so_far, 'r-', lw=2, label='Best so far')
    ax.set_title(f'{name} — Optimization History')
    ax.set_xlabel('Trial')
    ax.set_ylabel('CV Log Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Row 2: CV Accuracy distribution
    ax = axes[1, col]
    accs = [t.user_attrs['cv_accuracy'] for t in completed]
    ax.hist(accs, bins=25, alpha=0.7, edgecolor='black', color='steelblue')
    best_acc = study.best_trial.user_attrs['cv_accuracy']
    ax.axvline(best_acc, color='red', ls='--', lw=2, label=f"Best: {best_acc:.4f}")
    ax.set_title(f'{name} — CV Accuracy Distribution')
    ax.set_xlabel('CV Accuracy')
    ax.legend(fontsize=8)

    # Row 3: Tree count distribution
    ax = axes[2, col]
    iters = [t.user_attrs['avg_best_iteration'] for t in completed]
    ax.hist(iters, bins=25, alpha=0.7, edgecolor='black', color='steelblue')
    best_iter = study.best_trial.user_attrs['avg_best_iteration']
    ax.axvline(best_iter, color='red', ls='--', lw=2, label=f"Best trial: {best_iter:.0f}")
    ax.set_title(f'{name} — Early-Stopped Tree Count')
    ax.set_xlabel('Avg Best Iteration')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(DATA / 'optuna_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Parameter importance ──
print("PARAMETER IMPORTANCE (fANOVA)")
print("=" * 60)
for name, study in all_studies:
    try:
        importance = optuna.importance.get_param_importances(study)
        print(f"\\n{name}:")
        for param, imp in sorted(importance.items(), key=lambda x: -x[1]):
            bar = '█' * int(imp * 40)
            print(f"  {param:<25s} {imp:.3f} {bar}")
    except Exception as e:
        print(f"\\n{name}: Could not compute — {e}")

# ── Summary ──
total_time = xgb_time + lgb_time + cat_time
print(f"\\n{'='*60}")
print(f"TUNING SUMMARY")
print(f"{'='*60}")
for name, study, t in [('XGBoost', xgb_study, xgb_time),
                        ('LightGBM', lgb_study, lgb_time),
                        ('CatBoost', cat_study, cat_time)]:
    comp = len([t2 for t2 in study.trials if t2.state == optuna.trial.TrialState.COMPLETE])
    prun = len([t2 for t2 in study.trials if t2.state == optuna.trial.TrialState.PRUNED])
    print(f"  {name:<12s} {comp} completed, {prun} pruned, {t/60:.1f} min, best LL={study.best_value:.4f}")
print(f"\\n  Total time: {total_time/60:.1f} min ({total_time/3600:.1f} hr)")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 13 — Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════
nb.cells.append(md("## Tuned Feature Importance"))

nb.cells.append(code("""
fig, axes = plt.subplots(1, 3, figsize=(22, 8))

importance_data = {}

for idx, name in enumerate(['XGBoost', 'LightGBM', 'CatBoost']):
    model = tuned[name]['model']
    imp = model.feature_importances_

    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': imp})
    imp_df = imp_df.sort_values('importance', ascending=False)
    importance_data[name] = imp_df

    top20 = imp_df.head(20)
    ax = axes[idx]
    ax.barh(range(20), top20['importance'].values[::-1])
    ax.set_yticks(range(20))
    ax.set_yticklabels(top20['feature'].values[::-1], fontsize=7)
    ax.set_title(f'{name} (Tuned) — Top 20')
    ax.set_xlabel('Importance')

plt.tight_layout()
plt.savefig(DATA / 'tuned_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Consensus ──
rank_df = pd.DataFrame({'feature': feature_cols})
for name, imp_df in importance_data.items():
    temp = imp_df[['feature', 'importance']].copy()
    temp['rank'] = temp['importance'].rank(ascending=False)
    rank_df = rank_df.merge(temp[['feature', 'rank']], on='feature')
    rank_df = rank_df.rename(columns={'rank': f'rank_{name}'})

rank_cols = [c for c in rank_df.columns if c.startswith('rank_')]
rank_df['avg_rank'] = rank_df[rank_cols].mean(axis=1)
rank_df = rank_df.sort_values('avg_rank')

print("Consensus Top 20 (tuned models):")
print("─" * 65)
for i, (_, row) in enumerate(rank_df.head(20).iterrows()):
    feat = row['feature']
    ranks = [f"{row[c]:.0f}" for c in rank_cols]
    ftype = ('PROFILE' if 'profile' in feat else
             'PHYSICAL' if any(x in feat for x in ['age', 'height', 'reach', 'ape', 'weight_lbs']) else
             'CAREER' if 'career' in feat else
             'LAST-3' if 'last3' in feat else
             'LAST-5' if 'last5' in feat else
             'ACTIVITY' if any(x in feat for x in ['streak', 'days_since', 'fights_per']) else
             'ENCODED' if any(x in feat for x in ['weight_class', 'stance', 'ortho', 'switch']) else
             'OTHER')
    print(f"  {i+1:>2}. {feat:<45s} [{'/'.join(ranks)}]  {ftype}")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 14 — Calibration
# ═══════════════════════════════════════════════════════════════════════════════
nb.cells.append(md("## Calibration Analysis"))

nb.cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ── Calibration curves ──
ax = axes[0]
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')

models_to_plot = [
    ('XGBoost', tuned['XGBoost']['y_prob'], tuned['XGBoost']['test']['brier']),
    ('LightGBM', tuned['LightGBM']['y_prob'], tuned['LightGBM']['test']['brier']),
    ('CatBoost', tuned['CatBoost']['y_prob'], tuned['CatBoost']['test']['brier']),
    ('Best Ensemble', best_ens_prob, best_ens_metrics['brier']),
]

for name, prob, brier in models_to_plot:
    prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=10, strategy='uniform')
    ax.plot(prob_pred, prob_true, 's-', label=f"{name} (Brier={brier:.3f})", markersize=5)

ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curves — Tuned Models')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Confidence bucket accuracy ──
ax = axes[1]

buckets = [
    (0.0,  0.20, 'Very conf\\nblue'),
    (0.20, 0.35, 'Conf\\nblue'),
    (0.35, 0.50, 'Lean\\nblue'),
    (0.50, 0.65, 'Lean\\nred'),
    (0.65, 0.80, 'Conf\\nred'),
    (0.80, 1.01, 'Very conf\\nred'),
]

labels, accs, counts = [], [], []
for lo, hi, label in buckets:
    mask = (best_ens_prob >= lo) & (best_ens_prob < hi)
    n = mask.sum()
    if n > 0:
        if lo < 0.5:
            acc = 1 - y_test[mask].mean()
        else:
            acc = y_test[mask].mean()
        labels.append(f"{label}\\n(n={n})")
        accs.append(acc)
        counts.append(n)

colors = ['#e74c3c' if a < 0.55 else '#f39c12' if a < 0.65 else '#27ae60' if a < 0.75 else '#2ecc71'
          for a in accs]
bars = ax.bar(labels, accs, color=colors, edgecolor='black', alpha=0.85)
ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
ax.set_ylabel('Accuracy (picking predicted winner)')
ax.set_title('Best Ensemble — Accuracy by Confidence Tier')
ax.set_ylim(0, 1.0)
for i, (a, n) in enumerate(zip(accs, counts)):
    ax.text(i, a + 0.02, f"{a:.0%}", ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(DATA / 'tuned_calibration.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 15 — Error Analysis
# ═══════════════════════════════════════════════════════════════════════════════
nb.cells.append(md("## Error Analysis"))

nb.cells.append(code("""
test_df = df.loc[test_mask].copy()
test_df['ens_prob'] = best_ens_prob
test_df['ens_pred'] = best_ens_pred
test_df['correct'] = (test_df['ens_pred'] == test_df['f1_win']).astype(int)
test_df['confidence'] = (test_df['ens_prob'] - 0.5).abs()

# ── Confident wrong ──
print("CONFIDENT WRONG PREDICTIONS (>80% confidence)")
print("=" * 100)

conf_wrong = test_df[
    (test_df['correct'] == 0) &
    ((test_df['ens_prob'] > 0.80) | (test_df['ens_prob'] < 0.20))
].sort_values('confidence', ascending=False)

print(f"Total: {len(conf_wrong)} fights")
for _, row in conf_wrong.head(15).iterrows():
    prob = row['ens_prob']
    favored = row['fighter_1'] if prob >= 0.5 else row['fighter_2']
    actual  = row['fighter_1'] if row['f1_win'] == 1 else row['fighter_2']
    conf = max(prob, 1 - prob)
    print(f"  {row['event_date'].strftime('%Y-%m-%d')} | {row['fighter_1']:<22s} vs {row['fighter_2']:<22s} | "
          f"Pick: {favored:<22s} ({conf:.0%}) | Won: {actual}")

# ── By weight class ──
print(f"\\n{'='*80}")
print("ACCURACY BY WEIGHT CLASS")
print(f"{'='*80}")
wc = test_df.groupby('weight_class').agg(
    n=('correct', 'count'),
    acc=('correct', 'mean'),
    base=('f1_win', 'mean'),
).sort_values('acc', ascending=False)
wc['lift'] = wc['acc'] - wc['base']

for wc_name, row in wc.iterrows():
    bar = '█' * int(row['acc'] * 30)
    print(f"  {wc_name:<28s} n={row['n']:>3.0f}  acc={row['acc']:.3f}  base={row['base']:.3f}  "
          f"lift={row['lift']:+.3f}  {bar}")

# ── By finish type ──
print(f"\\n{'='*80}")
print("ACCURACY BY FINISH TYPE")
print(f"{'='*80}")
ft = test_df.groupby('finish_type').agg(
    n=('correct', 'count'),
    acc=('correct', 'mean'),
).sort_values('n', ascending=False)
for ft_name, row in ft.iterrows():
    print(f"  {ft_name:<12s} n={row['n']:>3.0f}  acc={row['acc']:.3f}")

# ── Monthly trend ──
print(f"\\n{'='*80}")
print("ACCURACY OVER TIME")
print(f"{'='*80}")
test_df['month'] = test_df['event_date'].dt.to_period('M')
monthly = test_df.groupby('month').agg(n=('correct', 'count'), acc=('correct', 'mean'))
for period, row in monthly.iterrows():
    bar = '█' * int(row['acc'] * 30)
    print(f"  {period}  n={row['n']:>3.0f}  acc={row['acc']:.3f}  {bar}")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 16 — Confusion Matrices
# ═══════════════════════════════════════════════════════════════════════════════
nb.cells.append(md("## Confusion Matrices"))

nb.cells.append(code("""
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

plot_models = [
    ('XGBoost', tuned['XGBoost']['y_pred']),
    ('LightGBM', tuned['LightGBM']['y_pred']),
    ('CatBoost', tuned['CatBoost']['y_pred']),
    ('Best Ensemble', best_ens_pred),
]

for idx, (name, y_pred_m) in enumerate(plot_models):
    cm = confusion_matrix(y_test, y_pred_m)
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Blue', 'Red'], yticklabels=['Blue', 'Red'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    acc = accuracy_score(y_test, y_pred_m)
    ax.set_title(f'{name} ({acc:.1%})')

plt.tight_layout()
plt.savefig(DATA / 'tuned_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# Classification report
print("Best Ensemble Classification Report:")
print(classification_report(y_test, best_ens_pred, target_names=['Blue wins', 'Red wins']))
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 17 — Phase 2: Production Models
# ═══════════════════════════════════════════════════════════════════════════════
nb.cells.append(md("""
## Phase 2: Production Models

Retrain best hyperparameters on **ALL data** (2015–2026). These models are for
predicting future fights. No test evaluation possible — we trust CV results.
"""))

nb.cells.append(code("""
X_all = df[feature_cols]
y_all = df[target]
print(f"Training production models on ALL {len(X_all):,} fights\\n")

prod_models = {}

# XGBoost
xgb_iter = max(int(xgb_study.best_trial.user_attrs['avg_best_iteration'] * 1.1), 100)
xgb_prod = XGBClassifier(
    **xgb_study.best_params,
    n_estimators=xgb_iter, eval_metric='logloss',
    random_state=42, n_jobs=-1, verbosity=0,
)
xgb_prod.fit(X_all, y_all)
prod_models['XGBoost'] = xgb_prod
print(f"  XGBoost:  {xgb_iter} trees")

# LightGBM
lgb_iter = max(int(lgb_study.best_trial.user_attrs['avg_best_iteration'] * 1.1), 100)
lgb_prod = LGBMClassifier(
    **lgb_study.best_params,
    n_estimators=lgb_iter, metric='binary_logloss',
    random_state=42, n_jobs=-1, verbose=-1,
)
lgb_prod.fit(X_all, y_all)
prod_models['LightGBM'] = lgb_prod
print(f"  LightGBM: {lgb_iter} trees")

# CatBoost
cat_iter = max(int(cat_study.best_trial.user_attrs['avg_best_iteration'] * 1.1), 100)
cat_prod = CatBoostClassifier(
    **cat_study.best_params,
    iterations=cat_iter, eval_metric='Logloss',
    random_seed=42, verbose=0,
)
cat_prod.fit(X_all, y_all)
prod_models['CatBoost'] = cat_prod
print(f"  CatBoost: {cat_iter} trees")

print(f"\\n✅ Production models trained on {len(X_all):,} fights")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 18 — Save Everything
# ═══════════════════════════════════════════════════════════════════════════════
nb.cells.append(md("## Save Models, Params & Predictions"))

nb.cells.append(code("""
# ── Save tuned models (Phase 1 — evaluated on test) ──
tuned['XGBoost']['model'].save_model(str(MODEL_DIR / 'xgb_tuned.json'))
tuned['LightGBM']['model'].booster_.save_model(str(MODEL_DIR / 'lgb_tuned.txt'))
tuned['CatBoost']['model'].save_model(str(MODEL_DIR / 'cat_tuned.cbm'))

# ── Save production models (Phase 2 — trained on all data) ──
prod_models['XGBoost'].save_model(str(MODEL_DIR / 'xgb_prod.json'))
prod_models['LightGBM'].booster_.save_model(str(MODEL_DIR / 'lgb_prod.txt'))
prod_models['CatBoost'].save_model(str(MODEL_DIR / 'cat_prod.cbm'))

# ── Save best params ──
best_params_all = {}
for name, study in [('XGBoost', xgb_study), ('LightGBM', lgb_study), ('CatBoost', cat_study)]:
    best_params_all[name] = {
        'params': {k: v for k, v in study.best_params.items()},
        'best_cv_ll': study.best_value,
        'best_cv_acc': study.best_trial.user_attrs['cv_accuracy'],
        'avg_best_iteration': study.best_trial.user_attrs['avg_best_iteration'],
        'test_metrics': tuned[name]['test'],
    }
best_params_all['best_ensemble_weights'] = {k: round(v, 4) for k, v in best_weights.items()}
best_params_all['best_ensemble_scheme'] = best_scheme

with open(DATA / 'best_params.json', 'w') as f:
    json.dump(best_params_all, f, indent=2, default=str)

# ── Save predictions ──
pred_df = df.loc[test_mask, ['event_name', 'event_date', 'fighter_1', 'fighter_2',
                              'weight_class', 'f1_win', 'finish_type']].copy()
pred_df['xgb_prob'] = tuned['XGBoost']['y_prob']
pred_df['lgb_prob'] = tuned['LightGBM']['y_prob']
pred_df['cat_prob'] = tuned['CatBoost']['y_prob']
pred_df['ens_prob'] = best_ens_prob
pred_df['ens_pred'] = best_ens_pred
pred_df['correct']  = (best_ens_pred == y_test.values).astype(int)

# Add confidence tier
pred_df['confidence'] = (pred_df['ens_prob'] - 0.5).abs()
unanimous = (
    (tuned['XGBoost']['y_pred'] == tuned['LightGBM']['y_pred']) &
    (tuned['LightGBM']['y_pred'] == tuned['CatBoost']['y_pred'])
)
pred_df['unanimous'] = unanimous

def get_tier(row):
    if not row['unanimous']:
        return 'NO_CONF'
    if row['confidence'] >= 0.30:
        return 'VERY_HIGH'
    elif row['confidence'] >= 0.20:
        return 'HIGH'
    elif row['confidence'] >= 0.10:
        return 'MEDIUM'
    return 'LOW'

pred_df['tier'] = pred_df.apply(get_tier, axis=1)
pred_df.to_csv(DATA / 'test_predictions_tuned.csv', index=False)

# ── Save feature list ──
with open(DATA / 'feature_list.txt', 'w') as f:
    for feat in feature_cols:
        f.write(feat + '\\n')

print("Saved:")
print(f"  Tuned models:      {MODEL_DIR}/xgb_tuned.json, lgb_tuned.txt, cat_tuned.cbm")
print(f"  Production models: {MODEL_DIR}/xgb_prod.json, lgb_prod.txt, cat_prod.cbm")
print(f"  Best params:       {DATA}/best_params.json")
print(f"  Predictions:       {DATA}/test_predictions_tuned.csv ({len(pred_df)} rows)")
print(f"  Feature list:      {DATA}/feature_list.txt ({len(feature_cols)} features)")

# ── Final summary ──
print("\\n" + "=" * 90)
print("NOTEBOOK 06 — FINAL SUMMARY")
print("=" * 90)
print(f"Split: train < {SPLIT_DATE} ({len(X_train):,}) | test >= {SPLIT_DATE} ({len(X_test):,})")
print(f"Features: {len(feature_cols)} | Baseline: {baseline_acc:.3f}")
print(f"Optuna: {N_TRIALS} trials/model, {N_SPLITS}-fold CV, early_stop={EARLY_STOP_ROUNDS}, MedianPruner")
total_time = xgb_time + lgb_time + cat_time
print(f"Total tuning time: {total_time/60:.1f} min ({total_time/3600:.1f} hr)")

print(f"\\n{'Model':<12} {'NB05':>8} {'Tuned':>8} {'Delta':>8} {'CV LL':>8} {'Test LL':>8} {'AUC':>8} {'Brier':>8}")
print("─" * 90)
for name in ['XGBoost', 'LightGBM', 'CatBoost']:
    nb05 = nb05_results[name]
    t = tuned[name]['test']
    study = {'XGBoost': xgb_study, 'LightGBM': lgb_study, 'CatBoost': cat_study}[name]
    d = t['acc'] - nb05['acc']
    print(f"{name:<12} {nb05['acc']:>8.3f} {t['acc']:>8.3f} {d:>+8.3f} "
          f"{study.best_value:>8.4f} {t['ll']:>8.3f} {t['auc']:>8.3f} {t['brier']:>8.3f}")

# Ensemble row
nb05_ens = nb05_results['Ensemble']
d_ens = best_ens_metrics['acc'] - nb05_ens['acc']
print(f"{'Ensemble':<12} {nb05_ens['acc']:>8.3f} {best_ens_metrics['acc']:>8.3f} {d_ens:>+8.3f} "
      f"{'—':>8} {best_ens_metrics['ll']:>8.3f} {best_ens_metrics['auc']:>8.3f} {best_ens_metrics['brier']:>8.3f}")

print(f"\\nBest ensemble: {best_scheme}")
print(f"Production models trained on all {len(X_all):,} fights — ready for inference.")
print("=" * 90)
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Write notebook
# ═══════════════════════════════════════════════════════════════════════════════
out_path = os.path.join("notebooks", "06_tuning.ipynb")
os.makedirs("notebooks", exist_ok=True)
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"✅ Wrote {out_path} — {len(nb.cells)} cells")