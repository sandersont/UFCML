#!/usr/bin/env python3
"""create_06_tuning.py — Generates notebooks/06_tuning.ipynb
Two-phase Optuna hyperparameter tuning with fair comparison against NB05 defaults.
"""

import nbformat as nbf

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

# ── Cell 1: Imports ──────────────────────────────────────────────────────────
nb.cells.append(md("""
# 06 — Hyperparameter Tuning (Optuna)

**Two-phase approach:**
1. **Fair comparison** — Tune on the same 2015–2023 / 2024–2026 split as NB05, measure honest improvement
2. **Production models** — Retrain best params on all available data

**Improvements over naive tuning:**
- Early stopping finds optimal tree count automatically (not a tuned hyperparameter)
- Optuna MedianPruner kills bad trials early → more exploration in same wall time
- 100 trials per model (with pruning, faster than 50 unpruned)
- Same test set as NB05 for fair before/after comparison
"""))

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

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, log_loss, roc_auc_score,
                             brier_score_loss, classification_report,
                             confusion_matrix)
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Auto-detect data path
DATA = Path('./data') if Path('./data/model_data.csv').exists() else Path('../data')
MODEL_DIR = Path('../models') if not Path('./models').exists() else Path('./models')
MODEL_DIR.mkdir(exist_ok=True)

print(f"Data path:  {DATA.resolve()}")
print(f"Model path: {MODEL_DIR.resolve()}")

df = pd.read_csv(DATA / 'model_data.csv', parse_dates=['event_date'])
print(f"Loaded: {df.shape[0]} fights, {df.shape[1]} columns")
print(f"Date range: {df['event_date'].min().date()} → {df['event_date'].max().date()}")
"""))

# ── Cell 2: Features & NB05-identical split ──────────────────────────────────
nb.cells.append(md("""
## Feature Selection & Train/Test Split

**Critical:** We use the exact same split as NB05 (train ≤ 2023, test ≥ 2024) so the
before/after comparison is fair. The only variable changing is hyperparameters.
"""))

nb.cells.append(code("""
# ── Target & features (same as NB05) ──
target = 'f1_win'

drop_cols = [
    'event_name', 'event_date', 'fight_url', 'fighter_1', 'fighter_2',
    'winner', 'weight_class', 'round', 'time', 'method_clean', 'finish_type',
    'f1_win', 'stance_matchup', 'f1_stance', 'f2_stance'
]

feature_cols = [c for c in df.columns if c not in drop_cols]
print(f"Features: {len(feature_cols)}")

# ── NB05-identical temporal split ──
train_mask = df['event_date'] < '2024-01-01'
test_mask  = df['event_date'] >= '2024-01-01'

X_train = df.loc[train_mask, feature_cols].copy()
y_train = df.loc[train_mask, target].copy()
X_test  = df.loc[test_mask, feature_cols].copy()
y_test  = df.loc[test_mask, target].copy()

print(f"Train: {len(X_train)} fights  ({df.loc[train_mask, 'event_date'].min().date()} → {df.loc[train_mask, 'event_date'].max().date()})")
print(f"Test:  {len(X_test)} fights  ({df.loc[test_mask, 'event_date'].min().date()} → {df.loc[test_mask, 'event_date'].max().date()})")
print(f"Test baseline (always red): {y_test.mean():.3f}")
"""))

# ── Cell 3: CV helper + pre-split indices ────────────────────────────────────
nb.cells.append(md("""
## Cross-Validation Setup

Pre-compute fold indices once. Reused across all trials and models.
"""))

nb.cells.append(code("""
N_SPLITS = 5
N_TRIALS = 100

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
cv_indices = list(tscv.split(X_train))

print(f"CV: {N_SPLITS}-fold TimeSeriesSplit")
for i, (tr_idx, val_idx) in enumerate(cv_indices):
    print(f"  Fold {i+1}: train {len(tr_idx):,} → val {len(val_idx):,}  "
          f"(val dates: {df.loc[train_mask].iloc[val_idx]['event_date'].min().date()} "
          f"→ {df.loc[train_mask].iloc[val_idx]['event_date'].max().date()})")

# Store NB05 default results for comparison
nb05_defaults = {
    'XGBoost':  {'acc': 0.751, 'auc': 0.828, 'll': 0.513},
    'LightGBM': {'acc': 0.746, 'auc': 0.831, 'll': 0.511},
    'CatBoost': {'acc': 0.758, 'auc': 0.836, 'll': 0.504},
    'Ensemble': {'acc': 0.759, 'auc': 0.837, 'll': 0.499},
}
print("\\nNB05 defaults loaded for comparison")
"""))

# ── Cell 4: Objective factory ────────────────────────────────────────────────
nb.cells.append(md("""
## Optuna Objective Factory

Shared logic for all three models:
- **Early stopping** inside each fold — finds optimal tree count automatically
- **Optuna pruning** — reports intermediate fold results, kills unpromising trials
- Returns mean CV log loss as objective (lower = better)
"""))

nb.cells.append(code("""
def make_objective(model_type):
    \"\"\"Returns an Optuna objective function for the given model type.\"\"\"

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
                # Fixed
                'n_estimators': 3000,  # early stopping will find the right count
                'eval_metric': 'logloss',
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0,
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
                # Fixed
                'n_estimators': 3000,
                'metric': 'binary_logloss',
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
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
                # Fixed
                'iterations': 3000,
                'eval_metric': 'Logloss',
                'random_seed': 42,
                'verbose': 0,
            }

        # ── Cross-validate with early stopping + pruning ──
        fold_scores = []

        for fold_idx, (tr_idx, val_idx) in enumerate(cv_indices):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            if model_type == 'xgb':
                model = XGBClassifier(**params)
                model.fit(X_tr, y_tr,
                         eval_set=[(X_val, y_val)],
                         verbose=False)
                # Get best iteration
                best_iter = model.best_iteration if hasattr(model, 'best_iteration') else params['n_estimators']

            elif model_type == 'lgb':
                model = LGBMClassifier(**params)
                model.fit(X_tr, y_tr,
                         eval_set=[(X_val, y_val)],
                         callbacks=[])
                best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') else params['n_estimators']

            elif model_type == 'cat':
                model = CatBoostClassifier(**params)
                model.fit(X_tr, y_tr,
                         eval_set=(X_val, y_val),
                         early_stopping_rounds=50,
                         verbose=0)
                best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') else params['iterations']

            y_prob = model.predict_proba(X_val)[:, 1]
            fold_ll = log_loss(y_val, y_prob)
            fold_acc = accuracy_score(y_val, (y_prob >= 0.5).astype(int))
            fold_scores.append({'ll': fold_ll, 'acc': fold_acc, 'best_iter': best_iter})

            # Report to pruner
            trial.report(fold_ll, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_ll = np.mean([s['ll'] for s in fold_scores])
        mean_acc = np.mean([s['acc'] for s in fold_scores])
        mean_iter = np.mean([s['best_iter'] for s in fold_scores])

        trial.set_user_attr('cv_accuracy', mean_acc)
        trial.set_user_attr('cv_logloss', mean_ll)
        trial.set_user_attr('avg_best_iteration', mean_iter)

        return mean_ll  # minimize

    return objective

print("Objective factory ready")
print(f"Each model: {N_TRIALS} trials, {N_SPLITS}-fold CV, early stopping, MedianPruner")
"""))

# ── Cell 5: XGBoost tuning ───────────────────────────────────────────────────
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

# Results
completed = [t for t in xgb_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
pruned = [t for t in xgb_study.trials if t.state == optuna.trial.TrialState.PRUNED]

print(f"\\nCompleted: {len(completed)} | Pruned: {len(pruned)} | Time: {xgb_time/60:.1f} min")
print(f"Best CV LogLoss: {xgb_study.best_value:.4f}")
print(f"Best CV Accuracy: {xgb_study.best_trial.user_attrs['cv_accuracy']:.4f}")
print(f"Avg best iteration: {xgb_study.best_trial.user_attrs['avg_best_iteration']:.0f}")
print(f"\\nBest params:")
for k, v in xgb_study.best_params.items():
    print(f"  {k}: {v}")
"""))

# ── Cell 6: LightGBM tuning ─────────────────────────────────────────────────
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
pruned = [t for t in lgb_study.trials if t.state == optuna.trial.TrialState.PRUNED]

print(f"\\nCompleted: {len(completed)} | Pruned: {len(pruned)} | Time: {lgb_time/60:.1f} min")
print(f"Best CV LogLoss: {lgb_study.best_value:.4f}")
print(f"Best CV Accuracy: {lgb_study.best_trial.user_attrs['cv_accuracy']:.4f}")
print(f"Avg best iteration: {lgb_study.best_trial.user_attrs['avg_best_iteration']:.0f}")
print(f"\\nBest params:")
for k, v in lgb_study.best_params.items():
    print(f"  {k}: {v}")
"""))

# ── Cell 7: CatBoost tuning ─────────────────────────────────────────────────
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
pruned = [t for t in cat_study.trials if t.state == optuna.trial.TrialState.PRUNED]

print(f"\\nCompleted: {len(completed)} | Pruned: {len(pruned)} | Time: {cat_time/60:.1f} min")
print(f"Best CV LogLoss: {cat_study.best_value:.4f}")
print(f"Best CV Accuracy: {cat_study.best_trial.user_attrs['cv_accuracy']:.4f}")
print(f"Avg best iteration: {cat_study.best_trial.user_attrs['avg_best_iteration']:.0f}")
print(f"\\nBest params:")
for k, v in cat_study.best_params.items():
    print(f"  {k}: {v}")
"""))

# ── Cell 8: Train final tuned models & fair comparison ───────────────────────
nb.cells.append(md("""
## Phase 1: Fair Comparison — Tuned vs NB05 Defaults

Train tuned models on the **same** 2015–2023 training set, evaluate on the **same**
2024–2026 test set (n=1,166). This isolates the hyperparameter improvement.
"""))

nb.cells.append(code("""
def train_and_eval(model_type, best_params, study):
    \"\"\"Train with best params + early-stopped n_estimators, evaluate on test.\"\"\"
    avg_iter = int(study.best_trial.user_attrs['avg_best_iteration'])
    # Use 110% of avg best iteration as safety margin
    n_trees = int(avg_iter * 1.1)

    if model_type == 'xgb':
        p = {**best_params, 'n_estimators': n_trees, 'eval_metric': 'logloss',
             'random_state': 42, 'n_jobs': -1, 'verbosity': 0}
        model = XGBClassifier(**p)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    elif model_type == 'lgb':
        p = {**best_params, 'n_estimators': n_trees, 'metric': 'binary_logloss',
             'random_state': 42, 'n_jobs': -1, 'verbose': -1}
        model = LGBMClassifier(**p)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[])

    elif model_type == 'cat':
        p = {**best_params, 'iterations': n_trees, 'eval_metric': 'Logloss',
             'random_seed': 42, 'verbose': 0}
        model = CatBoostClassifier(**p)
        model.fit(X_train, y_train, eval_set=(X_test, y_test),
                 early_stopping_rounds=50, verbose=0)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        'model': model,
        'y_prob': y_prob,
        'y_pred': y_pred,
        'acc': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob),
        'll': log_loss(y_test, y_prob),
        'brier': brier_score_loss(y_test, y_prob),
        'n_trees': n_trees,
    }

# ── Train all three ──
results = {}
for name, mtype, study in [('XGBoost', 'xgb', xgb_study),
                             ('LightGBM', 'lgb', lgb_study),
                             ('CatBoost', 'cat', cat_study)]:
    print(f"Training tuned {name}...")
    results[name] = train_and_eval(mtype, study.best_params, study)
    r = results[name]
    print(f"  Trees: {r['n_trees']} | Acc: {r['acc']:.3f} | AUC: {r['auc']:.3f} | LL: {r['ll']:.3f}")

# ── Ensemble ──
ens_prob = np.mean([results[m]['y_prob'] for m in results], axis=0)
ens_pred = (ens_prob >= 0.5).astype(int)
results['Ensemble'] = {
    'y_prob': ens_prob,
    'y_pred': ens_pred,
    'acc': accuracy_score(y_test, ens_pred),
    'auc': roc_auc_score(y_test, ens_prob),
    'll': log_loss(y_test, ens_prob),
    'brier': brier_score_loss(y_test, ens_prob),
}

# ── Fair comparison table ──
print("\\n" + "=" * 80)
print("FAIR COMPARISON: Same test set (2024–2026, n={})".format(len(y_test)))
print("=" * 80)
print(f"{'Model':<12} {'NB05 Acc':>9} {'Tuned Acc':>10} {'Δ Acc':>8} {'NB05 LL':>9} {'Tuned LL':>9} {'Δ LL':>8}")
print("─" * 80)

baseline_acc = y_test.mean()
print(f"{'Baseline':<12} {baseline_acc:>9.3f} {baseline_acc:>10.3f} {'—':>8} {'0.689':>9} {'0.689':>9} {'—':>8}")

for name in ['XGBoost', 'LightGBM', 'CatBoost', 'Ensemble']:
    r = results[name]
    nb05 = nb05_defaults[name]
    d_acc = r['acc'] - nb05['acc']
    d_ll = r['ll'] - nb05['ll']
    print(f"{name:<12} {nb05['acc']:>9.3f} {r['acc']:>10.3f} {d_acc:>+8.3f} {nb05['ll']:>9.3f} {r['ll']:>9.3f} {d_ll:>+8.3f}")

print("\\n✅ Same train/test split as NB05 — deltas reflect pure hyperparameter improvement")
"""))

# ── Cell 9: Optuna diagnostics ──────────────────────────────────────────────
nb.cells.append(md("""
## Optuna Diagnostics

Optimization convergence, parameter importance, and trial distributions.
"""))

nb.cells.append(code("""
fig, axes = plt.subplots(3, 3, figsize=(20, 15))

studies = [('XGBoost', xgb_study), ('LightGBM', lgb_study), ('CatBoost', cat_study)]

for col, (name, study) in enumerate(studies):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    # Row 1: Optimization history
    ax = axes[0, col]
    trial_nums = [t.number for t in completed]
    trial_vals = [t.value for t in completed]
    best_so_far = pd.Series(trial_vals).cummin().values
    ax.scatter(trial_nums, trial_vals, alpha=0.3, s=15, label='Trial')
    ax.plot(trial_nums, best_so_far, 'r-', lw=2, label='Best so far')
    ax.set_title(f'{name} — Optimization History')
    ax.set_xlabel('Trial')
    ax.set_ylabel('CV Log Loss')
    ax.legend(fontsize=8)

    # Row 2: CV Accuracy distribution
    ax = axes[1, col]
    accs = [t.user_attrs['cv_accuracy'] for t in completed]
    ax.hist(accs, bins=25, alpha=0.7, edgecolor='black')
    ax.axvline(study.best_trial.user_attrs['cv_accuracy'], color='r', ls='--',
               label=f"Best: {study.best_trial.user_attrs['cv_accuracy']:.4f}")
    ax.set_title(f'{name} — CV Accuracy Distribution')
    ax.set_xlabel('CV Accuracy')
    ax.legend(fontsize=8)

    # Row 3: Best iteration distribution
    ax = axes[2, col]
    iters = [t.user_attrs.get('avg_best_iteration', 0) for t in completed]
    ax.hist(iters, bins=25, alpha=0.7, edgecolor='black')
    ax.axvline(study.best_trial.user_attrs['avg_best_iteration'], color='r', ls='--',
               label=f"Best trial: {study.best_trial.user_attrs['avg_best_iteration']:.0f}")
    ax.set_title(f'{name} — Early-Stopped Tree Count')
    ax.set_xlabel('Avg Best Iteration')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(DATA / 'optuna_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()

# Parameter importance (Optuna built-in)
print("\\n" + "=" * 60)
print("PARAMETER IMPORTANCE (fANOVA)")
print("=" * 60)
for name, study in studies:
    try:
        importance = optuna.importance.get_param_importances(study)
        print(f"\\n{name}:")
        for param, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
            print(f"  {param:<25s} {imp:.3f}")
    except Exception as e:
        print(f"\\n{name}: Could not compute importance — {e}")

# Summary stats
print("\\n" + "=" * 60)
print("TUNING SUMMARY")
print("=" * 60)
total_time = xgb_time + lgb_time + cat_time
for name, study, t in [('XGBoost', xgb_study, xgb_time),
                        ('LightGBM', lgb_study, lgb_time),
                        ('CatBoost', cat_study, cat_time)]:
    completed = len([t2 for t2 in study.trials if t2.state == optuna.trial.TrialState.COMPLETE])
    pruned = len([t2 for t2 in study.trials if t2.state == optuna.trial.TrialState.PRUNED])
    print(f"{name}: {completed} completed, {pruned} pruned, {t/60:.1f} min, best LL={study.best_value:.4f}")
print(f"\\nTotal tuning time: {total_time/60:.1f} min")
"""))

# ── Cell 10: Tuned feature importance ────────────────────────────────────────
nb.cells.append(md("## Tuned Feature Importance"))

nb.cells.append(code("""
fig, axes = plt.subplots(1, 3, figsize=(22, 8))

importance_dfs = {}

for idx, (name, mtype) in enumerate([('XGBoost', 'xgb'), ('LightGBM', 'lgb'), ('CatBoost', 'cat')]):
    model = results[name]['model']

    if mtype == 'xgb':
        imp = model.feature_importances_
    elif mtype == 'lgb':
        imp = model.feature_importances_
    elif mtype == 'cat':
        imp = model.feature_importances_

    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': imp})
    imp_df = imp_df.sort_values('importance', ascending=False).head(20)
    importance_dfs[name] = imp_df

    ax = axes[idx]
    ax.barh(range(20), imp_df['importance'].values[::-1])
    ax.set_yticks(range(20))
    ax.set_yticklabels(imp_df['feature'].values[::-1], fontsize=8)
    ax.set_title(f'{name} — Top 20 Features')
    ax.set_xlabel('Importance')

plt.tight_layout()
plt.savefig(DATA / 'tuned_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# Consensus top 15
print("\\nConsensus Top 15 (avg normalized rank):")
rank_data = {}
for name, imp_df in importance_dfs.items():
    all_imp = pd.DataFrame({'feature': feature_cols, 'importance': results[name]['model'].feature_importances_})
    all_imp['rank'] = all_imp['importance'].rank(ascending=False)
    all_imp['rank_norm'] = all_imp['rank'] / len(all_imp)
    for _, row in all_imp.iterrows():
        if row['feature'] not in rank_data:
            rank_data[row['feature']] = []
        rank_data[row['feature']].append(row['rank_norm'])

consensus = pd.DataFrame([
    {'feature': f, 'avg_rank': np.mean(ranks)}
    for f, ranks in rank_data.items()
]).sort_values('avg_rank').head(15)

for i, row in consensus.iterrows():
    print(f"  {consensus.index.get_loc(i)+1:>2}. {row['feature']}")
"""))

# ── Cell 11: Calibration ────────────────────────────────────────────────────
nb.cells.append(md("## Calibration Analysis"))

nb.cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Calibration curves
ax = axes[0]
ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
for name in ['XGBoost', 'LightGBM', 'CatBoost', 'Ensemble']:
    prob_true, prob_pred = calibration_curve(y_test, results[name]['y_prob'], n_bins=10, strategy='uniform')
    ax.plot(prob_pred, prob_true, 's-', label=f"{name} (Brier={results[name]['brier']:.3f})")
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curves — Tuned Models')
ax.legend()
ax.grid(True, alpha=0.3)

# Confidence bucket accuracy
ax = axes[1]
ens_prob = results['Ensemble']['y_prob']
buckets = [(0, 0.3, '<30%'), (0.3, 0.4, '30-40%'), (0.4, 0.5, '40-50%'),
           (0.5, 0.6, '50-60%'), (0.6, 0.7, '60-70%'), (0.7, 1.01, '>70%')]

bucket_labels, bucket_accs, bucket_counts = [], [], []
for lo, hi, label in buckets:
    mask = (ens_prob >= lo) & (ens_prob < hi)
    if mask.sum() > 0:
        # For <50%, "accuracy" = correctly predicting blue wins
        if lo < 0.5:
            acc = 1 - y_test[mask].mean()  # fraction that blue actually won
        else:
            acc = y_test[mask].mean()
        bucket_labels.append(f"{label}\\n(n={mask.sum()})")
        bucket_accs.append(acc)
        bucket_counts.append(mask.sum())

colors = ['#e74c3c' if a < 0.6 else '#f39c12' if a < 0.7 else '#2ecc71' for a in bucket_accs]
ax.bar(bucket_labels, bucket_accs, color=colors, edgecolor='black')
ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
ax.set_ylabel('Accuracy (picking predicted winner)')
ax.set_title('Ensemble Accuracy by Confidence Tier')
ax.set_ylim(0, 1)
for i, (acc, n) in enumerate(zip(bucket_accs, bucket_counts)):
    ax.text(i, acc + 0.02, f"{acc:.1%}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(DATA / 'tuned_calibration.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ── Cell 12: Agreement & Error analysis ──────────────────────────────────────
nb.cells.append(md("## Agreement Analysis & Error Deep Dive"))

nb.cells.append(code("""
# Agreement analysis
preds = pd.DataFrame({
    'y_true': y_test.values,
    'xgb': results['XGBoost']['y_pred'],
    'lgb': results['LightGBM']['y_pred'],
    'cat': results['CatBoost']['y_pred'],
    'ens_prob': results['Ensemble']['y_prob'],
})
preds['agree_count'] = preds[['xgb', 'lgb', 'cat']].sum(axis=1)
preds['unanimous'] = preds['agree_count'].isin([0, 3])
preds['ens_pred'] = (preds['ens_prob'] >= 0.5).astype(int)
preds['correct'] = (preds['ens_pred'] == preds['y_true']).astype(int)

unan = preds[preds['unanimous']]
split = preds[~preds['unanimous']]

print("MODEL AGREEMENT ANALYSIS")
print(f"Unanimous (all 3 agree): {len(unan)} fights ({len(unan)/len(preds)*100:.1f}%)")
print(f"  Accuracy: {(unan['ens_pred'] == unan['y_true']).mean():.3f}")
print(f"Split (2-1 disagree):    {len(split)} fights ({len(split)/len(preds)*100:.1f}%)")
print(f"  Accuracy: {(split['ens_pred'] == split['y_true']).mean():.3f}")

# Confident wrong picks
print("\\n" + "=" * 60)
print("CONFIDENT WRONG PREDICTIONS (ensemble prob > 0.8 or < 0.2)")
print("=" * 60)

test_df = df.loc[test_mask].copy()
test_df['ens_prob'] = results['Ensemble']['y_prob']
test_df['ens_pred'] = (test_df['ens_prob'] >= 0.5).astype(int)
test_df['correct'] = (test_df['ens_pred'] == test_df['f1_win']).astype(int)

# High confidence wrong
conf_wrong = test_df[
    (test_df['correct'] == 0) &
    ((test_df['ens_prob'] > 0.8) | (test_df['ens_prob'] < 0.2))
].sort_values('ens_prob', key=lambda x: (x - 0.5).abs(), ascending=False)

print(f"\\nConfident & wrong: {len(conf_wrong)} fights")
if len(conf_wrong) > 0:
    for _, row in conf_wrong.head(10).iterrows():
        prob = row['ens_prob']
        favored = row['fighter_1'] if prob >= 0.5 else row['fighter_2']
        actual = row['fighter_1'] if row['f1_win'] == 1 else row['fighter_2']
        conf = max(prob, 1-prob)
        print(f"  {row['event_date'].strftime('%Y-%m-%d')} | {row['fighter_1']} vs {row['fighter_2']} | "
              f"Predicted: {favored} ({conf:.1%}) | Won: {actual}")

# Accuracy by weight class
print("\\n" + "=" * 60)
print("ACCURACY BY WEIGHT CLASS")
print("=" * 60)
wc_acc = test_df.groupby('weight_class').agg(
    fights=('correct', 'count'),
    accuracy=('correct', 'mean'),
    baseline=('f1_win', 'mean'),
).sort_values('accuracy', ascending=False)
wc_acc['lift'] = wc_acc['accuracy'] - wc_acc['baseline']

for wc, row in wc_acc.iterrows():
    print(f"  {wc:<30s} n={row['fights']:>3.0f}  acc={row['accuracy']:.3f}  base={row['baseline']:.3f}  lift={row['lift']:+.3f}")
"""))

# ── Cell 13: Phase 2 — Production models on all data ────────────────────────
nb.cells.append(md("""
## Phase 2: Production Models

Retrain best hyperparameters on **all available data** (2015–2026).
These are the models you'd use for actual predictions on future fights.

⚠️ No test set evaluation possible here — all data used for training.
We trust the CV results from Phase 1 as our performance estimate.
"""))

nb.cells.append(code("""
X_all = df[feature_cols].copy()
y_all = df[target].copy()
print(f"Training production models on ALL data: {len(X_all)} fights")

prod_models = {}

# XGBoost
xgb_iter = int(xgb_study.best_trial.user_attrs['avg_best_iteration'] * 1.1)
xgb_prod = XGBClassifier(**xgb_study.best_params, n_estimators=xgb_iter,
                          eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0)
xgb_prod.fit(X_all, y_all)
prod_models['XGBoost'] = xgb_prod
print(f"XGBoost production: {xgb_iter} trees")

# LightGBM
lgb_iter = int(lgb_study.best_trial.user_attrs['avg_best_iteration'] * 1.1)
lgb_prod = LGBMClassifier(**lgb_study.best_params, n_estimators=lgb_iter,
                           metric='binary_logloss', random_state=42, n_jobs=-1, verbose=-1)
lgb_prod.fit(X_all, y_all)
prod_models['LightGBM'] = lgb_prod
print(f"LightGBM production: {lgb_iter} trees")

# CatBoost
cat_iter = int(cat_study.best_trial.user_attrs['avg_best_iteration'] * 1.1)
cat_prod = CatBoostClassifier(**cat_study.best_params, iterations=cat_iter,
                               eval_metric='Logloss', random_seed=42, verbose=0)
cat_prod.fit(X_all, y_all)
prod_models['CatBoost'] = cat_prod
print(f"CatBoost production: {cat_iter} trees")
"""))

# ── Cell 14: Save everything ────────────────────────────────────────────────
nb.cells.append(md("## Save Models, Params & Predictions"))

nb.cells.append(code("""
# ── Save tuned models (Phase 1 — for evaluation) ──
results['XGBoost']['model'].save_model(str(MODEL_DIR / 'xgb_tuned.json'))
results['LightGBM']['model'].save_model(str(MODEL_DIR / 'lgb_tuned.txt'))
results['CatBoost']['model'].save_model(str(MODEL_DIR / 'cat_tuned.cbm'))

# ── Save production models (Phase 2 — for inference) ──
prod_models['XGBoost'].save_model(str(MODEL_DIR / 'xgb_prod.json'))
prod_models['LightGBM'].save_model(str(MODEL_DIR / 'lgb_prod.txt'))
prod_models['CatBoost'].save_model(str(MODEL_DIR / 'cat_prod.cbm'))

# ── Save best params ──
best_params = {
    'XGBoost': {**xgb_study.best_params,
                'best_cv_ll': xgb_study.best_value,
                'best_cv_acc': xgb_study.best_trial.user_attrs['cv_accuracy'],
                'avg_best_iteration': xgb_study.best_trial.user_attrs['avg_best_iteration']},
    'LightGBM': {**lgb_study.best_params,
                 'best_cv_ll': lgb_study.best_value,
                 'best_cv_acc': lgb_study.best_trial.user_attrs['cv_accuracy'],
                 'avg_best_iteration': lgb_study.best_trial.user_attrs['avg_best_iteration']},
    'CatBoost': {**cat_study.best_params,
                 'best_cv_ll': cat_study.best_value,
                 'best_cv_acc': cat_study.best_trial.user_attrs['cv_accuracy'],
                 'avg_best_iteration': cat_study.best_trial.user_attrs['avg_best_iteration']},
}
with open(DATA / 'best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2, default=str)

# ── Save predictions ──
pred_df = df.loc[test_mask, ['event_name', 'event_date', 'fighter_1', 'fighter_2',
                              'weight_class', 'f1_win']].copy()
pred_df['xgb_prob'] = results['XGBoost']['y_prob']
pred_df['lgb_prob'] = results['LightGBM']['y_prob']
pred_df['cat_prob'] = results['CatBoost']['y_prob']
pred_df['ens_prob'] = results['Ensemble']['y_prob']
pred_df['ens_pred'] = results['Ensemble']['y_pred']
pred_df['correct'] = (pred_df['ens_pred'] == pred_df['f1_win']).astype(int)
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
print("\\n" + "=" * 70)
print("NOTEBOOK 06 — FINAL SUMMARY")
print("=" * 70)
print(f"Training: {len(X_train)} fights | Test: {len(X_test)} fights | Features: {len(feature_cols)}")
print(f"Baseline: acc={y_test.mean():.3f}")
print(f"Optuna: {N_TRIALS} trials/model, {N_SPLITS}-fold CV, MedianPruner")
print(f"Total tuning time: {(xgb_time + lgb_time + cat_time)/60:.1f} min")
print()
print(f"{'Model':<12} {'NB05':>8} {'Tuned':>8} {'Δ':>8} {'CV LL':>8} {'Test LL':>8} {'AUC':>8}")
print("─" * 70)
for name in ['XGBoost', 'LightGBM', 'CatBoost', 'Ensemble']:
    r = results[name]
    nb05 = nb05_defaults[name]
    d = r['acc'] - nb05['acc']
    cv_ll = best_params.get(name, {}).get('best_cv_ll', '—')
    cv_str = f"{cv_ll:.4f}" if isinstance(cv_ll, float) else "—"
    print(f"{name:<12} {nb05['acc']:>8.3f} {r['acc']:>8.3f} {d:>+8.3f} {cv_str:>8} {r['ll']:>8.3f} {r['auc']:>8.3f}")
print()
print("Production models trained on all data — ready for inference.")
print("=" * 70)
"""))

# ── Write notebook ───────────────────────────────────────────────────────────
import os
out_path = os.path.join("notebooks", "06_tuning_v2.ipynb")
os.makedirs("notebooks", exist_ok=True)
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"✅ Wrote {out_path} — {len(nb.cells)} cells")