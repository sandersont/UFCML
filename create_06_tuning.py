#!/usr/bin/env python3
"""create_06_tuning.py — generates notebooks/06_tuning.ipynb"""

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
md("""# 06 — Hyperparameter Tuning (Optuna)
Tune XGBoost, LightGBM, CatBoost with Optuna using TimeSeriesSplit CV.  
Objective: minimize log loss (better probability calibration → better downstream use).""")

code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, time, json
warnings.filterwarnings('ignore')

from sklearn.metrics import (accuracy_score, log_loss, roc_auc_score,
                             brier_score_loss, classification_report,
                             confusion_matrix)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA = './data/' if os.path.exists('./data/model_data.csv') else '../data/'
df = pd.read_csv(f'{DATA}model_data.csv', parse_dates=['event_date'])
df = df.sort_values('event_date').reset_index(drop=True)

print(f"Loaded: {df.shape}")
print(f"Date range: {df.event_date.min().date()} to {df.event_date.max().date()}")
print(f"Target mean: {df.f1_win.mean():.3f}")
""")

# ── Cell 2 ── Feature Selection & Split (same as 05) ─────────────────
md("""## Feature Selection & Split
Identical to notebook 05 — same features, same temporal split.""")

code("""
TRAIN_END = '2023-12-31'
train = df[df.event_date <= TRAIN_END].copy()
test  = df[df.event_date >  TRAIN_END].copy()

IDENTITY = ['event_name', 'event_date', 'fight_url', 'fighter_1', 'fighter_2',
            'winner', 'f1_win', 'method_clean', 'finish_type', 'round', 'time',
            'time_seconds', 'total_time_seconds', 'weight_class',
            'stance_matchup', 'f1_stance', 'f2_stance']

IN_FIGHT = [c for c in df.columns if any(c.startswith(p) for p in [
    'f1_kd','f2_kd','f1_sub','f2_sub','f1_str_','f2_str_',
    'f1_total_str','f2_total_str','f1_td_','f2_td_',
    'f1_head_','f2_head_','f1_body_','f2_body_',
    'f1_leg_','f2_leg_','f1_distance_','f2_distance_',
    'f1_clinch_','f2_clinch_','f1_ground_','f2_ground_',
    'f1_ctrl_','f2_ctrl_','f1_rev','f2_rev'])]

EXCLUDE = set(IDENTITY + IN_FIGHT)
all_features = sorted([c for c in df.columns if c not in EXCLUDE])

TARGET = 'f1_win'
X_train, y_train = train[all_features], train[TARGET]
X_test,  y_test  = test[all_features],  test[TARGET]

baseline_acc = max(y_test.mean(), 1 - y_test.mean())
baseline_probs = np.full(len(y_test), y_train.mean())
baseline_ll = log_loss(y_test, baseline_probs)

print(f"Train: {len(train)} | Test: {len(test)}")
print(f"Features: {len(all_features)}")
print(f"Baseline acc: {baseline_acc:.3f} | Baseline ll: {baseline_ll:.3f}")

# Notebook 05 results for comparison
nb05 = {
    'XGBoost':  {'acc': 0.751, 'auc': 0.828, 'll': 0.513},
    'LightGBM': {'acc': 0.746, 'auc': 0.831, 'll': 0.511},
    'CatBoost': {'acc': 0.758, 'auc': 0.836, 'll': 0.504},
    'Ensemble': {'acc': 0.759, 'auc': 0.837, 'll': 0.499},
}
print(f"\\nNotebook 05 results (to beat):")
for name, m in nb05.items():
    print(f"  {name:<12} acc={m['acc']:.3f}  AUC={m['auc']:.3f}  ll={m['ll']:.3f}")
""")

# ── Cell 3 ── CV Objective Helper ─────────────────────────────────────
md("""## CV Objective
All three Optuna studies use the same TimeSeriesSplit with 5 folds.  
Objective: minimize mean CV log loss (rewards good probability calibration).""")

code("""
N_SPLITS = 5
TSCV = TimeSeriesSplit(n_splits=N_SPLITS)

# Pre-split indices (reused across all trials for consistency)
CV_SPLITS = list(TSCV.split(X_train))

def cv_score(model, X, y, splits=CV_SPLITS):
    \"\"\"Return mean CV log loss for a fitted-per-fold model factory.\"\"\"
    scores = []
    for tr_idx, val_idx in splits:
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        m = model(X_tr, y_tr)

        if hasattr(m, 'predict_proba'):
            probs = m.predict_proba(X_val)[:, 1]
        else:
            probs = m.predict(X_val)

        scores.append(log_loss(y_val, probs))
    return np.mean(scores)

print(f"CV splits ready: {N_SPLITS} folds")
print(f"Fold sizes: {[len(v) for _, v in CV_SPLITS]}")
""")

# ── Cell 4 ── XGBoost Tuning ─────────────────────────────────────────
md("""## XGBoost — Optuna Tuning
100 trials, optimizing log loss via TimeSeriesSplit.""")

code("""
def xgb_objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 200, 1500),
        'max_depth':         trial.suggest_int('max_depth', 3, 9),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_child_weight':  trial.suggest_int('min_child_weight', 1, 20),
        'gamma':             trial.suggest_float('gamma', 0.0, 5.0),
        'eval_metric':       'logloss',
        'random_state':      42,
        'n_jobs':            -1,
    }

    def make_model(X_tr, y_tr):
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr, verbose=False)
        return m

    return cv_score(make_model, X_train, y_train)

print("Tuning XGBoost (100 trials)...")
t0 = time.time()
xgb_study = optuna.create_study(direction='minimize', study_name='xgb',
                                 sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_objective, n_trials=100, show_progress_bar=True)

print(f"\\nDone in {time.time()-t0:.0f}s")
print(f"Best CV log loss: {xgb_study.best_value:.4f}")
print(f"Best params:")
for k, v in xgb_study.best_params.items():
    print(f"  {k}: {v}")
""")

# ── Cell 5 ── LightGBM Tuning ────────────────────────────────────────
md("""## LightGBM — Optuna Tuning""")

code("""
def lgb_objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 200, 1500),
        'max_depth':         trial.suggest_int('max_depth', 3, 12),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'num_leaves':        trial.suggest_int('num_leaves', 15, 127),
        'min_split_gain':    trial.suggest_float('min_split_gain', 0.0, 2.0),
        'random_state':      42,
        'n_jobs':            -1,
        'verbosity':         -1,
    }

    def make_model(X_tr, y_tr):
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr)
        return m

    return cv_score(make_model, X_train, y_train)

print("Tuning LightGBM (100 trials)...")
t0 = time.time()
lgb_study = optuna.create_study(direction='minimize', study_name='lgb',
                                 sampler=optuna.samplers.TPESampler(seed=42))
lgb_study.optimize(lgb_objective, n_trials=100, show_progress_bar=True)

print(f"\\nDone in {time.time()-t0:.0f}s")
print(f"Best CV log loss: {lgb_study.best_value:.4f}")
print(f"Best params:")
for k, v in lgb_study.best_params.items():
    print(f"  {k}: {v}")
""")

# ── Cell 6 ── CatBoost Tuning ────────────────────────────────────────
md("""## CatBoost — Optuna Tuning""")

code("""
def cat_objective(trial):
    params = {
        'iterations':        trial.suggest_int('iterations', 200, 1500),
        'depth':             trial.suggest_int('depth', 3, 9),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'l2_leaf_reg':       trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'min_data_in_leaf':  trial.suggest_int('min_data_in_leaf', 1, 50),
        'random_strength':   trial.suggest_float('random_strength', 0.0, 5.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 5.0),
        'border_count':      trial.suggest_int('border_count', 32, 255),
        'auto_class_weights': trial.suggest_categorical('auto_class_weights',
                              ['None', 'Balanced']),
        'random_seed':       42,
        'verbose':           0,
        'eval_metric':       'Logloss',
    }

    # CatBoost wants None not string 'None'
    if params['auto_class_weights'] == 'None':
        params['auto_class_weights'] = None

    def make_model(X_tr, y_tr):
        m = cb.CatBoostClassifier(**params)
        m.fit(X_tr, y_tr)
        return m

    return cv_score(make_model, X_train, y_train)

print("Tuning CatBoost (100 trials)...")
t0 = time.time()
cat_study = optuna.create_study(direction='minimize', study_name='cat',
                                 sampler=optuna.samplers.TPESampler(seed=42))
cat_study.optimize(cat_objective, n_trials=100, show_progress_bar=True)

print(f"\\nDone in {time.time()-t0:.0f}s")
print(f"Best CV log loss: {cat_study.best_value:.4f}")
print(f"Best params:")
for k, v in cat_study.best_params.items():
    print(f"  {k}: {v}")
""")

# ── Cell 7 ── Train Final Models with Best Params ────────────────────
md("""## Train Final Models
Retrain each model on full training set using Optuna's best params.""")

code("""
print("="*60)
print("TRAINING FINAL MODELS WITH TUNED PARAMS")
print("="*60)

# ── XGBoost ──
xgb_params = xgb_study.best_params.copy()
xgb_params.update({'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1})
xgb_tuned = xgb.XGBClassifier(**xgb_params)
xgb_tuned.fit(X_train, y_train, verbose=False)
xgb_probs = xgb_tuned.predict_proba(X_test)[:, 1]
xgb_preds = (xgb_probs >= 0.5).astype(int)
print(f"XGBoost  — acc={accuracy_score(y_test, xgb_preds):.3f}  "
      f"AUC={roc_auc_score(y_test, xgb_probs):.3f}  "
      f"ll={log_loss(y_test, xgb_probs):.3f}")

# ── LightGBM ──
lgb_params = lgb_study.best_params.copy()
lgb_params.update({'random_state': 42, 'n_jobs': -1, 'verbosity': -1})
lgb_tuned = lgb.LGBMClassifier(**lgb_params)
lgb_tuned.fit(X_train, y_train)
lgb_probs = lgb_tuned.predict_proba(X_test)[:, 1]
lgb_preds = (lgb_probs >= 0.5).astype(int)
print(f"LightGBM — acc={accuracy_score(y_test, lgb_preds):.3f}  "
      f"AUC={roc_auc_score(y_test, lgb_probs):.3f}  "
      f"ll={log_loss(y_test, lgb_probs):.3f}")

# ── CatBoost ──
cat_params = cat_study.best_params.copy()
cat_params.update({'random_seed': 42, 'verbose': 0, 'eval_metric': 'Logloss'})
if cat_params.get('auto_class_weights') == 'None':
    cat_params['auto_class_weights'] = None
cat_tuned = cb.CatBoostClassifier(**cat_params)
cat_tuned.fit(X_train, y_train)
cat_probs = cat_tuned.predict_proba(X_test)[:, 1]
cat_preds = (cat_probs >= 0.5).astype(int)
print(f"CatBoost — acc={accuracy_score(y_test, cat_preds):.3f}  "
      f"AUC={roc_auc_score(y_test, cat_probs):.3f}  "
      f"ll={log_loss(y_test, cat_probs):.3f}")

# ── Ensemble ──
ens_probs = (xgb_probs + lgb_probs + cat_probs) / 3
ens_preds = (ens_probs >= 0.5).astype(int)
ens_acc = accuracy_score(y_test, ens_preds)
ens_auc = roc_auc_score(y_test, ens_probs)
ens_ll  = log_loss(y_test, ens_probs)
print(f"Ensemble — acc={ens_acc:.3f}  AUC={ens_auc:.3f}  ll={ens_ll:.3f}")
""")

# ── Cell 8 ── Before vs After Comparison ─────────────────────────────
md("""## Before vs After
Side-by-side: notebook 05 defaults vs notebook 06 tuned.""")

code("""
print("="*60)
print("BEFORE (NB05 defaults) vs AFTER (NB06 tuned)")
print("="*60)

tuned = {
    'XGBoost':  {'acc': accuracy_score(y_test, xgb_preds),
                 'auc': roc_auc_score(y_test, xgb_probs),
                 'll':  log_loss(y_test, xgb_probs)},
    'LightGBM': {'acc': accuracy_score(y_test, lgb_preds),
                 'auc': roc_auc_score(y_test, lgb_probs),
                 'll':  log_loss(y_test, lgb_probs)},
    'CatBoost': {'acc': accuracy_score(y_test, cat_preds),
                 'auc': roc_auc_score(y_test, cat_probs),
                 'll':  log_loss(y_test, cat_probs)},
    'Ensemble': {'acc': ens_acc, 'auc': ens_auc, 'll': ens_ll},
}

print(f"\\n{'Model':<12} {'':>5} {'Accuracy':>9} {'AUC':>9} {'LogLoss':>9}")
print("─" * 50)
for name in ['XGBoost', 'LightGBM', 'CatBoost', 'Ensemble']:
    b = nb05[name]
    t = tuned[name]
    print(f"{name:<12} {'NB05':>5} {b['acc']:>9.3f} {b['auc']:>9.3f} {b['ll']:>9.3f}")
    acc_d = t['acc'] - b['acc']
    auc_d = t['auc'] - b['auc']
    ll_d  = t['ll'] - b['ll']
    print(f"{'':12} {'NB06':>5} {t['acc']:>9.3f} {t['auc']:>9.3f} {t['ll']:>9.3f}")
    print(f"{'':12} {'Δ':>5} {acc_d:>+9.3f} {auc_d:>+9.3f} {ll_d:>+9.3f}")
    print()

print(f"Baseline: acc={baseline_acc:.3f}  ll={baseline_ll:.3f}")
""")

# ── Cell 9 ── Optuna Visualization ────────────────────────────────────
md("""## Tuning Visualization
Optimization history and parameter importance for each model.""")

code("""
fig, axes = plt.subplots(2, 3, figsize=(22, 12))

studies = [('XGBoost', xgb_study), ('LightGBM', lgb_study), ('CatBoost', cat_study)]

# Row 1: optimization history
for ax, (name, study) in zip(axes[0], studies):
    trials = study.trials
    values = [t.value for t in trials]
    best_so_far = np.minimum.accumulate(values)

    ax.scatter(range(len(values)), values, alpha=0.3, s=15, color='#3498db')
    ax.plot(range(len(values)), best_so_far, color='#e74c3c', linewidth=2, label='Best so far')
    ax.set_xlabel('Trial')
    ax.set_ylabel('CV Log Loss')
    ax.set_title(f'{name} — Optimization History')
    ax.legend()

# Row 2: parameter importance
for ax, (name, study) in zip(axes[1], studies):
    try:
        importance = optuna.importance.get_param_importances(study)
        params = list(importance.keys())[:10]
        values = [importance[p] for p in params]
        ax.barh(params[::-1], values[::-1], color='#2ecc71')
        ax.set_xlabel('Importance')
        ax.set_title(f'{name} — Parameter Importance')
    except Exception as e:
        ax.text(0.5, 0.5, f'Could not compute\\n{e}',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{name} — Parameter Importance')

plt.tight_layout()
plt.savefig(f'{DATA}optuna_tuning.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ── Cell 10 ── Tuned Feature Importance ───────────────────────────────
md("""## Tuned Feature Importance
Has tuning changed which features matter?""")

code("""
fig, axes = plt.subplots(1, 3, figsize=(24, 10))

def get_color(f):
    if 'profile' in f:   return '#e74c3c'
    if 'last3' in f or 'last5' in f: return '#3498db'
    if 'career' in f:    return '#2ecc71'
    if any(k in f for k in ['age','height','reach','ape','weight']): return '#f39c12'
    if any(k in f for k in ['streak','days','fights_per']): return '#9b59b6'
    return '#95a5a6'

model_objs = [('XGBoost', xgb_tuned), ('LightGBM', lgb_tuned), ('CatBoost', cat_tuned)]

for ax, (name, model) in zip(axes, model_objs):
    if name == 'CatBoost':
        imp = model.get_feature_importance()
    else:
        imp = model.feature_importances_

    feat_imp = pd.Series(imp, index=all_features).sort_values(ascending=False)
    top20 = feat_imp.head(20)
    colors = [get_color(f) for f in top20.index]

    top20.plot.barh(ax=ax, color=colors)
    ax.set_title(f'{name} (tuned) — Top 20', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel('Importance')

from matplotlib.patches import Patch
legend_elements = [
    Patch(color='#e74c3c', label='Profile'),
    Patch(color='#3498db', label='Recent Form (L3/L5)'),
    Patch(color='#2ecc71', label='Career Rolling'),
    Patch(color='#f39c12', label='Physical'),
    Patch(color='#9b59b6', label='Activity/Streak'),
    Patch(color='#95a5a6', label='Other'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=11,
           bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f'{DATA}feature_importance_tuned.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Consensus ──
print("\\nCONSENSUS TOP 15 (tuned models)")
print("="*60)
ranks = pd.DataFrame()
for name, model in model_objs:
    imp = model.get_feature_importance() if name == 'CatBoost' else model.feature_importances_
    ranks[name] = pd.Series(imp, index=all_features).rank(ascending=False)
ranks['avg_rank'] = ranks.mean(axis=1)
consensus = ranks.sort_values('avg_rank').head(15)
consensus['group'] = [
    'PROFILE' if 'profile' in f else
    'RECENT' if ('last3' in f or 'last5' in f) else
    'CAREER' if 'career' in f else
    'PHYSICAL' if any(k in f for k in ['age','height','reach','ape','weight']) else
    'ACTIVITY' if any(k in f for k in ['streak','days','fights_per']) else
    'OTHER'
    for f in consensus.index
]
print(consensus[['avg_rank', 'group', 'XGBoost', 'LightGBM', 'CatBoost']].to_string())
""")

# ── Cell 11 ── Calibration Comparison ─────────────────────────────────
md("""## Calibration — Tuned Models""")

code("""
fig, axes = plt.subplots(1, 4, figsize=(22, 5))

all_models = [('XGBoost', xgb_probs), ('LightGBM', lgb_probs),
              ('CatBoost', cat_probs), ('Ensemble', ens_probs)]

for ax, (name, probs) in zip(axes, all_models):
    fraction_pos, mean_predicted = calibration_curve(y_test, probs, n_bins=10, strategy='uniform')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    ax.plot(mean_predicted, fraction_pos, 'o-', markersize=6, label=name)
    ax.fill_between(mean_predicted, fraction_pos, mean_predicted, alpha=0.15)
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction Positive')
    ax.set_title(f'{name} (tuned)')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(f'{DATA}calibration_tuned.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Confidence buckets ──
print("\\nCONFIDENCE BUCKETS (tuned ensemble)")
print("="*60)
bucket_df = pd.DataFrame({
    'prob': ens_probs,
    'pred': ens_preds,
    'actual': y_test.values,
    'correct': (ens_preds == y_test.values).astype(int),
    'confidence': np.abs(ens_probs - 0.5),
})
bins = [0, 0.03, 0.07, 0.12, 0.18, 0.25, 0.50]
labels = ['50-53%', '53-57%', '57-62%', '62-68%', '68-75%', '75%+']
bucket_df['bucket'] = pd.cut(bucket_df['confidence'], bins=bins, labels=labels)
bucket_stats = bucket_df.groupby('bucket', observed=True).agg(
    fights=('actual', 'count'),
    accuracy=('correct', 'mean'),
    avg_prob=('prob', lambda x: np.abs(x - 0.5).mean() + 0.5),
).round(3)
bucket_stats['pct'] = (bucket_stats['fights'] / len(y_test) * 100).round(1)
print(bucket_stats.to_string())
""")

# ── Cell 12 ── Agreement & Unanimous Analysis ────────────────────────
md("""## Model Agreement — Tuned""")

code("""
print("="*60)
print("MODEL AGREEMENT (tuned)")
print("="*60)

agree_all = (xgb_preds == lgb_preds) & (lgb_preds == cat_preds)
print(f"XGB-LGB agree: {(xgb_preds == lgb_preds).mean():.1%}")
print(f"XGB-CAT agree: {(xgb_preds == cat_preds).mean():.1%}")
print(f"LGB-CAT agree: {(lgb_preds == cat_preds).mean():.1%}")
print(f"All 3 agree:   {agree_all.mean():.1%} ({agree_all.sum()}/{len(y_test)})")

if agree_all.sum() > 0:
    unan_acc = accuracy_score(y_test[agree_all], xgb_preds[agree_all])
    print(f"Unanimous accuracy: {unan_acc:.3f}")

disagree = ~agree_all
if disagree.sum() > 0:
    dis_acc = accuracy_score(y_test[disagree], ens_preds[disagree])
    print(f"Disagreement fights ({disagree.sum()}): ensemble acc = {dis_acc:.3f}")
""")

# ── Cell 13 ── Save Everything ────────────────────────────────────────
md("""## Save Tuned Models & Results""")

code("""
MODELS_DIR = '../models/' if os.path.exists('../models/') else './models/'
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Save models ──
xgb_tuned.save_model(f'{MODELS_DIR}xgb_tuned.json')
lgb_tuned.booster_.save_model(f'{MODELS_DIR}lgb_tuned.txt')
cat_tuned.save_model(f'{MODELS_DIR}catboost_tuned.cbm')
print(f"Tuned models saved to {MODELS_DIR}")

# ── Save best params ──
best_params = {
    'xgb': xgb_study.best_params,
    'lgb': lgb_study.best_params,
    'cat': cat_study.best_params,
}
with open(f'{DATA}best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2, default=str)
print(f"Best params saved to {DATA}best_params.json")

# ── Save test predictions ──
test_out = test[['event_date', 'fighter_1', 'fighter_2', 'f1_win', 'weight_class']].copy()
test_out['xgb_prob'] = xgb_probs
test_out['lgb_prob'] = lgb_probs
test_out['cat_prob'] = cat_probs
test_out['ens_prob'] = ens_probs
test_out['ens_pred'] = ens_preds
test_out['correct']  = (ens_preds == y_test.values).astype(int)
test_out.to_csv(f'{DATA}test_predictions_tuned.csv', index=False)
print(f"Tuned predictions saved to {DATA}test_predictions_tuned.csv")

# ── FINAL SUMMARY ──
print(f"\\n{'='*60}")
print(f"FINAL SUMMARY — TUNED MODELS")
print(f"{'='*60}")
print(f"Training:  {len(train)} fights | Test: {len(test)} fights")
print(f"Features:  {len(all_features)}")
print(f"Baseline:  acc={baseline_acc:.3f}  ll={baseline_ll:.3f}")
print(f"")
print(f"{'Model':<12} {'NB05 Acc':>9} {'NB06 Acc':>9} {'Δ Acc':>7} "
      f"{'NB05 AUC':>9} {'NB06 AUC':>9} {'NB05 LL':>8} {'NB06 LL':>8}")
print("─" * 80)
for name in ['XGBoost', 'LightGBM', 'CatBoost', 'Ensemble']:
    b = nb05[name]
    t = tuned[name]
    print(f"{name:<12} {b['acc']:>9.3f} {t['acc']:>9.3f} {t['acc']-b['acc']:>+7.3f} "
          f"{b['auc']:>9.3f} {t['auc']:>9.3f} {b['ll']:>8.3f} {t['ll']:>8.3f}")

print(f"\\nOptuna trials per model: 100")
print(f"CV strategy: {N_SPLITS}-fold TimeSeriesSplit")
print(f"Objective: minimize CV log loss")
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

out = pathlib.Path("notebooks/06_tuning.ipynb")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(f"Created {out}  ({len(cells)} cells)")