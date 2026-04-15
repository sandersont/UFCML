## Last Updated
After completing notebooks 01–08 (scraping, cleaning, EDA, feature engineering, modeling, tuning, predictions, betting).
Notebook 09 (odds scraper) in progress.

## Architecture

/workspaces/UFCML/
├── setup_project.py                    # Creates dirs, .gitignore, requirements.txt
├── create_01_scraper.py                # Generates notebooks/01_scraper.ipynb
├── create_02_cleaning.py               # Generates notebooks/02_data_cleaning.ipynb
├── create_03_eda.py                    # Generates notebooks/03_eda.ipynb
├── create_04_feature_engineering.py    # Generates notebooks/04_feature_engineering.ipynb
├── create_05_modeling.py               # Generates notebooks/05_modeling.ipynb
├── create_06_tuning.py                 # Generates notebooks/06_tuning.ipynb
├── create_07_predict.py                # Generates notebooks/07_predict.ipynb
├── create_08_betting.py                # Generates notebooks/08_betting.ipynb
├── create_09_odds_scraper.py           # Generates notebooks/09_odds_scraper.ipynb (WIP)
├── requirements.txt
├── .gitignore
├── PROJECT_STATUS.md                   # This file
├── notebooks/
│   ├── 01_scraper.ipynb                # 8 cells — scrapes all data
│   ├── 02_data_cleaning.ipynb          # 7 cells — cleans and parses
│   ├── 03_eda.ipynb                    # 18 cells — exploratory analysis
│   ├── 04_feature_engineering.ipynb    # 15 cells — rolling features + diffs + feature EDA
│   ├── 05_modeling.ipynb               # 16 cells — baseline models + ablation
│   ├── 06_tuning.ipynb                 # 18 cells — Optuna hyperparameter tuning
│   ├── 07_predict.ipynb                # 11 cells — event predictions
│   ├── 08_betting.ipynb                # 16 cells — betting value finder
│   ├── 09_odds_scraper.ipynb           # WIP — historical odds from BestFightOdds
│   └── data/                           # All CSV/JSON files live here
│       ├── events.csv
│       ├── events_clean.csv
│       ├── fights_raw.csv
│       ├── fights.csv
│       ├── fights_clean.csv
│       ├── fighters.csv
│       ├── fighters_full.csv
│       ├── fighters_clean.csv
│       ├── fight_details.json
│       ├── fight_details_checkpoint.json
│       ├── model_data.csv
│       ├── test_predictions.csv
│       ├── test_predictions_tuned.csv
│       ├── feature_list.txt
│       ├── default_params.json
│       ├── nb05_results.json
│       ├── best_params.json
│       ├── optuna_studies.db
│       ├── optuna_trials_xgb.csv
│       ├── optuna_trials_lgb.csv
│       ├── optuna_trials_cat.csv
│       ├── optuna_all_trials.csv
│       ├── feature_importance_comparison.png
│       ├── tuned_feature_importance.png
│       ├── calibration_curves.png
│       ├── tuned_calibration.png
│       ├── confusion_matrices.png
│       ├── tuned_confusion_matrices.png
│       ├── probability_distributions.png
│       ├── optuna_diagnostics.png
│       ├── correlation_matrix_top30.png
│       ├── event_predictions.png
│       ├── betting_edge_chart.png
│       ├── predictions_*.csv           # Per-event prediction files
│       └── bets_*.csv                  # Per-event betting analysis files
├── data/                               # Empty (data lives in notebooks/data/)
├── models/
│   ├── xgb_baseline.json              # NB05 default XGBoost
│   ├── lgb_baseline.txt               # NB05 default LightGBM
│   ├── cat_baseline.cbm               # NB05 default CatBoost
│   ├── xgb_tuned.json                 # NB06 tuned XGBoost (train split only)
│   ├── lgb_tuned.txt                  # NB06 tuned LightGBM (train split only)
│   ├── cat_tuned.cbm                  # NB06 tuned CatBoost (train split only)
│   ├── xgb_prod.json                  # NB06 production XGBoost (all data)
│   ├── lgb_prod.txt                   # NB06 production LightGBM (all data)
│   └── cat_prod.cbm                   # NB06 production CatBoost (all data)
└── src/                                # Empty (future)

NOTE: Data path auto-detected in notebooks (`./data/` or `../data/`).
The `./data/` directory at root is empty. All data lives in `./notebooks/data/`.

**Model types:**
- **Baseline** — default hyperparameters, trained on train split. For NB05 evaluation.
- **Tuned** — Optuna best params, trained on train split. For fair NB05 vs NB06 comparison.
- **Production** — Optuna best params, trained on ALL data. For real predictions in NB07/NB08.

---

## Completed Notebooks

### 01_scraper.ipynb (8 cells)

| Cell | Stage             | Output File       | Records      |
|------|-------------------|-------------------|--------------|
| 1    | Imports & setup   | —                 | —            |
| 2    | Event list        | events.csv        | 769 events   |
| 3    | Event fights      | fights_raw.csv    | 8,637 fights |
| 4    | Fighter directory | fighters.csv      | 4,486 fighters |
| 5    | Fighter profiles  | fighters_full.csv | 4,486 fighters |
| 6    | Fight details     | fight_details.json| 8,637 records |
| 7    | Rebuild corners   | fights.csv        | 8,637 fights |
| 8    | Summary           | —                 | —            |

**Key scraping discoveries:**
- Event page always lists winner first (not red corner)
- Fight detail page lists red corner first with W/L status
- Must use fight detail pages to get correct red/blue corner assignment
- Fighter directory (`/statistics/fighters?char=X&page=all`) only has basic info
- Fighter profile pages (`/fighter-details/{id}`) have 8 career rate stats
- `get_text(separator='|')` leaves leading `|` on profile values — must strip
- Fight detail JSON has full "X of Y" format for all strike/TD stats
- JSON totals row: [0]=names [1]=KD [2]=sig_str [3]=sig_pct [4]=total_str [5]=TD [6]=TD_pct [7]=sub_att [8]=rev [9]=ctrl
- JSON sig strikes row: [0]=names [1]=sig_str [2]=sig_pct [3]=head [4]=body [5]=leg [6]=distance [7]=clinch [8]=ground
- Profile stat keys vary: "Str. Def" vs "Str. Def." — key_map handles both
- Uses 10 concurrent threads, tqdm progress bars
- Total runtime: ~20-30 minutes

### 02_data_cleaning.ipynb (7 cells)

| Cell | What                              | Output File        |
|------|-----------------------------------|--------------------|
| 1    | Imports & constants               | —                  |
| 2    | Load raw data                     | —                  |
| 3    | Year filter (2015+) + drop Draw/NC| —                  |
| 4    | Clean events                      | events_clean.csv   |
| 5    | Clean fighters (directory + profiles) | fighters_clean.csv |
| 6    | Clean fights (JSON stats)         | —                  |
| 7    | Save + verify                     | fights_clean.csv   |

**Key cleaning rules:**
- Pre-2015 data dropped (corner data unreliable; pre-2010 red WR = 100%)
- Draw/NC dropped (~1.5% of fights, not predictable)
- Draw/NC detection: direct comparison `winner != fighter_1 AND winner != fighter_2`
- NOT regex — `str.contains("NC")` falsely matches Francisco, Duncan, Vince
- Height/reach `"--"` → NaN; Weight `"-- lbs."` → NaN
- Profile stats have leading `|` from scraper — `clean_val()` strips it
- Profile pct fields `"|XX%"` → 0.XX float
- Profile float fields `"|X.XX"` → float
- Raw duplicate columns dropped after cleaning (only clean versions kept)
- Fighters: dropped height, weight, reach, dob, str_acc_pct, str_def_pct, td_acc_pct, td_def_pct
- Fights: dropped method (kept method_clean)
- Method field: embedded newlines stripped with regex
- `parse_pct` treats "0%" as NaN (no attempts = no accuracy)

**fighters_clean.csv final columns (23):**
fighter_url, first_name, last_name, nickname, stance, wins, losses, draws, slpm, sapm, td_avg, sub_avg, full_name, height_inches, reach_inches, weight_lbs, total_fights, win_pct, str_acc_career, str_def_career, td_acc_career, td_def_career, dob_parsed

**fights_clean.csv final columns (62):**
event_name, event_date, fight_url, fighter_1, fighter_2, winner, f1_kd, f2_kd, f1_sub, f2_sub, weight_class, round, time, f1_rev, f2_rev, f1_str_landed, f1_str_attempted, f1_str_acc, f2_str_landed, f2_str_attempted, f2_str_acc, f1_total_str_landed, f1_total_str_attempted, f2_total_str_landed, f2_total_str_attempted, f1_td_landed, f1_td_attempted, f1_td_acc, f2_td_landed, f2_td_attempted, f2_td_acc, f1_head_landed, f1_head_attempted, f1_body_landed, f1_body_attempted, f1_leg_landed, f1_leg_attempted, f1_distance_landed, f1_distance_attempted, f1_clinch_landed, f1_clinch_attempted, f1_ground_landed, f1_ground_attempted, f2_head_landed, f2_head_attempted, f2_body_landed, f2_body_attempted, f2_leg_landed, f2_leg_attempted, f2_distance_landed, f2_distance_attempted, f2_clinch_landed, f2_clinch_attempted, f2_ground_landed, f2_ground_attempted, f1_ctrl_seconds, f2_ctrl_seconds, time_seconds, total_time_seconds, f1_win, method_clean, finish_type

### 03_eda.ipynb (18 cells)

| Cell | Section |
|------|---------|
| 1    | Load data |
| 2    | Outcome overview (finish types, corner WR, rounds) |
| 3    | Fight length analysis |
| 4    | Red WR over time (yearly + 95% CI) |
| 5    | Red WR by finish type over time |
| 6    | Weight class deep dive |
| 7    | Winner vs loser stats |
| 8    | Stat differentials vs win probability |
| 9    | Strike location breakdown (head/body/leg) |
| 10   | Strike position breakdown (distance/clinch/ground) |
| 11   | Fighter career rate stats (SLpM, SApM, etc.) |
| 12   | Career stat diffs vs win rate |
| 13   | Physical attributes |
| 14   | Physical diffs vs win rate |
| 15   | Correlation matrix |
| 16   | Top correlations bar chart |
| 17   | Stance matchups |
| 18   | Key insights & next steps |

### 04_feature_engineering.ipynb (15 cells)

| Cell | What | Notes |
|------|------|-------|
| 1    | Imports & load data | Sort chronologically |
| 2    | Build fighter-centric history | Unpivot: each fight → 2 rows (one per fighter) |
| 3    | Career aggregates | Expanding window, all prior fights |
| 4    | Recent form (last 3, last 5) | Rolling windows with min_periods |
| 5    | Streaks & activity | Win/loss streak, days since last, fights/year |
| 6    | Opponent quality | Avg opponent WR at time of fight |
| 7    | Physical & profile stats | Height, reach, age, stance, ape index; dedup fix |
| 8    | Assemble F1 vs F2 + differentials | Pivot back + compute all diffs |
| 9    | Encode categoricals | Weight class ordinal, stance matchup |
| 10   | Feature EDA: top correlations | Bar chart + categorized table of top 30 diffs |
| 11   | Feature EDA: profile vs rolling | Side-by-side comparison, redundancy check |
| 12   | Feature EDA: NaN coverage | Complete rows per group, experience histograms |
| 13   | Feature EDA: age & physical deep dive | 6-panel viz, age is #3 feature |
| 14   | Feature EDA: multicollinearity | High-correlation pairs, top-20 heatmap |
| 15   | Leakage verification & save | 4 anti-leakage tests + model_data.csv |

**Anti-leakage protocol:**
- All fights sorted chronologically before any computation
- `expanding().mean().shift(1)` — shift(1) excludes current fight
- `shift(1).rolling(window, min_periods=window)` for recent form
- Fighter's first fight in dataset → NaN for all rolling features
- "True" accuracy ratios computed from cumulative landed/attempted (not averaging percentages)
- Opponent quality uses opponent's win rate at that same point in time

**Duplicate fighter names (Cell 7):**
- 7 pairs of different fighters with same full_name in fighters_clean
- Fix: deduplicate by keeping entry with most total_fights
- Bruno Silva — two active UFC fighters, can't disambiguate by name → stats set to NaN
- Other 6 pairs: only one version active in our fights data, dedup handles correctly

**Feature groups engineered:**

Career aggregates (expanding window, all prior fights):
- career_win_rate, career_fights
- career_avg_* for 24 stats: str_landed, str_attempted, str_acc, total_str_landed, total_str_attempted, td_landed, td_attempted, td_acc, kd, sub, rev, ctrl_seconds, head/body/leg landed & attempted, distance/clinch/ground landed & attempted
- career_avg_opp_* for absorbed stats: opp_str_landed, opp_str_attempted, opp_td_landed, opp_td_attempted, opp_kd, opp_ctrl_seconds, opp_head/body/leg_landed, opp_distance/clinch/ground_landed
- career_str_acc_true — cumulative landed / cumulative attempted (correct way)
- career_td_acc_true — same for takedowns
- career_str_def_true — 1 - (cumulative opp landed / cumulative opp attempted)
- career_td_def_true — same for takedown defense
- career_won_by_ko_rate, career_won_by_sub_rate, career_won_by_dec_rate

Recent form (last 3 & last 5 fights):
- last3_* / last5_* for: won, str_landed, str_attempted, str_acc, td_landed, td_attempted, kd, ctrl_seconds, head/body/leg_landed, distance/clinch/ground_landed, opp_str_landed, opp_kd, opp_ctrl_seconds, sub
- Requires exactly 3 (or 5) prior fights, otherwise NaN

Streaks & activity:
- win_streak — consecutive wins heading into fight (0 = no streak)
- loss_streak — consecutive losses heading into fight
- days_since_last — days since previous fight
- fights_per_year — career activity rate

Opponent quality:
- avg_opp_wr — average career win rate of all prior opponents (at time of each fight)
- last3_opp_wr — same but last 3 opponents only

Physical & profile (from fighters_clean):
- height_inches, reach_inches, weight_lbs
- age — computed from DOB at fight date
- ape_index — reach minus height
- profile_* — UFC career stats (slpm, sapm, str_acc_career, str_def_career, td_avg, td_acc_career, td_def_career, sub_avg, win_pct, total_fights) as cold-start fallback

Fight-level assembly:
- Every feature exists as f1_* (red/favorite) and f2_* (blue/underdog)
- Every numeric feature has a differential diff_* = f1 - f2
- Stance kept as categorical (f1_stance, f2_stance)

Categorical encoding:
- weight_class_ord — ordinal by weight (1=WSW through 12=HW)
- f1_stance_enc / f2_stance_enc — Orthodox=0, Southpaw=1, Switch=2, Unknown=-1
- stance_matchup — string "F1Stance_vs_F2Stance"
- ortho_vs_south — boolean flag
- has_switch — boolean flag

**Leakage verification (Cell 15):**
- First-fight fighters have NaN career stats ✅
- Career fight count increments by 1 per fight ✅
- Manual win rate verification against hand-counted prior results ✅
- Top correlations sanity check (no suspiciously high values) ✅

### 05_modeling.ipynb (16 cells)

| Cell | What | Notes |
|------|------|-------|
| 1    | Intro | Goals and approach |
| 2    | Imports & load | Shape, date range |
| 3    | Feature selection & temporal split | Train < 2025-07-01, Test ≥ 2025-07-01 |
| 4    | TimeSeriesSplit CV utility | 5-fold, per-fold + aggregate metrics |
| 5    | XGBoost | CV + test evaluation |
| 6    | LightGBM | CV + test evaluation |
| 7    | CatBoost | CV + test evaluation |
| 8    | Model comparison table | All models vs baseline + CV progression |
| 9    | Ensemble | Average ensemble + agreement analysis |
| 10   | Feature importance | 3-panel chart + consensus top 15 |
| 11   | Ablation study | 9 feature subsets |
| 12   | Calibration | Curves + confidence buckets |
| 13   | Error analysis | Confident wrong picks, by weight class, finish type, monthly |
| 14   | Confusion matrices | 4-panel + classification report |
| 15   | Probability distributions | Histogram by outcome + separation plot |
| 16   | Save & summary | Models, predictions, params, nb05_results.json |

**NB05 Train/Test Split:**
- Train: 2015-01-03 → 2025-06-28 (5,093 fights)
- Test: 2025-07-05 → 2026-04-12 (392 fights)
- Test baseline (always red): 55.4%

**NB05 Results (default hyperparameters):**

| Model    | CV Acc | Test Acc | Lift    | AUC   | LogLoss | Brier |
|----------|--------|----------|---------|-------|---------|-------|
| Baseline | —      | 0.554    | —       | 0.500 | 0.688   | —     |
| XGBoost  | 0.693  | 0.768    | +0.214  | 0.864 | 0.458   | 0.232 |
| LightGBM | 0.688  | 0.781    | +0.227  | 0.868 | 0.452   | 0.219 |
| CatBoost | 0.703  | 0.786    | +0.232  | 0.872 | 0.457   | 0.214 |
| Ensemble | —      | 0.773    | +0.219  | 0.871 | 0.448   | 0.146 |

**CV Fold Accuracy Progression:**
- XGBoost:  0.637 → 0.688 → 0.678 → 0.726 → 0.737 ↑
- LightGBM: 0.647 → 0.666 → 0.659 → 0.735 → 0.730 ↑
- CatBoost: 0.666 → 0.673 → 0.686 → 0.746 → 0.744 ↑
- Clear pattern: more training data → better performance

**NB05 Key Findings:**

CatBoost dominates individually (0.786) but ensemble underperforms it (0.773) because XGBoost drags down the average. LightGBM has best calibration (lowest LL 0.452, lowest Brier 0.219 among individuals). Ensemble has best Brier overall (0.146) due to probability smoothing.

Agreement analysis:
- Unanimous (3-0): 349 fights (89.0%) → 81.4% accuracy
- Split (2-1): 43 fights (11.0%) → 44.2% accuracy (worse than coin flip)
- When models disagree, the ensemble prediction is unreliable

**NB05 Ablation Results:**

| Feature Set                        | Features | Accuracy | AUC   | LogLoss | Lift    |
|------------------------------------|----------|----------|-------|---------|---------|
| All features                       | 310      | varies   | 0.831 | 0.511   | —       |
| All − profile                      | 280      | lower    | 0.651 | 0.670   | −0.112  |
| Diffs only                         | 101      | similar  | 0.821 | 0.516   | −0.007  |
| Diffs − profile                    | 91       | lower    | 0.632 | 0.679   | −0.140  |
| Profile only                       | 35       | strong   | 0.812 | 0.535   | —       |
| Profile diffs only                 | ~10      | strong   | —     | —       | —       |
| Career rolling only                | varies   | moderate | —     | —       | —       |
| Recent form only (L3+L5)           | varies   | moderate | —     | —       | —       |
| Physical + activity + categoricals | 32       | weak     | 0.627 | 0.687   | —       |

**NB05 saves:**
- `models/xgb_baseline.json`, `lgb_baseline.txt`, `cat_baseline.cbm`
- `data/test_predictions.csv`
- `data/feature_list.txt` (310 features)
- `data/default_params.json`
- `data/nb05_results.json` (for NB06 fair comparison)

### 06_tuning.ipynb (18 cells)

| Cell | What | Notes |
|------|------|-------|
| 1    | Intro | Two-phase approach description |
| 2    | Imports & load | + Optuna, load NB05 baselines from JSON |
| 3    | Features & split | Same mid-2025 split as NB05 |
| 4    | CV setup | 5-fold TimeSeriesSplit, pre-split indices |
| 5    | Objective factory | Early stopping + pruning + SQLite storage |
| 6    | XGBoost Optuna | 100 trials, resumable |
| 7    | LightGBM Optuna | 100 trials, resumable |
| 8    | CatBoost Optuna | 100 trials, resumable |
| 8.5  | Export trials to CSV | Per-model + combined CSV |
| 9    | Train tuned models & fair comparison | Same test set as NB05 |
| 10   | Ensemble strategies | Equal, CatBoost-heavy, AUC-weighted, LL-weighted, drop-weakest |
| 11   | Agreement & confidence tiers | VERY_HIGH / HIGH / MEDIUM / LOW / NO_CONF |
| 12   | Optuna diagnostics | Convergence, accuracy distribution, tree counts, param importance |
| 13   | Tuned feature importance | Top 20 per model + consensus |
| 14   | Calibration | Curves + confidence buckets |
| 15   | Error analysis | Confident wrong, by weight class, finish type, monthly |
| 16   | Confusion matrices | 4-panel |
| 17   | Phase 2: production models | Retrain on ALL data |
| 18   | Save & summary | Native formats, params JSON, predictions CSV |

**NB06 Configuration:**
- Split: Same as NB05 — Train < 2025-07-01 (5,093) / Test ≥ 2025-07-01 (392)
- CV: 5-fold TimeSeriesSplit within training set
- Optuna: 100 trials per model, TPE sampler (seed=42), minimize CV log loss
- Early stopping: 50 rounds (n_estimators not tuned — set to 3000, early stop finds optimal)
- Pruner: MedianPruner (n_startup_trials=10, n_warmup_steps=2)
- Storage: SQLite (`optuna_studies.db`) — resumable if interrupted
- Same 310 features as NB05

**Optuna search spaces:**

XGBoost (8 hyperparameters):
- max_depth: 3–9
- learning_rate: 0.01–0.3 (log)
- subsample: 0.5–1.0
- colsample_bytree: 0.3–1.0
- reg_alpha: 1e-3–10.0 (log)
- reg_lambda: 1e-3–10.0 (log)
- min_child_weight: 1–20
- gamma: 0.0–5.0

LightGBM (9 hyperparameters):
- max_depth: 3–12
- learning_rate: 0.01–0.3 (log)
- subsample: 0.5–1.0
- colsample_bytree: 0.3–1.0
- reg_alpha: 1e-3–10.0 (log)
- reg_lambda: 1e-3–10.0 (log)
- min_child_samples: 5–50
- num_leaves: 15–127
- min_split_gain: 0.0–2.0

CatBoost (9 hyperparameters):
- depth: 3–9
- learning_rate: 0.01–0.3 (log)
- subsample: 0.5–1.0
- l2_leaf_reg: 1e-3–10.0 (log)
- min_data_in_leaf: 1–50
- random_strength: 0.0–5.0
- bagging_temperature: 0.0–5.0
- border_count: 32–255
- auto_class_weights: None or Balanced

**NB06 Results (fair comparison — same test set as NB05):**

| Model    | NB05 Acc | Tuned Acc | Δ Acc  | CV LL  | Test LL | AUC   | Brier |
|----------|----------|-----------|--------|--------|---------|-------|-------|
| XGBoost  | 0.768    | 0.791     | +0.023 | 0.5621 | 0.459   | 0.874 | 0.147 |
| LightGBM | 0.781    | 0.791     | +0.010 | 0.5669 | 0.453   | 0.885 | 0.145 |
| CatBoost | 0.786    | 0.781     | -0.005 | 0.5652 | 0.465   | 0.878 | 0.149 |
| Ensemble | 0.773    | 0.791     | +0.018 | —      | 0.458   | 0.881 | 0.146 |

**Tuning impact:**
- XGBoost biggest winner (+2.3%) — default params were suboptimal
- LightGBM modest gain (+1.0%) — defaults already reasonable
- CatBoost slightly worse (-0.5%) — defaults famously good, Optuna overfit CV slightly
- Tuning compressed the gap: all three models converged to 0.791
- Best ensemble: Equal (1/3 each) — models perform identically, weighting doesn't help
- Total tuning time: 24.7 minutes (100 trials × 3 models with pruning)

**NB06 saves:**
- Tuned models: `models/xgb_tuned.json`, `lgb_tuned.txt`, `cat_tuned.cbm`
- Production models: `models/xgb_prod.json`, `lgb_prod.txt`, `cat_prod.cbm`
- `data/best_params.json` (params + ensemble weights + test metrics)
- `data/test_predictions_tuned.csv` (392 rows)
- `data/optuna_studies.db` (SQLite — all trials persisted)
- `data/optuna_trials_xgb.csv`, `optuna_trials_lgb.csv`, `optuna_trials_cat.csv`
- `data/optuna_all_trials.csv` (combined)

### 07_predict.ipynb (11 cells)

| Cell | What | Notes |
|------|------|-------|
| 1    | Intro | —  |
| 2    | Config | Model files, ensemble weights, event URL |
| 3    | Imports & load data | model_data, fighters_clean, feature_list |
| 4    | Load models | XGB + LGB + CAT, auto-load ensemble weights |
| 5    | Scrape event | Auto-detect latest or use custom URL |
| 6    | Match fighters & build features | Profile-only fallback for cold-start fighters |
| 7    | Predict | Per-model probs + weighted ensemble + tiers |
| 8    | Prediction card | Full card + full-features-only card |
| 9    | Individual model breakdown | Per-model picks + disagreement detail |
| 10   | Probability chart | Visual with model dots + ensemble bars |
| 11   | Save predictions | Per-event CSV |

**NB07 Features:**
- Config cell to swap model sets (baseline / tuned / production)
- Auto-detects latest event or accepts custom URL
- Profile-only fallback: fighters in `fighters_clean` but not in `model_data` get ~15 profile + physical features, rest NaN. Tree models handle NaN natively.
- Coverage tagging: each fight marked as `full` (>250 features) or `profile_only`
- Separate output for full-feature fights only
- Confidence tiers: VERY_HIGH (≥80%), HIGH (≥65%), MEDIUM (≥55%), LOW (<55%), NO_CONF (models disagree)
- Automatic result scoring if event has completed
- Per-event CSV saved with predictions, tiers, and results

**NB07 Prediction pipeline:**
1. Scrape event card from UFCStats
2. For each fight, scrape detail page for correct red/blue corners + winner
3. Match fighter names to historical data
4. Pull each fighter's latest feature snapshot from model_data
5. For unmatched fighters, fall back to profile stats from fighters_clean
6. Compute all differentials (diff_* = f1 - f2)
7. Run through all three models → individual probabilities
8. Weighted ensemble → pick + confidence + tier
9. Display + save

### 08_betting.ipynb (16 cells)

| Cell | What | Notes |
|------|------|-------|
| 1 | md | Intro + betting rules |
| 2 | md | Config instructions |
| 3 | code | **Edit: EVENT_URL, BANKROLL, ODDS** |
| 4 | code | Imports, load production models + data |
| 5 | code | Odds conversion (American → decimal → remove vig) |
| 6 | code | Scrape event card from UFCStats |
| 7 | code | Match fighters, build features, compute diffs |
| 8 | code | Run 3 models → ensemble → picks + tiers |
| 9 | code | Calculate edge + half-Kelly stakes + filter |
| 10 | code | **Bet card display** |
| 11 | code | Full analysis table (all fights) |
| 12 | code | Edge visualization (bar chart + scatter) |
| 13 | code | Risk summary + scenario analysis |
| 14 | md | Results instructions |
| 15 | code | **Score results after event** |
| 16 | code | Save CSV |

**NB08 Betting Rules:**
- Only bet when all 3 models are **unanimous**
- Only bet when ensemble confidence **>= 55%** (MEDIUM tier or above)
- Only bet when **positive edge** exists (model prob > market implied prob)
- Stake = **half-Kelly** = (edge / (decimal_odds - 1)) / 2
- User inputs American odds directly in config cell

**NB08 Odds Pipeline:**
1. User pastes American moneyline odds per fight in config cell
2. Convert American → decimal → raw implied probability
3. Remove vig: divide each implied prob by sum (overround)
4. Edge = model probability - fair implied probability
5. Half-Kelly fraction = (edge / (decimal_odds - 1)) / 2
6. Stake = kelly_frac * BANKROLL

**NB08 Workflow:**
1. Edit config cell: paste EVENT_URL, set BANKROLL, fill in ODDS
2. Run All → get bet card with qualified picks, stakes, edges
3. After event completes → re-run Cell 6 + Cell 15 to score results

**NB08 First Live Results — UFC 327: Prochazka vs. Ulberg:**

| Pick | Odds | Model | Market | Edge | Stake | Result | P&L |
|------|------|-------|--------|------|-------|--------|-----|
| Carlos Ulberg | -110 | 82.6% | 50.0% | +32.6% | \$179 | WIN | +\$163 |
| Azamat Murzakanov | -245 | 78.7% | 68.1% | +10.6% | \$130 | LOSS | -\$130 |
| Josh Hokit | -120 | 72.8% | 52.2% | +20.6% | \$124 | WIN | +\$103 |
| Dominick Reyes | -162 | 64.7% | 59.3% | +5.4% | \$44 | WIN | +\$27 |
| Cub Swanson | -122 | 70.1% | 52.6% | +17.5% | \$107 | WIN | +\$88 |

**UFC 327 Summary: 4W-1L | +\$250.40 | ROI +42.9%**

**NB08 saves:**
- `data/bets_{event_slug}.csv` (per-event betting analysis)
- `data/betting_edge_chart.png`

---

## In Progress

### 09_odds_scraper.ipynb (WIP)

**Goal:** Scrape historical odds from BestFightOdds for numbered UFC events, enabling backtesting.

**BFO Site Structure (discovered):**
- Archive page (`/archive`) only shows ~20 recent events across all promotions
- Search (`/search?query=...`) returns max 25 results, fuzzy matching, unreliable for automated use
- Event URLs require full slug: `/events/ufc-327-4074` (ID-only URLs 302 redirect to homepage)
- Event page has two tables: Table 0 = left headers, Table 1 = odds data
- Event name in `<h1>` tag, date in `<span class="table-header-date">`
- Title format: `"UFC 300 Odds: Pereira vs Hill for April 14 | Best Fight Odds"`
- Fighter names in `<span class="t-b-fcc">`
- Moneyline odds in `<td class="but-sg">` (props use `but-sgp` — skip)
- Prop rows have `class="pr"` — skip
- Fights come in consecutive row pairs (fighter 1 then fighter 2)
- Odds text includes arrows: `+207▲`, `-255▼` — must strip
- Batch search works: searching "ufc 327 326 325 324 323" finds multiple events
- Search results in Table 1 with date in td[0] and event name+link in td[1]

**BFO Event URLs Found So Far:**

| Event | BFO URL |
|-------|---------|
| UFC 311 | /events/ufc-311-3596 |
| UFC 313 | /events/ufc-313-3650 |
| UFC 315 | /events/ufc-315-3708 |
| UFC 316 | /events/ufc-316-3702 |
| UFC 319 | /events/ufc-319-3800 |
| UFC 320 | /events/ufc-320-3853 |
| UFC 321 | /events/ufc-321-odds-3780 |
| UFC 322 | /events/ufc-322-3924 |
| UFC 323 | /events/ufc-323-3951 |
| UFC 324 | /events/ufc-324-3973 |
| UFC 326 | /events/ufc-326-4065 |
| UFC 327 | /events/ufc-327-4074 |

**Still missing:** UFC 312, 314, 317, 318, 325 (user collecting URLs manually)

**Scope change:** Only scraping 2025+ numbered UFC events (~17 events) instead of all events back to 2015. Sufficient for backtest of production model period.

**Next steps:**
- User provides remaining BFO event URLs
- Build scraper to parse odds from each event page
- Match BFO fighter names to UFCStats fighter names (fuzzy matching)
- Save `odds_historical.csv`
- Then build notebook 10: backtest analysis using model predictions + real odds

---

## Data Summary

**model_data.csv** — 5,485 rows × 322 columns
- Date range: 2015 to 2026
- F1 = Red corner = Favorite | F2 = Blue corner = Underdog
- Target: f1_win (1 = red wins, 0 = blue wins)
- Baseline: 57.1% (always pick red, full dataset)
- All features are pre-fight (no in-fight data leakage)
- 101 differential features (diff_*)
- Contains: identity columns + f1_* features + f2_* features + diff_* differentials + categoricals

**fights_clean.csv** — 5,485 rows
- 62 columns (no raw duplicates)
- 0 nulls in all core stat columns
- In-fight stats only — used as input for feature engineering

**fighters_clean.csv** — 4,486 rows
- 23 columns (no raw duplicates — pipe columns dropped)
- 100% name match rate to fights
- 7 duplicate full_name pairs (different fighters, same name)
- Profile career rate stats:

| Stat            | Coverage   | Mean  |
|-----------------|------------|-------|
| slpm            | 4486/4486  | 2.486 |
| sapm            | 4486/4486  | 3.231 |
| str_acc_career  | 3659/4486  | 0.439 |
| str_def_career  | 3727/4486  | 0.513 |
| td_avg          | 4486/4486  | 1.240 |
| td_acc_career   | 2607/4486  | 0.445 |
| td_def_career   | 2913/4486  | 0.616 |
| sub_avg         | 4486/4486  | 0.567 |
| dob_parsed      | 3978/4486  | —     |

**events_clean.csv** — 467 rows
- event_name, event_url, event_date, location

**fight_details.json** — 8,637 records
- Full per-fight data including per-round breakdowns
- Totals table + significant strikes breakdown table

---

## Key EDA Findings

### Baseline
- Always pick red corner: 57.1% — any model must beat this
- Consistent ~55-60% across all years (2015-2026)
- Split decisions: ~51% red WR (essentially coin flips — good sanity check)

### Finish Types

| Type    | Count | Pct   |
|---------|-------|-------|
| DEC     | 2,737 | 49.9% |
| KO/TKO  | 1,760 | 32.1% |
| SUB     | 975   | 17.8% |
| OTHER   | 13    | 0.2%  |

Average fight length: 11.1 minutes

### Pre-Fight Signal Strength

**Tier 1** — Strongest pre-fight signals (from EDA career stat diffs):

| Feature              | Correlation |
|----------------------|-------------|
| Win% diff            | +0.377      |
| Str Acc diff         | +0.238      |
| SLpM diff            | +0.189      |
| SApM diff (lower=better) | -0.180  |
| Str Def diff         | +0.161      |

**Tier 2** — Moderate:

| Feature     | Correlation |
|-------------|-------------|
| TD Avg diff | +0.112      |

**Tier 3** — Weak but real:

| Feature     | Correlation |
|-------------|-------------|
| Reach diff  | +0.052      |
| Height diff | +0.047      |

### Weight Classes
- Heavyweight: highest red WR (60%) and KO rate (47%)
- Women's Strawweight: lowest KO rate (14%), still ~57% red WR
- 13 weight classes total, no title/championship indicator in data

### Stance Matchups (n≥20)
- Switch vs Southpaw: 73% red WR (n=78, notable but small)
- Orthodox vs Switch: ~50% — switch fighters neutralize favorite advantage
- Orthodox vs Orthodox: 57% (matches baseline)

---

## Feature Engineering Results (from notebook 04)

### Top 20 Engineered Feature Correlations with f1_win

| Feature                        | Correlation | Type     |
|--------------------------------|-------------|----------|
| diff_profile_win_pct           | +0.376      | PROFILE  |
| diff_profile_str_acc_career    | +0.236      | PROFILE  |
| diff_age                       | -0.204      | PHYSICAL |
| diff_profile_slpm              | +0.189      | PROFILE  |
| diff_last5_won                 | +0.181      | LAST-5   |
| diff_profile_sapm              | -0.179      | PROFILE  |
| diff_profile_str_def_career    | +0.161      | PROFILE  |
| diff_last5_opp_str_landed      | -0.158      | LAST-5   |
| diff_last3_won                 | +0.151      | LAST-3   |
| diff_profile_td_def_career     | +0.132      | PROFILE  |
| diff_last5_ctrl_seconds        | +0.131      | LAST-5   |
| diff_career_win_rate           | +0.130      | CAREER   |
| diff_last5_str_acc             | +0.129      | LAST-5   |
| diff_last3_opp_str_landed      | -0.128      | LAST-3   |
| diff_win_streak                | +0.126      | ACTIVITY |
| diff_last5_td_attempted        | +0.119      | LAST-5   |
| diff_last3_ctrl_seconds        | +0.119      | LAST-3   |
| diff_profile_td_avg            | +0.115      | PROFILE  |
| diff_last5_td_landed           | +0.110      | LAST-5   |
| diff_last5_ground_landed       | +0.109      | LAST-5   |

### NaN Coverage

| Feature Group    | Features | Avg % Null | Notes                              |
|------------------|----------|------------|------------------------------------|
| Profile diffs    | 10       | ~2.4%      | Best coverage                      |
| Career diffs     | ~44      | 26.8%      | NaN when either fighter's 1st fight|
| Physical diffs   | 5        | ~1.5%      | Missing height/reach/DOB           |
| Activity diffs   | 4        | 12.9%      | NaN at 1st fight                   |
| Last-3 diffs     | 19       | 58.2%      | Needs both fighters 3+ prior fights|
| Last-5 diffs     | 18       | 74.2%      | Needs both fighters 5+ prior fights|
| Opp quality diffs| 2        | 54.8%      | Needs 2+ prior fights              |

NaN strategy: XGBoost/LightGBM/CatBoost handle NaN natively — no imputation needed. Profile features provide fallback signal for cold-start fighters.

---

## Modeling Results

### NB05 — Baseline Models (Default Hyperparameters)

**Configuration:**
- Split: Train < 2025-07-01 (5,093 fights) / Test ≥ 2025-07-01 (392 fights)
- CV: 5-fold TimeSeriesSplit
- Features: 310
- Default params: depth 5, 500 trees, lr 0.05

| Model    | CV Acc | Test Acc | Lift    | AUC   | LogLoss | Brier |
|----------|--------|----------|---------|-------|---------|-------|
| Baseline | —      | 0.554    | —       | 0.500 | 0.688   | —     |
| XGBoost  | 0.693  | 0.768    | +0.214  | 0.864 | 0.458   | 0.232 |
| LightGBM | 0.688  | 0.781    | +0.227  | 0.868 | 0.452   | 0.219 |
| CatBoost | 0.703  | 0.786    | +0.232  | 0.872 | 0.457   | 0.214 |
| Ensemble | —      | 0.773    | +0.219  | 0.871 | 0.448   | 0.146 |

### NB06 — Tuned Models (Optuna, 100 trials/model)

**Configuration:**
- Same split as NB05 (fair comparison)
- 100 trials/model, early stopping, MedianPruner
- SQLite storage for resumability
- Total tuning time: 24.7 minutes

| Model    | NB05 Acc | Tuned Acc | Δ Acc  | CV LL  | Test LL | AUC   | Brier |
|----------|----------|-----------|--------|--------|---------|-------|-------|
| XGBoost  | 0.768    | 0.791     | +0.023 | 0.5621 | 0.459   | 0.874 | 0.147 |
| LightGBM | 0.781    | 0.791     | +0.010 | 0.5669 | 0.453   | 0.885 | 0.145 |
| CatBoost | 0.786    | 0.781     | -0.005 | 0.5652 | 0.465   | 0.878 | 0.149 |
| Ensemble | 0.773    | 0.791     | +0.018 | —      | 0.458   | 0.881 | 0.146 |

**Key findings:**
- Tuning compressed model gap: all three converged to ~0.791
- XGBoost gained most (+2.3%), CatBoost defaults were already near-optimal
- Equal-weight ensemble is best (models perform identically after tuning)
- LightGBM: best AUC (0.885) and best Brier (0.145) — best probability estimates
- Production models trained on all 5,485 fights with tuned params

### Confidence Tier System

| Tier | Condition | Typical Accuracy |
|------|-----------|-----------------|
| VERY_HIGH | Unanimous + conf >= 80% | ~80%+ |
| HIGH | Unanimous + conf 65-80% | ~75% |
| MEDIUM | Unanimous + conf 55-65% | ~65% |
| LOW | Unanimous + conf < 55% | ~55% |
| NO_CONF | Models disagree (2-1 split) | ~45% (skip) |

### NB08 — Live Betting Results

**UFC 327: Prochazka vs. Ulberg (first live card)**
- 5 qualified bets from 5 fights (all numbered event main card)
- Record: 4W-1L (80%)
- Total staked: \$584.13 (58.4% of bankroll)
- P&L: +\$250.40
- ROI: +42.9%
- Only loss: Murzakanov (-245) — heaviest favorite, Costa upset

---

## Known Data Issues
- Fighter name merges can create duplicates (5493 > 5485 in stance analysis)
- "KO/TKO Punches" vs "KO/TKO Punch" are duplicates in method field
- `parse_pct` treats "0%" as NaN — intentional (0 attempts = no accuracy)
- Profile career stats are UFC's current totals (see ablation analysis in NB05)
- 7 duplicate fighter names in fighters_clean — handled by dedup + Bruno Silva → NaN
- `career_td_acc_true` has 50.2% null (many fighters never attempt takedowns)
- LGBMClassifier has no `save_model()` — use `model.booster_.save_model()` instead
- Profile-only predictions in NB07/NB08 are less reliable than full-feature predictions
- NB08 half-Kelly can exceed 100% bankroll when many +EV bets on same card (scale down or cap)

---

## Bugs Found & Fixed

| Bug | Where | Fix |
|-----|-------|-----|
| Event page lists winner first, not red corner | 01_scraper Cell 3/7 | Use fight detail page for corner assignment |
| Fighter directory doesn't have career rate stats | 01_scraper Cell 4 | Added Cell 5: follow fighter_url to profile page |
| `str.contains("NC")` matches Francisco, Duncan, Vince | 02_cleaning Cell 3 | Use direct comparison instead of regex |
| Profile values have leading `\|` from `get_text(separator='\|')` | 02_cleaning Cell 5 | Added `clean_val()` to strip `\|` |
| `parse_float` choked on `"\|3.29"` | 02_cleaning Cell 5 | `clean_val()` runs before `float()` |
| Height/reach `"--"` appears non-null | 02_cleaning Cell 5 | Explicit check in parse functions |
| Method field has embedded newlines | 02_cleaning Cell 6 | Regex `\\s*\\n+\\s*` → space |
| Data path inconsistency (`./data/` vs `./notebooks/data/`) | All notebooks | Auto-detect `./data/` or `../data/` |
| Pre-2010 red WR = 100% (broken corner data) | 02_cleaning Cell 3 | Year cutoff at 2015 |
| Raw duplicate columns kept alongside clean versions | 02_cleaning Cell 5/6 | Drop raw columns before saving |
| Profile career stats reflect current totals (leakage) | 04_feature_eng | Compute own rolling versions; profile as fallback |
| Averaging per-fight accuracy % is mathematically wrong | 04_feature_eng Cell 3 | Compute "true" accuracy from cumulative landed/attempted |
| 7 duplicate fighter names cause InvalidIndexError | 04_feature_eng Cell 7 | Dedup by most total_fights; Bruno Silva → NaN |
| `calibration_curve` import location changed in sklearn | 05_modeling Cell 1 | Import from `sklearn.calibration` not `sklearn.metrics` |
| LGBMClassifier.save_model() doesn't exist | 05_modeling Cell 16 | Use `model.booster_.save_model()` |
| Optuna trials lost on kernel crash | 06_tuning | Added SQLite storage + `load_if_exists=True` |
| Mixing model sets in NB07 causes disagreements | 07_predict | Keep model sets matched (all baseline OR all prod) |
| BFO ID-only URLs 302 redirect to homepage | 09_odds_scraper | Must use full slug URLs |
| BFO search returns max 25 results with fuzzy matching | 09_odds_scraper | Batch numbered event searches; collect URLs manually |

---

## Dependencies
requests, beautifulsoup4, pandas, numpy, matplotlib, seaborn,
scikit-learn, lxml, tqdm, ipykernel, xgboost, lightgbm, catboost,
optuna, joblib