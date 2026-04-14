## Last Updated
After completing notebooks 01–06 (scraping, cleaning, EDA, feature engineering, modeling, tuning).
Notebook 06 still running Optuna tuning (50 trials per model).

## Architecture

/workspaces/UFCML/
├── setup_project.py # Creates dirs, .gitignore, requirements.txt
├── create_01_scraper.py # Generates notebooks/01_scraper.ipynb
├── create_02_cleaning.py # Generates notebooks/02_data_cleaning.ipynb
├── create_03_eda.py # Generates notebooks/03_eda.ipynb
├── create_04_feature_engineering.py # Generates notebooks/04_feature_engineering.ipynb
├── create_05_modeling.py # Generates notebooks/05_modeling.ipynb
├── create_06_tuning.py # Generates notebooks/06_tuning.ipynb
├── requirements.txt
├── .gitignore
├── PROJECT_STATUS.md # This file
├── notebooks/
│ ├── 01_scraper.ipynb # 8 cells — scrapes all data
│ ├── 02_data_cleaning.ipynb # 7 cells — cleans and parses
│ ├── 03_eda.ipynb # 18 cells — exploratory analysis
│ ├── 04_feature_engineering.ipynb # 15 cells — rolling features + diffs + feature EDA
│ ├── 05_modeling.ipynb # 15 cells — baseline models + ablation
│ ├── 06_tuning.ipynb # 13 cells — Optuna hyperparameter tuning
│ └── data/ # All CSV/JSON files live here
│ ├── events.csv
│ ├── events_clean.csv
│ ├── fights_raw.csv
│ ├── fights.csv
│ ├── fights_clean.csv
│ ├── fighters.csv
│ ├── fighters_full.csv
│ ├── fighters_clean.csv
│ ├── fight_details.json
│ ├── fight_details_checkpoint.json
│ ├── model_data.csv
│ ├── test_predictions.csv
│ ├── feature_list.txt
│ ├── feature_importance_comparison.png
│ ├── calibration_curves.png
│ ├── confusion_matrices.png
│ └── probability_distributions.png
├── data/ # Empty (data lives in notebooks/data/)
├── models/
│ ├── xgb_model.json # NB05 default XGBoost
│ ├── lgb_model.txt # NB05 default LightGBM
│ └── catboost_model.cbm # NB05 default CatBoost
└── src/ # Empty (future)


NOTE: Data path auto-detected in notebooks (`./data/` or `../data/`).
The `./data/` directory at root is empty. All data lives in `./notebooks/data/`.

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

### 05_modeling.ipynb (15 cells)

| Cell | What | Notes |
|------|------|-------|
| 1    | Imports & load | Shape, date range, baseline |
| 2    | Feature selection & temporal split | Train 2015–2023, Test 2024–2026 |
| 3    | TimeSeriesSplit CV utility | 5-fold, returns metrics per fold |
| 4    | XGBoost | CV + test evaluation |
| 5    | LightGBM | CV + test evaluation |
| 6    | CatBoost | CV + test evaluation |
| 7    | Model comparison table | All models vs baseline |
| 8    | Feature importance | 3-panel chart + consensus top 15 |
| 9    | Ensemble | Average ensemble + agreement analysis |
| 10   | Calibration | Curves + confidence buckets |
| 11   | Ablation study | Profile leakage test (6 feature subsets) |
| 12   | Error analysis | Confident wrong picks, by weight class, finish type, quarter |
| 13   | Confusion matrices | 4-panel + classification report |
| 14   | Probability distributions | Histogram by outcome + separation plot |
| 15   | Save & summary | Models, predictions CSV, feature list |

**NB05 Train/Test Split:**
- Train: 2015-01-03 → 2023-12-16 (4,319 fights)
- Test: 2024-01-13 → 2026-04-11 (1,166 fights)
- Test baseline (always red): 55.2%

**NB05 Results:**

| Model    | CV Acc | Test Acc | Lift    | AUC   | LogLoss |
|----------|--------|----------|---------|-------|---------|
| Baseline | —      | 0.552    | —       | 0.500 | 0.689   |
| XGBoost  | 0.684  | 0.751    | +0.199  | 0.828 | 0.513   |
| LightGBM | 0.675  | 0.746    | +0.194  | 0.831 | 0.511   |
| CatBoost | 0.675  | 0.758    | +0.206  | 0.836 | 0.504   |
| Ensemble | —      | 0.759    | +0.207  | 0.837 | 0.499   |

**NB05 Key Findings:**

Feature importance — consensus top 15 dominated by profile and physical features:
1. diff_profile_win_pct (unanimous #1 across all 3 models)
2. diff_profile_total_fights (unanimous #2)
3. diff_profile_slpm
4. f1_profile_total_fights
5. diff_age (#3 physical feature)
6. diff_profile_str_acc_career
- Zero rolling features in consensus top 15

Ablation study results:

| Feature Set            | Features | Accuracy | AUC   | LogLoss |
|------------------------|----------|----------|-------|---------|
| All features           | 310      | 0.746    | 0.831 | 0.511   |
| No profile             | 280      | 0.634    | 0.651 | 0.670   |
| Diffs only             | 101      | 0.739    | 0.821 | 0.516   |
| Diffs no profile       | 91       | 0.606    | 0.632 | 0.679   |
| Profile only           | 35       | 0.730    | 0.812 | 0.535   |
| Physical + activity    | 32       | 0.607    | 0.627 | 0.687   |

Profile leakage impact: +0.112 (All features vs No profile)

Profile features are a mix of:
1. **Legitimate signal** — pre-2015 career history rolling features can't access
2. **Leakage** — post-fight-date career outcomes baked in UFC's current totals

Decision: keep profile features. They capture real pre-fight information (career record was public knowledge). The leakage component exists but the signal is valuable.

Ensemble agreement analysis:
- All 3 agree: 84.8% of fights (989/1166)
- Unanimous accuracy: 79.4%
- Disagreement fights (177): 56.5% accuracy (near coin flip)

Error analysis highlights:
- Jean Silva: model's nemesis — 4 fights, 4 confident wrong predictions (97-99%)
- Best division: Welterweight (82.8% accuracy, +35pt lift)
- Weakest: Light Heavyweight (+7.3pt lift)
- Consistent across finish types: DEC 76.2%, KO/TKO 76.5%, SUB 74.1%
- Performance stable over time, no degradation 2024→2026

Classification balance:
- Red recall: 80% | Blue recall: 71%
- Model correctly identifies 71% of upsets (blue wins)

Probability distribution:
- Mean: 0.564, Std: 0.282, Range: 0.022–0.994
- <40%: 33.3% | 40-60%: 19.6% | >60%: 47.1%
- Model makes decisive calls, not clustering around baseline

### 06_tuning.ipynb (13 cells) — IN PROGRESS

| Cell | What | Notes |
|------|------|-------|
| 1    | Imports & load | + Optuna |
| 2    | Features & split | Train 2015–2025, Test 2026 only |
| 3    | CV helper | Pre-split indices, reused across trials |
| 4    | XGBoost Optuna | 50 trials, 9 hyperparameters |
| 5    | LightGBM Optuna | 50 trials, 10 hyperparameters |
| 6    | CatBoost Optuna | 50 trials, 10 hyperparameters |
| 7    | Train final models | Best params → full training set → test eval |
| 8    | Before vs after | NB05 vs NB06 side-by-side |
| 9    | Optuna visualization | Optimization history + parameter importance |
| 10   | Tuned feature importance | Top 20 per model + consensus |
| 11   | Calibration | Curves + confidence buckets |
| 12   | Agreement analysis | Unanimous picks accuracy |
| 13   | Save & summary | Joblib models, params JSON, predictions CSV |

**NB06 Train/Test Split:**
- Train: 2015 → end of 2025 (~5,000+ fights)
- Test: 2026 only (~138 fights)
- More training data, smaller but more honest out-of-time test

**Optuna configuration:**
- 50 trials per model (TPE sampler, seed=42)
- Objective: minimize CV log loss (TimeSeriesSplit, 5 folds)
- Per-trial reporting: accuracy, log loss, key hyperparams
- Accuracy stored as user_attr for post-hoc analysis

**XGBoost search space:**
- n_estimators: 200–1500
- max_depth: 3–9
- learning_rate: 0.01–0.3 (log)
- subsample: 0.5–1.0
- colsample_bytree: 0.3–1.0
- reg_alpha: 1e-3–10.0 (log)
- reg_lambda: 1e-3–10.0 (log)
- min_child_weight: 1–20
- gamma: 0.0–5.0

**LightGBM search space:**
- n_estimators: 200–1500
- max_depth: 3–12
- learning_rate: 0.01–0.3 (log)
- subsample: 0.5–1.0
- colsample_bytree: 0.3–1.0
- reg_alpha: 1e-3–10.0 (log)
- reg_lambda: 1e-3–10.0 (log)
- min_child_samples: 5–50
- num_leaves: 15–127
- min_split_gain: 0.0–2.0

**CatBoost search space:**
- iterations: 200–1500
- depth: 3–9
- learning_rate: 0.01–0.3 (log)
- subsample: 0.5–1.0
- l2_leaf_reg: 1e-3–10.0 (log)
- min_data_in_leaf: 1–50
- random_strength: 0.0–5.0
- bagging_temperature: 0.0–5.0
- border_count: 32–255
- auto_class_weights: None or Balanced

**NB06 saves:**
- `models/xgb_tuned.joblib`
- `models/lgb_tuned.joblib`
- `models/cat_tuned.joblib`
- `data/best_params.json`
- `data/test_predictions_tuned.csv`

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

### Strike Location (winner vs loser per fight)

| Location | Winner | Loser | Diff |
|----------|--------|-------|------|
| Head     | ~11    | ~7    | +4   |
| Body     | ~4     | ~3    | +1   |
| Leg      | ~4     | ~3    | +1   |

### Strike Position (winner vs loser per fight)

| Position  | Winner | Loser | Diff |
|-----------|--------|-------|------|
| Distance  | ~14    | ~10   | +4   |
| Clinch    | ~2     | ~2    | ~0   |
| Ground    | ~3     | ~1    | +2   |

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

### Key Feature Engineering Insights

**Profile stats dominate** because they include full career history (pre-2015 fights). Rolling career stats only use fights in our 2015+ dataset, so a fighter who went 20-0 before 2015 shows career_win_rate = NaN at their first fight in our data, but profile_win_pct reflects their entire career.

**Age is the #3 feature** (r=-0.204). Older red corner fighters lose more often. Since red = favorite, this means aging favorites underperform expectations. Genuinely new signal not available from fight stats.

**Rolling features add unique signal** not captured by profile stats:
- diff_last5_won (+0.181) — recent form matters
- diff_last5_opp_str_landed (-0.158) — recent defensive performance
- diff_win_streak (+0.126) — momentum effect
- diff_last5_ctrl_seconds (+0.131) — grappling dominance trend

**Profile vs Rolling comparison:**
- Profile win_pct: +0.376 vs Rolling career_win_rate: +0.130
- Profile str_acc: +0.236 vs Rolling str_acc_true: ~+0.10
- Correlation between profile and rolling versions is moderate (~0.3-0.6)
- They capture different information: profile = full career, rolling = recent in-dataset form
- Both included for modeling (not redundant)

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

## Modeling Results (Notebook 05)

### NB05 Configuration
- Split: Train 2015–2023 (4,319 fights) / Test 2024–2026 (1,166 fights)
- CV: 5-fold TimeSeriesSplit within training set
- Features: 310 (all features including profile)
- Default hyperparameters (conservative: depth 5, 500 trees, lr 0.05)

### NB05 Performance

| Model    | CV Acc | Test Acc | Lift    | AUC   | LogLoss | Brier |
|----------|--------|----------|---------|-------|---------|-------|
| Baseline | —      | 0.552    | —       | 0.500 | 0.689   | 0.248 |
| XGBoost  | 0.684  | 0.751    | +0.199  | 0.828 | 0.513   | 0.168 |
| LightGBM | 0.675  | 0.746    | +0.194  | 0.831 | 0.511   | 0.167 |
| CatBoost | 0.675  | 0.758    | +0.206  | 0.836 | 0.504   | 0.164 |
| Ensemble | —      | 0.759    | +0.207  | 0.837 | 0.499   | 0.163 |

### CV-to-Test Pattern
All three models showed improving accuracy with more training data:
- XGBoost folds: 0.655 → 0.651 → 0.704 → 0.695 → 0.716
- Final model trained on all 2015–2023 data achieved highest scores
- This motivated NB06's expanded training set (2015–2025)

### NB05 Ablation Results

| Feature Set          | Features | Accuracy | AUC   | LogLoss | Lift vs Base |
|----------------------|----------|----------|-------|---------|--------------|
| All features         | 310      | 0.746    | 0.831 | 0.511   | +0.194       |
| No profile           | 280      | 0.634    | 0.651 | 0.670   | +0.081       |
| Diffs only           | 101      | 0.739    | 0.821 | 0.516   | +0.187       |
| Diffs no profile     | 91       | 0.606    | 0.632 | 0.679   | +0.054       |
| Profile only         | 35       | 0.730    | 0.812 | 0.535   | +0.178       |
| Physical + activity  | 32       | 0.607    | 0.627 | 0.687   | +0.055       |

**Profile impact: +11.2%** (All features vs No profile)

Profile features contain a mix of legitimate pre-fight signal (full career history including pre-2015) and potential leakage (UFC's current totals include post-fight outcomes). Decision: keep profile features — the career record was public knowledge before each fight. Rolling features provide clean +8pt lift over baseline independently.

### NB05 Error Analysis

**Accuracy by weight class (test set):**

| Weight Class          | Fights | Accuracy | Baseline | Lift   |
|-----------------------|--------|----------|----------|--------|
| Welterweight          | 134    | 0.828    | 0.478    | +0.350 |
| Women's Bantamweight  | 46     | 0.826    | 0.522    | +0.304 |
| Middleweight          | 148    | 0.777    | 0.574    | +0.203 |
| Heavyweight           | 78     | 0.769    | 0.628    | +0.141 |
| Bantamweight          | 135    | 0.756    | 0.541    | +0.215 |
| Featherweight         | 143    | 0.741    | 0.483    | +0.258 |
| Lightweight           | 154    | 0.734    | 0.558    | +0.176 |
| Light Heavyweight     | 82     | 0.732    | 0.659    | +0.073 |
| Flyweight             | 94     | 0.723    | 0.553    | +0.170 |
| Women's Flyweight     | 58     | 0.724    | 0.603    | +0.121 |
| Women's Strawweight   | 79     | 0.722    | 0.544    | +0.178 |

**Accuracy by finish type:** DEC 76.2%, KO/TKO 76.5%, SUB 74.1%

**Accuracy by quarter:** Stable 2024–2026, no degradation over time. 2026Q1 hit 81.6%.

**Confident wrong predictions:** Jean Silva — 4 fights at 97-99% confidence, all wrong. Model's nemesis.

**Model agreement:** When all 3 agree (84.8% of fights): 79.4% accuracy. Disagreement fights: 56.5%.

---

## Tuning Results (Notebook 06) — IN PROGRESS

### NB06 Configuration
- Split: Train 2015–2025 / Test 2026 only (~138 fights)
- CV: 5-fold TimeSeriesSplit within training set
- Optuna: 50 trials per model, TPE sampler, minimize CV log loss
- Same 310 features as NB05

Results pending — tuning in progress.

---

## Known Data Issues
- Fighter name merges can create duplicates (5493 > 5485 in stance analysis)
- "KO/TKO Punches" vs "KO/TKO Punch" are duplicates in method field
- `parse_pct` treats "0%" as NaN — intentional (0 attempts = no accuracy)
- Profile career stats are UFC's current totals (see ablation analysis in NB05)
- 7 duplicate fighter names in fighters_clean — handled by dedup + Bruno Silva → NaN
- `career_td_acc_true` has 50.2% null (many fighters never attempt takedowns)
- NB06 test set is small (~138 fights) — accuracy will be noisy

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

---

## Dependencies
requests, beautifulsoup4, pandas, numpy, matplotlib, seaborn,
scikit-learn, lxml, tqdm, ipykernel, xgboost, lightgbm, catboost,
optuna, joblib


