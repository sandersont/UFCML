# UFC Fight Outcome Predictor

Predicting the outcome of UFC fights using machine learning, built on historical data scraped from [ufcstats.com](http://www.ufcstats.com).

---

## Project Overview

The goal of this project is to build a model that can predict the winner of upcoming UFC fights. We approach this by:

1. **Collecting** comprehensive historical fight data from ufcstats.com
2. **Cleaning** and processing the raw data into analysis-ready formats
3. **Exploring** the data to understand patterns and relationships
4. **Engineering** features that capture fighter strengths and matchup dynamics
5. **Training** machine learning models to predict fight outcomes
6. **Evaluating** model performance against baselines

---

## Key Discovery: Red Corner = Favorite

The UFC assigns the **red corner** to the higher-ranked or favored fighter and the **blue corner** to the underdog. Through investigation of the scraped data, we discovered:

| Source | Fighter Order | Use |
|---|---|---|
| **Event page** | Winner listed first, loser second | Getting fight stats |
| **Fight detail page** | Red corner first, blue corner second (with W/L) | Getting correct corner assignment |

We cross-reference both sources to build our final dataset where:
- **Fighter 1 (F1)** = Red corner = Favorite
- **Fighter 2 (F2)** = Blue corner = Underdog
- **Red corner win rate ≈ 55%** — this is our **naive baseline** to beat

---

## Project Structure

UFCML/
├── README.md
├── setup_project.py # Generates all notebooks
├── requirements.txt # Python dependencies
├── .gitignore
├── data/ # Scraped & processed data (not tracked in git)
│ ├── events.csv # All UFC events with dates & locations
│ ├── fights_raw.csv # Raw fights (winner-first order from event pages)
│ ├── fights.csv # Corrected fights (red corner-first from detail pages)
│ ├── fighters.csv # Fighter directory with physical attributes
│ ├── fight_details.json # Detailed round-by-round fight stats
│ ├── events_clean.csv # Cleaned events (2015+)
│ ├── fighters_clean.csv # Cleaned fighters
│ └── fights_clean.csv # Cleaned fights (2015+)
├── notebooks/
│ ├── 01_scraper.ipynb # Data collection + corner assignment fix
│ ├── 02_data_cleaning.ipynb # Data cleaning, year cutoff, feature parsing
│ └── 03_eda.ipynb # Exploratory data analysis
├── models/ # Saved models (future)
└── src/ # Utility modules (future)


---

## Data Pipeline

### Notebook 1: Scraper (`01_scraper.ipynb`)
1. Scrape all events (fixed date parsing — date is inside TD[0] with event name)
2. Scrape fights from event pages (note: winner is always listed first)
3. Scrape fighter directory (A-Z)
4. Scrape detailed fight stats (gives us true red/blue corner order + W/L status)
5. **Rebuild fights.csv** — cross-reference event page stats with detail page corner order

### Notebook 2: Data Cleaning (`02_data_cleaning.ipynb`)
1. Load raw data
2. Data quality analysis by year
3. Drop pre-2015 data (less relevant to modern UFC)
4. Clean events, fighters, fights
5. Parse strikes ("X of Y" → landed/attempted/accuracy)
6. Create target variable (`f1_win` = did red corner win?)

### Notebook 3: EDA (`03_eda.ipynb`)
1. Fight outcome distribution (finish methods, corner win rates)
2. Time trends (fights per year, red corner win rate over time)
3. Weight class analysis (fight counts, KO rates)
4. Striking analysis (winner vs loser patterns)
5. Physical attribute distributions
6. Correlation analysis
7. Key insights and data quality report

---

## Data Collected

| Dataset | Records | Description |
|---|---|---|
| **Events** | ~769 | All completed UFC events (name, date, location) |
| **Fights** | ~8,637 total / ~5,589 post-2015 | Every fight with correct corner assignment |
| **Fighters** | ~4,486 | Fighter directory (height, weight, reach, stance, record) |
| **Fight Details** | ~8,637 | Detailed per-fight stats with W/L and corner info |

### Fight Data Fields (after cleaning)

| Field | Description |
|---|---|
| `fighter_1` | 🔴 Red corner (favorite) |
| `fighter_2` | 🔵 Blue corner (underdog) |
| `winner` | Name of winner, or "Draw/NC" |
| `f1_win` | Target variable: 1 if red corner won, 0 if not |
| `f1_kd` / `f2_kd` | Knockdowns landed |
| `f1_str_landed` / `f1_str_attempted` / `f1_str_acc` | Significant strikes |
| `f1_td_landed` / `f1_td_attempted` | Takedowns |
| `f1_sub` / `f2_sub` | Submission attempts |
| `weight_class` | Division |
| `finish_type` | KO/TKO, SUB, DEC, or OTHER |
| `round` / `time_seconds` / `total_time_seconds` | When the fight ended |

---

## Setup & Reproduction

### Prerequisites
- GitHub Codespace (or any Python 3.10+ environment)
- No virtual environment needed

### Quick Start

```bash
# 1. Generate project files
python setup_project.py

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks in order:
#    notebooks/01_scraper.ipynb    → Run All (~20 min with 10 threads)
#    notebooks/02_data_cleaning.ipynb → Run All (~5 sec)
#    notebooks/03_eda.ipynb        → Run All (~10 sec)

Pipeline Status
Step	Notebook	Status	Notes
Data Collection	01_scraper.ipynb	✅ Complete	8,637 fights, correct corner assignment
Data Cleaning	02_data_cleaning.ipynb	✅ Complete	2015+ cutoff, parsed stats
EDA	03_eda.ipynb	🔄 In Progress	
Feature Engineering	TBD	⬜ Not Started	
Modeling	TBD	⬜ Not Started	
Evaluation	TBD	⬜ Not Started	
Technical Notes
Scraping Strategy
requests.Session() for connection pooling
10 concurrent threads via ThreadPoolExecutor
Checkpoint saves every 500 fight details
Full scrape completes in ~20 minutes
Data Integrity
Event page lists winner first (not corner order) — discovered and fixed
Fight detail page lists red corner first with W/L status — used for correct assignment
Cross-referenced both sources to ensure stats match correct fighters
Planned Model Baselines
Red corner baseline — always predict red corner wins (~55%)
Logistic regression — simple linear model
Random forest / XGBoost — tree-based ensembles
Neural network — if data supports it
Known Considerations
Red corner (F1) ≈ 55% win rate (baseline)
Draw/No Contest fights are rare but exist
Pre-2015 data dropped for modern relevance
Some fighters have missing physical attributes (height, reach)
License
This project is for educational and research purposes only. Fight data is sourced from ufcstats.com.

