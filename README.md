# UFCML

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
6. **Evaluating** model performance against baselines (e.g., always picking the red corner favorite)

### Key Insight: Red Corner = Favorite

The UFC assigns the **red corner** to the higher-ranked or favored fighter and the **blue corner** to the underdog. In our data:
- **Fighter 1 (f1)** = Red corner = Favorite
- **Fighter 2 (f2)** = Blue corner = Underdog

This means the red corner win rate (~55-60%) serves as our **naive baseline** — any useful model must beat it.

---

## Project Structure

UFCML/
├── README.md
├── setup_project.py # Script to generate all notebooks
├── requirements.txt # Python dependencies
├── .gitignore
├── data/ # Scraped & processed data (not tracked in git)
│ ├── events.csv # All UFC events
│ ├── fights.csv # All fights with summary stats
│ ├── fighters.csv # Fighter directory with physical attributes
│ ├── fight_details.json # Detailed round-by-round fight stats
│ ├── events_clean.csv # Cleaned events
│ ├── fighters_clean.csv # Cleaned fighters
│ └── fights_clean.csv # Cleaned fights
├── notebooks/
│ ├── 01_scraper.ipynb # ✅ Data collection from ufcstats.com
│ ├── 02_data_cleaning.ipynb # Data cleaning & processing
│ └── 03_eda.ipynb # Exploratory data analysis
├── models/ # Saved models (future)
└── src/ # Utility modules (future)


---

## Data Collected

Scraped on **$(date)** from ufcstats.com using 10 concurrent threads.

| Dataset | Records | Description |
|---|---|---|
| **Events** | 769 | All completed UFC events (name, date, location) |
| **Fights** | 8,637 | Every fight with winner, method, round, time, and summary stats |
| **Fighters** | 4,486 | Fighter directory (height, weight, reach, stance, record) |
| **Fight Details** | 8,637 | Detailed per-fight stats (round-by-round strikes, takedowns, etc.) |

### Fight Data Fields

| Field | Description |
|---|---|
| `fighter_1` / `fighter_2` | Red corner (favorite) / Blue corner (underdog) |
| `winner` | Name of the winning fighter, or "Draw/NC" |
| `f1_kd` / `f2_kd` | Knockdowns landed |
| `f1_str` / `f2_str` | Significant strikes (landed of attempted) |
| `f1_td` / `f2_td` | Takedowns (landed of attempted) |
| `f1_sub` / `f2_sub` | Submission attempts |
| `weight_class` | Division (e.g., Lightweight, Welterweight) |
| `method` | Win method (KO/TKO, Submission, Decision) |
| `round` | Round the fight ended |
| `time` | Time in the round the fight ended |

### Fighter Data Fields

| Field | Description |
|---|---|
| `first_name` / `last_name` | Fighter name |
| `height` / `weight` / `reach` | Physical attributes |
| `stance` | Orthodox, Southpaw, Switch |
| `wins` / `losses` / `draws` | Career record |

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

# 3. Run notebooks in order
#    Open notebooks/01_scraper.ipynb → Run All (~15-20 min)
#    Open notebooks/02_data_cleaning.ipynb → Run All (~5 sec)
#    Open notebooks/03_eda.ipynb → Run All (~10 sec)

dependencies

requests
beautifulsoup4
pandas
numpy
matplotlib
seaborn
scikit-learn
lxml
tqdm
ipykernel

Pipeline Status
Step	Notebook	Status	Notes
Data Collection	01_scraper.ipynb	✅ Complete	8,637 fights, 4,486 fighters scraped
Data Cleaning	02_data_cleaning.ipynb	🔄 In Progress	
EDA	03_eda.ipynb	⬜ Not Started	
Feature Engineering	TBD	⬜ Not Started	
Modeling	TBD	⬜ Not Started	
Evaluation	TBD	⬜ Not Started	
Technical Notes
Scraping Strategy
Uses requests.Session() for connection pooling
10 concurrent threads via concurrent.futures.ThreadPoolExecutor
Checkpoint saves every 500 fight details to prevent data loss
Full scrape completes in ~15-20 minutes
Planned Model Baselines
Red corner baseline — always predict Fighter 1 (red corner / favorite) wins
Logistic regression — simple linear model on fight stats
Random forest / XGBoost — tree-based ensemble methods
Neural network — if data supports it
Known Considerations
Red corner (Fighter 1) is typically the UFC-designated favorite
Draw/No Contest fights exist but are rare
Early UFC events (pre-2001) have sparser stats
Some fighters have missing physical attribute data (height, reach)

License
This project is for educational and research purposes only. Fight data is sourced from ufcstats.com.


