# March Machine Learning Mania 2026

A probabilistic forecasting system for the [Kaggle March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) competition. The goal is to predict the outcome probabilities for every possible NCAA Men's and Women's basketball tournament matchup, evaluated by **Brier score** (lower is better).

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Detailed File Descriptions](#detailed-file-descriptions)
  - [Configuration](#1-srcconfigpy--configuration)
  - [Data Loading](#2-srcdata_loaderpy--data-loading--canonicalization)
  - [Data Validation](#3-srcvalidationpy--data-integrity-checks)
  - [Rating Systems](#4-srcfeatureselopy--rating-systems)
  - [Efficiency Metrics](#5-srcfeaturesefficiencypy--efficiency-metrics)
  - [Seed Features](#6-srcfeaturesseedspy--seed--tournament-features)
  - [Massey Ordinals](#7-srcfeaturesmasseypy--massey-ordinals-mens-only)
  - [Interaction Features](#8-srcfeaturesinteractionspy--interaction-features)
  - [Logistic Regression](#9-srcmodelslogisticpy--logistic-regression)
  - [LightGBM](#10-srcmodelslgbmpy--lightgbm)
  - [Calibration](#11-srccalibrationpy--probability-calibration)
  - [Ensemble](#12-srcensemblepy--model-ensembling)
  - [Submission](#13-srcsubmissionpy--submission-generation)
  - [Pipeline](#14-srcpipelinepy--main-orchestrator)
- [How to Run](#how-to-run)
  - [Prerequisites](#prerequisites)
  - [Step 1 – Install Dependencies](#step-1--install-dependencies)
  - [Step 2 – Download Competition Data](#step-2--download-competition-data)
  - [Step 3 – Run the Pipeline](#step-3--run-the-pipeline)
  - [Expected Runtime](#expected-runtime)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Outputs](#outputs)
- [Example Output](#example-output)
- [License](#license)

---

## Overview

This system runs two independent end-to-end pipelines — one for **Men's** and one for **Women's** NCAA basketball tournaments — producing a single `submission.csv` for Kaggle upload. Each pipeline follows these steps:

1. **Load & validate** historical game data from Kaggle CSV files
2. **Build canonical game records** (standardize winner/loser format so TeamA < TeamB)
3. **Compute features** — ELO ratings, efficiency metrics, seed stats, Massey rankings, interaction terms
4. **Build matchup-level training data** from historical tournament games
5. **Rolling-origin temporal cross-validation** (never train on future data)
6. **Train two models** — Elastic Net Logistic Regression + LightGBM
7. **Calibrate** predicted probabilities (isotonic, Platt, or temperature scaling)
8. **Ensemble** the two calibrated models (weighted average or Ridge stacking)
9. **Generate** the final submission CSV with clamped probabilities

---

## Repository Structure

```
.
├── competition files/           # Kaggle competition data (CSV files)
│   ├── MRegularSeasonCompactResults.csv
│   ├── MRegularSeasonDetailedResults.csv
│   ├── MNCAATourneyCompactResults.csv
│   ├── MNCAATourneySeeds.csv
│   ├── WRegularSeasonCompactResults.csv
│   ├── WNCAATourneyCompactResults.csv
│   ├── Cities.csv
│   └── ... (all Kaggle-provided CSV files)
│
├── src/                         # All source code
│   ├── __init__.py              # Makes src a Python package
│   ├── config.py                # Paths, constants, hyperparameters
│   ├── data_loader.py           # CSV loading, seed parsing, game canonicalization
│   ├── pipeline.py              # Main orchestrator / entry point
│   ├── validation.py            # Data integrity checks
│   ├── calibration.py           # Probability calibration (isotonic, Platt, temperature)
│   ├── ensemble.py              # Model ensembling (weighted avg, Ridge stacking)
│   ├── submission.py            # Submission CSV generation
│   │
│   ├── features/                # Feature engineering modules
│   │   ├── __init__.py
│   │   ├── elo.py               # ELO, Margin-ELO, Glicko ratings, momentum
│   │   ├── efficiency.py        # Offensive/defensive efficiency, Four Factors, pace
│   │   ├── seeds.py             # Seed difference, upset history, conference strength
│   │   ├── massey.py            # Massey ordinal rankings (Men's only)
│   │   └── interactions.py      # Cross-feature interactions, stability filtering
│   │
│   └── models/                  # Model training modules
│       ├── __init__.py
│       ├── logistic.py          # Elastic Net Logistic Regression
│       └── lgbm.py              # LightGBM with Optuna Bayesian HPO
│
├── outputs/                     # Generated outputs (after running pipeline)
│   ├── submission.csv           # Final Kaggle submission
│   ├── oof_predictions.csv      # Out-of-fold predictions
│   ├── feature_importance_report.md
│   ├── calibration_report.md
│   ├── full_methodology_report.md
│   └── model_artifacts/         # Serialized model objects (.pkl)
│       ├── m/                   # Men's models
│       └── w/                   # Women's models
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## Detailed File Descriptions

### 1. `src/config.py` — Configuration

The central configuration file. All other modules import paths, constants, and hyperparameters from here. Nothing is hard-coded elsewhere.

**What it defines:**

| Setting | Purpose |
|---------|---------|
| `DATA_DIR` | Path to the `competition files/` directory containing Kaggle CSVs |
| `OUTPUT_DIR` / `ARTIFACT_DIR` | Where outputs and saved models are written |
| `RANDOM_SEED` | `42` — ensures reproducibility across all random operations |
| `M_FIRST_SEASON` / `W_FIRST_SEASON` | First season with data (1985 / 1998) |
| `COVID_SEASON` | `2020` — tournament was cancelled; excluded from validation |
| `CURRENT_SEASON` | `2026` — the season we predict for (no tournament data yet) |
| `M_CV_TRAIN_CUTOFFS` / `W_CV_TRAIN_CUTOFFS` | List of season cutoffs for rolling-origin cross-validation |
| `ELO_*` / `GLICKO_*` | Rating system hyperparameters (K-factor, initial rating, season reversion, etc.) |
| `LGBM_SEARCH` | Optuna search space for LightGBM hyperparameter optimization |
| `PROB_FLOOR` / `PROB_CEIL` | `0.02` / `0.98` — clamp range for final probabilities |
| `FILES` | Dictionary mapping short names (e.g., `"m_reg_compact"`) to full CSV file paths |

**How it contributes:** Every module reads its configuration from this single file, so changing a parameter (e.g., the ELO K-factor) requires editing only one place.

---

### 2. `src/data_loader.py` — Data Loading & Canonicalization

Handles reading Kaggle CSV files and converting raw game results into a standardized format.

**Key functions:**

| Function | What it does |
|----------|-------------|
| `load_csv(key)` | Loads a CSV file by its `FILES` dictionary key |
| `parse_seed("W01")` | Parses seed strings like `"W01"` or `"X16a"` → `(region, seed_number, is_playin)` |
| `seed_to_int("X16a")` | Converts seed string to numeric (e.g., `16.5` for play-in teams) |
| `load_all(gender)` | Loads ALL relevant CSV files for Men's (`"m"`) or Women's (`"w"`) into a dictionary of DataFrames |
| `canonicalize_game(df)` | **Core function** — converts winner/loser format to canonical form where `TeamA < TeamB` by ID. This ensures every matchup has a consistent representation regardless of who won. The `target` column is `1` if TeamA won, `0` if TeamB won. Also handles location flipping (Home/Away from TeamA's perspective) and detailed box-score stats. |
| `build_all_games(data)` | Combines regular season, tournament, conference tourney, and secondary tourney games into one canonical DataFrame |
| `build_tourney_dataset(data)` | Extracts only NCAA tournament games in canonical form |
| `build_season_aggregates(all_games)` | Computes per-team, per-season statistics (games played, wins, avg points, margin, win%) |

**How it contributes:** Provides the cleaned, standardized game data that all downstream feature engineering and model training depends on. The canonical ordering (`TeamA < TeamB`) is critical because Kaggle submission IDs use this same convention.

---

### 3. `src/validation.py` — Data Integrity Checks

Runs quality checks on the loaded data to catch issues before they propagate into the model.

**Checks performed:**

1. **No duplicate games** — verifies uniqueness on `(Season, DayNum, WTeamID, LTeamID)`
2. **Winner score > Loser score** — ensures `WScore > LScore` in all game records
3. **All TeamIDs exist in master** — every team in game data appears in the teams table
4. **COVID season** — confirms no tournament games exist for 2020 (cancelled season)
5. **Current season** — confirms no tournament games for 2026 (hasn't happened yet)
6. **Seed integrity** — no duplicate `(Season, TeamID)` in seeds; every tournament team has a seed

**How it contributes:** Acts as a guard rail — if data has issues (e.g., a corrupted CSV), validation catches them early with clear warnings before the pipeline spends time computing features.

---

### 4. `src/features/elo.py` — Rating Systems

Implements four distinct rating systems, all computed strictly chronologically (no future data leakage).

**Classes:**

| Class | Description |
|-------|-------------|
| `EloRating` | Standard season-reset ELO. After each game, the winner gains points and the loser loses points proportional to K-factor (20) and expected outcome. At the start of each season, ratings carry over 75% of the previous rating and revert 25% toward the mean (1500). Handles home-court advantage (+100 for home team). |
| `MarginElo` | Same as ELO but the K-factor is scaled by a **margin-of-victory multiplier** (FiveThirtyEight-style: `log(|margin| + 1)` with autocorrelation dampening). Blowout wins produce larger rating changes. |
| `GlickoRating` | Tracks both a rating AND a **rating deviation (RD)** — a measure of uncertainty. Teams with fewer recent games have higher RD. RD increases during the off-season and decreases with each game played. |
| `MomentumTracker` | Tracks the **linear slope of ELO** over the last 5 games. A positive slope means the team is "heating up"; negative means trending down. |

**Master function:**

`compute_all_ratings(all_games)` processes every game chronologically, updating all four rating systems. Returns:
- **`snapshot_df`**: One row per `(Season, TeamID)` with post-regular-season rating values (Elo, MarginElo, Glicko, GlickoRD, Momentum)
- **`game_ratings_df`**: Game-level ratings (pre-game and post-game) for analysis

**How it contributes:** Produces the most predictive features in the system. ELO difference between two teams is the single strongest predictor of tournament outcomes.

---

### 5. `src/features/efficiency.py` — Efficiency Metrics

Computes advanced basketball analytics from detailed box-score data.

**Key functions:**

| Function | What it does |
|----------|-------------|
| `estimate_possessions(fga, oreb, to, fta)` | Dean Oliver formula: `FGA − OREB + TO + 0.475×FTA` |
| `compute_game_stats(all_games)` | From canonical games with box-score columns, computes per-game stats: Offensive Efficiency, Defensive Efficiency, Pace, eFG%, Turnover Rate, Offensive Rebound %, Free Throw Rate, and opponent versions |
| `compute_rolling_features(game_stats)` | For each team-season, computes exponentially-weighted moving averages over windows of 5, 10, and full season. Recent games are weighted more heavily. |
| `compute_adjusted_efficiency(game_stats)` | **Iterative opponent adjustment** (5 rounds): a team's offensive efficiency is adjusted based on how good/bad their opponents' defenses were, and vice versa. Converges to "how good would this team be against an average opponent?" |
| `compute_sos(game_stats, elo_snapshot)` | **Strength of Schedule**: average opponent ELO across the season |
| `compute_points_only_features(season_agg)` | Fallback for seasons without detailed box-score data — derives proxy features from scoring averages only |
| `build_efficiency_features(...)` | Master function that orchestrates all of the above and merges results |

**How it contributes:** Captures HOW teams win (efficient shooting vs. rebounding vs. pace), not just that they won. These features help the model understand team styles and predict matchup-specific outcomes.

---

### 6. `src/features/seeds.py` — Seed & Tournament Features

Builds features related to tournament seeding and historical matchup patterns.

**Key functions:**

| Function | What it does |
|----------|-------------|
| `build_seed_features(...)` | Returns a dictionary containing: |
| | • `seed_lookup`: Maps `(Season, TeamID)` → numeric seed |
| | • `upset_table`: For every `(FavSeed, DogSeed)` pairing, the historical win rate of the favorite |
| | • `conf_strength`: Average ELO of teams in each conference per season |
| | • `tourney_appearances`: Count of prior tournament appearances per team |
| `get_upset_prior(upset_table, seed_a, seed_b)` | Returns the historical probability that the better-seeded team wins for a given seed pairing (e.g., 1-vs-16 historically ~99%) |
| `get_tourney_experience(tourney_appearances, tid, season)` | Number of prior tournament appearances for a team before a given season |

**How it contributes:** Seeds are the most widely-known predictor in March Madness. The upset table adds historical base rates (e.g., 5-vs-12 upsets happen ~35% of the time). Conference strength captures whether a team plays in a strong or weak conference.

---

### 7. `src/features/massey.py` — Massey Ordinals (Men's only)

Reads the massive `MMasseyOrdinals.csv` file (millions of rows) containing computer rankings from dozens of ranking systems.

**Key functions:**

| Function | What it does |
|----------|-------------|
| `load_massey_for_season(season)` | Reads the CSV in 500k-row chunks, filtering to selected systems and pre-tournament dates only (DayNum ≤ 133) |
| `build_massey_features(seasons)` | For each season, takes the LAST available ranking per system per team, pivots to wide format, and computes a composite mean rank |
| `get_massey_diff_features(massey_df, team_a, team_b, season)` | Computes rank difference for a specific matchup |

**Top ranking systems used:** POM (Pomeroy), SAG (Sagarin), MOR (Moore), DOL (Dolphin), COL (Colley)

**How it contributes:** These computer rankings aggregate information (margin, schedule, opponents) that may not be fully captured by our own ELO/efficiency features. Only available for Men's tournaments.

---

### 8. `src/features/interactions.py` — Interaction Features

Creates non-linear feature combinations and performs feature stability analysis.

**Key functions:**

| Function | What it does |
|----------|-------------|
| `add_interaction_features(matchup_df)` | Adds computed columns: |
| | • `Elo_x_Seed`: ELO difference × seed difference |
| | • `AdjEM_x_Pace`: Adjusted efficiency margin × pace difference |
| | • `UpsetProne`: Binary indicator for large seed differential (≥5) |
| | • `UnderdogVolatility`: Margin standard deviation of the lower-seeded team |
| | • `RatingUncertainty`: Absolute Glicko RD difference |
| `analyze_feature_stability(matchup_df, feature_cols)` | Drops features with high cross-season coefficient of variation or frequent sign flips — keeps only stable, reliable features |

**How it contributes:** Interaction terms capture non-linear relationships (e.g., a large ELO gap matters MORE when the seed gap is also large). Stability filtering removes noisy features that would hurt generalization.

---

### 9. `src/models/logistic.py` — Logistic Regression

A regularized logistic regression using scikit-learn's `LogisticRegressionCV`.

**Key functions:**

| Function | What it does |
|----------|-------------|
| `build_logistic_model()` | Creates a `StandardScaler` → `LogisticRegressionCV` pipeline with elastic net penalty (L1+L2), searching over 7 L1 ratios and 20 regularization strengths via 5-fold CV |
| `train_logistic(X_train, y_train)` | Fits the model, returns the pipeline + metadata (chosen C, l1_ratio, coefficients) |
| `predict_logistic(model, X)` | Returns P(TeamA wins) for each matchup |

**How it contributes:** Provides a strong, interpretable baseline model. The elastic net penalty performs automatic feature selection (L1 zeroes out irrelevant features). Coefficients show which features matter most.

---

### 10. `src/models/lgbm.py` — LightGBM

A gradient-boosted tree model with Bayesian hyperparameter optimization via Optuna.

**Key functions:**

| Function | What it does |
|----------|-------------|
| `optimize_lgbm(X_train, y_train)` | Runs 60–80 Optuna trials with TPE sampler, each trial doing 5-fold stratified CV with early stopping on Brier score. Searches over: num_leaves, max_depth, learning_rate, min_child_samples, regularization, subsample, colsample_bytree |
| `train_lgbm(X_train, y_train, ...)` | Trains a LightGBM model with given parameters, optional validation set for early stopping, returns model + feature importances (by gain) |
| `predict_lgbm(model, X)` | Returns predicted probabilities |
| `_lgb_brier_eval(y_pred, dtrain)` | Custom evaluation metric: Brier score (lower = better) |

**How it contributes:** LightGBM captures non-linear relationships and feature interactions that logistic regression misses. The Bayesian HPO finds near-optimal hyperparameters efficiently.

---

### 11. `src/calibration.py` — Probability Calibration

Ensures predicted probabilities are well-calibrated (a prediction of 0.7 should mean ~70% of those games are won).

**Methods implemented:**

| Method | Description |
|--------|-------------|
| **Isotonic Regression** | Non-parametric monotonic function fit to map raw predictions → calibrated probabilities |
| **Platt Scaling** | Fits a logistic regression on the raw logits (log-odds of predictions) |
| **Temperature Scaling** | Learns a single temperature parameter T to divide logits by, sharpening or softening predictions |

**Selection process:** `select_calibration()` cross-validates all three methods (plus "no calibration") with 5-fold CV, picks the method with the lowest average Brier score, then refits on all data.

**How it contributes:** Models often produce over-confident or under-confident probabilities. Calibration corrects this, directly improving Brier score.

---

### 12. `src/ensemble.py` — Model Ensembling

Combines predictions from Logistic Regression and LightGBM into a single, stronger prediction.

**Methods:**

| Method | Description |
|--------|-------------|
| **Weighted Average** | Grid-searches the optimal weight `w` such that `blend = w × LR + (1-w) × LGB` minimizes Brier score. For >2 models, uses scipy constrained optimization |
| **Ridge Stacking** | Trains a Ridge regression meta-learner that takes both models' predictions as inputs and learns optimal combination coefficients |

**Selection:** `select_best_ensemble()` compares both methods by Brier score and fold-level variance, picks the better one.

**How it contributes:** Ensembling almost always improves over individual models because each model captures different patterns. The weighted average or stacking learns the optimal combination.

---

### 13. `src/submission.py` — Submission Generation

Generates the final CSV file for Kaggle upload.

**Key functions:**

| Function | What it does |
|----------|-------------|
| `parse_submission_ids(sub_df)` | Parses Kaggle submission IDs (`"2026_1101_1102"`) into Season, TeamA, TeamB, Gender columns |
| `build_matchup_features_for_submission(...)` | Computes ALL features for a single matchup using only pre-tournament data. Called once per submission row. |
| `generate_submission(...)` | Orchestrates: build features → predict → clamp to [0.02, 0.98] → validate → save CSV |

**How it contributes:** The final step that turns model predictions into the format Kaggle requires. The `[0.02, 0.98]` clamping prevents extreme probabilities that would be severely punished by Brier score.

---

### 14. `src/pipeline.py` — Main Orchestrator

The entry point that ties everything together. Runs the full end-to-end pipeline.

**Key functions:**

| Function | What it does |
|----------|-------------|
| `build_matchup_features(...)` | For each historical tournament game, builds the complete feature vector by calling `build_matchup_features_for_submission()` + `add_interaction_features()` |
| `get_feature_cols(matchup_df)` | Extracts feature column names, excluding identifiers, targets, and redundant raw values (keeps `_diff` columns) |
| `seed_baseline(matchup_df)` | Computes a baseline Brier score using only historical seed-pairing win rates |
| `rolling_origin_cv(...)` | **Core CV loop**: for each cutoff season S, trains on seasons ≤ S, validates on S+1 tournament. Collects out-of-fold predictions from both models. Also runs LightGBM Optuna HPO on the full training set. |
| `run_gender_pipeline(gender)` | **Full pipeline for one gender**: Load → Validate → Features → CV → Calibrate → Ensemble → Save artifacts. Returns everything needed for submission. |
| `generate_final_submission(artifacts_m, artifacts_w)` | Uses trained models from both pipelines to predict on the 2026 submission matchups |
| `save_oof_predictions(...)` / `save_reports(...)` | Saves CSV and markdown reports to `outputs/` |
| `main()` | Entry point: runs Men's pipeline, Women's pipeline, generates submission, saves reports |

**How it contributes:** Orchestrates the entire system from raw data to final submission. This is the only file you need to run.

---

## How to Run

### Prerequisites

- **Python 3.9+** (tested with 3.10, 3.11, 3.12)
- **~8 GB RAM** (the Massey ordinals file is large; LightGBM HPO runs many trials)
- **Kaggle account** to download the competition data

### Step 1 — Install Dependencies

```bash
pip install numpy pandas scikit-learn lightgbm optuna scipy joblib
```

### Step 2 — Download Competition Data

1. Go to [Kaggle March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data)
2. Download the dataset (or use the Kaggle CLI: `kaggle competitions download -c march-machine-learning-mania-2026`)
3. Extract all CSV files into the `competition files/` directory at the repository root:

```
competition files/
├── MRegularSeasonCompactResults.csv
├── MRegularSeasonDetailedResults.csv
├── MNCAATourneyCompactResults.csv
├── MNCAATourneyDetailedResults.csv
├── MNCAATourneySeeds.csv
├── MNCAATourneySlots.csv
├── MNCAATourneySeedRoundSlots.csv
├── MSeasons.csv
├── MTeams.csv
├── MTeamCoaches.csv
├── MTeamConferences.csv
├── MMasseyOrdinals.csv
├── MConferenceTourneyGames.csv
├── MSecondaryTourneyCompactResults.csv
├── MSecondaryTourneyTeams.csv
├── MTeamSpellings.csv
├── MGameCities.csv
├── WRegularSeasonCompactResults.csv
├── WRegularSeasonDetailedResults.csv
├── WNCAATourneyCompactResults.csv
├── WNCAATourneyDetailedResults.csv
├── WNCAATourneySeeds.csv
├── WNCAATourneySlots.csv
├── WSeasons.csv
├── WTeams.csv
├── WTeamConferences.csv
├── WConferenceTourneyGames.csv
├── WSecondaryTourneyCompactResults.csv
├── WSecondaryTourneyTeams.csv
├── WTeamSpellings.csv
├── WGameCities.csv
├── Conferences.csv
├── Cities.csv
├── SampleSubmissionStage1.csv
└── SampleSubmissionStage2.csv
```

### Step 3 — Run the Pipeline

From the repository root:

```bash
python -m src.pipeline
```

This will:
1. Train models for both Men's and Women's tournaments
2. Save model artifacts to `outputs/model_artifacts/`
3. Write `outputs/submission.csv` (ready for Kaggle upload)
4. Write OOF predictions and markdown reports to `outputs/`

### Expected Runtime

| Stage | Approximate Time |
|-------|-----------------|
| Data loading & validation | ~5 seconds |
| ELO / rating computation | ~30 seconds |
| Efficiency features | ~1 minute |
| Massey ordinals (Men's) | ~1–2 minutes |
| Matchup feature building | ~1 minute |
| LightGBM Optuna HPO | ~10–15 minutes |
| Rolling-origin CV | ~5–10 minutes |
| Calibration + Ensemble | ~30 seconds |
| Submission generation | ~2 minutes |
| **Total** | **~20–30 minutes** |

---

## Pipeline Walkthrough

Here is what happens when you run `python -m src.pipeline`:

```
Step  1  →  config.py defines all paths and constants
Step  2  →  data_loader.py reads all CSV files for Men's
Step  3  →  validation.py checks data integrity
Step  4  →  data_loader.py canonicalizes all games (TeamA < TeamB)
Step  5  →  elo.py computes ELO, Margin-ELO, Glicko, Momentum for every game
Step  6  →  efficiency.py computes offensive/defensive efficiency, Four Factors
Step  7  →  seeds.py builds seed features, upset table, conference strength
Step  8  →  massey.py loads computer rankings (Men's only)
Step  9  →  pipeline.py + submission.py builds matchup-level feature vectors
Step 10  →  interactions.py adds cross-feature interactions
Step 11  →  interactions.py filters unstable features
Step 12  →  pipeline.py computes seed-only baseline Brier score
Step 13  →  pipeline.py runs rolling-origin CV:
               → logistic.py trains Elastic Net LR per fold
               → lgbm.py runs Optuna HPO, trains LightGBM per fold
Step 14  →  calibration.py selects and fits best calibration method per model
Step 15  →  ensemble.py selects weighted-avg vs stacking, optimizes blend
Step 16  →  logistic.py + lgbm.py retrain final models on ALL historical data
Step 17  →  pipeline.py saves model artifacts (.pkl files)
             ─── Repeat Steps 2–17 for Women's ───
Step 18  →  submission.py + pipeline.py generates predictions for 2026 matchups
Step 19  →  pipeline.py saves submission.csv, OOF predictions, markdown reports
```

---

## Outputs

After running the pipeline, the `outputs/` directory contains:

| File | Description |
|------|-------------|
| `submission.csv` | Final Kaggle submission — two columns: `ID` (e.g., `2026_1101_1102`) and `Pred` (probability TeamA wins). All predictions clamped to [0.02, 0.98]. |
| `oof_predictions.csv` | Out-of-fold predictions for both genders. Contains raw LR/LGB predictions, calibrated predictions, and ensemble predictions per tournament game. Useful for diagnosing model performance. |
| `feature_importance_report.md` | Top-30 feature importances for LightGBM (by gain) and Logistic Regression (by absolute coefficient), per gender. Shows which features drive predictions. |
| `calibration_report.md` | Fold-level Brier scores for each cross-validation fold, seed baseline comparison, calibration slope. |
| `full_methodology_report.md` | Detailed summary of the entire methodology with final results. |
| `model_artifacts/m/` | Serialized Men's models: `logistic.pkl`, `lgbm.pkl`, `cal_lr.pkl`, `cal_lgb.pkl`, `ensemble.pkl` |
| `model_artifacts/w/` | Serialized Women's models (same files) |

---

## Example Output

Running the validation demo (with synthetic data) confirms all components work:

```
============================================================
  1. Configuration
============================================================
  ✓ DATA_DIR is a Path
  ✓ RANDOM_SEED is 42
  ✓ Prob clamp range valid
  ✓ COVID season not in M CV cutoffs (as val)
  ✓ COVID season not in W CV cutoffs (as val)

============================================================
  2. Data Loader
============================================================
  ✓ parse_seed W01 → region=W, seed=1, not playin
  ✓ parse_seed X16a → region=X, seed=16, is playin
  ✓ seed_to_int W01 = 1.0
  ✓ seed_to_int X16a = 16.5
  ✓ canonicalize: TeamA < TeamB enforced
  ✓ canonicalize: target values correct

============================================================
  3. ELO / Rating Systems
============================================================
  ✓ ELO initial rating = 1500
  ✓ ELO winner increases, loser decreases, zero-sum
  ✓ ELO season reset moves toward mean
  ✓ Margin ELO responds to victory margin
  ✓ Glicko rating + RD tracking works
  ✓ Momentum detects rising ELO trend

============================================================
  4. Efficiency / Seeds / Interactions / Calibration / Ensemble
============================================================
  ✓ Possession estimate reasonable (50–90 range)
  ✓ Offensive/Defensive efficiency computed
  ✓ Upset prior defaults to 0.5 for unknown pairings
  ✓ Interaction features (Elo×Seed, UpsetProne) created
  ✓ Isotonic, Platt, and Temperature calibration all produce [0,1] output
  ✓ Ensemble weights sum to 1.0

============================================================
  5. Models
============================================================
  ✓ Logistic Regression produces valid probabilities
  ✓ LightGBM produces valid probabilities
  ✓ Both models achieve Brier < 0.3 on synthetic data

  RESULTS: 68/68 checks passed, 0 failed
  ✅ All validation checks passed!
```

---

## License

This project is licensed under the [MIT License](LICENSE).
