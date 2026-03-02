# March Machine Learning Mania 2026

A probabilistic forecasting system for the [Kaggle March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) competition. The goal is to predict the win probabilities for every possible NCAA Men's and Women's basketball tournament matchup, evaluated by Brier score.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Methodology](#methodology)
  - [Feature Engineering](#feature-engineering)
  - [Models](#models)
  - [Calibration](#calibration)
  - [Ensemble](#ensemble)
  - [Temporal Validation](#temporal-validation)
- [Outputs](#outputs)
- [License](#license)

---

## Overview

Two independent end-to-end pipelines are run — one for Men's and one for Women's tournaments — each following these steps:

1. Load and validate historical data
2. Build canonical game records
3. Compute rating and efficiency features
4. Build matchup-level training data from historical tournaments
5. Rolling-origin temporal cross-validation
6. Train Logistic Regression + LightGBM models
7. Calibrate predicted probabilities
8. Ensemble model predictions
9. Generate the final Kaggle submission CSV

---

## Repository Structure

```
.
├── march-machine-learning-mania-2026/   # Kaggle competition data (CSV files)
│   ├── MRegularSeasonCompactResults.csv
│   ├── MNCAATourneyCompactResults.csv
│   ├── MNCAATourneySeeds.csv
│   ├── WRegularSeasonCompactResults.csv
│   ├── WNCAATourneyCompactResults.csv
│   └── ... (all Kaggle-provided data files)
├── src/
│   ├── config.py          # Paths, constants, and hyperparameter defaults
│   ├── data_loader.py     # Data loading, schema validation, matchup canonicalization
│   ├── pipeline.py        # Main orchestrator (entry point)
│   ├── backtest_2025.py   # 5-year backtesting for 2025 predictions
│   ├── validation.py      # Data integrity checks
│   ├── calibration.py     # Probability calibration methods
│   ├── ensemble.py        # Model ensembling (weighted average, stacking)
│   ├── submission.py      # Submission file generation
│   ├── features/
│   │   ├── elo.py         # ELO, margin-ELO, and Glicko rating systems
│   │   ├── efficiency.py  # Offensive/defensive efficiency and Four Factors
│   │   ├── seeds.py       # Seed features and historical upset rates
│   │   ├── massey.py      # Massey ordinal rankings (Men's only)
│   │   └── interactions.py # Engineered interaction features
│   └── models/
│       ├── logistic.py    # Elastic Net Logistic Regression
│       └── lgbm.py        # LightGBM with Optuna Bayesian HPO
├── outputs/
│   ├── submission.csv                  # Final Kaggle submission
│   ├── oof_predictions.csv             # Out-of-fold predictions
│   ├── feature_importance_report.md    # LightGBM and LR feature importances
│   ├── calibration_report.md           # Fold-level Brier scores
│   ├── full_methodology_report.md      # Detailed methodology summary
│   └── model_artifacts/                # Saved model objects (.pkl)
├── LICENSE
└── README.md
```

---

## Setup

**Python 3.9+** is required. Install dependencies with:

```bash
pip install numpy pandas scikit-learn lightgbm optuna scipy joblib
```

Place the competition data files downloaded from Kaggle into the `march-machine-learning-mania-2026/` directory.

---

## Usage

Run the full pipeline from the repository root:

```bash
python -m src.pipeline
```

This will:
- Train models for both Men's and Women's tournaments
- Save model artifacts to `outputs/model_artifacts/`
- Write `outputs/submission.csv` (ready for Kaggle upload)
- Write OOF predictions and markdown reports to `outputs/`

### 2025 Backtesting (5-Year Window)

To evaluate model accuracy using only the past 5 years of data (2021-2024, excluding 2020 due to COVID) to predict and compare against actual 2025 tournament results:

```bash
python -m src.backtest_2025
```

This backtesting pipeline:
1. Uses only data from 2020-2024 (effectively 2021-2024 due to COVID cancellation)
2. Trains models on this limited 5-year dataset
3. Predicts 2025 tournament matchup outcomes
4. Compares predictions against actual 2025 results
5. Reports comprehensive accuracy metrics including:
   - **Brier Score**: Mean squared error of probability predictions
   - **Log Loss**: Cross-entropy loss
   - **Accuracy**: Percentage of correct win/loss predictions
   - **Expected Calibration Error (ECE)**: Measures how well predicted probabilities align with observed outcomes
   - **Upset Analysis**: How well the model predicts upsets vs favorites

Output files are saved to `outputs/backtest_2025/`:
- `predictions_m.csv`: Men's game-by-game predictions vs actuals
- `predictions_w.csv`: Women's game-by-game predictions vs actuals

---

## Methodology

### Feature Engineering

**Rating Systems** (`src/features/elo.py`)
- Season-reset ELO (K=20, 75% reversion to mean each season)
- Margin-of-victory adjusted ELO (FiveThirtyEight-style MOV multiplier)
- Simplified Glicko (tracks rating + rating deviation)
- Momentum (linear slope of ELO over the last 5 games)

**Efficiency Metrics** (`src/features/efficiency.py`)
- Offensive and defensive efficiency (points per 100 possessions)
- Four Factors: effective FG%, turnover rate, offensive rebounding %, free-throw rate
- Opponent-adjusted efficiency (5 iterative rounds)
- Pace and Strength of Schedule
- Exponentially weighted rolling windows (spans: 5, 10, full season)

**Seed Features** (`src/features/seeds.py`)
- Seed difference between matchup opponents
- Historical upset frequency by seed pairing
- Tournament experience and conference strength index

**Massey Ordinals** (`src/features/massey.py`) — Men's only
- Pre-tournament computer rankings from top systems (POM, SAG, MOR, DOL, COL)
- Mean ordinal difference across ranking systems

**Interaction Features** (`src/features/interactions.py`)
- ELO difference × seed difference
- Adjusted efficiency × pace
- Upset-prone indicator (|seed diff| ≥ 5)
- Underdog volatility and rating uncertainty

### Models

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Elastic Net (L1+L2), standardized features, path search over regularization strength |
| **LightGBM** | Gradient boosted trees; hyperparameters tuned with Bayesian optimization via Optuna (80 trials, 5-fold CV, early stopping on Brier score) |

### Calibration

Three calibration methods are evaluated per model using cross-validated Brier score; the best is automatically selected:
- Isotonic regression
- Platt scaling (logistic on raw logits)
- Temperature scaling (single learnable parameter)

Predictions are clamped to `[0.02, 0.98]` to avoid overconfident extremes.

### Ensemble

Predictions from both calibrated models are combined using one of:
- **Weighted average** — weights optimized by grid search
- **Ridge stacking** — Ridge regression meta-learner

The method with the lowest mean Brier score and lowest fold variance is selected automatically.

### Temporal Validation

Rolling-origin cross-validation is used to respect the time ordering of seasons:
- Train on all seasons ≤ S, validate on the season S+1 tournament
- **Men's**: ~15 folds covering 2009–2025 (excluding 2020, when the tournament was cancelled)
- **Women's**: ~12 folds covering 2013–2025 (excluding 2020)

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/submission.csv` | Final submission file for Kaggle (ID, Pred) |
| `outputs/oof_predictions.csv` | Out-of-fold predictions for both genders |
| `outputs/feature_importance_report.md` | Top feature importances for LightGBM and LR coefficients |
| `outputs/calibration_report.md` | Fold-level Brier scores and calibration slope per gender |
| `outputs/full_methodology_report.md` | Full methodology summary with final results |
| `outputs/model_artifacts/` | Serialized model, calibration, and ensemble objects |
| `outputs/backtest_2025/predictions_m.csv` | 2025 Men's backtesting predictions vs actuals |
| `outputs/backtest_2025/predictions_w.csv` | 2025 Women's backtesting predictions vs actuals |

---

## License

This project is licensed under the [MIT License](LICENSE).
