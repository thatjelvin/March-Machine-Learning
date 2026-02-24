"""
Configuration for March Machine Learning Mania 2026 forecasting system.
All paths, constants, and hyperparameter defaults.
"""

import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "march-machine-learning-mania-2026"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ARTIFACT_DIR = OUTPUT_DIR / "model_artifacts"

# ─── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Season boundaries ───────────────────────────────────────────────────────
M_FIRST_SEASON = 1985
W_FIRST_SEASON = 1998
M_DETAILED_START = 2003
W_DETAILED_START = 2010
COVID_SEASON = 2020          # tournament cancelled
CURRENT_SEASON = 2026        # partial – no tournament yet

# Stage 1 backtesting seasons (tournaments occurred)
STAGE1_SEASONS = [2022, 2023, 2024, 2025]

# Rolling-origin CV folds: train ≤ S, validate on S+1 tournament
# Skip 2020 (no tourney) – we never validate on 2020
M_CV_TRAIN_CUTOFFS = [y for y in range(2008, 2025) if y != 2019]
# (validate on 2009..2025 excl 2020)
W_CV_TRAIN_CUTOFFS = [y for y in range(2012, 2025) if y != 2019]

# ─── ELO defaults ────────────────────────────────────────────────────────────
ELO_INITIAL = 1500.0
ELO_K = 20.0
ELO_SEASON_REVERT = 0.75        # revert toward mean each season
ELO_HOME_ADV = 100.0            # points added for home team
MARGIN_ELO_K = 24.0

# Glicko
GLICKO_INITIAL_RD = 350.0
GLICKO_MIN_RD = 50.0
GLICKO_RD_INCREASE_PER_SEASON = 100.0
GLICKO_Q = 0.00575646273  # ln(10)/400

# Momentum window
MOMENTUM_WINDOW = 5

# ─── Efficiency ──────────────────────────────────────────────────────────────
EFF_ROLLING_SPANS = [5, 10, 9999]   # 9999 = full season
ITERATIVE_ADJ_ROUNDS = 5

# ─── Probability clamp ──────────────────────────────────────────────────────
PROB_FLOOR = 0.02
PROB_CEIL = 0.98

# ─── File mappings ───────────────────────────────────────────────────────────
def _p(name):
    return DATA_DIR / name

FILES = {
    # ── Men's ──
    "m_reg_compact": _p("MRegularSeasonCompactResults.csv"),
    "m_reg_detailed": _p("MRegularSeasonDetailedResults.csv"),
    "m_tourney_compact": _p("MNCAATourneyCompactResults.csv"),
    "m_tourney_detailed": _p("MNCAATourneyDetailedResults.csv"),
    "m_seeds": _p("MNCAATourneySeeds.csv"),
    "m_slots": _p("MNCAATourneySlots.csv"),
    "m_seed_round_slots": _p("MNCAATourneySeedRoundSlots.csv"),
    "m_seasons": _p("MSeasons.csv"),
    "m_teams": _p("MTeams.csv"),
    "m_coaches": _p("MTeamCoaches.csv"),
    "m_conferences": _p("MTeamConferences.csv"),
    "m_massey": _p("MMasseyOrdinals.csv"),
    "m_conf_tourney": _p("MConferenceTourneyGames.csv"),
    "m_secondary_results": _p("MSecondaryTourneyCompactResults.csv"),
    "m_secondary_teams": _p("MSecondaryTourneyTeams.csv"),
    "m_spellings": _p("MTeamSpellings.csv"),
    "m_game_cities": _p("MGameCities.csv"),
    # ── Women's ──
    "w_reg_compact": _p("WRegularSeasonCompactResults.csv"),
    "w_reg_detailed": _p("WRegularSeasonDetailedResults.csv"),
    "w_tourney_compact": _p("WNCAATourneyCompactResults.csv"),
    "w_tourney_detailed": _p("WNCAATourneyDetailedResults.csv"),
    "w_seeds": _p("WNCAATourneySeeds.csv"),
    "w_slots": _p("WNCAATourneySlots.csv"),
    "w_seasons": _p("WSeasons.csv"),
    "w_teams": _p("WTeams.csv"),
    "w_conferences": _p("WTeamConferences.csv"),
    "w_conf_tourney": _p("WConferenceTourneyGames.csv"),
    "w_secondary_results": _p("WSecondaryTourneyCompactResults.csv"),
    "w_secondary_teams": _p("WSecondaryTourneyTeams.csv"),
    "w_spellings": _p("WTeamSpellings.csv"),
    "w_game_cities": _p("WGameCities.csv"),
    # ── Shared ──
    "conferences": _p("Conferences.csv"),
    "cities": _p("Cities.csv"),
    # ── Submission ──
    "sub_stage1": _p("SampleSubmissionStage1.csv"),
    "sub_stage2": _p("SampleSubmissionStage2.csv"),
}

# ─── LightGBM search space (Optuna) ─────────────────────────────────────────
LGBM_SEARCH = {
    "num_leaves": (8, 64),
    "max_depth": (3, 8),
    "learning_rate": (0.01, 0.1),
    "min_child_samples": (10, 100),
    "reg_alpha": (0.0, 5.0),
    "reg_lambda": (0.0, 5.0),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.5, 1.0),
}
LGBM_N_TRIALS = 80
LGBM_EARLY_STOP = 50
