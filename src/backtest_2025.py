"""
Backtesting script for 2025 tournament predictions using only the past 5 years of data.

This script:
1. Uses only data from 2020-2024 (actually 2021-2024 due to COVID cancellation in 2020)
2. Trains models on this limited dataset  
3. Predicts 2025 tournament matchup outcomes
4. Compares predictions against actual 2025 results
5. Reports accuracy metrics (Brier score, accuracy, log loss, etc.)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict
from pathlib import Path

from src.config import (
    FILES, OUTPUT_DIR, ARTIFACT_DIR, RANDOM_SEED,
    COVID_SEASON, CURRENT_SEASON, PROB_FLOOR, PROB_CEIL,
    M_DETAILED_START, W_DETAILED_START,
    BACKTEST_TARGET_YEAR, BACKTEST_YEARS_BACK, BACKTEST_START_YEAR, BACKTEST_TRAIN_SEASONS,
)
from src.data_loader import (
    load_all, load_csv, build_all_games, build_tourney_dataset,
    build_season_aggregates, seed_to_int,
)
from src.validation import validate_data
from src.features.elo import compute_all_ratings
from src.features.efficiency import build_efficiency_features
from src.features.seeds import (
    build_seed_features, get_upset_prior, get_tourney_experience,
)
from src.features.interactions import add_interaction_features, analyze_feature_stability
from src.models.logistic import train_logistic, predict_logistic
from src.models.lgbm import optimize_lgbm, train_lgbm, predict_lgbm
from src.calibration import select_calibration, apply_calibration
from src.ensemble import select_best_ensemble, optimize_weights_grid
from src.submission import (
    parse_submission_ids, build_matchup_features_for_submission,
)

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════════════════════
#  Build matchup-level features for tournament games
# ═══════════════════════════════════════════════════════════════════════════════

def build_matchup_features(tourney_games: pd.DataFrame,
                            elo_snap: pd.DataFrame,
                            eff_features: pd.DataFrame,
                            seed_info: dict,
                            conferences_map: pd.DataFrame,
                            massey_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    For each historical tournament matchup, build feature vector.
    Returns DataFrame with features + target.
    """
    rows = []
    seed_lookup = seed_info["seed_lookup"]
    upset_table = seed_info["upset_table"]
    conf_strength = seed_info["conf_strength"]
    tourney_appearances = seed_info["tourney_appearances"]

    for _, game in tourney_games.iterrows():
        season = int(game["Season"])
        team_a = int(game["TeamA"])
        team_b = int(game["TeamB"])
        target = int(game["target"])

        feats = build_matchup_features_for_submission(
            team_a, team_b, season,
            elo_snap, eff_features, seed_lookup,
            conf_strength, conferences_map,
            upset_table, tourney_appearances,
            massey_df=massey_df,
        )
        feats["Season"] = season
        feats["TeamA"] = team_a
        feats["TeamB"] = team_b
        feats["target"] = target
        rows.append(feats)

    df = pd.DataFrame(rows)

    # Add interaction features
    df = add_interaction_features(df)

    return df


def get_feature_cols(matchup_df: pd.DataFrame) -> list:
    """Extract feature columns (everything except identifiers and target)."""
    exclude = {"Season", "TeamA", "TeamB", "target", "SeedNum_A", "SeedNum_B",
               "StdMargin_A", "StdMargin_B", "Elo_A", "Elo_B",
               "MarginElo_A", "MarginElo_B", "Glicko_A", "Glicko_B",
               "GlickoRD_A", "GlickoRD_B", "Momentum_A", "Momentum_B"}
    cols = [c for c in matchup_df.columns if c not in exclude]
    return cols


# ═══════════════════════════════════════════════════════════════════════════════
#  Seed-only baseline
# ═══════════════════════════════════════════════════════════════════════════════

def seed_baseline(matchup_df: pd.DataFrame) -> float:
    """Predict using seed-pairing historical win rates. Return Brier."""
    preds = matchup_df["UpsetPrior"].values.copy()
    actual_preds = np.where(
        matchup_df["SeedNum_A"] <= matchup_df["SeedNum_B"],
        preds,
        1 - preds
    )
    same = matchup_df["SeedNum_A"] == matchup_df["SeedNum_B"]
    actual_preds[same] = 0.5

    brier = np.mean((matchup_df["target"].values - actual_preds) ** 2)
    return brier


# ═══════════════════════════════════════════════════════════════════════════════
#  Accuracy metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute various accuracy metrics for probability predictions.
    """
    # Brier score
    brier = np.mean((y_true - y_pred) ** 2)
    
    # Log loss
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    log_loss = -np.mean(y_true * np.log(y_pred_clipped) + 
                        (1 - y_true) * np.log(1 - y_pred_clipped))
    
    # Accuracy (using 0.5 threshold)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    accuracy = np.mean(y_pred_binary == y_true)
    
    # Calibration: expected calibration error (binned)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_pred[mask].mean()
            ece += mask.sum() / len(y_true) * np.abs(bin_acc - bin_conf)
    
    return {
        "brier_score": brier,
        "log_loss": log_loss,
        "accuracy": accuracy,
        "expected_calibration_error": ece,
        "n_samples": len(y_true),
        "n_correct": int(np.sum(y_pred_binary == y_true)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main backtesting pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest_pipeline(gender: str) -> dict:
    """
    Run backtesting for 2025 predictions using only past 5 years (2020-2024).
    
    Pipeline:
    1. Load all data
    2. Filter to use only 2020-2024 data for training (2021-2024 due to COVID)
    3. Compute features using only this limited data
    4. Train models
    5. Predict 2025 tournament results
    6. Compare with actual 2025 results
    """
    g = gender.lower()
    
    print(f"\n{'#'*60}")
    print(f"  BACKTEST PIPELINE: {gender.upper()}")
    print(f"  Training on: {BACKTEST_TRAIN_SEASONS}")
    print(f"  Predicting: {BACKTEST_TARGET_YEAR}")
    print(f"{'#'*60}")

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    t0 = time.time()
    data = load_all(g)
    print(f"    Loaded in {time.time()-t0:.1f}s")

    # ── 2. Validate ───────────────────────────────────────────────────────
    print("\n[2] Validating data...")
    validate_data(data, g)

    # ── 3. Build canonical games (using ALL historical data for rating systems) ──
    # Note: Ratings need historical data to be meaningful, but we'll only use
    # recent years for training the matchup prediction models
    print("\n[3] Building canonical game datasets...")
    all_games = build_all_games(data)
    tourney_games = build_tourney_dataset(data)
    season_agg = build_season_aggregates(all_games)
    print(f"    Total games: {len(all_games)}, Tournament: {len(tourney_games)}")

    # ── 4. Compute ratings (using all historical data for continuity) ─────
    print("\n[4] Computing ELO / Glicko / Margin ratings...")
    t0 = time.time()
    elo_snap, game_ratings = compute_all_ratings(all_games)
    print(f"    Computed for {len(elo_snap)} team-seasons in {time.time()-t0:.1f}s")

    # ── 5. Compute efficiency features ────────────────────────────────────
    print("\n[5] Computing efficiency features...")
    t0 = time.time()
    eff_features = build_efficiency_features(all_games, elo_snap, season_agg)
    print(f"    Efficiency features: {eff_features.shape} in {time.time()-t0:.1f}s")

    # ── 6. Seed features ─────────────────────────────────────────────────
    print("\n[6] Building seed & tournament features...")
    seed_info = build_seed_features(
        data["seeds"], tourney_games, elo_snap, data["conferences_map"]
    )

    # ── 7. Massey ordinals (men's only) ──────────────────────────────────
    massey_df = pd.DataFrame()
    if g == "m":
        print("\n[7] Loading Massey ordinals...")
        t0 = time.time()
        try:
            from src.features.massey import build_massey_features
            massey_seasons = [s for s in range(2003, CURRENT_SEASON)]
            massey_df = build_massey_features(massey_seasons)
            print(f"    Massey features: {massey_df.shape} in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"    ⚠ Massey loading failed: {e}. Proceeding without.")
            massey_df = pd.DataFrame()
    else:
        print("\n[7] Skipping Massey ordinals (women's)")

    # ── 8. Build matchup-level features for TRAINING data ─────────────────
    # Only use 2021-2024 tournament games for training
    print(f"\n[8] Building matchup features for training ({BACKTEST_TRAIN_SEASONS})...")
    t0 = time.time()
    
    train_tourney = tourney_games[tourney_games["Season"].isin(BACKTEST_TRAIN_SEASONS)]
    print(f"    Training tournament games: {len(train_tourney)}")
    
    train_matchup_df = build_matchup_features(
        train_tourney, elo_snap, eff_features, seed_info,
        data["conferences_map"], massey_df
    )
    print(f"    Training matchup features: {train_matchup_df.shape} in {time.time()-t0:.1f}s")

    # ── 9. Build matchup features for TEST data (2025) ────────────────────
    print(f"\n[9] Building matchup features for testing ({BACKTEST_TARGET_YEAR})...")
    t0 = time.time()
    
    test_tourney = tourney_games[tourney_games["Season"] == BACKTEST_TARGET_YEAR]
    print(f"    Test tournament games: {len(test_tourney)}")
    
    if len(test_tourney) == 0:
        print(f"    ⚠ No tournament games found for {BACKTEST_TARGET_YEAR}!")
        print(f"    Available seasons: {sorted(tourney_games['Season'].unique())}")
        return None
    
    test_matchup_df = build_matchup_features(
        test_tourney, elo_snap, eff_features, seed_info,
        data["conferences_map"], massey_df
    )
    print(f"    Test matchup features: {test_matchup_df.shape} in {time.time()-t0:.1f}s")

    # ── 10. Feature stability analysis ────────────────────────────────────
    print("\n[10] Analyzing feature stability...")
    candidate_cols = get_feature_cols(train_matchup_df)
    stable_cols = analyze_feature_stability(train_matchup_df, candidate_cols)
    stable_cols = [c for c in stable_cols if c not in
                   {"Season", "TeamA", "TeamB", "target"}]
    print(f"    Stable feature count: {len(stable_cols)}")

    # ── 11. Seed baseline ────────────────────────────────────────────────
    print("\n[11] Computing seed-only baseline for 2025...")
    if "UpsetPrior" in test_matchup_df.columns and "SeedNum_A" in test_matchup_df.columns:
        baseline_brier = seed_baseline(test_matchup_df)
        print(f"    Seed-only baseline Brier: {baseline_brier:.4f}")
    else:
        baseline_brier = None
        print("    ⚠ Cannot compute seed baseline (missing columns)")

    # ── 12. Train models on 5-year data ──────────────────────────────────
    print(f"\n[12] Training models on {len(train_matchup_df)} samples from {BACKTEST_TRAIN_SEASONS}...")
    
    X_train = train_matchup_df[stable_cols].fillna(0.0)
    y_train = train_matchup_df["target"].values
    
    # LightGBM HPO
    print("    Running LightGBM Bayesian HPO...")
    lgb_params = optimize_lgbm(X_train, y_train, n_trials=40, n_folds=4)
    
    # Train Logistic Regression
    print("    Training Logistic Regression...")
    lr_model, lr_info = train_logistic(X_train, y_train)
    lr_train_preds = predict_logistic(lr_model, X_train)
    train_brier_lr = np.mean((y_train - lr_train_preds) ** 2)
    print(f"    LR training Brier: {train_brier_lr:.4f}")
    
    # Train LightGBM
    print("    Training LightGBM...")
    lgb_model, lgb_imp = train_lgbm(X_train, y_train, params=lgb_params, num_boost_round=300)
    lgb_train_preds = predict_lgbm(lgb_model, X_train)
    train_brier_lgb = np.mean((y_train - lgb_train_preds) ** 2)
    print(f"    LGB training Brier: {train_brier_lgb:.4f}")

    # ── 13. Calibrate predictions ────────────────────────────────────────
    print("\n[13] Calibrating predictions...")
    cal_method_lr, cal_lr = select_calibration(y_train, lr_train_preds)
    cal_method_lgb, cal_lgb = select_calibration(y_train, lgb_train_preds)
    print(f"    LR calibration method: {cal_method_lr}")
    print(f"    LGB calibration method: {cal_method_lgb}")

    # ── 14. Build ensemble ───────────────────────────────────────────────
    print("\n[14] Building ensemble...")
    lr_train_cal = apply_calibration(cal_method_lr, cal_lr, lr_train_preds)
    lgb_train_cal = apply_calibration(cal_method_lgb, cal_lgb, lgb_train_preds)
    
    # Simple weighted average (grid search for best weights)
    best_weight_lr = 0.5
    best_brier = float('inf')
    for w in np.arange(0.0, 1.05, 0.1):
        ens_pred = w * lr_train_cal + (1 - w) * lgb_train_cal
        brier = np.mean((y_train - ens_pred) ** 2)
        if brier < best_brier:
            best_brier = brier
            best_weight_lr = w
    print(f"    Best ensemble weights: LR={best_weight_lr:.2f}, LGB={1-best_weight_lr:.2f}")

    # ── 15. Predict 2025 tournament results ──────────────────────────────
    print(f"\n[15] Predicting {BACKTEST_TARGET_YEAR} tournament results...")
    
    # Ensure test data has same columns
    for col in stable_cols:
        if col not in test_matchup_df.columns:
            test_matchup_df[col] = 0.0
    X_test = test_matchup_df[stable_cols].fillna(0.0)
    y_test = test_matchup_df["target"].values
    
    # Raw predictions
    lr_preds = predict_logistic(lr_model, X_test)
    lgb_preds = predict_lgbm(lgb_model, X_test)
    
    # Calibrated predictions
    lr_preds_cal = apply_calibration(cal_method_lr, cal_lr, lr_preds)
    lgb_preds_cal = apply_calibration(cal_method_lgb, cal_lgb, lgb_preds)
    
    # Ensemble prediction
    ensemble_preds = best_weight_lr * lr_preds_cal + (1 - best_weight_lr) * lgb_preds_cal
    ensemble_preds = np.clip(ensemble_preds, PROB_FLOOR, PROB_CEIL)

    # ── 16. Compute accuracy metrics ─────────────────────────────────────
    print(f"\n[16] Computing accuracy metrics for {BACKTEST_TARGET_YEAR}...")
    
    metrics_lr = compute_metrics(y_test, lr_preds_cal)
    metrics_lgb = compute_metrics(y_test, lgb_preds_cal)
    metrics_ens = compute_metrics(y_test, ensemble_preds)

    # ── 17. Print results ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS: {gender.upper()} - {BACKTEST_TARGET_YEAR}")
    print(f"  Training on: {BACKTEST_TRAIN_SEASONS}")
    print(f"{'='*60}")
    
    print(f"\n  Number of 2025 tournament games: {len(test_matchup_df)}")
    
    if baseline_brier is not None:
        print(f"\n  Seed-only baseline:")
        print(f"    Brier Score: {baseline_brier:.4f}")
    
    print(f"\n  Logistic Regression:")
    print(f"    Brier Score: {metrics_lr['brier_score']:.4f}")
    print(f"    Log Loss: {metrics_lr['log_loss']:.4f}")
    print(f"    Accuracy: {metrics_lr['accuracy']:.1%} ({metrics_lr['n_correct']}/{metrics_lr['n_samples']})")
    print(f"    ECE: {metrics_lr['expected_calibration_error']:.4f}")
    
    print(f"\n  LightGBM:")
    print(f"    Brier Score: {metrics_lgb['brier_score']:.4f}")
    print(f"    Log Loss: {metrics_lgb['log_loss']:.4f}")
    print(f"    Accuracy: {metrics_lgb['accuracy']:.1%} ({metrics_lgb['n_correct']}/{metrics_lgb['n_samples']})")
    print(f"    ECE: {metrics_lgb['expected_calibration_error']:.4f}")
    
    print(f"\n  Ensemble (LR={best_weight_lr:.2f}, LGB={1-best_weight_lr:.2f}):")
    print(f"    Brier Score: {metrics_ens['brier_score']:.4f}")
    print(f"    Log Loss: {metrics_ens['log_loss']:.4f}")
    print(f"    Accuracy: {metrics_ens['accuracy']:.1%} ({metrics_ens['n_correct']}/{metrics_ens['n_samples']})")
    print(f"    ECE: {metrics_ens['expected_calibration_error']:.4f}")
    
    if baseline_brier is not None:
        improvement = baseline_brier - metrics_ens['brier_score']
        print(f"\n  Improvement over seed baseline: {improvement:.4f} ({improvement/baseline_brier:.1%})")

    # ── 18. Game-by-game predictions ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  GAME-BY-GAME PREDICTIONS vs ACTUALS")
    print(f"{'='*60}")
    
    results_df = test_matchup_df[["Season", "TeamA", "TeamB", "target"]].copy()
    results_df["pred_lr"] = lr_preds_cal
    results_df["pred_lgb"] = lgb_preds_cal
    results_df["pred_ensemble"] = ensemble_preds
    results_df["pred_binary"] = (ensemble_preds >= 0.5).astype(int)
    results_df["correct"] = (results_df["pred_binary"] == results_df["target"]).astype(int)
    
    # Add seed info if available
    if "SeedNum_A" in test_matchup_df.columns:
        results_df["SeedA"] = test_matchup_df["SeedNum_A"].values
        results_df["SeedB"] = test_matchup_df["SeedNum_B"].values
    
    print("\n  Sample predictions (first 20 games):")
    print("  " + "-"*80)
    print(f"  {'TeamA':>6} {'TeamB':>6} {'SeedA':>5} {'SeedB':>5} {'Pred':>6} {'Actual':>6} {'Correct':>7}")
    print("  " + "-"*80)
    
    for idx, row in results_df.head(20).iterrows():
        seed_a = row.get('SeedA', '?')
        seed_b = row.get('SeedB', '?')
        if isinstance(seed_a, float):
            seed_a = f"{seed_a:.0f}" if not np.isnan(seed_a) else "?"
        if isinstance(seed_b, float):
            seed_b = f"{seed_b:.0f}" if not np.isnan(seed_b) else "?"
        correct_marker = "✓" if row['correct'] else "✗"
        print(f"  {int(row['TeamA']):>6} {int(row['TeamB']):>6} {seed_a:>5} {seed_b:>5} "
              f"{row['pred_ensemble']:.3f} {int(row['target']):>6} {correct_marker:>7}")

    # ── 19. Upset analysis ───────────────────────────────────────────────
    if "SeedA" in results_df.columns:
        print(f"\n{'='*60}")
        print(f"  UPSET ANALYSIS")
        print(f"{'='*60}")
        
        # Define upset: lower seed (higher number) beats higher seed
        results_df["expected_winner"] = np.where(
            results_df["SeedA"] <= results_df["SeedB"], 0, 1
        )  # 0 = TeamA expected, 1 = TeamB expected (based on seed)
        results_df["is_upset"] = (results_df["target"] != results_df["expected_winner"]).astype(int)
        
        total_upsets = results_df["is_upset"].sum()
        total_games = len(results_df)
        print(f"\n  Total upsets in 2025: {total_upsets}/{total_games} ({total_upsets/total_games:.1%})")
        
        # How many upsets did we correctly predict?
        upset_games = results_df[results_df["is_upset"] == 1]
        if len(upset_games) > 0:
            upset_pred_correct = (upset_games["pred_binary"] == upset_games["target"]).sum()
            print(f"  Upsets correctly predicted: {upset_pred_correct}/{len(upset_games)} ({upset_pred_correct/len(upset_games):.1%})")
        
        # How many favorites did we correctly predict?
        fav_games = results_df[results_df["is_upset"] == 0]
        if len(fav_games) > 0:
            fav_pred_correct = (fav_games["pred_binary"] == fav_games["target"]).sum()
            print(f"  Favorites correctly predicted: {fav_pred_correct}/{len(fav_games)} ({fav_pred_correct/len(fav_games):.1%})")

    # ── 20. Save results ─────────────────────────────────────────────────
    print(f"\n[17] Saving results...")
    backtest_dir = OUTPUT_DIR / "backtest_2025"
    os.makedirs(backtest_dir, exist_ok=True)
    
    results_path = backtest_dir / f"predictions_{g}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"    Saved predictions: {results_path}")
    
    # Save summary
    summary = {
        "gender": g,
        "target_year": BACKTEST_TARGET_YEAR,
        "training_seasons": BACKTEST_TRAIN_SEASONS,
        "n_train_games": len(train_matchup_df),
        "n_test_games": len(test_matchup_df),
        "baseline_brier": baseline_brier,
        "lr_brier": metrics_lr['brier_score'],
        "lgb_brier": metrics_lgb['brier_score'],
        "ensemble_brier": metrics_ens['brier_score'],
        "ensemble_accuracy": metrics_ens['accuracy'],
        "ensemble_log_loss": metrics_ens['log_loss'],
        "lr_weight": best_weight_lr,
        "lgb_weight": 1 - best_weight_lr,
    }
    
    return {
        "summary": summary,
        "results_df": results_df,
        "metrics_lr": metrics_lr,
        "metrics_lgb": metrics_lgb,
        "metrics_ens": metrics_ens,
    }


def main():
    """Run backtesting for both Men's and Women's tournaments."""
    start = time.time()
    
    print(f"\n{'#'*60}")
    print(f"  2025 TOURNAMENT BACKTESTING")
    print(f"  Using only data from past 5 years: {BACKTEST_TRAIN_SEASONS}")
    print(f"{'#'*60}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = {}
    
    # Run for Men's
    try:
        results["m"] = run_backtest_pipeline("m")
    except Exception as e:
        print(f"\n⚠ Men's backtest failed: {e}")
        import traceback
        traceback.print_exc()
        results["m"] = None
    
    # Run for Women's
    try:
        results["w"] = run_backtest_pipeline("w")
    except Exception as e:
        print(f"\n⚠ Women's backtest failed: {e}")
        import traceback
        traceback.print_exc()
        results["w"] = None
    
    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'#'*60}")
    
    for gender, res in results.items():
        if res is None:
            print(f"\n  {gender.upper()}: Failed to complete")
            continue
            
        s = res["summary"]
        print(f"\n  {gender.upper()}:")
        print(f"    Training games: {s['n_train_games']}")
        print(f"    Test games: {s['n_test_games']}")
        print(f"    Seed baseline Brier: {s['baseline_brier']:.4f}" if s['baseline_brier'] else "    Seed baseline: N/A")
        print(f"    Ensemble Brier: {s['ensemble_brier']:.4f}")
        print(f"    Ensemble Accuracy: {s['ensemble_accuracy']:.1%}")
    
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  BACKTEST COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    main()
