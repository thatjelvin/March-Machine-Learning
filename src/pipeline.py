"""
Pipeline orchestrator: end-to-end execution of the forecasting system.
  1. Load & validate data
  2. Build canonical games
  3. Compute all features (ELO, efficiency, seeds, Massey, interactions)
  4. Build matchup-level training data (historical tournament matchups)
  5. Rolling-origin temporal CV
  6. Train models (Logistic + LightGBM)
  7. Calibrate
  8. Ensemble
  9. Generate submission
 10. Save reports
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
    M_CV_TRAIN_CUTOFFS, W_CV_TRAIN_CUTOFFS,
    COVID_SEASON, CURRENT_SEASON, PROB_FLOOR, PROB_CEIL,
    M_DETAILED_START, W_DETAILED_START,
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
    generate_submission,
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
    # Keep _diff, interactions, priors — drop raw A/B values that are redundant
    # (we keep diffs which encode the matchup asymmetry)
    cols = [c for c in matchup_df.columns if c not in exclude]
    return cols


# ═══════════════════════════════════════════════════════════════════════════════
#  Seed-only baseline
# ═══════════════════════════════════════════════════════════════════════════════

def seed_baseline(matchup_df: pd.DataFrame) -> float:
    """Predict using seed-pairing historical win rates. Return Brier."""
    preds = matchup_df["UpsetPrior"].values.copy()
    # UpsetPrior = FavWinRate for (min_seed, max_seed) pairing
    # If SeedA < SeedB → A is favorite → P(A wins) = FavWinRate = UpsetPrior
    actual_preds = np.where(
        matchup_df["SeedNum_A"] <= matchup_df["SeedNum_B"],
        preds,       # A is fav, P(A wins) = FavWinRate
        1 - preds    # B is fav, P(A wins) = 1 - FavWinRate
    )
    # For same seeds, use 0.5
    same = matchup_df["SeedNum_A"] == matchup_df["SeedNum_B"]
    actual_preds[same] = 0.5

    brier = np.mean((matchup_df["target"].values - actual_preds) ** 2)
    return brier


# ═══════════════════════════════════════════════════════════════════════════════
#  Rolling-origin temporal CV
# ═══════════════════════════════════════════════════════════════════════════════

def rolling_origin_cv(matchup_df: pd.DataFrame,
                       feature_cols: list,
                       cv_cutoffs: list,
                       gender: str) -> dict:
    """
    For each cutoff season S in cv_cutoffs:
      Train on seasons ≤ S
      Validate on season S+1 tournament
    
    Returns dict with OOF predictions, fold metrics, and best params.
    """
    print(f"\n{'='*60}")
    print(f"  ROLLING-ORIGIN CV ({gender.upper()})")
    print(f"{'='*60}")

    oof_preds_lr = []
    oof_preds_lgb = []
    oof_actuals = []
    oof_seasons = []
    fold_indices = []
    fold_metrics = []

    # First, run Optuna on the full historical training set to get good params
    all_train = matchup_df[matchup_df["Season"] <= max(cv_cutoffs)]
    X_all = all_train[feature_cols].fillna(0.0)
    y_all = all_train["target"].values

    print(f"\n  Running LightGBM Bayesian HPO on {len(X_all)} samples...")
    lgb_params = optimize_lgbm(X_all, y_all, n_trials=60, n_folds=5)

    idx_offset = 0
    for cutoff in cv_cutoffs:
        val_season = cutoff + 1
        if val_season == COVID_SEASON:
            continue  # skip 2020

        train_mask = matchup_df["Season"] <= cutoff
        val_mask = matchup_df["Season"] == val_season

        if val_mask.sum() == 0:
            continue

        X_train = matchup_df.loc[train_mask, feature_cols].fillna(0.0)
        y_train = matchup_df.loc[train_mask, "target"].values
        X_val = matchup_df.loc[val_mask, feature_cols].fillna(0.0)
        y_val = matchup_df.loc[val_mask, "target"].values

        if len(X_train) < 20 or len(X_val) < 5:
            continue

        # Model A: Logistic Regression
        lr_model, lr_info = train_logistic(X_train, y_train)
        lr_preds = predict_logistic(lr_model, X_val)

        # Model B: LightGBM
        lgb_model, lgb_imp = train_lgbm(X_train, y_train, X_val, y_val,
                                         params=lgb_params)
        lgb_preds = predict_lgbm(lgb_model, X_val)

        # Metrics
        brier_lr = np.mean((y_val - lr_preds) ** 2)
        brier_lgb = np.mean((y_val - lgb_preds) ** 2)

        fold_metrics.append({
            "val_season": val_season,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "brier_lr": brier_lr,
            "brier_lgb": brier_lgb,
        })
        print(f"  Fold val={val_season}: LR Brier={brier_lr:.4f}, "
              f"LGB Brier={brier_lgb:.4f} (train={len(X_train)}, val={len(X_val)})")

        # Store OOF predictions
        oof_preds_lr.extend(lr_preds)
        oof_preds_lgb.extend(lgb_preds)
        oof_actuals.extend(y_val)
        oof_seasons.extend([val_season] * len(y_val))
        fold_indices.append(np.arange(idx_offset, idx_offset + len(y_val)))
        idx_offset += len(y_val)

    oof_preds_lr = np.array(oof_preds_lr)
    oof_preds_lgb = np.array(oof_preds_lgb)
    oof_actuals = np.array(oof_actuals)
    oof_seasons = np.array(oof_seasons)

    # Overall Brier scores
    overall_brier_lr = np.mean((oof_actuals - oof_preds_lr) ** 2)
    overall_brier_lgb = np.mean((oof_actuals - oof_preds_lgb) ** 2)
    print(f"\n  Overall OOF Brier: LR={overall_brier_lr:.4f}, LGB={overall_brier_lgb:.4f}")

    return {
        "oof_preds_lr": oof_preds_lr,
        "oof_preds_lgb": oof_preds_lgb,
        "oof_actuals": oof_actuals,
        "oof_seasons": oof_seasons,
        "fold_indices": fold_indices,
        "fold_metrics": fold_metrics,
        "lgb_params": lgb_params,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Calibration slope (reliability diagnostic)
# ═══════════════════════════════════════════════════════════════════════════════

def calibration_slope(y_true, y_pred):
    """Logistic regression of outcome on predicted prob → slope ≈ 1.0 is ideal."""
    from sklearn.linear_model import LogisticRegression
    eps = 1e-7
    logits = np.log(np.clip(y_pred, eps, 1-eps) / (1 - np.clip(y_pred, eps, 1-eps)))
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(logits.reshape(-1, 1), y_true)
    return lr.coef_[0][0]


# ═══════════════════════════════════════════════════════════════════════════════
#  Full pipeline for one gender
# ═══════════════════════════════════════════════════════════════════════════════

def run_gender_pipeline(gender: str) -> dict:
    """
    Full pipeline for men's or women's:
    load → validate → features → CV → train → calibrate → ensemble.
    Returns all artifacts needed for submission.
    """
    g = gender.lower()
    cv_cutoffs = M_CV_TRAIN_CUTOFFS if g == "m" else W_CV_TRAIN_CUTOFFS
    detailed_start = M_DETAILED_START if g == "m" else W_DETAILED_START

    print(f"\n{'#'*60}")
    print(f"  PIPELINE: {gender.upper()}")
    print(f"{'#'*60}")

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    t0 = time.time()
    data = load_all(g)
    print(f"    Loaded in {time.time()-t0:.1f}s")

    # ── 2. Validate ───────────────────────────────────────────────────────
    print("\n[2] Validating data...")
    validate_data(data, g)

    # ── 3. Build canonical games ──────────────────────────────────────────
    print("\n[3] Building canonical game datasets...")
    all_games = build_all_games(data)
    tourney_games = build_tourney_dataset(data)
    season_agg = build_season_aggregates(all_games)
    print(f"    Total games: {len(all_games)}, Tournament: {len(tourney_games)}")

    # ── 4. Compute ratings ────────────────────────────────────────────────
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

    # ── 8. Build matchup-level features ──────────────────────────────────
    print("\n[8] Building matchup-level features...")
    t0 = time.time()
    matchup_df = build_matchup_features(
        tourney_games, elo_snap, eff_features, seed_info,
        data["conferences_map"], massey_df
    )
    print(f"    Matchup features: {matchup_df.shape} in {time.time()-t0:.1f}s")

    # ── 9. Feature stability ─────────────────────────────────────────────
    print("\n[9] Analyzing feature stability...")
    candidate_cols = get_feature_cols(matchup_df)
    stable_cols = analyze_feature_stability(matchup_df, candidate_cols)

    # Remove non-feature columns that might have slipped in
    stable_cols = [c for c in stable_cols if c not in
                   {"Season", "TeamA", "TeamB", "target"}]

    # ── 10. Seed baseline ────────────────────────────────────────────────
    print("\n[10] Computing seed-only baseline...")
    baseline_brier = seed_baseline(matchup_df)
    print(f"    Seed-only baseline Brier: {baseline_brier:.4f}")

    # ── 11. Rolling-origin CV ────────────────────────────────────────────
    print("\n[11] Running rolling-origin cross-validation...")
    cv_results = rolling_origin_cv(matchup_df, stable_cols, cv_cutoffs, g)

    # ── 12. Calibration ──────────────────────────────────────────────────
    print("\n[12] Calibrating probabilities...")
    oof_lr = cv_results["oof_preds_lr"]
    oof_lgb = cv_results["oof_preds_lgb"]
    y_oof = cv_results["oof_actuals"]

    cal_method_lr, cal_lr = select_calibration(y_oof, oof_lr)
    cal_method_lgb, cal_lgb = select_calibration(y_oof, oof_lgb)

    oof_lr_cal = apply_calibration(cal_method_lr, cal_lr, oof_lr)
    oof_lgb_cal = apply_calibration(cal_method_lgb, cal_lgb, oof_lgb)

    brier_lr_cal = np.mean((y_oof - oof_lr_cal) ** 2)
    brier_lgb_cal = np.mean((y_oof - oof_lgb_cal) ** 2)
    print(f"    Calibrated OOF Brier: LR={brier_lr_cal:.4f}, LGB={brier_lgb_cal:.4f}")

    # ── 13. Ensemble ─────────────────────────────────────────────────────
    print("\n[13] Building ensemble...")
    model_preds = {"lr": oof_lr_cal, "lgb": oof_lgb_cal}
    ens_method, ens_obj, ens_metrics = select_best_ensemble(
        y_oof, model_preds, cv_results["fold_indices"]
    )

    # Compute final ensemble OOF Brier
    if ens_method == "weighted_avg":
        ens_oof = sum(ens_obj[n] * model_preds[n]
                       for n in model_preds.keys())
    else:
        ens_oof = np.clip(
            ens_obj.predict(
                np.column_stack([model_preds[n] for n in ["lr", "lgb"]])
            ), PROB_FLOOR, PROB_CEIL
        )
    ens_brier = np.mean((y_oof - ens_oof) ** 2)

    # Calibration slope
    cal_slope = calibration_slope(y_oof, ens_oof)

    print(f"\n  {'='*50}")
    print(f"  RESULTS SUMMARY ({gender.upper()})")
    print(f"  {'='*50}")
    print(f"  Seed baseline Brier:   {baseline_brier:.4f}")
    print(f"  Ensemble OOF Brier:    {ens_brier:.4f}")
    print(f"  Improvement:           {baseline_brier - ens_brier:.4f}")
    print(f"  Calibration slope:     {cal_slope:.3f} (target ≈ 1.0)")
    print(f"  Fold Brier std:        {ens_metrics.get(ens_method, {}).get('fold_std', 'N/A')}")

    # ── 14. Train final models on ALL data ───────────────────────────────
    print("\n[14] Training final models on all historical data...")
    X_all = matchup_df[stable_cols].fillna(0.0)
    y_all = matchup_df["target"].values

    final_lr, lr_info = train_logistic(X_all, y_all)
    final_lgb, lgb_imp = train_lgbm(X_all, y_all, params=cv_results["lgb_params"],
                                     num_boost_round=500)

    # ── Save artifacts ────────────────────────────────────────────────────
    print("\n[15] Saving artifacts...")
    os.makedirs(ARTIFACT_DIR / g, exist_ok=True)
    joblib.dump(final_lr, ARTIFACT_DIR / g / "logistic.pkl")
    joblib.dump(final_lgb, ARTIFACT_DIR / g / "lgbm.pkl")
    joblib.dump({"method": cal_method_lr, "obj": cal_lr}, ARTIFACT_DIR / g / "cal_lr.pkl")
    joblib.dump({"method": cal_method_lgb, "obj": cal_lgb}, ARTIFACT_DIR / g / "cal_lgb.pkl")
    joblib.dump({"method": ens_method, "obj": ens_obj}, ARTIFACT_DIR / g / "ensemble.pkl")

    return {
        "gender": g,
        "feature_cols": stable_cols,
        "elo_snap": elo_snap,
        "eff_features": eff_features,
        "seed_info": seed_info,
        "conferences_map": data["conferences_map"],
        "massey_df": massey_df,
        "final_lr": final_lr,
        "final_lgb": final_lgb,
        "cal_lr": (cal_method_lr, cal_lr),
        "cal_lgb": (cal_method_lgb, cal_lgb),
        "ensemble": (ens_method, ens_obj),
        "baseline_brier": baseline_brier,
        "ens_brier": ens_brier,
        "cal_slope": cal_slope,
        "cv_results": cv_results,
        "lgb_importances": lgb_imp,
        "lr_info": lr_info,
        "oof_preds": pd.DataFrame({
            "Season": cv_results["oof_seasons"],
            "actual": y_oof,
            "pred_lr": oof_lr,
            "pred_lgb": oof_lgb,
            "pred_lr_cal": oof_lr_cal,
            "pred_lgb_cal": oof_lgb_cal,
            "pred_ensemble": ens_oof,
        }),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Generate final submission
# ═══════════════════════════════════════════════════════════════════════════════

def generate_final_submission(artifacts_m: dict, artifacts_w: dict):
    """
    Generate submission.csv for Stage 2 (2026 predictions).
    """
    print(f"\n{'#'*60}")
    print(f"  GENERATING FINAL SUBMISSION")
    print(f"{'#'*60}")

    sub_raw = pd.read_csv(FILES["sub_stage2"])
    sub = parse_submission_ids(sub_raw)

    sub_m = sub[sub["Gender"] == "m"].copy()
    sub_w = sub[sub["Gender"] == "w"].copy()

    all_preds = []

    for sub_g, art in [(sub_m, artifacts_m), (sub_w, artifacts_w)]:
        if len(sub_g) == 0:
            continue

        g = art["gender"]
        print(f"\n  Processing {g.upper()}: {len(sub_g)} matchups")

        # Build features for each matchup
        feature_rows = []
        for _, row in sub_g.iterrows():
            feats = build_matchup_features_for_submission(
                int(row["TeamA"]), int(row["TeamB"]), int(row["Season"]),
                art["elo_snap"], art["eff_features"], art["seed_info"]["seed_lookup"],
                art["seed_info"]["conf_strength"], art["conferences_map"],
                art["seed_info"]["upset_table"], art["seed_info"]["tourney_appearances"],
                massey_df=art["massey_df"],
            )
            feature_rows.append(feats)

        feat_df = pd.DataFrame(feature_rows)

        # Add interaction features
        feat_df = add_interaction_features(feat_df)

        # Ensure correct columns
        for col in art["feature_cols"]:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        X = feat_df[art["feature_cols"]].fillna(0.0)

        # Predict with both models
        lr_preds = predict_logistic(art["final_lr"], X)
        lgb_preds = predict_lgbm(art["final_lgb"], X)

        # Calibrate
        cal_m_lr, cal_o_lr = art["cal_lr"]
        cal_m_lgb, cal_o_lgb = art["cal_lgb"]
        lr_cal = apply_calibration(cal_m_lr, cal_o_lr, lr_preds)
        lgb_cal = apply_calibration(cal_m_lgb, cal_o_lgb, lgb_preds)

        # Ensemble
        ens_method, ens_obj = art["ensemble"]
        if ens_method == "weighted_avg":
            preds = sum(ens_obj[n] * p for n, p in
                        [("lr", lr_cal), ("lgb", lgb_cal)])
        else:
            preds = np.clip(
                ens_obj.predict(np.column_stack([lr_cal, lgb_cal])),
                PROB_FLOOR, PROB_CEIL
            )

        # Clamp
        preds = np.clip(preds, PROB_FLOOR, PROB_CEIL)

        sub_g = sub_g.copy()
        sub_g["Pred"] = preds
        all_preds.append(sub_g[["ID", "Pred"]])

    result = pd.concat(all_preds, ignore_index=True)

    # Merge back with original submission to keep ordering
    final = sub_raw[["ID"]].merge(result, on="ID", how="left")

    # Any missing predictions (shouldn't happen) → 0.5
    missing = final["Pred"].isna().sum()
    if missing > 0:
        print(f"  ⚠ {missing} matchups missing predictions → defaulting to 0.5")
        final["Pred"] = final["Pred"].fillna(0.5)

    # Validate
    assert len(final) == len(sub_raw), f"Row mismatch: {len(final)} vs {len(sub_raw)}"
    assert final["Pred"].isna().sum() == 0, "NaN predictions!"

    out_path = OUTPUT_DIR / "submission.csv"
    final.to_csv(out_path, index=False)
    print(f"\n  ✓ Final submission: {out_path} ({len(final)} rows)")
    print(f"    Pred range: [{final['Pred'].min():.4f}, {final['Pred'].max():.4f}]")
    print(f"    Pred mean: {final['Pred'].mean():.4f}")

    return final


# ═══════════════════════════════════════════════════════════════════════════════
#  Save OOF and reports
# ═══════════════════════════════════════════════════════════════════════════════

def save_oof_predictions(artifacts_m, artifacts_w):
    """Save combined OOF predictions."""
    frames = []
    for art in [artifacts_m, artifacts_w]:
        oof = art["oof_preds"].copy()
        oof["Gender"] = art["gender"]
        frames.append(oof)
    combined = pd.concat(frames, ignore_index=True)
    path = OUTPUT_DIR / "oof_predictions.csv"
    combined.to_csv(path, index=False)
    print(f"  ✓ OOF predictions: {path}")


def save_reports(artifacts_m, artifacts_w):
    """Generate markdown reports."""

    # ── Feature importance ────────────────────────────────────────────────
    lines = ["# Feature Importance Report\n"]
    for art in [artifacts_m, artifacts_w]:
        g = art["gender"].upper()
        lines.append(f"\n## {g} — LightGBM Feature Importance (Gain)\n")
        imp = sorted(art["lgb_importances"].items(), key=lambda x: -x[1])
        lines.append("| Rank | Feature | Gain |")
        lines.append("|------|---------|------|")
        for i, (f, v) in enumerate(imp[:30], 1):
            lines.append(f"| {i} | {f} | {v:.1f} |")

        lines.append(f"\n## {g} — Logistic Regression Coefficients\n")
        coefs = sorted(art["lr_info"]["coefs"].items(), key=lambda x: -abs(x[1]))
        lines.append("| Feature | Coefficient |")
        lines.append("|---------|-------------|")
        for f, v in coefs[:30]:
            lines.append(f"| {f} | {v:.4f} |")

    with open(OUTPUT_DIR / "feature_importance_report.md", "w") as f:
        f.write("\n".join(lines))

    # ── Calibration report ────────────────────────────────────────────────
    lines = ["# Calibration Report\n"]
    for art in [artifacts_m, artifacts_w]:
        g = art["gender"].upper()
        lines.append(f"\n## {g}\n")
        lines.append(f"- Seed baseline Brier: {art['baseline_brier']:.4f}")
        lines.append(f"- Ensemble OOF Brier: {art['ens_brier']:.4f}")
        lines.append(f"- Improvement: {art['baseline_brier'] - art['ens_brier']:.4f}")
        lines.append(f"- Calibration slope: {art['cal_slope']:.3f}")
        lines.append(f"\n### Fold-level metrics\n")
        lines.append("| Val Season | LR Brier | LGB Brier | N_val |")
        lines.append("|------------|----------|-----------|-------|")
        for fm in art["cv_results"]["fold_metrics"]:
            lines.append(f"| {fm['val_season']} | {fm['brier_lr']:.4f} | "
                         f"{fm['brier_lgb']:.4f} | {fm['n_val']} |")

    with open(OUTPUT_DIR / "calibration_report.md", "w") as f:
        f.write("\n".join(lines))

    # ── Full methodology ──────────────────────────────────────────────────
    lines = [
        "# Full Methodology Report\n",
        "## System Overview\n",
        "Probabilistic tournament forecasting system targeting Brier score minimisation.",
        "Two independent pipelines for Men's and Women's NCAA tournaments.\n",
        "## Feature Engineering\n",
        "### Rating Systems",
        "- Season-reset ELO (K=20, 75% reversion to mean)",
        "- Margin-of-victory adjusted ELO (FiveThirtyEight-style MOV multiplier)",
        "- Simplified Glicko (tracks rating + rating deviation)",
        "- Momentum (linear slope of last-5-game ELO)\n",
        "### Efficiency Metrics",
        "- Offensive/Defensive efficiency (pts per 100 possessions)",
        "- Four Factors: eFG%, TO rate, ORB%, FT rate",
        "- Opponent-adjusted efficiency (5 iterative rounds)",
        "- Pace, Strength of Schedule",
        "- Exponentially weighted rolling windows (span 5, 10, full season)\n",
        "### Seed Features",
        "- Seed difference, historical upset frequency by seed pairing",
        "- Tournament experience, conference strength index\n",
        "### Massey Ordinals (Men's only)",
        "- Pre-tournament rankings from top-5 systems (POM, SAG, MOR, DOL, COL)",
        "- Mean ordinal difference across systems\n",
        "### Interactions",
        "- Elo_diff × Seed_diff, AdjEM × Pace",
        "- Upset-prone indicator (|seed_diff| ≥ 5)",
        "- Underdog volatility, rating uncertainty\n",
        "## Model Architecture\n",
        "- Model A: Elastic Net Logistic Regression (L1+L2 path search, standardized)",
        "- Model B: LightGBM (Bayesian HPO via Optuna, early stopping on Brier)\n",
        "## Calibration\n",
        "- Isotonic regression, Platt scaling, temperature scaling",
        "- Selected per model by cross-validated Brier\n",
        "## Ensemble\n",
        "- Weighted average (grid search) vs Ridge stacking",
        "- Selected by lowest mean Brier + lowest fold variance\n",
        "## Temporal Validation\n",
        "- Rolling-origin: train ≤ season S, validate on S+1 tournament",
        "- Men's: ~15 folds (2009–2025 excl 2020)",
        "- Women's: ~12 folds (2013–2025 excl 2020)",
        "- Metrics: Brier, calibration slope, sharpness\n",
    ]

    for art in [artifacts_m, artifacts_w]:
        g = art["gender"].upper()
        lines.append(f"## {g} Results\n")
        lines.append(f"- Seed baseline: {art['baseline_brier']:.4f}")
        lines.append(f"- Ensemble Brier: {art['ens_brier']:.4f}")
        lines.append(f"- Calibration slope: {art['cal_slope']:.3f}")
        lines.append(f"- Feature count: {len(art['feature_cols'])}\n")

    with open(OUTPUT_DIR / "full_methodology_report.md", "w") as f:
        f.write("\n".join(lines))

    print("  ✓ Reports saved")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Run the full pipeline."""
    start = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # Run pipelines
    artifacts_m = run_gender_pipeline("m")
    artifacts_w = run_gender_pipeline("w")

    # Generate submission
    generate_final_submission(artifacts_m, artifacts_w)

    # Save OOF
    save_oof_predictions(artifacts_m, artifacts_w)

    # Save reports
    save_reports(artifacts_m, artifacts_w)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
