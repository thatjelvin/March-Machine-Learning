"""
Interaction features and nonlinear combinations.
Feature stability analysis (cross-season variance).
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
#  Interaction features
# ═══════════════════════════════════════════════════════════════════════════════

def add_interaction_features(matchup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a matchup-level feature DataFrame (one row per matchup),
    add interaction and nonlinear features.

    Expected columns: Elo_diff, SeedNum_diff, AdjEM_diff, Pace_diff, etc.
    """
    df = matchup_df.copy()

    # Cross-feature interactions
    if "Elo_diff" in df.columns and "SeedNum_diff" in df.columns:
        df["Elo_x_Seed"] = df["Elo_diff"] * df["SeedNum_diff"]

    if "AdjEM_diff" in df.columns and "Pace_full_diff" in df.columns:
        df["AdjEM_x_Pace"] = df["AdjEM_diff"] * df["Pace_full_diff"]

    # Upset-prone indicator
    if "SeedNum_diff" in df.columns:
        df["UpsetProne"] = (df["SeedNum_diff"].abs() >= 5).astype(int)

    # Underdog volatility (std of margin for lower-seeded team)
    if "StdMargin_A" in df.columns and "StdMargin_B" in df.columns:
        if "SeedNum_A" in df.columns and "SeedNum_B" in df.columns:
            # Higher seed number = lower seed = underdog
            df["UnderdogVolatility"] = np.where(
                df["SeedNum_A"] > df["SeedNum_B"],
                df["StdMargin_A"],
                df["StdMargin_B"]
            )
        else:
            df["UnderdogVolatility"] = (df["StdMargin_A"] + df["StdMargin_B"]) / 2

    # Rating confidence spread
    if "GlickoRD_diff" in df.columns:
        df["RatingUncertainty"] = df["GlickoRD_diff"].abs()

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature stability analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_feature_stability(matchup_df: pd.DataFrame,
                               feature_cols: list,
                               max_cv: float = 1.5,
                               max_sign_flip_rate: float = 0.30
                               ) -> list:
    """
    For each feature, compute season-wise mean/variance.
    Drop features with high cross-season coefficient of variation
    or frequent sign flips.

    Returns list of stable feature names.
    """
    if "Season" not in matchup_df.columns:
        return feature_cols  # can't analyze without season

    stable = []
    report = []

    for col in feature_cols:
        if col not in matchup_df.columns:
            continue

        season_means = matchup_df.groupby("Season")[col].mean()
        if len(season_means) < 3:
            stable.append(col)
            continue

        overall_mean = season_means.mean()
        overall_std = season_means.std()

        if abs(overall_mean) < 1e-10:
            cv = 0.0 if overall_std < 1e-10 else float("inf")
        else:
            cv = overall_std / abs(overall_mean)

        # Sign flip rate
        signs = np.sign(season_means.values)
        sign_changes = (np.diff(signs) != 0).sum()
        sign_flip_rate = sign_changes / max(len(signs) - 1, 1)

        is_stable = cv <= max_cv and sign_flip_rate <= max_sign_flip_rate
        report.append({
            "feature": col,
            "cv": round(cv, 4),
            "sign_flip_rate": round(sign_flip_rate, 4),
            "stable": is_stable,
        })
        if is_stable:
            stable.append(col)

    print(f"  Feature stability: {len(stable)}/{len(feature_cols)} features retained")
    unstable = [r for r in report if not r["stable"]]
    if unstable:
        print(f"  Dropped: {[r['feature'] for r in unstable]}")

    return stable
