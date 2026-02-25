"""
Submission generation: parse IDs, compute features, generate predictions.
"""

import numpy as np
import pandas as pd

from src.config import FILES, PROB_FLOOR, PROB_CEIL, CURRENT_SEASON


def parse_submission_ids(sub_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse submission ID column (Season_TeamA_TeamB) into components.
    Adds columns: Season, TeamA, TeamB, Gender.
    """
    parts = sub_df["ID"].str.split("_", expand=True)
    sub_df = sub_df.copy()
    sub_df["Season"] = parts[0].astype(int)
    sub_df["TeamA"] = parts[1].astype(int)
    sub_df["TeamB"] = parts[2].astype(int)

    # Kaggle requires TeamA < TeamB in submission IDs
    invalid = sub_df["TeamA"] >= sub_df["TeamB"]
    if invalid.any():
        raise ValueError(
            f"Submission has {invalid.sum()} matchups where TeamA >= TeamB"
        )

    # Gender from TeamID range: 1xxx = Men's, 3xxx = Women's
    sub_df["Gender"] = np.where(sub_df["TeamA"] < 3000, "m", "w")
    return sub_df


def build_matchup_features_for_submission(
    team_a: int, team_b: int, season: int,
    elo_snap, eff_features, seed_lookup, conf_strength, conferences_map,
    upset_table, tourney_appearances,
    massey_df=None,
):
    """
    Compute ALL features for one matchup using only pre-tournament data.
    Returns dict of feature values.
    """
    feats = {}

    # ── ELO features ──────────────────────────────────────────────────────
    for col in ["Elo", "MarginElo", "Glicko", "GlickoRD", "Momentum"]:
        val_a = elo_snap.loc[
            (elo_snap["Season"] == season) & (elo_snap["TeamID"] == team_a), col
        ]
        val_b = elo_snap.loc[
            (elo_snap["Season"] == season) & (elo_snap["TeamID"] == team_b), col
        ]
        va = val_a.values[0] if len(val_a) > 0 else (1500.0 if "Elo" in col else 0.0)
        vb = val_b.values[0] if len(val_b) > 0 else (1500.0 if "Elo" in col else 0.0)
        feats[f"{col}_diff"] = va - vb
        feats[f"{col}_A"] = va
        feats[f"{col}_B"] = vb

    # ── Efficiency features ───────────────────────────────────────────────
    eff_cols = [c for c in eff_features.columns
                if c not in ("Season", "TeamID")]
    for col in eff_cols:
        val_a = eff_features.loc[
            (eff_features["Season"] == season) & (eff_features["TeamID"] == team_a), col
        ]
        val_b = eff_features.loc[
            (eff_features["Season"] == season) & (eff_features["TeamID"] == team_b), col
        ]
        va = val_a.values[0] if len(val_a) > 0 else np.nan
        vb = val_b.values[0] if len(val_b) > 0 else np.nan
        feats[f"{col}_diff"] = va - vb if pd.notna(va) and pd.notna(vb) else 0.0

    # ── Seed features ─────────────────────────────────────────────────────
    seed_a_row = seed_lookup[
        (seed_lookup["Season"] == season) & (seed_lookup["TeamID"] == team_a)
    ]
    seed_b_row = seed_lookup[
        (seed_lookup["Season"] == season) & (seed_lookup["TeamID"] == team_b)
    ]
    seed_a = seed_a_row["SeedNum"].values[0] if len(seed_a_row) > 0 else 8.5
    seed_b = seed_b_row["SeedNum"].values[0] if len(seed_b_row) > 0 else 8.5
    feats["SeedNum_diff"] = seed_a - seed_b
    feats["SeedNum_A"] = seed_a
    feats["SeedNum_B"] = seed_b

    # Historical upset rate for this seed pairing
    from src.features.seeds import get_upset_prior, get_tourney_experience
    feats["UpsetPrior"] = get_upset_prior(
        upset_table, int(round(seed_a)), int(round(seed_b))
    )

    # Tournament experience
    feats["TourneyExp_diff"] = (
        get_tourney_experience(tourney_appearances, team_a, season) -
        get_tourney_experience(tourney_appearances, team_b, season)
    )

    # ── Conference strength ───────────────────────────────────────────────
    conf_a_row = conferences_map[
        (conferences_map["Season"] == season) & (conferences_map["TeamID"] == team_a)
    ]
    conf_b_row = conferences_map[
        (conferences_map["Season"] == season) & (conferences_map["TeamID"] == team_b)
    ]
    if len(conf_a_row) > 0 and len(conf_b_row) > 0:
        ca = conf_a_row["ConfAbbrev"].values[0]
        cb = conf_b_row["ConfAbbrev"].values[0]
        cs_a = conf_strength.loc[
            (conf_strength["Season"] == season) & (conf_strength["ConfAbbrev"] == ca),
            "ConfStrength"
        ]
        cs_b = conf_strength.loc[
            (conf_strength["Season"] == season) & (conf_strength["ConfAbbrev"] == cb),
            "ConfStrength"
        ]
        feats["ConfStrength_diff"] = (
            (cs_a.values[0] if len(cs_a) > 0 else 1500.0) -
            (cs_b.values[0] if len(cs_b) > 0 else 1500.0)
        )
    else:
        feats["ConfStrength_diff"] = 0.0

    # ── Massey ordinals (men's only) ──────────────────────────────────────
    if massey_df is not None and not massey_df.empty:
        from src.features.massey import get_massey_diff_features
        massey_feats = get_massey_diff_features(massey_df, team_a, team_b, season)
        feats.update(massey_feats)

    return feats


def generate_submission(sub_df: pd.DataFrame,
                         predict_fn,
                         feature_builder_fn,
                         feature_cols: list,
                         output_path: str) -> pd.DataFrame:
    """
    Generate final submission CSV.
    
    sub_df: parsed submission DataFrame (with Season, TeamA, TeamB, Gender)
    predict_fn: function(features_df) → probabilities
    feature_builder_fn: function(row) → feature dict
    feature_cols: ordered list of feature column names
    """
    print(f"  Generating predictions for {len(sub_df)} matchups...")

    feature_rows = []
    for _, row in sub_df.iterrows():
        feats = feature_builder_fn(row)
        feature_rows.append(feats)

    features_df = pd.DataFrame(feature_rows)

    # Ensure column ordering + fill missing
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0
    features_df = features_df[feature_cols].fillna(0.0)

    # Predict
    probs = predict_fn(features_df)

    # Clamp
    probs = np.clip(probs, PROB_FLOOR, PROB_CEIL)

    # Build output
    out = pd.DataFrame({"ID": sub_df["ID"].values, "Pred": probs})

    # Validate
    assert out["Pred"].isna().sum() == 0, "NaN predictions found!"
    assert len(out) == len(sub_df), "Row count mismatch!"
    assert (out["Pred"] >= PROB_FLOOR).all() and (out["Pred"] <= PROB_CEIL).all(), \
        "Predictions out of clamp range!"

    out.to_csv(output_path, index=False)
    print(f"  ✓ Submission saved: {output_path} ({len(out)} rows)")

    return out
