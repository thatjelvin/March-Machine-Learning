"""
Efficiency metrics from detailed box-score data:
  - Offensive / Defensive efficiency (points per 100 possessions)
  - Four Factors: eFG%, TO rate, OR%, FTrate
  - Pace
  - Opponent-adjusted efficiency (iterative)
  - Strength of Schedule
  - Exponentially weighted rolling windows
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

from src.config import EFF_ROLLING_SPANS, ITERATIVE_ADJ_ROUNDS


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-game stats (team-level, from canonical games)
# ═══════════════════════════════════════════════════════════════════════════════

_DETAIL_COLS = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
                "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]


def _has_detail(df):
    return "A_FGA" in df.columns


def estimate_possessions(fga, ora, to, fta):
    """Dean Oliver approximation: Poss ≈ FGA - OR + TO + 0.475*FTA"""
    return fga - ora + to + 0.475 * fta


def compute_game_stats(all_games: pd.DataFrame) -> pd.DataFrame:
    """
    From canonical all_games with A_*/B_* box-score columns, compute
    per-game efficiency stats for each team-game.

    Returns long-format DataFrame: one row per (Season, DayNum, TeamID)
    with efficiency columns.
    """
    if not _has_detail(all_games):
        return pd.DataFrame()  # no detail available

    reg = all_games[
        (all_games["source"] == "regular") & all_games["A_FGA"].notna()
    ].copy()

    rows = []
    for _, g in reg.iterrows():
        season = g["Season"]
        day = g["DayNum"]

        # --- Team A stats ---
        poss_a = estimate_possessions(g["A_FGA"], g["A_OR"], g["A_TO"], g["A_FTA"])
        poss_b = estimate_possessions(g["B_FGA"], g["B_OR"], g["B_TO"], g["B_FTA"])
        poss_a = max(poss_a, 1)
        poss_b = max(poss_b, 1)
        pace = (poss_a + poss_b) / 2.0

        for side, opp in [("A", "B"), ("B", "A")]:
            p = estimate_possessions(g[f"{side}_FGA"], g[f"{side}_OR"],
                                     g[f"{side}_TO"], g[f"{side}_FTA"])
            p = max(p, 1)
            p_opp = estimate_possessions(g[f"{opp}_FGA"], g[f"{opp}_OR"],
                                          g[f"{opp}_TO"], g[f"{opp}_FTA"])
            p_opp = max(p_opp, 1)

            score = g[f"Score{side}"]
            opp_score = g[f"Score{opp}"]

            off_eff = score / p * 100.0
            def_eff = opp_score / p_opp * 100.0

            efg = (g[f"{side}_FGM"] + 0.5 * g[f"{side}_FGM3"]) / max(g[f"{side}_FGA"], 1)
            to_rate = g[f"{side}_TO"] / max(p, 1)
            orb_pct = g[f"{side}_OR"] / max(g[f"{side}_OR"] + g[f"{opp}_DR"], 1)
            ft_rate = g[f"{side}_FTM"] / max(g[f"{side}_FGA"], 1)

            # Opponent's four factors (for defensive profile)
            opp_efg = (g[f"{opp}_FGM"] + 0.5 * g[f"{opp}_FGM3"]) / max(g[f"{opp}_FGA"], 1)
            opp_to_rate = g[f"{opp}_TO"] / max(p_opp, 1)

            tid = int(g[f"Team{side}"])
            opp_tid = int(g[f"Team{opp}"])

            rows.append({
                "Season": int(season),
                "DayNum": int(day),
                "TeamID": tid,
                "OppID": opp_tid,
                "Score": score,
                "OppScore": opp_score,
                "Poss": p,
                "Pace": pace,
                "OffEff": off_eff,
                "DefEff": def_eff,
                "eFG": efg,
                "TO_rate": to_rate,
                "ORB_pct": orb_pct,
                "FT_rate": ft_rate,
                "Opp_eFG": opp_efg,
                "Opp_TO_rate": opp_to_rate,
                "FGM3": g[f"{side}_FGM3"],
                "FGA3": g[f"{side}_FGA3"],
                "Ast": g[f"{side}_Ast"],
                "Stl": g[f"{side}_Stl"],
                "Blk": g[f"{side}_Blk"],
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  Exponentially weighted rolling aggregates
# ═══════════════════════════════════════════════════════════════════════════════

_STAT_FEATURES = [
    "OffEff", "DefEff", "Pace", "eFG", "TO_rate", "ORB_pct", "FT_rate",
    "Opp_eFG", "Opp_TO_rate",
]


def compute_rolling_features(game_stats: pd.DataFrame,
                             spans=EFF_ROLLING_SPANS) -> pd.DataFrame:
    """
    For each team-season, compute exponentially-weighted rolling stats.
    Returns season-level features per team.
    """
    if game_stats.empty:
        return pd.DataFrame()

    records = []
    for (season, tid), group in game_stats.groupby(["Season", "TeamID"]):
        grp = group.sort_values("DayNum")
        rec = {"Season": int(season), "TeamID": int(tid)}

        for span in spans:
            suffix = f"_{span}" if span < 9000 else "_full"
            for col in _STAT_FEATURES:
                vals = grp[col].values
                if len(vals) == 0:
                    rec[f"{col}{suffix}"] = np.nan
                    continue
                if span >= 9000:
                    # Full season mean
                    rec[f"{col}{suffix}"] = np.mean(vals)
                else:
                    # EWM: use last `span` games
                    alpha = 2.0 / (span + 1.0)
                    s = pd.Series(vals)
                    ewm = s.ewm(span=span, adjust=True).mean().iloc[-1]
                    rec[f"{col}{suffix}"] = ewm

        # Also store game count and scoring stats
        rec["Games_detail"] = len(grp)
        rec["AvgScore"] = grp["Score"].mean()
        rec["AvgOppScore"] = grp["OppScore"].mean()

        records.append(rec)

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
#  Opponent-adjusted efficiency (iterative)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_adjusted_efficiency(game_stats: pd.DataFrame,
                                 rounds: int = ITERATIVE_ADJ_ROUNDS
                                 ) -> pd.DataFrame:
    """
    Iterative opponent adjustment: regress efficiency against opponent
    quality. Converges in a few iterations.

    Returns DataFrame with (Season, TeamID, AdjOff, AdjDef, AdjEM).
    """
    if game_stats.empty:
        return pd.DataFrame()

    records = []
    for season, sdf in game_stats.groupby("Season"):
        # Initialize with raw efficiency
        team_off = sdf.groupby("TeamID")["OffEff"].mean().to_dict()
        team_def = sdf.groupby("TeamID")["DefEff"].mean().to_dict()

        league_off = sdf["OffEff"].mean()
        league_def = sdf["DefEff"].mean()

        for _ in range(rounds):
            new_off = {}
            new_def = {}

            for tid, tdf in sdf.groupby("TeamID"):
                # Adjust offensive efficiency by opponents' defensive quality
                opp_def_quality = tdf["OppID"].map(
                    lambda x: team_def.get(x, league_def)
                ).values
                adj_factor = opp_def_quality / league_def
                adj_off = (tdf["OffEff"].values / adj_factor).mean() if len(adj_factor) > 0 else league_off

                # Adjust defensive efficiency by opponents' offensive quality
                opp_off_quality = tdf["OppID"].map(
                    lambda x: team_off.get(x, league_off)
                ).values
                adj_factor_d = opp_off_quality / league_off
                adj_def = (tdf["DefEff"].values / adj_factor_d).mean() if len(adj_factor_d) > 0 else league_def

                new_off[tid] = adj_off
                new_def[tid] = adj_def

            team_off = new_off
            team_def = new_def

        for tid in team_off:
            records.append({
                "Season": int(season),
                "TeamID": int(tid),
                "AdjOff": team_off[tid],
                "AdjDef": team_def[tid],
                "AdjEM": team_off[tid] - team_def[tid],
            })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
#  Strength of Schedule from ELO snapshot
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sos(game_stats: pd.DataFrame,
                elo_snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Average opponent ELO across the season → SOS feature.
    """
    if game_stats.empty:
        return pd.DataFrame()

    merged = game_stats.merge(
        elo_snapshot[["Season", "TeamID", "Elo"]].rename(
            columns={"TeamID": "OppID", "Elo": "OppElo"}
        ),
        on=["Season", "OppID"],
        how="left",
    )
    merged["OppElo"] = merged["OppElo"].fillna(1500.0)

    sos = merged.groupby(["Season", "TeamID"])["OppElo"].mean().reset_index()
    sos.rename(columns={"OppElo": "SOS"}, inplace=True)
    return sos


# ═══════════════════════════════════════════════════════════════════════════════
#  Points-only proxy features (for seasons without detailed stats)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_points_only_features(season_agg: pd.DataFrame) -> pd.DataFrame:
    """
    For seasons without detailed data, derive proxy features from
    scoring and win/loss data only (from season_agg).
    """
    out = season_agg[["Season", "TeamID", "AvgPts", "AvgOppPts",
                       "AvgMargin", "StdMargin", "WinPct", "Games"]].copy()
    out["PtsProxy_OffEff"] = out["AvgPts"]
    out["PtsProxy_DefEff"] = out["AvgOppPts"]
    out["PtsProxy_EM"] = out["AvgMargin"]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Master: build all efficiency features per (Season, TeamID)
# ═══════════════════════════════════════════════════════════════════════════════

def build_efficiency_features(all_games: pd.DataFrame,
                               elo_snapshot: pd.DataFrame,
                               season_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Full efficiency feature pipeline.
    Returns DataFrame with (Season, TeamID, ... efficiency columns ...).
    """
    game_stats = compute_game_stats(all_games)

    if game_stats.empty:
        # Fallback to points-only
        return compute_points_only_features(season_agg)

    rolling = compute_rolling_features(game_stats)
    adj = compute_adjusted_efficiency(game_stats)
    sos = compute_sos(game_stats, elo_snapshot)

    # Merge everything
    features = rolling.copy()
    if not adj.empty:
        features = features.merge(adj, on=["Season", "TeamID"], how="left")
    if not sos.empty:
        features = features.merge(sos, on=["Season", "TeamID"], how="left")

    # For seasons WITHOUT detailed data, fill from points-only proxy
    pts_proxy = compute_points_only_features(season_agg)
    # Identify teams with no detailed features
    features = features.merge(
        pts_proxy[["Season", "TeamID", "PtsProxy_OffEff", "PtsProxy_DefEff",
                    "PtsProxy_EM", "WinPct", "StdMargin"]],
        on=["Season", "TeamID"],
        how="outer",
    )

    # Fill missing adjusted efficiency from points proxy
    features["AdjOff"] = features["AdjOff"].fillna(features["PtsProxy_OffEff"])
    features["AdjDef"] = features["AdjDef"].fillna(features["PtsProxy_DefEff"])
    features["AdjEM"] = features["AdjEM"].fillna(features["PtsProxy_EM"])
    features["SOS"] = features["SOS"].fillna(1500.0)

    return features
