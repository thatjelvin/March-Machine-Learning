"""
Multiple ELO / rating systems computed from chronological game data.
  - Season-reset ELO
  - Margin-of-victory adjusted ELO
  - Glicko-style rating with uncertainty (RD)
  - Momentum (rolling slope of ELO)

All systems process games in strict chronological order and never use future data.
"""

import math
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Tuple

from src.config import (
    ELO_INITIAL, ELO_K, ELO_SEASON_REVERT, ELO_HOME_ADV,
    MARGIN_ELO_K, GLICKO_INITIAL_RD, GLICKO_MIN_RD,
    GLICKO_RD_INCREASE_PER_SEASON, GLICKO_Q, MOMENTUM_WINDOW,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Standard season-reset ELO
# ═══════════════════════════════════════════════════════════════════════════════

class EloRating:
    """Classic ELO with season-mean reversion."""

    def __init__(self, k=ELO_K, init=ELO_INITIAL, revert=ELO_SEASON_REVERT,
                 home_adv=ELO_HOME_ADV):
        self.k = k
        self.init = init
        self.revert = revert
        self.home_adv = home_adv
        self.ratings: Dict[int, float] = defaultdict(lambda: self.init)
        self._current_season = None

    def _expected(self, ra, rb):
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def season_reset(self, season):
        """Revert ratings toward mean at the start of a new season."""
        if self._current_season is not None and season != self._current_season:
            for tid in list(self.ratings.keys()):
                self.ratings[tid] = (self.revert * self.ratings[tid]
                                     + (1 - self.revert) * self.init)
        self._current_season = season

    def update(self, team_w, team_l, wloc="N", season=None):
        """Update after a game. Returns pre-game ratings."""
        if season is not None:
            self.season_reset(season)

        ra = self.ratings[team_w]
        rb = self.ratings[team_l]

        # Home advantage adjustment for expected calculation
        ha = 0.0
        if wloc == "H":
            ha = self.home_adv
        elif wloc == "A":
            ha = -self.home_adv

        exp_w = self._expected(ra + ha, rb)
        self.ratings[team_w] = ra + self.k * (1.0 - exp_w)
        self.ratings[team_l] = rb + self.k * (0.0 - (1.0 - exp_w))

        return ra, rb

    def get(self, tid):
        return self.ratings[tid]


# ═══════════════════════════════════════════════════════════════════════════════
#  Margin-of-Victory ELO
# ═══════════════════════════════════════════════════════════════════════════════

class MarginElo:
    """ELO where K scales with margin of victory, autocorrelation-dampened."""

    def __init__(self, k=MARGIN_ELO_K, init=ELO_INITIAL, revert=ELO_SEASON_REVERT,
                 home_adv=ELO_HOME_ADV):
        self.k = k
        self.init = init
        self.revert = revert
        self.home_adv = home_adv
        self.ratings: Dict[int, float] = defaultdict(lambda: self.init)
        self._current_season = None

    def _expected(self, ra, rb):
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def _mov_multiplier(self, margin, elo_diff):
        """FiveThirtyEight-style MOV multiplier with autocorrelation correction."""
        return math.log(abs(margin) + 1.0) * (2.2 / (abs(elo_diff) * 0.001 + 2.2))

    def season_reset(self, season):
        if self._current_season is not None and season != self._current_season:
            for tid in list(self.ratings.keys()):
                self.ratings[tid] = (self.revert * self.ratings[tid]
                                     + (1 - self.revert) * self.init)
        self._current_season = season

    def update(self, team_w, team_l, margin, wloc="N", season=None):
        if season is not None:
            self.season_reset(season)

        ra = self.ratings[team_w]
        rb = self.ratings[team_l]

        ha = 0.0
        if wloc == "H":
            ha = self.home_adv
        elif wloc == "A":
            ha = -self.home_adv

        exp_w = self._expected(ra + ha, rb)
        elo_diff = ra - rb
        mov_mult = self._mov_multiplier(margin, elo_diff)

        self.ratings[team_w] = ra + self.k * mov_mult * (1.0 - exp_w)
        self.ratings[team_l] = rb + self.k * mov_mult * (0.0 - (1.0 - exp_w))

        return ra, rb

    def get(self, tid):
        return self.ratings[tid]


# ═══════════════════════════════════════════════════════════════════════════════
#  Glicko-style rating (simplified)
# ═══════════════════════════════════════════════════════════════════════════════

class GlickoRating:
    """
    Simplified Glicko: tracks rating + rating deviation (RD).
    RD increases during off-season and decreases with each game.
    """

    def __init__(self, init=ELO_INITIAL, init_rd=GLICKO_INITIAL_RD,
                 min_rd=GLICKO_MIN_RD, rd_season_increase=GLICKO_RD_INCREASE_PER_SEASON,
                 q=GLICKO_Q):
        self.init = init
        self.init_rd = init_rd
        self.min_rd = min_rd
        self.rd_season_increase = rd_season_increase
        self.q = q
        self.ratings: Dict[int, float] = defaultdict(lambda: self.init)
        self.rds: Dict[int, float] = defaultdict(lambda: self.init_rd)
        self._current_season = None

    def _g(self, rd):
        return 1.0 / math.sqrt(1.0 + 3.0 * (self.q ** 2) * (rd ** 2) / (math.pi ** 2))

    def _expected(self, r, rj, rdj):
        return 1.0 / (1.0 + 10.0 ** (-self._g(rdj) * (r - rj) / 400.0))

    def season_reset(self, season):
        if self._current_season is not None and season != self._current_season:
            for tid in list(self.rds.keys()):
                self.rds[tid] = min(
                    self.init_rd,
                    math.sqrt(self.rds[tid] ** 2 + self.rd_season_increase ** 2)
                )
            # Slight revert of rating toward mean
            for tid in list(self.ratings.keys()):
                self.ratings[tid] = 0.85 * self.ratings[tid] + 0.15 * self.init
        self._current_season = season

    def update(self, team_w, team_l, season=None):
        if season is not None:
            self.season_reset(season)

        rw, rdw = self.ratings[team_w], self.rds[team_w]
        rl, rdl = self.ratings[team_l], self.rds[team_l]

        # Pre-game values for feature extraction
        pre_rw, pre_rdw = rw, rdw
        pre_rl, pre_rdl = rl, rdl

        # Update winner
        gj = self._g(rdl)
        ej = self._expected(rw, rl, rdl)
        d2_w = 1.0 / (self.q ** 2 * gj ** 2 * ej * (1 - ej))
        self.ratings[team_w] = rw + (self.q / (1.0 / rdw ** 2 + 1.0 / d2_w)) * gj * (1.0 - ej)
        self.rds[team_w] = max(self.min_rd, math.sqrt(1.0 / (1.0 / rdw ** 2 + 1.0 / d2_w)))

        # Update loser
        gj = self._g(rdw)
        ej = self._expected(rl, rw, rdw)
        d2_l = 1.0 / (self.q ** 2 * gj ** 2 * ej * (1 - ej))
        self.ratings[team_l] = rl + (self.q / (1.0 / rdl ** 2 + 1.0 / d2_l)) * gj * (0.0 - ej)
        self.rds[team_l] = max(self.min_rd, math.sqrt(1.0 / (1.0 / rdl ** 2 + 1.0 / d2_l)))

        return pre_rw, pre_rdw, pre_rl, pre_rdl

    def get(self, tid):
        return self.ratings[tid], self.rds[tid]


# ═══════════════════════════════════════════════════════════════════════════════
#  Momentum tracker
# ═══════════════════════════════════════════════════════════════════════════════

class MomentumTracker:
    """Track rolling ELO slope (last N games) as a momentum feature."""

    def __init__(self, window=MOMENTUM_WINDOW):
        self.window = window
        self.history: Dict[int, list] = defaultdict(list)
        self._current_season = None

    def season_reset(self, season):
        """Clear histories at season boundary."""
        if self._current_season is not None and season != self._current_season:
            self.history.clear()
        self._current_season = season

    def record(self, tid, elo, season=None):
        if season is not None:
            self.season_reset(season)
        self.history[tid].append(elo)
        # Keep only last window * 2 entries (memory efficiency)
        if len(self.history[tid]) > self.window * 3:
            self.history[tid] = self.history[tid][-self.window * 2:]

    def get_momentum(self, tid):
        """Slope of ELO over last N games via linear regression."""
        vals = self.history[tid]
        n = min(len(vals), self.window)
        if n < 3:
            return 0.0
        y = np.array(vals[-n:])
        x = np.arange(n, dtype=float)
        slope = np.polyfit(x, y, 1)[0]
        return slope


# ═══════════════════════════════════════════════════════════════════════════════
#  Master function: compute all ratings across all games
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_ratings(all_games: pd.DataFrame) -> pd.DataFrame:
    """
    Process all games chronologically, computing ELO/Glicko/Margin-ELO.
    
    Inputs: all_games DataFrame with columns:
        Season, DayNum, WTeamID, LTeamID, WScore, LScore, WLoc, source
    
    Returns: DataFrame with one row per (Season, TeamID) containing
    pre-tournament snapshot of all rating features.
    """
    # Sort strictly chronologically
    games = all_games.sort_values(["Season", "DayNum"]).reset_index(drop=True)

    elo = EloRating()
    melo = MarginElo()
    glicko = GlickoRating()
    momentum = MomentumTracker()

    # We'll store snapshots: for each (Season, TeamID) we keep
    # the rating AFTER all regular season games (before tournament)
    # We also store game-level ratings for later matchup features

    game_ratings = []

    for _, row in games.iterrows():
        season = int(row["Season"])
        wid = int(row["WTeamID"])
        lid = int(row["LTeamID"])
        wscore = row["WScore"]
        lscore = row["LScore"]
        margin = wscore - lscore if pd.notna(wscore) and pd.notna(lscore) else 0
        wloc = row.get("WLoc", "N")
        if pd.isna(wloc):
            wloc = "N"

        # Get pre-game ratings
        elo_w_pre, elo_l_pre = elo.update(wid, lid, wloc=wloc, season=season)
        melo_w_pre, melo_l_pre = melo.update(wid, lid, margin=margin, wloc=wloc, season=season)
        gr = glicko.update(wid, lid, season=season)
        glicko_w_pre, glicko_rd_w_pre, glicko_l_pre, glicko_rd_l_pre = gr

        # Record momentum
        momentum.season_reset(season)
        momentum.record(wid, elo.get(wid), season=season)
        momentum.record(lid, elo.get(lid), season=season)

        game_ratings.append({
            "Season": season,
            "DayNum": int(row["DayNum"]),
            "WTeamID": wid,
            "LTeamID": lid,
            "source": row.get("source", ""),
            "elo_W": elo_w_pre,
            "elo_L": elo_l_pre,
            "melo_W": melo_w_pre,
            "melo_L": melo_l_pre,
            "glicko_W": glicko_w_pre,
            "glicko_rd_W": glicko_rd_w_pre,
            "glicko_L": glicko_l_pre,
            "glicko_rd_L": glicko_rd_l_pre,
            # Post-game ratings (after this game's update)
            "elo_W_post": elo.get(wid),
            "elo_L_post": elo.get(lid),
            "melo_W_post": melo.get(wid),
            "melo_L_post": melo.get(lid),
            "glicko_W_post": glicko.ratings[wid],
            "glicko_rd_W_post": glicko.rds[wid],
            "glicko_L_post": glicko.ratings[lid],
            "glicko_rd_L_post": glicko.rds[lid],
        })

    game_ratings_df = pd.DataFrame(game_ratings)

    # ── Build season-level snapshots: last rating before tournament ────────
    # For each season, take the LAST regular-season rating per team
    snapshots = []
    for season in games["Season"].unique():
        season = int(season)
        # Get all teams that played this season
        sg = games[games["Season"] == season]
        all_teams = set(sg["WTeamID"].unique()) | set(sg["LTeamID"].unique())

        # Get pre-tournament ratings (current state of rating objects
        # after processing all games up to this point would need resimulation…)
        # Instead, use game_ratings_df to get last regular-season game per team
        sg_rat = game_ratings_df[
            (game_ratings_df["Season"] == season) &
            (game_ratings_df["source"].isin(["regular", "conf_tourney"]))
        ]

        # Last game for each team (as winner or loser)
        for tid in all_teams:
            as_w = sg_rat[sg_rat["WTeamID"] == tid]
            as_l = sg_rat[sg_rat["LTeamID"] == tid]

            last_day_w = as_w["DayNum"].max() if len(as_w) > 0 else -1
            last_day_l = as_l["DayNum"].max() if len(as_l) > 0 else -1

            # Take whichever is later
            if last_day_w >= last_day_l and len(as_w) > 0:
                last = as_w[as_w["DayNum"] == last_day_w].iloc[-1]
                e, me, g, grd = last["elo_W_post"], last["melo_W_post"], last["glicko_W_post"], last["glicko_rd_W_post"]
            elif len(as_l) > 0:
                last = as_l[as_l["DayNum"] == last_day_l].iloc[-1]
                e, me, g, grd = last["elo_L_post"], last["melo_L_post"], last["glicko_L_post"], last["glicko_rd_L_post"]
            else:
                continue  # no regular season data

            snapshots.append({
                "Season": season,
                "TeamID": int(tid),
                "Elo": e,
                "MarginElo": me,
                "Glicko": g,
                "GlickoRD": grd,
            })

    snapshot_df = pd.DataFrame(snapshots)

    # ── Add momentum (recomputed per season) ──────────────────────────────
    # Rebuild momentum per team from the game-level data
    mom_records = []
    for season in snapshot_df["Season"].unique():
        sg = game_ratings_df[
            (game_ratings_df["Season"] == season) &
            (game_ratings_df["source"].isin(["regular", "conf_tourney"]))
        ].sort_values("DayNum")

        team_elos = defaultdict(list)
        for _, row in sg.iterrows():
            team_elos[int(row["WTeamID"])].append(row["elo_W"])
            team_elos[int(row["LTeamID"])].append(row["elo_L"])

        for tid, vals in team_elos.items():
            n = min(len(vals), MOMENTUM_WINDOW)
            if n >= 3:
                y = np.array(vals[-n:])
                x = np.arange(n, dtype=float)
                slope = np.polyfit(x, y, 1)[0]
            else:
                slope = 0.0
            mom_records.append({"Season": int(season), "TeamID": tid, "Momentum": slope})

    mom_df = pd.DataFrame(mom_records)
    snapshot_df = snapshot_df.merge(mom_df, on=["Season", "TeamID"], how="left")
    snapshot_df["Momentum"] = snapshot_df["Momentum"].fillna(0.0)

    return snapshot_df, game_ratings_df
