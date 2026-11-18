# %% [markdown]
# # Yo-Yo Clubs vs. Stable Ligue 1: Step 1, scope audit & cohort definition
#
# **Why this step matters:** before computing a single KPI we need to know
# 1. which French competitions the dataset actually covers,
# 2. which seasons are reliable for each data source (matches, line-ups, transfers, valuations),
# 3. which clubs belong to which cohort, and how many promotion events we can study.
#
# Every chart in later steps inherits these choices, so we document them explicitly.

# %%
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

# ---------------- Parameters: change them here, never inside the code ----------------
# Paths are relative to the repository root: open the repo folder in VS Code and run from there
DATA_DIR = Path("data")      # the 7 Kaggle CSVs go here (this folder is not pushed to GitHub)
OUTPUT_DIR = Path("outputs")  # created automatically

N_SEASONS = 10              # length of the analysis window, in seasons
COMPLETENESS_MIN = 0.95     # share of expected league games needed to call a season "complete"
MIN_L1_SEASONS_STABLE = 3   # a promoted club needs 3+ L1 seasons (and no relegation) to count as stable
KNOWN_SHORT_SEASONS = {2019: "COVID stop (round 28)"}  # explains a low match_pct, not a data gap

OUTPUT_DIR.mkdir(exist_ok=True)


# %%
def season_from_date(dates: pd.Series) -> pd.Series:
    """Season start year from a date: Aug-2023 -> 2023, Mar-2024 -> 2023 (i.e. season 2023/24)."""
    return dates.dt.year.where(dates.dt.month >= 7, dates.dt.year - 1).astype("Int64")


def season_from_label(labels: pd.Series) -> pd.Series:
    """Transfermarkt 'transfer_season' label to start year: '23/24' -> 2023, '99/00' -> 1999."""
    yy = pd.to_numeric(labels.astype(str).str[:2], errors="coerce")
    return (2000 + yy).where(yy <= 50, 1900 + yy).astype("Int64")


def season_label(season: int) -> str:
    """2023 -> '2023/24' (for display only)."""
    return f"{season}/{str(season + 1)[-2:]}"


# %% [markdown]
# ## 0. Load the data

# %%
competitions = pd.read_csv(DATA_DIR / "competitions.csv")
clubs = pd.read_csv(DATA_DIR / "clubs.csv")
games = pd.read_csv(DATA_DIR / "games.csv", parse_dates=["date"])
transfers = pd.read_csv(DATA_DIR / "transfers.csv", parse_dates=["transfer_date"])
valuations = pd.read_csv(DATA_DIR / "player_valuations.csv", parse_dates=["date"])
# appearances is the biggest file (~1.7M rows): we only load the 3 columns we need
appearances = pd.read_csv(DATA_DIR / "appearances.csv", usecols=["game_id", "player_id", "player_club_id"])

tables = {"competitions": competitions, "clubs": clubs, "games": games,
          "transfers": transfers, "valuations": valuations}
for name, df in tables.items():
    print(f"{name:<13}{len(df):>10,} rows | {', '.join(df.columns)}")
print(f"\nLatest date in games.csv: {games['date'].max():%d %b %Y}")


# %% [markdown]
# ## A. French scope: which competitions and clubs are in the data?

# %%
fr_comps = competitions.loc[competitions["country_name"] == "France",
                            ["competition_id", "name", "type", "sub_type"]]
print(fr_comps.to_string(index=False))

FR_LEAGUES = [c for c in ["FR1", "FR2"] if c in set(games["competition_id"])]
FR_CUPS = fr_comps.loc[fr_comps["type"] == "domestic_cup", "competition_id"].tolist()
HAS_FR2 = "FR2" in FR_LEAGUES

print(f"\nLeagues with match data: {FR_LEAGUES} | Domestic cups: {FR_CUPS}")
if not HAS_FR2:
    print("WARNING: no Ligue 2 (FR2) games in the data. A club-season outside Ligue 1 is read as "
          "'Ligue 2 or lower', and Ligue 2-season KPIs will rely on transfers and valuations only.")

# %%
fr_games = games[games["competition_id"].isin(FR_LEAGUES + FR_CUPS)]

# One row per club x season x competition, built from home AND away games
club_comp_season = (
    pd.concat([
        fr_games[["season", "competition_id", "home_club_id"]].rename(columns={"home_club_id": "club_id"}),
        fr_games[["season", "competition_id", "away_club_id"]].rename(columns={"away_club_id": "club_id"}),
    ])
    .dropna(subset=["club_id"])
    .astype({"club_id": int, "season": int})
    .groupby(["club_id", "season", "competition_id"])
    .size()
    .rename("games")
    .reset_index()
)
fr_club_ids = club_comp_season["club_id"].unique()

# Ligue 1 membership: a club counts as an L1 member of a season only if it played a real share of it.
# This guards against a few stray games (e.g. a Ligue 2 club in the L1/L2 relegation play-off).
fr1_club_games = club_comp_season[club_comp_season["competition_id"] == "FR1"]
median_games = fr1_club_games.groupby("season")["games"].transform("median")
l1_members = fr1_club_games.loc[fr1_club_games["games"] >= 0.5 * median_games, ["club_id", "season"]]

# Readable club names: clubs.csv first, games.csv as a fallback (cup-only clubs are not in clubs.csv)
names_from_games = (
    pd.concat([
        games[["home_club_id", "home_club_name"]].set_axis(["club_id", "club_name"], axis=1),
        games[["away_club_id", "away_club_name"]].set_axis(["club_id", "club_name"], axis=1),
    ])
    .dropna()
    .astype({"club_id": int})
    .drop_duplicates("club_id")
    .set_index("club_id")["club_name"]
)
club_names = clubs.set_index("club_id")["name"].combine_first(names_from_games)

print(f"French clubs found in league + cup games: {len(fr_club_ids)}")


# %% [markdown]
# ## B. Coverage audit: which seasons can we trust, source by source?

# %%
fr1 = games[games["competition_id"] == "FR1"]

# 1) Matches: Ligue 1 games played vs. the theoretical number, n_clubs x (n_clubs - 1)
match_cov = pd.DataFrame({
    "l1_clubs": l1_members.groupby("season").size(),
    "games_played": fr1.groupby("season")["home_club_goals"].count(),  # count() skips games without a score
})
match_cov["games_expected"] = match_cov["l1_clubs"] * (match_cov["l1_clubs"] - 1)
match_cov["match_pct"] = match_cov["games_played"] / match_cov["games_expected"]

# 2) Line-ups: share of Ligue 1 games with player appearances recorded
lineup_cov = (
    fr1.assign(has_lineup=fr1["game_id"].isin(appearances["game_id"]))
    .groupby("season")["has_lineup"].mean()
    .rename("lineup_pct")
)

# 3) Transfers touching a French club (incoming or outgoing)
transfers["season"] = season_from_label(transfers["transfer_season"])
fr_transfers = transfers[transfers["from_club_id"].isin(fr_club_ids)
                         | transfers["to_club_id"].isin(fr_club_ids)]
transfer_cov = (
    fr_transfers.assign(fee_known=fr_transfers["transfer_fee"].notna(),
                        fee_paid=fr_transfers["transfer_fee"] > 0)
    .groupby("season")
    .agg(transfers=("player_id", "size"),
         fee_known_pct=("fee_known", "mean"),
         fee_paid_pct=("fee_paid", "mean"),
         fees_eur_m=("transfer_fee", "sum"))
)
transfer_cov["fees_eur_m"] = transfer_cov["fees_eur_m"] / 1e6

# 4) Valuations tagged with a French club
valuations["season"] = season_from_date(valuations["date"])
valuation_cov = (
    valuations[valuations["current_club_id"].isin(fr_club_ids)]
    .groupby("season")
    .agg(players_valued=("player_id", "nunique"), clubs_valued=("current_club_id", "nunique"))
)

# 5) Sanity check for Step 2: is a valuation tagged with the club the player was at AT THAT TIME,
#    or with his CURRENT club? We compare it with the club he actually played for (line-ups).
#    ~85-95% every season = historical tag. High only for recent seasons = current club only.
fr1_rosters = (
    appearances.merge(fr1[["game_id", "season"]], on="game_id")
    [["player_id", "player_club_id", "season"]]
    .drop_duplicates()
)
last_valuation = (
    valuations.sort_values("date")
    .drop_duplicates(["player_id", "season"], keep="last")
    [["player_id", "season", "current_club_id"]]
)
attribution = (
    fr1_rosters.merge(last_valuation, on=["player_id", "season"])
    .assign(same_club=lambda d: d["player_club_id"] == d["current_club_id"])
    .groupby("season")["same_club"].mean()
    .rename("val_club_match")
)

# %%
coverage = (
    match_cov.join([lineup_cov, transfer_cov, valuation_cov, attribution])
    .astype({"l1_clubs": "Int64", "games_played": "Int64", "games_expected": "Int64",
             "transfers": "Int64", "players_valued": "Int64", "clubs_valued": "Int64"})
)

LAST_SEASON = int(coverage.index[coverage["match_pct"] >= COMPLETENESS_MIN].max())
WINDOW = list(range(LAST_SEASON - N_SEASONS + 1, LAST_SEASON + 1))

coverage["note"] = coverage.index.map(KNOWN_SHORT_SEASONS).fillna("")
coverage.loc[coverage.index > LAST_SEASON, "note"] = "in progress"

audit = coverage.loc[coverage.index >= WINDOW[0]].copy()  # analysis window + season in progress
audit.index = audit.index.map(season_label)
print(f"Analysis window: {season_label(WINDOW[0])} to {season_label(WINDOW[-1])} ({N_SEASONS} seasons)\n")
print(audit.round(2).to_string())


# %% [markdown]
# ## C. Cohorts: one label per club, based on its Ligue 1 trajectory over the window
#
# An **L1 spell** is a run of consecutive Ligue 1 seasons.
# - **Stable Ligue 1**: in L1 every season, OR promoted once, never relegated, 3+ L1 seasons
# - **Yo-Yo**: 2 or more separate L1 spells (the club keeps bouncing between divisions)
# - **Promoted, then relegated**: a single L1 spell that started with a promotion and ended with a relegation
# - **Fallen from Ligue 1**: already in L1 before the window started, relegated once, never back
# - **Not classified yet**: promoted, still in L1, but fewer than 3 L1 seasons (outcome unknown):
#   excluded from cohort comparisons, but kept in the promotion events (section D)
# - **Ligue 2 / Cup-only**: never in L1 during the window (benchmark group)

# %%
in_window = club_comp_season[club_comp_season["season"].isin(WINDOW)]

# Club x season grid: 1 = played Ligue 1 that season, 0 = did not.
# It also includes the season BEFORE the window, so that a club promoted into the first
# window season (e.g. Nancy, Dijon in 2016/17) is seen as promoted, not as an established L1 club.
pre_window = [WINDOW[0] - 1] if (WINDOW[0] - 1) in set(l1_members["season"]) else []
l1_grid_ext = (
    l1_members[l1_members["season"].isin(pre_window + WINDOW)]
    .assign(in_l1=1)
    .pivot_table(index="club_id", columns="season", values="in_l1", fill_value=0)
    .reindex(index=in_window["club_id"].unique(), columns=pre_window + WINDOW, fill_value=0)
    .astype(int)
)
l1_grid = l1_grid_ext[WINDOW]  # the window only

# Promotions: moves INTO a window season (including the first one, thanks to the extra season).
# Relegations: only from an L1 season INSIDE the window (a drop at the end of 2015/16 is not ours).
promotion_moves = l1_grid_ext.diff(axis=1)[WINDOW] == 1
relegation_moves = l1_grid.diff(axis=1) == -1
new_spell = (l1_grid == 1) & (l1_grid.shift(1, axis=1, fill_value=0) == 0)  # start of an L1 spell

cohorts = pd.DataFrame({
    "club": club_names.reindex(l1_grid.index),
    "timeline": l1_grid.replace({1: "■", 0: "·"}).agg("".join, axis=1),
    "l1_seasons": l1_grid.sum(axis=1),
    "promotions": promotion_moves.sum(axis=1),
    "relegations": relegation_moves.sum(axis=1),
    "l1_spells": new_spell.sum(axis=1),
})
cohorts["switches"] = cohorts["promotions"] + cohorts["relegations"]


def assign_cohort(club: pd.Series) -> tuple[str, str]:
    """Return (cohort, detail) for one club, from its Ligue 1 trajectory over the window."""
    if club["l1_seasons"] == 0:
        return "Ligue 2 / Cup-only", "Never in L1 during the window"
    if club["l1_spells"] >= 2:
        return "Yo-Yo", f"{club['l1_spells']} separate L1 spells"
    # From here: exactly one L1 spell
    if club["switches"] == 0:
        return "Stable Ligue 1", "Core: L1 every season"
    if club["promotions"] == 1 and club["relegations"] == 1:
        return "Promoted, then relegated", f"{club['l1_seasons']} L1 season(s) before going down"
    if club["promotions"] == 0 and club["relegations"] == 1:
        return "Fallen from Ligue 1", "Established L1 club, relegated, not back"
    # Last case: promoted once (promotions == 1, relegations == 0) and still in Ligue 1
    if club["l1_seasons"] >= MIN_L1_SEASONS_STABLE:
        return "Stable Ligue 1", "Promoted, then stabilised"
    return "Not classified yet", f"Promoted, only {club['l1_seasons']} L1 season(s) so far"


cohorts[["cohort", "cohort_detail"]] = cohorts.apply(assign_cohort, axis=1, result_type="expand")

COHORT_ORDER = ["Stable Ligue 1", "Yo-Yo", "Promoted, then relegated", "Fallen from Ligue 1",
                "Not classified yet", "Ligue 2 / Cup-only"]
cohorts["cohort"] = pd.Categorical(cohorts["cohort"], categories=COHORT_ORDER, ordered=True)
# Filter to use in Steps 2-3: True for every club we compare across cohorts
cohorts["in_comparison"] = cohorts["cohort"] != "Not classified yet"

# %%
print(cohorts["cohort"].value_counts().reindex(COHORT_ORDER).to_string(), "\n")

print(f"Timeline: one character per season, {season_label(WINDOW[0])} to {season_label(WINDOW[-1])}"
      "  (■ = Ligue 1, · = not in Ligue 1)\n")
l1_clubs = cohorts[cohorts["l1_seasons"] > 0].sort_values(["cohort", "l1_seasons"], ascending=[True, False])
print(l1_clubs[["club", "timeline", "l1_seasons", "promotions", "relegations",
                "cohort", "cohort_detail"]].to_string(index=False))

games_in_window = in_window.groupby("club_id")["games"].sum().rename("games_in_window")
cup_only = cohorts[cohorts["cohort"] == "Ligue 2 / Cup-only"].join(games_in_window)
print(f"\nLigue 2 / Cup-only clubs: {len(cup_only)}")
if len(cup_only) > 0:
    print(cup_only.nlargest(10, "games_in_window")[["club", "games_in_window"]].to_string(index=False))


# %% [markdown]
# ## D. Promotion events: the true sample size for "survival after promotion"
#
# The threshold question is tested on **promotions**, not on clubs: each promoted club-season is
# one observation, and the outcome is "still in Ligue 1 the following season?".

# %%
l1_long = l1_members.drop_duplicates()
l1_keys = pd.MultiIndex.from_frame(l1_long)


def was_in_l1(df: pd.DataFrame, offset: int) -> np.ndarray:
    """VLOOKUP-style: was this club in Ligue 1 `offset` seasons away from the row's season?"""
    keys = pd.MultiIndex.from_arrays([df["club_id"], df["season"] + offset])
    return keys.isin(l1_keys)


first_l1, last_l1 = l1_long["season"].min(), l1_long["season"].max()
promotions = l1_long[
    l1_long["season"].isin(WINDOW)
    & (l1_long["season"] > first_l1)   # we need the previous season to call it a promotion
    & ~was_in_l1(l1_long, -1)
].copy()
promotions["outcome_year_1"] = np.select(
    [was_in_l1(promotions, +1), promotions["season"] + 1 > last_l1],
    ["Survived", "Unknown (next season not in data)"],
    default="Relegated",
)
promotions["club"] = club_names.reindex(promotions["club_id"]).to_numpy()
promotions["season_label"] = promotions["season"].map(season_label)

# Reconciliation, like a controller would do: both tables must count the same promotions
check = "OK" if cohorts["promotions"].sum() == len(promotions) else "MISMATCH, investigate"
print(f"Reconciliation: {cohorts['promotions'].sum()} promotions in the cohort table vs "
      f"{len(promotions)} promotion events -> {check}\n")

print(f"Promotion events in the window: {len(promotions)}")
print(promotions["outcome_year_1"].value_counts().to_string(), "\n")
print(promotions.sort_values(["season", "club"])[["season_label", "club", "outcome_year_1"]]
      .to_string(index=False))


# %% [markdown]
# ## E. Save the perimeter for Step 2

# %%
# Long version of the Ligue 1 grid: one row per club x season. This is the backbone of Step 2.
club_season_grid = (
    l1_grid.rename_axis(index="club_id", columns=None)
    .reset_index()
    .melt(id_vars="club_id", var_name="season", value_name="in_l1")
    .merge(cohorts[["club", "cohort"]], left_on="club_id", right_index=True)
    .sort_values(["club_id", "season"])
)

cohorts.to_csv(OUTPUT_DIR / "step1_club_cohorts.csv")
coverage.to_csv(OUTPUT_DIR / "step1_coverage_audit.csv")
promotions.to_csv(OUTPUT_DIR / "step1_promotion_events.csv", index=False)
club_season_grid.to_csv(OUTPUT_DIR / "step1_club_season_grid.csv", index=False)
l1_members.to_csv(OUTPUT_DIR / "step1_l1_membership.csv", index=False)  # every L1 club-season in the data
print(f"Saved 5 files to {OUTPUT_DIR.resolve()}")
