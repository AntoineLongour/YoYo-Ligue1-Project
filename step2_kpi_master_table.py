# %% [markdown]
# # Yo-Yo Clubs vs. Stable Ligue 1: Step 2, KPI engineering
#
# Builds three tables, each KPI defined once and reused in Steps 3-4:
# 1. **League panel** (`step2_l1_panel.csv`): one row per Ligue 1 club x season, for EVERY season in
#    the data (not just the window). Used for league-wide comparisons and the survival test.
# 2. **Master table** (`step2_master_table.csv`): one row per club x season for the clubs of the
#    10-season window (Step 1), including their seasons outside Ligue 1.
# 3. **Promotion events** (`step2_promotion_events.csv`): every promoted club-season and its outcome.
#
# | KPI | Definition | Available for |
# |---|---|---|
# | `points`, `ppg`, `gd`, `rank`, `bottom_half` | Computed from league results; rank on points per game, then goal difference, then goals scored | Ligue 1 seasons |
# | `squad_value_m` | Sum of the market values on 1 Sept of every player who played at least one league game | Ligue 1 seasons |
# | `xi_value_m` | Value of the "average eleven on the pitch": minutes-weighted average player value x 11 (not inflated by rotation) | Ligue 1 seasons |
# | `squad_value_index` | `xi_value_m` / median of the Ligue 1 clubs that season (1.0 = median club): comparable across years | Ligue 1 seasons |
# | `ppg_per_100m` | Points per game per €100M of `xi_value_m`: capital efficiency | Ligue 1 seasons |
# | `valuation_coverage` | Share of league minutes played by players who have a valuation (quality flag) | Ligue 1 seasons |
# | `avg_age` | Minutes-weighted age on 1 Sept | Ligue 1 seasons |
# | `new_signings_min_pct` | Share of league minutes played by players signed during that season's windows (squad turnover) | Ligue 1 seasons |
# | `turnover_index` | `new_signings_min_pct` / median of the Ligue 1 clubs that season (corrects the transfer-coverage drift) | Ligue 1 seasons |
# | `n_in`, `n_out`, `fees_in_m`, `fees_out_m`, `net_spend_m` | Recorded transfers and known fees (€M); net spend > 0 = net buyer | All seasons |
# | `net_spend_pctile` | Percentile of net spend among Ligue 1 clubs that season (1.0 = biggest net buyer) | Ligue 1 seasons |
# | `preseason_net_spend_m`, `preseason_net_spend_pctile` | Same, summer window only (deals dated up to 5 Sept): spending decided before any result | Ligue 1 seasons |
# | `tag_value_m` | Sum of valuations tagged with the club, closest to 1 Sept (lower quality, see Step 1) | All seasons, window clubs |
#
# **Why value squads on 1 September?** The summer window is closed and the season has barely started,
# so the value reflects the resources the club chose to start with, not how well the season went.
# Using a value from May would leak the season's results into the "cause" (look-ahead bias).

# %%
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

# ---------------- Parameters ----------------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

VALUATION_MONTH, VALUATION_DAY = 9, 1  # squads are valued on 1 September of each season
MAX_STALE_DAYS = 400       # ignore a valuation older than this on the valuation date
FORWARD_GRACE_DAYS = 180   # no earlier valuation (e.g. a young player)? accept one up to 6 months later
PRESEASON_CUTOFF = (9, 5)  # summer-window transfers = dated up to 5 September (deadline + registration lag)

# Documented manual input: the relegation play-off of the LAST window season is not in the data.
# 2025/26 source: ligue1.com, list of play-off winners, "2025/26 : OGC Nice (maintenu)" -> stayed up.
PLAYOFF_CLUB_RELEGATED = {2025: False}  # season -> was the play-off club relegated?


def season_from_date(dates: pd.Series) -> pd.Series:
    """Season start year from a date: Aug-2023 -> 2023, Mar-2024 -> 2023."""
    return dates.dt.year - (dates.dt.month < 7).astype(int)


def season_from_label(labels: pd.Series) -> pd.Series:
    """Transfermarkt 'transfer_season' label to start year: '23/24' -> 2023, '99/00' -> 1999."""
    yy = pd.to_numeric(labels.astype(str).str[:2], errors="coerce")
    return (2000 + yy).where(yy <= 50, 1900 + yy).astype("Int64")


def valuation_date(seasons: pd.Series) -> pd.Series:
    """Valuation date of a season: 2023 -> 2023-09-01."""
    parts = pd.DataFrame({"year": seasons, "month": VALUATION_MONTH, "day": VALUATION_DAY})
    return pd.to_datetime(parts).astype("datetime64[ns]")


def season_label(season: int) -> str:
    """2023 -> '2023/24' (for display only)."""
    return f"{season}/{str(season + 1)[-2:]}"


# %% [markdown]
# ## 0. Load Step 1 outputs and raw data

# %%
grid = pd.read_csv(OUTPUT_DIR / "step1_club_season_grid.csv")        # window clubs x seasons, cohort
l1_members = pd.read_csv(OUTPUT_DIR / "step1_l1_membership.csv")     # every L1 club-season in the data
step1_promotions = pd.read_csv(OUTPUT_DIR / "step1_promotion_events.csv")

WINDOW = sorted(grid["season"].unique())
LAST_SEASON = WINDOW[-1]
PANEL_SEASONS = sorted(s for s in l1_members["season"].unique() if s <= LAST_SEASON)
l1_keys = pd.MultiIndex.from_frame(l1_members[["club_id", "season"]])  # includes a season in progress, if any

games = pd.read_csv(DATA_DIR / "games.csv", parse_dates=["date"])
clubs = pd.read_csv(DATA_DIR / "clubs.csv", usecols=["club_id", "name"])
appearances = pd.read_csv(DATA_DIR / "appearances.csv",
                          usecols=["game_id", "player_id", "player_club_id", "minutes_played"])
players = pd.read_csv(DATA_DIR / "players.csv", usecols=["player_id", "date_of_birth"],
                      parse_dates=["date_of_birth"])
valuations = pd.read_csv(DATA_DIR / "player_valuations.csv", parse_dates=["date"])
transfers = pd.read_csv(DATA_DIR / "transfers.csv", parse_dates=["transfer_date"])
transfers["season"] = season_from_label(transfers["transfer_season"])

# Club names: clubs.csv first, games.csv as a fallback (same rule as Step 1)
names_from_games = (
    pd.concat([
        games[["home_club_id", "home_club_name"]].set_axis(["club_id", "club"], axis=1),
        games[["away_club_id", "away_club_name"]].set_axis(["club_id", "club"], axis=1),
    ])
    .dropna().astype({"club_id": int}).drop_duplicates("club_id").set_index("club_id")["club"]
)
club_names = clubs.set_index("club_id")["name"].combine_first(names_from_games)

print(f"Window: {season_label(WINDOW[0])} to {season_label(LAST_SEASON)} ({len(grid)} club-seasons) | "
      f"League panel: {season_label(PANEL_SEASONS[0])} to {season_label(PANEL_SEASONS[-1])}")


# %% [markdown]
# ## A. Sporting KPIs (every Ligue 1 season)

# %%
fr1 = games[(games["competition_id"] == "FR1") & games["season"].isin(PANEL_SEASONS)
            & games["home_club_goals"].notna()]
# Keep only games between two Ligue 1 members of that season: drops L1/L2 relegation play-off games
home_is_l1 = pd.MultiIndex.from_arrays([fr1["home_club_id"], fr1["season"]]).isin(l1_keys)
away_is_l1 = pd.MultiIndex.from_arrays([fr1["away_club_id"], fr1["season"]]).isin(l1_keys)
fr1 = fr1[home_is_l1 & away_is_l1]

# One row per club per game: the home view and the away view of each match
club_games = pd.concat([
    fr1.rename(columns={"home_club_id": "club_id", "home_club_goals": "gf", "away_club_goals": "ga"}),
    fr1.rename(columns={"away_club_id": "club_id", "away_club_goals": "gf", "home_club_goals": "ga"}),
])[["club_id", "season", "game_id", "gf", "ga"]]
club_games["points"] = np.select(
    [club_games["gf"] > club_games["ga"], club_games["gf"] == club_games["ga"]], [3, 1], default=0)

sporting = (
    club_games.groupby(["club_id", "season"])
    .agg(games=("game_id", "size"), points=("points", "sum"), gf=("gf", "sum"), ga=("ga", "sum"))
    .reset_index()
)
sporting["gd"] = sporting["gf"] - sporting["ga"]
sporting["ppg"] = sporting["points"] / sporting["games"]

# Rank on points per game (same as points in a full season, and fair in the COVID season),
# then goal difference, then goals scored. Official tables can differ (head-to-head tie-breaks,
# points deductions): e.g. 2023/24, where Metz finished ahead of Lorient on head-to-head.
sporting = sporting.sort_values(["season", "ppg", "gd", "gf"], ascending=[True, False, False, False])
sporting["rank"] = sporting.groupby("season").cumcount() + 1
sporting["bottom_half"] = sporting["rank"] > sporting.groupby("season")["club_id"].transform("size") / 2


# %% [markdown]
# ## B. Squad KPIs from line-ups (every Ligue 1 season)
# The squad = every player who played at least one league game for the club that season.

# %%
league_apps = appearances.merge(fr1[["game_id", "season"]], on="game_id")
squad = (
    league_apps.groupby(["player_club_id", "season", "player_id"], as_index=False)
    .agg(minutes=("minutes_played", "sum"), apps=("game_id", "size"))
    .rename(columns={"player_club_id": "club_id"})
)

# 1) Market value on 1 September: the latest valuation before that date (not older than MAX_STALE_DAYS),
#    or, if none, the first one within FORWARD_GRACE_DAYS after it
squad["value_date"] = valuation_date(squad["season"])
squad = squad.sort_values("value_date").reset_index(drop=True)  # merge_asof needs sorted dates
vals = (valuations[["player_id", "date", "market_value_in_eur"]].dropna()
        .astype({"date": "datetime64[ns]"})
        .sort_values("date"))

value_before = pd.merge_asof(squad, vals, left_on="value_date", right_on="date", by="player_id",
                             direction="backward", tolerance=pd.Timedelta(days=MAX_STALE_DAYS))
value_after = pd.merge_asof(squad, vals, left_on="value_date", right_on="date", by="player_id",
                            direction="forward", tolerance=pd.Timedelta(days=FORWARD_GRACE_DAYS))
squad["value_eur"] = value_before["market_value_in_eur"].fillna(value_after["market_value_in_eur"])

# 2) Age on 1 September
squad = squad.merge(players, on="player_id", how="left")
squad["age"] = (squad["value_date"] - squad["date_of_birth"]).dt.days / 365.25

# 3) New signing = a transfer INTO this club during this season's windows (summer or winter)
arrival_keys = pd.MultiIndex.from_frame(
    transfers[["player_id", "to_club_id", "season"]].dropna().astype("int64"))
squad["new_signing"] = pd.MultiIndex.from_frame(squad[["player_id", "club_id", "season"]]).isin(arrival_keys)

# %%
# Minutes-weighted building blocks (a player who played 3,000 minutes weighs more than one who played 30)
squad["valued_min"] = squad["minutes"].where(squad["value_eur"].notna(), 0)
squad["aged_min"] = squad["minutes"].where(squad["age"].notna(), 0)
squad["new_min"] = squad["minutes"].where(squad["new_signing"], 0)
squad["value_x_min"] = squad["value_eur"].fillna(0) * squad["minutes"]
squad["age_x_min"] = squad["age"].fillna(0) * squad["minutes"]

squad_kpis = squad.groupby(["club_id", "season"]).agg(
    players_used=("player_id", "size"),
    new_signings_used=("new_signing", "sum"),
    squad_value_m=("value_eur", "sum"),
    minutes=("minutes", "sum"),
    valued_min=("valued_min", "sum"),
    aged_min=("aged_min", "sum"),
    new_min=("new_min", "sum"),
    value_x_min=("value_x_min", "sum"),
    age_x_min=("age_x_min", "sum"),
)
squad_kpis["squad_value_m"] = squad_kpis["squad_value_m"] / 1e6
squad_kpis["xi_value_m"] = squad_kpis["value_x_min"] / squad_kpis["valued_min"] * 11 / 1e6
squad_kpis["valuation_coverage"] = squad_kpis["valued_min"] / squad_kpis["minutes"]
squad_kpis["avg_age"] = squad_kpis["age_x_min"] / squad_kpis["aged_min"]
squad_kpis["new_signings_min_pct"] = squad_kpis["new_min"] / squad_kpis["minutes"]
squad_kpis = squad_kpis[["players_used", "new_signings_used", "squad_value_m", "xi_value_m",
                         "valuation_coverage", "avg_age", "new_signings_min_pct"]].reset_index()


# %% [markdown]
# ## C. Transfer KPIs (every club, every season of the panel)

# %%
panel_transfers = transfers[transfers["season"].isin(PANEL_SEASONS)]
transfers_in = (
    panel_transfers.groupby(["to_club_id", "season"])
    .agg(n_in=("player_id", "size"), fees_in_m=("transfer_fee", "sum"))
    .rename_axis(["club_id", "season"])
)
transfers_out = (
    panel_transfers.groupby(["from_club_id", "season"])
    .agg(n_out=("player_id", "size"), fees_out_m=("transfer_fee", "sum"))
    .rename_axis(["club_id", "season"])
)
transfer_kpis = transfers_in.join(transfers_out, how="outer").fillna(0)
transfer_kpis[["fees_in_m", "fees_out_m"]] = transfer_kpis[["fees_in_m", "fees_out_m"]] / 1e6
transfer_kpis["net_spend_m"] = transfer_kpis["fees_in_m"] - transfer_kpis["fees_out_m"]
transfer_kpis = transfer_kpis.reset_index().dropna().astype({"club_id": "int64", "season": "int64"})

# Pre-season version: summer-window transfers only (dated up to PRESEASON_CUTOFF). This spending is
# decided before any result comes in, so it cannot be a reaction to a bad start (unlike January deals).
summer_cutoff = pd.to_datetime(pd.DataFrame({"year": panel_transfers["season"].astype("int64"),
                                             "month": PRESEASON_CUTOFF[0], "day": PRESEASON_CUTOFF[1]}))
summer = panel_transfers[panel_transfers["transfer_date"] <= summer_cutoff]
summer_in = summer.groupby(["to_club_id", "season"])["transfer_fee"].sum().rename("summer_in")
summer_out = summer.groupby(["from_club_id", "season"])["transfer_fee"].sum().rename("summer_out")
summer_net = (summer_in.rename_axis(["club_id", "season"]).to_frame()
              .join(summer_out.rename_axis(["club_id", "season"]), how="outer").fillna(0))
summer_net["preseason_net_spend_m"] = (summer_net["summer_in"] - summer_net["summer_out"]) / 1e6
summer_net = (summer_net[["preseason_net_spend_m"]].reset_index().dropna()
              .astype({"club_id": "int64", "season": "int64"}))

transfer_kpis = transfer_kpis.merge(summer_net, on=["club_id", "season"], how="outer").fillna(0)
TRANSFER_COLS = ["n_in", "n_out", "fees_in_m", "fees_out_m", "net_spend_m", "preseason_net_spend_m"]


# %% [markdown]
# ## D. League panel: every Ligue 1 club-season, with relative KPIs and lifecycle flags

# %%
panel = (
    l1_members[l1_members["season"].isin(PANEL_SEASONS)]
    .merge(sporting, on=["club_id", "season"], how="left")
    .merge(squad_kpis, on=["club_id", "season"], how="left")
    .merge(transfer_kpis, on=["club_id", "season"], how="left")
    .sort_values(["club_id", "season"])
    .reset_index(drop=True)
)
panel["club"] = club_names.reindex(panel["club_id"]).to_numpy()
panel[TRANSFER_COLS] = panel[TRANSFER_COLS].fillna(0)  # no recorded transfer means zero

# Relative KPIs: each club against the other Ligue 1 clubs of the SAME season
season_median = lambda col: panel.groupby("season")[col].transform("median")
panel["squad_value_index"] = panel["xi_value_m"] / season_median("xi_value_m")
panel["turnover_index"] = panel["new_signings_min_pct"] / season_median("new_signings_min_pct")
panel["net_spend_pctile"] = panel.groupby("season")["net_spend_m"].rank(pct=True)
panel["preseason_net_spend_pctile"] = panel.groupby("season")["preseason_net_spend_m"].rank(pct=True)
panel["ppg_per_100m"] = panel["ppg"] / panel["xi_value_m"] * 100

# Lifecycle flags, looked up in the full membership table (VLOOKUP-style)
prev_keys = pd.MultiIndex.from_arrays([panel["club_id"], panel["season"] - 1])
next_keys = pd.MultiIndex.from_arrays([panel["club_id"], panel["season"] + 1])
panel["promoted_season"] = (panel["season"] > PANEL_SEASONS[0]) & ~prev_keys.isin(l1_keys)
panel["relegated_at_end"] = pd.array(~next_keys.isin(l1_keys), dtype="boolean")

# Last window season: if the next season is not in the data, apply the competition rules to our
# computed final table. Last two = relegated directly; third from bottom = play-off (manual input above).
if (LAST_SEASON + 1) not in set(l1_members["season"]):
    last = panel["season"] == LAST_SEASON
    n_last = int(last.sum())
    direct = last & (panel["rank"] >= n_last - 1)
    playoff = last & (panel["rank"] == n_last - 2)
    panel.loc[last, "relegated_at_end"] = False
    panel.loc[direct, "relegated_at_end"] = True
    panel.loc[playoff, "relegated_at_end"] = PLAYOFF_CLUB_RELEGATED.get(LAST_SEASON, pd.NA)
    playoff_result = {True: "relegated", False: "stayed up"}.get(PLAYOFF_CLUB_RELEGATED.get(LAST_SEASON), "unknown")
    print(f"{season_label(LAST_SEASON)} outcome: relegated directly = {', '.join(panel.loc[direct, 'club'])} | "
          f"play-off = {', '.join(panel.loc[playoff, 'club'])} ({playoff_result})")

count_cols = ["games", "points", "gd", "rank", "players_used", "new_signings_used"]
panel[count_cols] = panel[count_cols].astype("Int64")


# %% [markdown]
# ## E. Promotion events (every season of the panel) and the master table (window clubs)

# %%
events = panel[panel["promoted_season"]].copy()
events["in_window"] = events["season"].isin(WINDOW)
events["outcome"] = events["relegated_at_end"].map({True: "Relegated", False: "Survived"}).fillna("Unknown")

# Reconciliation with Step 1: same promotions inside the window?
check = "OK" if events["in_window"].sum() == len(step1_promotions) else "MISMATCH, investigate"
print(f"Reconciliation: {events['in_window'].sum()} promotions in the window vs {len(step1_promotions)} "
      f"in Step 1 -> {check} | total promotion events in the panel: {len(events)}")

# Tag-based squad value (window clubs, all their seasons): the only option outside Ligue 1.
# For each player tagged with the club during the season, keep the valuation closest to 1 September.
OUR_CLUBS = grid["club_id"].unique()
tagged = valuations[valuations["current_club_id"].isin(OUR_CLUBS)].copy()
tagged["season"] = season_from_date(tagged["date"])
tagged = tagged[tagged["season"].isin(WINDOW)]
tagged["gap_days"] = (tagged["date"] - valuation_date(tagged["season"])).abs().dt.days
tag_kpis = (
    tagged.sort_values("gap_days")
    .drop_duplicates(["player_id", "current_club_id", "season"])  # one valuation per player and club
    .groupby(["current_club_id", "season"])
    .agg(tag_players=("player_id", "size"), tag_value_m=("market_value_in_eur", "sum"))
    .rename_axis(["club_id", "season"])
    .reset_index()
    .astype({"club_id": "int64", "season": "int64"})
)
tag_kpis["tag_value_m"] = tag_kpis["tag_value_m"] / 1e6

# Master table = Step 1 grid + panel KPIs (Ligue 1 seasons) + transfers and tags (all seasons)
panel_cols = [c for c in panel.columns if c not in TRANSFER_COLS + ["club"]]
master = (
    grid.merge(panel[panel_cols], on=["club_id", "season"], how="left")
    .merge(transfer_kpis, on=["club_id", "season"], how="left")
    .merge(tag_kpis, on=["club_id", "season"], how="left")
    .sort_values(["club_id", "season"])
    .reset_index(drop=True)
)
master["division"] = np.where(master["in_l1"] == 1, "Ligue 1", "Ligue 2 or lower")
master[TRANSFER_COLS] = master[TRANSFER_COLS].fillna(0)
master["promoted_season"] = master["promoted_season"].eq(True)  # empty outside Ligue 1 -> False
master.loc[master["division"] != "Ligue 1", "relegated_at_end"] = False
master["tag_players"] = master["tag_players"].astype("Int64")

MASTER_COLUMNS = [
    "club_id", "club", "season", "division", "cohort", "promoted_season", "relegated_at_end",
    "games", "points", "ppg", "gd", "rank", "bottom_half",
    "squad_value_m", "xi_value_m", "squad_value_index", "ppg_per_100m", "valuation_coverage",
    "avg_age", "players_used", "new_signings_used", "new_signings_min_pct", "turnover_index",
    "n_in", "n_out", "fees_in_m", "fees_out_m", "net_spend_m", "net_spend_pctile",
    "preseason_net_spend_m", "preseason_net_spend_pctile", "tag_players", "tag_value_m",
]
master = master[MASTER_COLUMNS]

panel.to_csv(OUTPUT_DIR / "step2_l1_panel.csv", index=False)
master.to_csv(OUTPUT_DIR / "step2_master_table.csv", index=False)
events.to_csv(OUTPUT_DIR / "step2_promotion_events.csv", index=False)
squad.drop(columns=["valued_min", "aged_min", "new_min", "value_x_min", "age_x_min"]).to_csv(
    OUTPUT_DIR / "step2_player_squads.csv", index=False)
print(f"Saved: league panel {panel.shape}, master table {master.shape}, promotion events {events.shape}")


# %% [markdown]
# ## F. Validation checks (paste these back)

# %%
# 1) Does the computed table match reality? Champion and bottom 3 of each season
print("\n1) Computed league tables (* = inside the analysis window)")
for season, table in panel.sort_values("rank").groupby("season"):
    star = "*" if season in WINDOW else " "
    print(f"  {star}{season_label(season)}  champion: {table['club'].iloc[0]:<22} "
          f"bottom 3: {', '.join(table['club'].tail(3))}")

# 2) Data quality and orders of magnitude by season (medians across Ligue 1 clubs)
print("\n2) Squad KPIs by season (median Ligue 1 club)")
quality = panel.groupby("season").agg(
    coverage_med=("valuation_coverage", "median"), coverage_min=("valuation_coverage", "min"),
    players_used=("players_used", "median"), xi_value_m=("xi_value_m", "median"),
    avg_age=("avg_age", "median"), new_signings_min_pct=("new_signings_min_pct", "median"),
    transfers_in=("n_in", "median"),
)
quality.index = [f"{'*' if s in WINDOW else ' '}{season_label(s)}" for s in quality.index]
print(quality.round(2).to_string())

# 3) Transfer sanity check: the biggest recorded fee paid by a Ligue 1 club each season
print("\n3) Biggest recorded incoming fee per season")
l1_arrivals = pd.MultiIndex.from_arrays([panel_transfers["to_club_id"].fillna(-1).astype("int64"),
                                         panel_transfers["season"].fillna(-1).astype("int64")]).isin(l1_keys)
biggest = (
    panel_transfers[l1_arrivals].dropna(subset=["transfer_fee"])
    .sort_values("transfer_fee", ascending=False)
    .drop_duplicates("season")
    .sort_values("season")
)
for row in biggest.itertuples():
    print(f"   {season_label(int(row.season))}  {row.player_name:<24} to {row.to_club_name:<24} "
          f"€{row.transfer_fee / 1e6:,.1f}M")

# 4) Missing values by division in the master table (Ligue 1 rows should be close to 0%)
print("\n4) Share of missing values by division (master table)")
kpis = ["ppg", "xi_value_m", "avg_age", "turnover_index", "net_spend_pctile", "tag_value_m"]
print(master.groupby("division")[kpis].agg(lambda s: s.isna().mean()).round(2).to_string())

# 5) Every promotion event: squad value index on 1 Sept, final rank and outcome
print("\n5) Promotion events (* = inside the analysis window)")
show = events.sort_values(["season", "squad_value_index"]).assign(
    season=lambda d: [f"{'*' if w else ' '}{season_label(s)}" for s, w in zip(d["season"], d["in_window"])])
print(show[["season", "club", "squad_value_index", "valuation_coverage", "rank", "outcome"]]
      .round(2).to_string(index=False))
