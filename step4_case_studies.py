# %% [markdown]
# # Yo-Yo Clubs vs. Stable Ligue 1: Step 4, case studies
#
# Three clubs, three stories, read through the KPIs of Steps 2-3:
# - each club's seasons side by side (division, rank, squad value, efficiency, age, turnover, spending);
# - a **promotion scorecard**: for each promotion, squad value and summer spending vs. the benchmark
#   found in Step 3 (summer net spend above the median promoted club -> 88% survival, vs 47% below);
# - one chart of the cycle: squad value and summer net spend, season by season, Ligue 2 years shaded.
#
# To study other clubs, change `CASE_STUDIES` (club names exactly as in the master table).

# %%
import sys
from pathlib import Path

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # running as a script: write charts to files, open no windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

# ---------------- Parameters ----------------
OUTPUT_DIR = Path("outputs")
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CASE_STUDIES = {               # club name -> the story it illustrates
    "FC Metz": "the yo-yo archetype",
    "RC Lens": "the promotion that stuck",
    "FC Nantes": "the slow decline",
}

# Chart style (same as Step 3)
BLUE, ORANGE, CONTEXT = "#2a78d6", "#eb6834", "#c3c2b7"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": CONTEXT, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "text.color": INK, "font.size": 9.5, "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
})
INDEX_TICKS = [0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 4, 8]


def season_label(season: int) -> str:
    """2023 -> '2023/24'."""
    return f"{season}/{str(season + 1)[-2:]}"


# %% [markdown]
# ## 0. Load and prepare

# %%
master = pd.read_csv(OUTPUT_DIR / "step2_master_table.csv")
panel = pd.read_csv(OUTPUT_DIR / "step2_l1_panel.csv")
events = pd.read_csv(OUTPUT_DIR / "step2_promotion_events.csv")
WINDOW = sorted(master["season"].unique())

missing = [club for club in CASE_STUDIES if club not in set(master["club"])]
if missing:
    raise ValueError(f"Clubs not found: {missing}. Available: {sorted(master['club'].unique())}")

# Same efficiency model as Step 3: expected points per game from the (log) squad value index
window_l1 = panel[panel["season"].isin(WINDOW)]
slope, intercept = np.polyfit(np.log(window_l1["squad_value_index"]), window_l1["ppg"], 1)
master["ppg_vs_expected"] = master["ppg"] - (intercept + slope * np.log(master["squad_value_index"]))

# Tag-based value index: the squad REGISTERED with the club on 1 Sept vs the median Ligue 1 club.
# Lower quality than the line-up measure (Step 1), but it exists in both divisions, so it shows
# the whole cycle, Ligue 2 years included.
l1_tag_median = master[master["division"] == "Ligue 1"].groupby("season")["tag_value_m"].median()
master["tag_value_index"] = master["tag_value_m"] / master["season"].map(l1_tag_median)

# Benchmark from Step 3: summer net spend percentile of the median promoted club
SPEND_BENCHMARK = events["preseason_net_spend_pctile"].median()
print(f"Benchmark: median promoted club's summer net spend percentile = {SPEND_BENCHMARK:.2f} "
      f"(clubs above it survived 88% of the time in Step 3, 47% below)")


def season_flag(row: pd.Series) -> str:
    """Promotion / relegation markers for the season table."""
    flags = []
    if row["promoted_season"]:
        flags.append("promoted")
    if row["relegated_at_end"] in (True, "True"):  # True, never "unknown"
        flags.append("relegated at end")
    return " + ".join(flags)


# %% [markdown]
# ## A. Season-by-season tables and promotion scorecards

# %%
TABLE_COLUMNS = {
    "season": "season", "div": "div", "rank": "rank", "ppg": "ppg",
    "squad_value_index": "value_idx", "ppg_vs_expected": "ppg_vs_exp", "avg_age": "age",
    "turnover_index": "turnover_idx", "preseason_net_spend_m": "summer_net_€m",
    "preseason_net_spend_pctile": "summer_pctile", "tag_value_index": "tag_value_idx", "flag": "event",
}
for club, story in CASE_STUDIES.items():
    rows = master[master["club"] == club].sort_values("season").copy()
    rows["flag"] = rows.apply(season_flag, axis=1)
    rows["div"] = np.where(rows["division"] == "Ligue 1", "L1", "L2-")
    rows["rank"] = rows["rank"].astype("Int64")  # whole numbers (empty outside Ligue 1)
    rows["season"] = rows["season"].map(season_label)
    print(f"\n=== {club}: {story} ({rows['cohort'].iloc[0]}) ===")
    print(rows[list(TABLE_COLUMNS)].rename(columns=TABLE_COLUMNS).round(2).to_string(index=False))

    scorecard = events[events["club"] == club].sort_values("season").copy()
    if len(scorecard):
        scorecard["summer_spend_vs_benchmark"] = np.where(
            scorecard["preseason_net_spend_pctile"] >= SPEND_BENCHMARK, "above", "below")
        scorecard["season"] = scorecard["season"].map(season_label)
        print("  Promotion scorecard:")
        print(scorecard[["season", "squad_value_index", "preseason_net_spend_pctile",
                         "summer_spend_vs_benchmark", "rank", "outcome"]]
              .round(2).to_string(index=False))


# %% [markdown]
# ## B. The cycle in one chart
# Top: squad value index (log scale). Blue line = registered squad (tag-based, every season);
# orange dots = players who actually played (line-ups, Ligue 1 seasons only).
# Bottom: summer net spend in €M (deals dated up to 5 September). Shaded = season outside Ligue 1.
# Note: older transfer data is incomplete (Step 2), so early seasons may understate spending.

# %%
clubs = list(CASE_STUDIES)
fig, axes = plt.subplots(2, len(clubs), figsize=(5.2 * len(clubs), 6.6), sharex="col", sharey="row",
                         gridspec_kw={"height_ratios": [1.6, 1]})
axes = np.array(axes).reshape(2, len(clubs))
for col, club in enumerate(clubs):
    rows = master[master["club"] == club].sort_values("season")
    x = np.arange(len(rows))
    top, bottom = axes[0, col], axes[1, col]
    for ax in (top, bottom):  # shade the seasons spent outside Ligue 1
        for xi, division in zip(x, rows["division"]):
            if division != "Ligue 1":
                ax.axvspan(xi - 0.5, xi + 0.5, color=GRID, alpha=0.7, linewidth=0, zorder=0)
    top.plot(x, rows["tag_value_index"], color=BLUE, linewidth=2, marker="o", markersize=4.5, zorder=3,
             label="Registered squad (tag-based, every season)")
    top.scatter(x, rows["squad_value_index"], s=34, color=ORANGE, edgecolor=SURFACE, linewidth=1, zorder=4,
                label="Players who played (line-ups, Ligue 1 only)")
    top.axhline(1, color=CONTEXT, linewidth=1, zorder=1)
    top.set_title(f"{club}: {CASE_STUDIES[club]}")
    bottom.bar(x, rows["preseason_net_spend_m"], width=0.55, color=MUTED, zorder=3)
    bottom.axhline(0, color=INK_2, linewidth=1, zorder=2)
    bottom.set_xticks(x, [season_label(s)[2:] for s in rows["season"]], rotation=45, ha="right")
    bottom.grid(axis="x", visible=False)
    top.grid(axis="x", visible=False)

axes[0, 0].set_yscale("log")
low, high = axes[0, 0].get_ylim()
ticks = [t for t in INDEX_TICKS if low <= t <= high]
axes[0, 0].set_yticks(ticks, [f"{t:g}" for t in ticks])
axes[0, 0].minorticks_off()
axes[0, 0].set_ylabel("Squad value index\n(1 = median Ligue 1 club, log)")
axes[1, 0].set_ylabel("Summer net spend (€M)")
handles, labels = axes[0, 0].get_legend_handles_labels()
handles.append(Patch(facecolor=GRID, alpha=0.7, label="Season outside Ligue 1"))
axes[0, 0].legend(handles=handles, loc="upper left", fontsize=8)
fig.suptitle(f"Case studies: squad value and summer spending through the cycle, {season_label(WINDOW[0])} "
             f"to {season_label(WINDOW[-1])}", x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_f_case_studies.png")
print("[chart saved: outputs/figures/fig_f_case_studies.png]")
if "ipykernel" in sys.modules:
    plt.show()
else:
    plt.close(fig)
