# %% [markdown]
# # Yo-Yo Clubs vs. Stable Ligue 1: Step 3, exploratory analysis & hypothesis tests
#
# Four questions, one section each:
# - **A. The tier cliff**: how far behind do promoted clubs start?
# - **B. Capital efficiency**: who turns squad value into points, and does net spending buy results?
# - **C. Cohort profiles**: how do yo-yo clubs differ from stable ones?
# - **D. The survival test**: what predicts staying up after promotion (squad value, age, turnover, spending)?
#
# Every chart is saved in `outputs/figures/`, and the numbers behind it are printed below it.
#
# **Statistics with small samples.** We have around 30 promotions and 32 clubs. So we use rank-based
# tests (no normality assumption), report effect sizes with bootstrap intervals, and treat p-values
# as indicative rather than as proof. Seasons of the same club are not independent observations.

# %%
import sys
from pathlib import Path

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # running as a script: write charts to files, open no windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

# ---------------- Parameters ----------------
OUTPUT_DIR = Path("outputs")
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

MIN_EVENT_COVERAGE = 0.80  # drop a promotion event if valued players played < 80% of its minutes
N_BOOT = 2000              # bootstrap resamples
rng = np.random.default_rng(42)  # fixed seed: same results at every run

# Chart style. Blue and orange pass the colour-blind checks together; grey marks are context only.
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


def auc(higher: pd.Series, lower: pd.Series) -> float:
    """Probability that a random value from `higher` beats one from `lower` (0.5 = no signal)."""
    return stats.mannwhitneyu(higher, lower).statistic / (len(higher) * len(lower))


def bootstrap_auc(values, positive) -> np.ndarray:
    """90% bootstrap interval of the AUC: resample the events with replacement, recompute."""
    values, positive = np.asarray(values), np.asarray(positive)
    results = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(values), len(values))
        if positive[i].all() or not positive[i].any():
            continue  # a resample needs both survivors and relegated clubs
        results.append(auc(values[i][positive[i]], values[i][~positive[i]]))
    return np.percentile(results, [5, 95])


def best_threshold(values: np.ndarray, positive: np.ndarray) -> tuple[float, float]:
    """Cut-off c maximising Youden's J (hit rate - false-alarm rate) for 'survives if value >= c'."""
    best_c, best_j = np.nan, -1.0
    for c in np.unique(values):
        j = (values[positive] >= c).mean() - (values[~positive] >= c).mean()
        if j > best_j:
            best_c, best_j = c, j
    return best_c, best_j


def index_axis(ax, axis: str = "x") -> None:
    """Log scale for the squad value index, with readable ticks (1 = median Ligue 1 club)."""
    set_scale, set_ticks, get_lim = ((ax.set_xscale, ax.set_xticks, ax.get_xlim) if axis == "x"
                                     else (ax.set_yscale, ax.set_yticks, ax.get_ylim))
    set_scale("log")
    low, high = get_lim()
    ticks = [t for t in INDEX_TICKS if low <= t <= high]
    set_ticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()


def titles(ax, title: str, subtitle: str) -> None:
    """Bold title with a one-line grey subtitle underneath, both above the plot."""
    ax.set_title(title, pad=22)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")


def save(fig, name: str) -> None:
    """Save a chart; show it too when running cell by cell in VS Code."""
    fig.savefig(FIG_DIR / name)
    print(f"[chart saved: outputs/figures/{name}]")
    if "ipykernel" in sys.modules:
        plt.show()
    else:
        plt.close(fig)


# %% [markdown]
# ## 0. Load Step 2 outputs

# %%
panel = pd.read_csv(OUTPUT_DIR / "step2_l1_panel.csv")
master = pd.read_csv(OUTPUT_DIR / "step2_master_table.csv")
events = pd.read_csv(OUTPUT_DIR / "step2_promotion_events.csv")

WINDOW = sorted(master["season"].unique())
cohort_of = master.drop_duplicates("club_id").set_index("club_id")["cohort"]
window_l1 = panel[panel["season"].isin(WINDOW)].copy()  # every Ligue 1 club-season of the window
window_l1["cohort"] = window_l1["club_id"].map(cohort_of)
window_label = f"{season_label(WINDOW[0])} to {season_label(WINDOW[-1])}"
print(f"Window {window_label}: {len(window_l1)} Ligue 1 club-seasons | promotion events in the panel: {len(events)}")


# %% [markdown]
# ## A. The tier cliff: how far behind do promoted clubs start?
# Squad value index on 1 September for three groups of Ligue 1 club-seasons: promoted clubs in their
# first season up, established clubs that finished in the bottom half, and the rest.

# %%
GROUPS = ["Promoted (1st season up)", "Established, bottom half", "Established, top half"]
window_l1["group"] = np.select(
    [window_l1["promoted_season"].astype(bool), window_l1["bottom_half"].astype(bool)],
    GROUPS[:2], default=GROUPS[2])

cliff = (window_l1.groupby("group")["squad_value_index"]
         .describe(percentiles=[0.25, 0.5, 0.75])[["count", "25%", "50%", "75%"]]
         .reindex(GROUPS))
promoted_idx = window_l1.loc[window_l1["group"] == GROUPS[0], "squad_value_index"]
bottom_idx = window_l1.loc[window_l1["group"] == GROUPS[1], "squad_value_index"]
p_cliff = stats.mannwhitneyu(promoted_idx, bottom_idx).pvalue

print(cliff.round(2).to_string())
print(f"\nMedian promoted club: {promoted_idx.median():.2f} x the median Ligue 1 club, i.e. "
      f"{promoted_idx.median() / bottom_idx.median():.0%} of the median established bottom-half club.")
print(f"Probability a random bottom-half established club out-values a random promoted club: "
      f"{auc(bottom_idx, promoted_idx):.0%} (Mann-Whitney p = {p_cliff:.3f})")

# %%
fig, ax = plt.subplots(figsize=(9, 3.6))
for row, group in enumerate(GROUPS):
    values = window_l1.loc[window_l1["group"] == group, "squad_value_index"]
    jitter = rng.uniform(-0.18, 0.18, len(values))
    color = BLUE if row == 0 else CONTEXT
    ax.scatter(values, row + jitter, s=26, color=color, edgecolor=SURFACE, linewidth=1, zorder=3)
    median = values.median()
    ax.plot([median, median], [row - 0.32, row + 0.32], color=INK, linewidth=2, zorder=4)
    ax.annotate(f"median {median:.2f}", (median, row - 0.36), ha="center", va="bottom",
                fontsize=8.5, color=INK_2)
ax.set_yticks(range(len(GROUPS)), GROUPS)
ax.set_ylim(len(GROUPS) - 0.5, -0.6)  # first group on top
ax.grid(axis="y", visible=False)
index_axis(ax, "x")
ax.set_xlabel("Squad value index on 1 September (1 = median Ligue 1 club, log scale)")
titles(ax, f"The tier cliff: promoted clubs start at {promoted_idx.median():.2f}x the median Ligue 1 squad",
       f"Ligue 1 club-seasons, {window_label}. Each dot is one club in one season; "
       "the black tick is the group median.")
save(fig, "fig_a_tier_cliff.png")


# %% [markdown]
# ## B. Capital efficiency: who turns squad value into points?
# 1. Points per game against squad value (log): the fitted line is the "expected" return on squad
#    value; clubs above it over-perform. 2. Does net spending (as a percentile within its season)
#    go with better results?

# %%
log_value = np.log(window_l1["squad_value_index"])
slope, intercept = np.polyfit(log_value, window_l1["ppg"], 1)
r_value = np.corrcoef(log_value, window_l1["ppg"])[0, 1]
window_l1["ppg_expected"] = intercept + slope * log_value
window_l1["ppg_vs_expected"] = window_l1["ppg"] - window_l1["ppg_expected"]
window_l1["ppg_pctile"] = window_l1.groupby("season")["ppg"].rank(pct=True)

rho_value, p_value = stats.spearmanr(window_l1["squad_value_index"], window_l1["ppg_pctile"])
rho_spend, p_spend = stats.spearmanr(window_l1["net_spend_pctile"], window_l1["ppg_pctile"])

print(f"Doubling the squad value index goes with {slope * np.log(2):+.2f} points per game "
      f"(R² = {r_value ** 2:.2f}: squad value explains {r_value ** 2:.0%} of the differences in PPG).")
print(f"Rank correlation with results (within-season percentile): squad value {rho_value:.2f} "
      f"(p = {p_value:.3f}) vs net spend {rho_spend:.2f} (p = {p_spend:.3f})")

cols = ["club", "season", "cohort", "squad_value_index", "ppg", "ppg_expected", "ppg_vs_expected"]
ranked = window_l1.sort_values("ppg_vs_expected").assign(season=lambda d: d["season"].map(season_label))
print("\nBiggest over-performers (points per game above what their squad value predicts)")
print(ranked.tail(5).iloc[::-1][cols].round(2).to_string(index=False))
print("\nBiggest under-performers")
print(ranked.head(5)[cols].round(2).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(9, 5.2))
styles = [("Other cohorts", ~window_l1["cohort"].isin(["Stable Ligue 1", "Yo-Yo"]), CONTEXT),
          ("Stable Ligue 1", window_l1["cohort"] == "Stable Ligue 1", BLUE),
          ("Yo-Yo", window_l1["cohort"] == "Yo-Yo", ORANGE)]
for label, mask, color in styles:
    ax.scatter(window_l1.loc[mask, "squad_value_index"], window_l1.loc[mask, "ppg"], s=30, color=color,
               edgecolor=SURFACE, linewidth=1, label=label, zorder=3)
x_line = np.linspace(log_value.min(), log_value.max(), 50)
ax.plot(np.exp(x_line), intercept + slope * x_line, color=INK_2, linewidth=1.5, zorder=2,
        label="Expected PPG for that squad value")
for _, row in pd.concat([ranked.tail(3), ranked.head(3)]).iterrows():  # label only the extremes
    ax.annotate(f"{row['club']} {row['season'][2:]}", (row["squad_value_index"], row["ppg"]),
                xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK_2)
index_axis(ax, "x")
ax.set_xlabel("Squad value index on 1 September (1 = median Ligue 1 club, log scale)")
ax.set_ylabel("Points per game")
titles(ax, f"Squad value explains {r_value ** 2:.0%} of the differences in points per game",
       f"Ligue 1 club-seasons, {window_label}. Labels: the three biggest over- and under-performers.")
ax.legend(loc="upper left", fontsize=8.5)
save(fig, "fig_b_capital_efficiency.png")




# %% [markdown]
# ## C. Cohort profiles: how do yo-yo clubs differ from stable ones?
# Medians of five KPIs over each cohort's Ligue 1 seasons. Kruskal-Wallis tests whether the four
# cohorts differ at all; Mann-Whitney compares Yo-Yo with Stable Ligue 1 directly.
#
# Efficiency is measured as **PPG vs expected** (section B), not as a ratio: "points per €M" always
# flatters cheap squads, because each extra euro buys fewer points (the curve in chart B flattens).
# Caution: cohorts are defined by results (staying up), so stable clubs beating their budget is
# partly true by construction. Read this section as a description, not as a cause.

# %%
COHORTS = ["Stable Ligue 1", "Yo-Yo", "Promoted, then relegated", "Fallen from Ligue 1"]
PROFILE_KPIS = {
    "squad_value_index": "Squad value index",
    "ppg_vs_expected": "PPG vs expected",
    "avg_age": "Average age (years)",
    "turnover_index": "Turnover index",
    "preseason_net_spend_pctile": "Pre-season net spend pctile",
}
comparable = window_l1[window_l1["cohort"].isin(COHORTS)]
profile = comparable.groupby("cohort")[list(PROFILE_KPIS)].median().reindex(COHORTS)
profile.insert(0, "club_seasons", comparable.groupby("cohort").size().reindex(COHORTS))

tests = []
for kpi in PROFILE_KPIS:
    groups = [comparable.loc[comparable["cohort"] == c, kpi].dropna() for c in COHORTS]
    yoyo, stable = groups[1], groups[0]
    tests.append({"kpi": kpi,
                  "kruskal_p_4_cohorts": stats.kruskal(*[g for g in groups if len(g) > 0]).pvalue,
                  "yoyo_vs_stable_p": stats.mannwhitneyu(yoyo, stable).pvalue})
print(profile.round(2).to_string())
print()
print(pd.DataFrame(tests).round(3).to_string(index=False))

# %%
fig, axes = plt.subplots(1, len(PROFILE_KPIS), figsize=(15, 3.2), sharey=True)
bar_colors = [BLUE, ORANGE, CONTEXT, CONTEXT]
for ax, (kpi, title) in zip(axes, PROFILE_KPIS.items()):
    values = profile[kpi]
    ax.barh(range(len(COHORTS)), values, height=0.4, color=bar_colors, zorder=3)
    for i, v in enumerate(values):  # value at the bar tip, on the correct side for negative bars
        ax.annotate(f"{v:.1f}" if kpi == "avg_age" else f"{v:+.2f}" if kpi == "ppg_vs_expected" else f"{v:.2f}",
                    (v, i), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if v >= 0 else "right", va="center", fontsize=8.5, color=INK_2)
    low, high = min(0, values.min()), max(0, values.max())
    pad = (high - low) * 0.35
    ax.set_xlim(low - (pad if low < 0 else 0), high + pad)
    ax.axvline(0, color=CONTEXT, linewidth=1)
    ax.set_title(title, fontsize=9.5)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(COHORTS)), COHORTS)
axes[0].invert_yaxis()
fig.suptitle(f"Cohort profiles: median of each cohort's Ligue 1 seasons, {window_label}", x=0.01, y=1.06,
             ha="left", fontsize=11, fontweight="bold")
save(fig, "fig_c_cohort_profiles.png")


# %% [markdown]
# ## D. The survival test: what predicts staying up after promotion?
# One observation = one promotion. We measure how well each pre-season KPI separates the clubs that
# stayed up from those that went straight back down, using the **AUC**: the probability that a random
# survivor scores higher than a random relegated club. 0.5 = coin flip, 1.0 = perfect separation.
# (Credit-risk teams use the same metric to judge a scoring model: Gini = 2 x AUC - 1.)
#
# Two robustness checks before believing any signal:
# 1. **Summer-only spending** (`preseason_net_spend_pctile`): decided before any result, so it cannot be a
#    reaction to a bad start, unlike January deals (reverse causality).
# 2. **Window promotions only** (2016/17 onwards): transfer data is more complete in recent seasons.

# %%
usable = events[(events["outcome"] != "Unknown") & (events["valuation_coverage"] >= MIN_EVENT_COVERAGE)].copy()
usable["survived"] = usable["outcome"] == "Survived"
print(f"Promotion events: {len(events)} in total, {len(usable)} usable "
      f"({(events['outcome'] == 'Unknown').sum()} with unknown outcome, "
      f"{((events['outcome'] != 'Unknown') & (events['valuation_coverage'] < MIN_EVENT_COVERAGE)).sum()} "
      f"with valuation coverage < {MIN_EVENT_COVERAGE:.0%}) | seasons {season_label(usable['season'].min())} "
      f"to {season_label(usable['season'].max())} | survived {usable['survived'].sum()}, "
      f"relegated {(~usable['survived']).sum()}")

SURVIVAL_KPIS = {
    "squad_value_index": "Squad value index",
    "avg_age": "Average age",
    "turnover_index": "Turnover index",
    "net_spend_pctile": "Net spend pctile (full season)",
    "preseason_net_spend_pctile": "Net spend pctile (summer only)",
}


def separation_table(data: pd.DataFrame) -> pd.DataFrame:
    """For each KPI: medians by outcome, AUC with its 90% bootstrap interval, Mann-Whitney p-value."""
    rows = []
    for kpi, label in SURVIVAL_KPIS.items():
        d = data.dropna(subset=[kpi])
        survivors, relegated = d.loc[d["survived"], kpi], d.loc[~d["survived"], kpi]
        low, high = bootstrap_auc(d[kpi], d["survived"])
        rows.append({"kpi": label, "median_survived": survivors.median(), "median_relegated": relegated.median(),
                     "auc": auc(survivors, relegated), "auc_90pct_low": low, "auc_90pct_high": high,
                     "mann_whitney_p": stats.mannwhitneyu(survivors, relegated).pvalue})
    return pd.DataFrame(rows)


survival_table = separation_table(usable)
print("\nWhich pre-season KPI separates survivors from relegated clubs? (AUC > 0.5: higher = safer)")
print(survival_table.round(2).to_string(index=False))

window_events = usable[usable["in_window"]]
window_table = separation_table(window_events)
print(f"\nRobustness: same test on the {len(window_events)} window promotions only "
      f"({(~window_events['survived']).sum()} relegated)")
print(window_table[["kpi", "auc", "auc_90pct_low", "auc_90pct_high", "mann_whitney_p"]].round(2).to_string(index=False))

# %%
THIRDS = ["Lowest third", "Middle third", "Highest third"]


def survival_chart(kpi: str, short_name: str, x_label: str, title: str, filename: str,
                   log_scale: bool = False) -> None:
    """Left: each promotion by outcome (dots). Right: survival rate by third of the KPI (bars)."""
    data = usable.dropna(subset=[kpi]).copy()
    survived = data["survived"].to_numpy()
    cut, _ = best_threshold(data[kpi].to_numpy(), survived)
    # Thirds are set on the KPI alone, before looking at outcomes (rank first: ties never break the bins)
    data["third"] = pd.qcut(data[kpi].rank(method="first"), 3, labels=THIRDS)
    bands = data.groupby("third", observed=True).agg(
        events=("survived", "size"), survival_rate=("survived", "mean"),
        value_from=(kpi, "min"), value_to=(kpi, "max"))
    print(f"\nSurvival rate by third of {short_name}")
    print(bands.round(2).to_string())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [2.2, 1]})
    data["y"] = np.where(survived, 0, 1) + rng.uniform(-0.2, 0.2, len(data))  # row + jitter
    for outcome, color in [("Survived", BLUE), ("Relegated", ORANGE)]:
        part = data[data["outcome"] == outcome]
        ax1.scatter(part[kpi], part["y"], s=34, color=color, edgecolor=SURFACE, linewidth=1, zorder=3)
    # Label only the surprises: the 2 lowest values that stayed up and the 2 highest that went down
    # (labels alternate above / below their dot so close neighbours don't collide)
    for group in [data[data["survived"]].nsmallest(2, kpi), data[~data["survived"]].nlargest(2, kpi)]:
        for k, (_, event) in enumerate(group.sort_values(kpi).iterrows()):
            ax1.annotate(f"{event['club']} {season_label(event['season'])[2:]}", (event[kpi], event["y"]),
                         xytext=(5, 7 if k % 2 == 0 else -14), textcoords="offset points",
                         fontsize=8, color=INK_2)
    if log_scale:
        index_axis(ax1, "x")
    else:
        ax1.set_xlim(-0.03, 1.05)
    ax1.axvline(cut, color=INK_2, linewidth=1, zorder=2)
    low, high = ax1.get_xlim()
    middle = np.sqrt(low * high) if log_scale else (low + high) / 2
    on_right = cut < middle  # put the label on the side where there is room
    ax1.annotate(f"best in-sample split: {cut:.2f}", (cut, -0.45), xytext=(4 if on_right else -4, 0),
                 textcoords="offset points", ha="left" if on_right else "right", fontsize=8, color=INK_2)
    ax1.set_yticks([0, 1], [f"Survived ({survived.sum()})", f"Relegated ({(~survived).sum()})"])
    ax1.set_ylim(1.5, -0.6)
    ax1.grid(axis="y", visible=False)
    ax1.set_xlabel(x_label)
    titles(ax1, title, "Labels: the 2 lowest values that stayed up and the 2 highest that went down.")

    ax2.bar(range(len(bands)), bands["survival_rate"] * 100, width=0.5, color=BLUE, zorder=3)
    for i, (rate, n) in enumerate(zip(bands["survival_rate"], bands["events"])):
        ax2.annotate(f"{rate:.0%}\n(n = {n})", (i, rate * 100), xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=8.5, color=INK_2)
    ax2.set_xticks(range(len(bands)), bands.index.astype(str))
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Stayed up after one season (%)")
    ax2.grid(axis="x", visible=False)
    titles(ax2, f"Survival rate by third of {short_name}", "Thirds are set before looking at outcomes.")
    save(fig, filename)


survival_chart("squad_value_index", "squad value",
               "Squad value index on 1 September of the promoted season (log scale)",
               "Promoted clubs: squad value on day one vs. outcome", "fig_d_survival_squad_value.png",
               log_scale=True)

# How stable is the "best" squad value threshold when the sample changes? (bootstrap)
values, survived = usable["squad_value_index"].to_numpy(), usable["survived"].to_numpy()
cut, youden = best_threshold(values, survived)
boot_cuts = []
for _ in range(N_BOOT):
    i = rng.integers(0, len(values), len(values))
    if survived[i].all() or not survived[i].any():
        continue
    boot_cuts.append(best_threshold(values[i], survived[i])[0])
cut_low, cut_mid, cut_high = np.percentile(boot_cuts, [5, 50, 95])
print(f"\nBest in-sample squad value threshold: index >= {cut:.2f} -> survival {survived[values >= cut].mean():.0%} "
      f"(n = {(values >= cut).sum()}) vs {survived[values < cut].mean():.0%} below (n = {(values < cut).sum()})")
print(f"Bootstrap: the best threshold moves between {cut_low:.2f} and {cut_high:.2f} (90% range, median "
      f"{cut_mid:.2f}). A wide range means the exact cut-off is not reliable, even if the direction is.")

# %%
survival_chart("preseason_net_spend_pctile", "summer net spend",
               "Summer net spend percentile among Ligue 1 clubs that season (1 = biggest net buyer)",
               "Promoted clubs: summer net spending vs. outcome", "fig_e_survival_summer_spend.png")

# Simple splits for the executive summary, at the median of the PROMOTED clubs (not of Ligue 1:
# nearly every promoted club out-spends most Ligue 1 clubs in the summer, so that split is lopsided)
usable["spend_half"] = np.where(
    usable["preseason_net_spend_pctile"] >= usable["preseason_net_spend_pctile"].median(),
    "Summer net spend: above promoted median", "Summer net spend: below promoted median")
usable["value_half"] = np.where(usable["squad_value_index"] >= usable["squad_value_index"].median(),
                                "Squad value: above promoted median", "Squad value: below promoted median")
spend_split = usable.groupby("spend_half").agg(events=("survived", "size"), survival_rate=("survived", "mean"))
print("\nSurvival rate by summer net spend (above vs below the median promoted club)")
print(spend_split.round(2).to_string())

cells = usable.groupby(["value_half", "spend_half"])["survived"].agg(["mean", "size"])
two_by_two = (cells["mean"].map("{:.0%}".format) + " (n=" + cells["size"].astype(str) + ")").unstack()
print("\nSurvival rate: squad value x summer spending (small cells: indicative only)")
print(two_by_two.to_string())

# The surprises: material for the case studies of Step 4
surprise_cols = ["season", "club", "squad_value_index", "preseason_net_spend_pctile", "turnover_index", "rank"]
print("\nSurprises: weakest squads that stayed up")
print(usable[usable["survived"]].nsmallest(3, "squad_value_index")
      .assign(season=lambda d: d["season"].map(season_label))[surprise_cols].round(2).to_string(index=False))
print("Surprises: strongest squads that went down")
print(usable[~usable["survived"]].nlargest(3, "squad_value_index")
      .assign(season=lambda d: d["season"].map(season_label))[surprise_cols].round(2).to_string(index=False))


# %% [markdown]
# ## E. Key numbers (paste these back)

# %%
def auc_line(table: pd.DataFrame, label: str) -> str:
    row = table[table["kpi"] == label].iloc[0]
    return f"AUC {row['auc']:.2f} (90% CI {row['auc_90pct_low']:.2f}-{row['auc_90pct_high']:.2f}, p = {row['mann_whitney_p']:.3f})"


print("KEY NUMBERS")
print(f"- Tier cliff: median promoted club starts at {promoted_idx.median():.2f}x the median Ligue 1 squad "
      f"({promoted_idx.median() / bottom_idx.median():.0%} of an established bottom-half club).")
print(f"- Efficiency: doubling squad value = {slope * np.log(2):+.2f} PPG; value explains {r_value ** 2:.0%} "
      f"of PPG differences. Rank correlation with results: value {rho_value:.2f}, net spend {rho_spend:.2f}.")
print(f"- Survival, {len(usable)} promotions ({(~usable['survived']).sum()} relegated):")
for label in SURVIVAL_KPIS.values():
    print(f"    {label:<32} all: {auc_line(survival_table, label)} | window only: {auc_line(window_table, label)}")
print(f"- Squad value threshold: best in-sample {cut:.2f}, bootstrap 90% range {cut_low:.2f}-{cut_high:.2f}.")
print("- Summer spending: " + ", ".join(f"{half} {rate:.0%} (n={n})" for half, rate, n in
                                         zip(spend_split.index, spend_split["survival_rate"], spend_split["events"])))
