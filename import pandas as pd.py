"""
Life expectancy: EDA and data visualization (local version).

Requirements:  pip install pandas numpy matplotlib
Run:           python life_expectancy_viz.py
Also works cell by cell in VS Code or Jupyter (the "# %%" markers).
"""


# %% [markdown]
# # Life Expectancy: EDA and Data Visualization
#
# **Dataset:** life expectancy at birth by country and region (Our World in Data).
#
# **Plan**
# 1. Load the data and explore its contents.
# 2. Find the grain (what one row represents).
# 3. Pick the promising columns and exclude the ones that are hard to read or have quality problems.
# 4. Build pivot tables to understand the shape of the data.
# 5. Decide on chart types and the table format each one needs.
# 6. Prepare the tables and create the visualizations.


# %%
import glob
import logging
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def show(table):
    """Pretty table in a notebook, plain print in a terminal."""
    try:
        from IPython.display import display
        display(table)
    except ImportError:
        print(table)


# %%
# ---------------------------------------------------------------
# 1. LOAD THE DATA
# The path can point to the CSV itself or to the folder that holds it.
# ---------------------------------------------------------------
DATA_PATH = "/Users/joaogolin/Desktop/AMU/FALL 2026/DATA VISUALISATION/life-expectancy"


def resolve_csv(path):
    if os.path.isfile(path):
        return path
    if os.path.isfile(path + ".csv"):
        return path + ".csv"
    if os.path.isdir(path):
        hits = sorted(glob.glob(os.path.join(path, "*life*expectancy*.csv"))) or \
               sorted(glob.glob(os.path.join(path, "*.csv")))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Could not find the life expectancy CSV at: {path}")


CSV_FILE = resolve_csv(DATA_PATH)
BASE_DIR = DATA_PATH if os.path.isdir(DATA_PATH) else os.path.dirname(CSV_FILE)
OUT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

raw = pd.read_csv(CSV_FILE)
print("Loaded:", CSV_FILE)
print("Figures will be saved to:", OUT_DIR)


# %% [markdown]
# ## 1. Explore the contents
#
# A first look at size, column types, missing values and value ranges.


# %%
# The value column is always the 4th one; rename everything to short, clean names
raw = raw.rename(columns={raw.columns[0]: "entity", raw.columns[1]: "code",
                          raw.columns[2]: "year", raw.columns[3]: "life_exp"})

print("Shape (rows, columns):", raw.shape)
print()
raw.info()
show(raw.head(10))
show(raw.describe(include="all").T)

print("Missing values per column:")
print(raw.isna().sum())
print()
print("Distinct entities:", raw["entity"].nunique())
print("Year range:", raw["year"].min(), "to", raw["year"].max())


# %% [markdown]
# ## 2. Grain of the dataset
#
# The test: if `entity` + `year` never repeats, then **one row = one entity in one year**,
# holding that entity's life expectancy at birth.
#
# The tricky part is that "entity" mixes two kinds of things:
# - **Countries and territories**, which have a 3 letter ISO code (BRA, USA, SWE).
# - **Aggregates**, such as continents, income groups and UN groupings. These have either an `OWID_` code or no code at all.
#
# Mixing them in one chart would double count, so we label each row with its type.


# %%
dupes = raw.duplicated(["entity", "year"]).sum()
print("Duplicate entity-year pairs:", dupes, "-> grain confirmed" if dupes == 0 else "-> check!")

# Kosovo uses an OWID code but it is a country; the USSR is a historical entity with 2 rows.
OWID_COUNTRIES = {"OWID_KOS"}
HISTORICAL = {"USSR"}

def entity_type(row):
    if row["entity"] in HISTORICAL:
        return "historical"
    if pd.isna(row["code"]):
        return "aggregate (no code)"
    if str(row["code"]).startswith("OWID_") and row["code"] not in OWID_COUNTRIES:
        return "aggregate (OWID)"
    return "country"

raw["entity_type"] = raw.apply(entity_type, axis=1)

summary = (raw.groupby("entity_type")
              .agg(entities=("entity", "nunique"), rows=("entity", "size"))
              .sort_values("rows", ascending=False))
show(summary)

print("Aggregates in the file:")
print(sorted(raw.loc[raw["entity_type"].str.startswith("aggregate"), "entity"].unique()))


# %% [markdown]
# ## 3. Columns: keep, hide, exclude
#
# | Column | Decision | Why |
# |---|---|---|
# | `entity` | **keep** | Country or region name. Our main dimension. |
# | `year` | **keep** | Time dimension. |
# | `life_exp` | **keep** | The measure: expected years of life at birth. |
# | `code` | **hide** | Only needed to tell countries from aggregates and to draw the map. Missing for 18 aggregates. |
#
# Rows and groups we exclude:
# - **Unclear aggregates** such as "No income group available" or "Less developed regions, excluding China". Their definitions are hard to explain on a chart.
# - **Overlapping UK pieces** (England and Wales, Scotland, Northern Ireland). The United Kingdom already covers them.
# - **The USSR**, which has only 2 rows.
# - **Years before 1950** in the main charts. Before 1950 only a few rich countries have data, and the regional numbers are rough estimates with gaps of up to 50 years. We keep that period for one "long view" chart only, using countries with real yearly records.


# %%
# How uneven is coverage before 1950?
coverage = (raw.assign(period=np.where(raw["year"] < 1950, "before 1950", "1950 onward"))
               .groupby("period")
               .agg(entities=("entity", "nunique"), rows=("entity", "size"),
                    first_year=("year", "min"), last_year=("year", "max")))
show(coverage)

world_years = raw.loc[raw["entity"] == "World", "year"]
print("World data points before 1950:", world_years[world_years < 1950].tolist())

# Tidy tables we will use from here on
UK_PARTS = {"England and Wales", "Scotland", "Northern Ireland"}

countries = raw[(raw["entity_type"] == "country") & (~raw["entity"].isin(UK_PARTS))]
countries = countries[["entity", "code", "year", "life_exp"]].copy()
countries_modern = countries[countries["year"] >= 1950]

REGIONS = ["Africa", "Asia", "Europe", "Latin America and the Caribbean",
           "Northern America", "Oceania"]
regions = raw[raw["entity"].isin(REGIONS + ["World"]) & (raw["year"] >= 1950)]
regions = regions[["entity", "year", "life_exp"]].copy()

print("Countries kept:", countries_modern["entity"].nunique(),
      "| years per country (1950+):", countries_modern.groupby("entity").size().unique())


# %% [markdown]
# ## 4. Pivot tables
#
# Pivots let us see the shape of the data before drawing anything: how regions compare
# over time, whether countries are drawing closer together, and which ones moved the most.


# %%
# Pivot 1: regions x decade (average life expectancy)
regions["decade"] = (regions["year"] // 10 * 10).astype(str) + "s"
pv_region_decade = regions.pivot_table(index="entity", columns="decade",
                                       values="life_exp", aggfunc="mean").round(1)
pv_region_decade = pv_region_decade.loc[["World"] + REGIONS]
print("Pivot 1: average life expectancy by region and decade")
show(pv_region_decade)

# Pivot 2: country spread by year (are countries converging?)
pv_spread = (countries_modern.pivot_table(index="year", values="life_exp",
                                          aggfunc=["count", "min", "median", "max", "std"])
                             .round(1))
pv_spread.columns = ["countries", "min", "median", "max", "std_dev"]
pv_spread["max_minus_min"] = (pv_spread["max"] - pv_spread["min"]).round(1)
print("Pivot 2: spread across countries, selected years")
show(pv_spread.loc[[1950, 1960, 1970, 1980, 1990, 2000, 2010, 2019, 2021, 2023]])

# Pivot 3: countries x key years (wide), plus change columns
pv_wide = countries_modern.pivot_table(index="entity", columns="year", values="life_exp")
key = pv_wide[[1950, 1990, 2019, 2021, 2023]].copy()
key["gain_1950_2023"] = key[2023] - key[1950]
key["covid_change_2019_2021"] = key[2021] - key[2019]
key = key.round(1)
print("Pivot 3a: 10 biggest gains since 1950")
show(key.sort_values("gain_1950_2023", ascending=False).head(10))
print("Pivot 3b: 10 biggest drops from 2019 to 2021")
show(key.sort_values("covid_change_2019_2021").head(10))
print("Pivot 3c: highest and lowest in 2023")
show(pd.concat([key[2023].nlargest(5), key[2023].nsmallest(5)]).to_frame("life_exp_2023"))


# %% [markdown]
# ## 5. Chart choices and the table each one needs
#
# | # | Question | Chart | Table format it needs |
# |---|---|---|---|
# | 1 | How long did people live before modern medicine? | Long-run line chart | Long: one row per country-year |
# | 2 | How has each region changed since 1950? | Small multiples (one panel per region) | Long: region, year, value |
# | 3 | Are countries getting closer to each other? | Ridgeline of distributions | Long: country, year, value, for 5 snapshot years |
# | 4 | How much did the biggest countries gain? | Dumbbell chart | Wide: one row per country, columns 1950 and 2023 |
# | 5 | What did COVID-19 do? | Dot plot with three markers | Wide: one row per country, columns 2019, 2021, 2023 |
# | 6 | Regional summary at a glance | Annotated heatmap | Matrix: regions as rows, decades as columns (Pivot 1) |
# | 7 | How does one country compare, over time and on the map? | Interactive explorer: map + country menu + time lapse + side panel | Wide: one row per country (ISO code), one column per year |
#
# **Tools:** matplotlib for the static figures (full control over every detail), and Plotly.js inside a self-contained HTML page for the interactive explorer.
#
# **Design rules used throughout:** one accent color per message, gray for context, thin 2px lines,
# light hairline gridlines, direct labels where they help, a clear title that states the finding,
# and a source note on every chart.


# %%
# ---------------------------------------------------------------
# Visual system: colors, fonts and small helpers shared by every chart
# ---------------------------------------------------------------
SURFACE = "#fcfcfb"   # background
INK     = "#0b0b0b"   # titles
INK_2   = "#52514e"   # subtitles, labels
MUTED   = "#8a8984"   # axis ticks, notes
GRID    = "#e6e5e1"   # hairline gridlines
CONTEXT = "#d3d2cc"   # gray context lines

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SOURCE = "Source: Our World in Data, life expectancy at birth."

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial",
                        "Liberation Sans", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.fontsize": 10,
})


def headline(fig, title, subtitle, top=0.80, left=0.07):
    """Left-aligned title + subtitle block above the plot area."""
    fig.text(left, 0.955, title, fontsize=17, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(left, 0.895, subtitle, fontsize=11.5, color=INK_2, ha="left", va="top",
             linespacing=1.4)
    fig.subplots_adjust(top=top)


def footer(fig, note=SOURCE, left=0.07):
    fig.text(left, 0.02, note, fontsize=8.5, color=MUTED, ha="left", va="bottom")


def finish(fig):
    """Show the figure inline in a notebook; just close it when run as a script."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None and type(ip).__name__ != "TerminalInteractiveShell":
            plt.show()
            return
    except ImportError:
        pass
    plt.close(fig)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200)
    print("Saved:", path)


# %% [markdown]
# ## 6. Prepare the data and create the visualizations
#
# ### Chart 1: The long view
#
# For most of history, life expectancy at birth sat around 30 to 40 years, mostly because so many children died young.
# Countries with long yearly records show when that changed and how wars and pandemics cut it short.


# %%
LONG_RUN = {"Sweden": BLUE, "France": ORANGE, "United States": AQUA}
long_df = countries[countries["entity"].isin(LONG_RUN) & (countries["year"] >= 1750)]
world_long = raw[(raw["entity"] == "World")][["year", "life_exp"]]

fig, ax = plt.subplots(figsize=(12, 6.6))
for name, color in LONG_RUN.items():
    s = long_df[long_df["entity"] == name]
    ax.plot(s["year"], s["life_exp"], color=color, lw=1.8, label=name, zorder=3)

ax.plot(world_long["year"], world_long["life_exp"], color=INK_2, lw=2, zorder=4, label="World")
sparse = world_long[world_long["year"] < 1950]
ax.plot(sparse["year"], sparse["life_exp"], "o", ms=6, color=INK_2, mec=SURFACE, mew=1.5, zorder=5)

ax.annotate("1918 flu pandemic", xy=(1918, 48), xytext=(1900, 72),
            fontsize=9.5, color=INK_2, ha="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.annotate("World War II", xy=(1944, 47), xytext=(1932, 24),
            fontsize=9.5, color=INK_2, ha="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.annotate("Famine in Sweden, 1773", xy=(1773, 18.5), xytext=(1782, 13),
            fontsize=9.5, color=INK_2, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.text(1806, 25, "Before 1950 the world line joins a few rough\nestimates (dots), so it hides the ups and downs",
        fontsize=9, color=MUTED, ha="left", va="top")

ax.set_xlim(1750, 2026)
ax.set_ylim(10, 90)
ax.set_yticks(range(20, 91, 20))
ax.set_ylabel("Years")
ax.legend(loc="upper left", ncol=4, handlelength=1.6, bbox_to_anchor=(0, 1.02))

headline(fig, "Life expectancy barely moved for centuries, then doubled",
         "Life expectancy at birth in years, 1750 to 2023. Countries with long yearly records, plus the world estimate.")
footer(fig)
save(fig, "01_long_view.png")
finish(fig)


# %% [markdown]
# ### Chart 2: Every region gained, at a different pace
#
# One panel per region keeps the lines readable. The blue line is the region,
# the dark gray line is the world average, and the light gray lines are the other regions for context.


# %%
fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharex=True, sharey=True)
world = regions[regions["entity"] == "World"]

for ax, region in zip(axes.flat, REGIONS):
    for other in REGIONS:
        if other != region:
            s = regions[regions["entity"] == other]
            ax.plot(s["year"], s["life_exp"], color=CONTEXT, lw=1, zorder=1)
    ax.plot(world["year"], world["life_exp"], color=INK_2, lw=1.4, zorder=2)
    s = regions[regions["entity"] == region]
    ax.plot(s["year"], s["life_exp"], color=BLUE, lw=2.4, zorder=3)

    first, last = s.iloc[0], s.iloc[-1]
    for pt, ha, dx in [(first, "right", -2), (last, "left", 2)]:
        ax.plot(pt["year"], pt["life_exp"], "o", ms=7, color=BLUE, mec=SURFACE, mew=2, zorder=4)
        ax.text(pt["year"] + dx, pt["life_exp"], f"{pt['life_exp']:.0f}", ha=ha, va="center",
                fontsize=10, color=INK, fontweight="bold")

    title = region.replace("Latin America and the Caribbean", "Latin America & Caribbean")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_xlim(1942, 2031)
    ax.set_ylim(30, 88)
    ax.set_xticks([1950, 1990, 2023])
    ax.set_yticks([30, 50, 70])

axes[0, 0].text(1972, 63, "World", fontsize=9, color=INK_2, ha="left")

headline(fig, "Every region gained, and the gaps narrowed",
         "Life expectancy at birth in years, 1950 to 2023. Blue: region. Dark gray: world. Light gray: other regions.",
         top=0.80)
fig.subplots_adjust(hspace=0.35, wspace=0.12, left=0.07, right=0.97, bottom=0.09)
footer(fig)
save(fig, "02_regions_small_multiples.png")
finish(fig)


# %% [markdown]
# ### Chart 3: The world's countries moved closer together
#
# Each ridge is the distribution of all countries in one year. The ridge moves right (longer lives)
# and gets narrower (less inequality between countries). A Gaussian kernel density is computed with numpy, so no extra libraries are needed.


# %%
SNAPSHOTS = [1950, 1970, 1990, 2010, 2023]
ridge_colors = [BLUE_RAMP[i] for i in (3, 5, 7, 9, 11)]   # light (old) to dark (recent)
grid_x = np.linspace(10, 95, 600)


def kde(values, x):
    values = np.asarray(values, dtype=float)
    bw = 1.06 * values.std(ddof=1) * len(values) ** (-1 / 5)   # Silverman's rule
    z = (x[:, None] - values[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (len(values) * bw * np.sqrt(2 * np.pi))


snap = countries_modern[countries_modern["year"].isin(SNAPSHOTS)]
densities = {y: kde(snap.loc[snap["year"] == y, "life_exp"], grid_x) for y in SNAPSHOTS}
scale = 0.9 / max(d.max() for d in densities.values())

fig, ax = plt.subplots(figsize=(12, 7))
ax.grid(False)
for i, (year, color) in enumerate(zip(SNAPSHOTS, ridge_colors)):
    base = (len(SNAPSHOTS) - 1 - i) * 0.62
    d = densities[year] * scale
    ax.fill_between(grid_x, base, base + d, color=color, zorder=10 + i, lw=0)
    ax.plot(grid_x, base + d, color=SURFACE, lw=2, zorder=10 + i)
    ax.axhline(base, color=GRID, lw=0.8, zorder=1)
    med = snap.loc[snap["year"] == year, "life_exp"].median()
    h = np.interp(med, grid_x, d)
    ax.plot([med, med], [base, base + h], color=INK, lw=1.4, zorder=10 + i + 0.5)
    ax.text(med, base + h + 0.05, f"median {med:.0f}", ha="center", va="bottom",
            fontsize=9.5, color=INK, zorder=10 + i + 0.6,
            bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.9))
    ax.text(11, base + 0.08, str(year), fontsize=13, fontweight="bold", color=INK, va="bottom")

ax.set_xlim(10, 95)
ax.set_ylim(-0.05, (len(SNAPSHOTS) - 1) * 0.62 + 1.15)
ax.set_yticks([])
ax.set_xticks(range(20, 91, 10))
ax.set_xlabel("Life expectancy at birth (years)")
ax.spines["bottom"].set_visible(False)

headline(fig, "Countries are living longer, and the gap between them is shrinking",
         f"Distribution of life expectancy across {snap['entity'].nunique()} countries and territories in five snapshot years.")
footer(fig)
save(fig, "03_convergence_ridgeline.png")
finish(fig)


# %% [markdown]
# ### Chart 4: Gains in the most populous countries
#
# A dumbbell chart shows a start and an end point for each country. These 15 countries hold
# most of the world's people, so their progress drives the global average.


# %%
POPULOUS = ["India", "China", "United States", "Indonesia", "Pakistan", "Nigeria", "Brazil",
            "Bangladesh", "Russia", "Ethiopia", "Mexico", "Japan", "Egypt", "Philippines",
            "Democratic Republic of Congo"]
db = pv_wide.loc[POPULOUS, [1950, 2023]].copy()
db["gain"] = db[2023] - db[1950]
db = db.sort_values(2023)
db.index = db.index.str.replace("Democratic Republic of Congo", "DR Congo")
y = np.arange(len(db))

C1950, C2023 = BLUE_RAMP[3], BLUE_RAMP[9]
fig, ax = plt.subplots(figsize=(12, 8))
ax.grid(axis="x", color=GRID, lw=0.8)
ax.grid(axis="y", visible=False)
ax.hlines(y, db[1950], db[2023], color=CONTEXT, lw=2.5, zorder=1)
ax.scatter(db[1950], y, s=90, color=C1950, edgecolor=SURFACE, linewidth=2, zorder=3)
ax.scatter(db[2023], y, s=90, color=C2023, edgecolor=SURFACE, linewidth=2, zorder=3)

for yi, (_, row) in zip(y, db.iterrows()):
    ax.text(row[2023] + 1.2, yi, f"{row[2023]:.0f}", va="center", fontsize=10, color=INK)
    ax.text(97, yi, f"+{row['gain']:.0f} yrs", va="center", ha="right", fontsize=10, color=INK_2)

ax.text(97, len(db) - 0.45, "Gain", ha="right", va="bottom", fontsize=10, color=MUTED, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels(db.index, fontsize=10.5, color=INK)
ax.set_xlim(20, 97)
ax.set_ylim(-0.7, len(db) - 0.1)
ax.set_xticks(range(20, 91, 10))
ax.set_xlabel("Life expectancy at birth (years)")

handles = [Line2D([], [], marker="o", ls="", ms=9, color=c, mec=SURFACE, mew=2, label=l)
           for c, l in [(C1950, "1950"), (C2023, "2023")]]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)

n30 = int((db["gain"] >= 30).sum())
headline(fig, f"{n30} of the 15 biggest countries added 30 or more years of life",
         "Life expectancy at birth in 1950 and 2023, the 15 most populous countries, sorted by 2023 value.",
         left=0.17)
fig.subplots_adjust(left=0.17, right=0.97, bottom=0.1)
footer(fig, left=0.17)
save(fig, "04_dumbbell_populous.png")
finish(fig)


# %% [markdown]
# ### Chart 5: The COVID-19 shock
#
# Most countries lost ground between 2019 and 2021. This chart shows the countries with the largest drops,
# and whether they had recovered by 2023. The world average is the first row for reference.


# %%
cv = key.dropna(subset=[2019, 2021, 2023]).sort_values("covid_change_2019_2021").head(12)
n_down = int((key["covid_change_2019_2021"] < 0).sum())
n_all = int(key["covid_change_2019_2021"].notna().sum())

w = regions[regions["entity"] == "World"].set_index("year")["life_exp"]
world_row = pd.DataFrame({2019: [w[2019]], 2021: [w[2021]], 2023: [w[2023]]}, index=["World"])
plot_df = pd.concat([world_row, cv[[2019, 2021, 2023]]])
y = np.arange(len(plot_df))[::-1].astype(float)
y[0] += 0.6   # extra space under the World row

C19, C21, C23 = "#a3a29d", RED, BLUE
fig, ax = plt.subplots(figsize=(12, 7.8))
ax.grid(axis="x", color=GRID, lw=0.8)
ax.grid(axis="y", visible=False)
lo = plot_df.min(axis=1)
hi = plot_df.max(axis=1)
ax.hlines(y, lo, hi, color=CONTEXT, lw=2.5, zorder=1)
ax.scatter(plot_df[2021], y, s=90, color=C21, edgecolor=SURFACE, linewidth=2, zorder=3)
ax.scatter(plot_df[2023], y, s=90, color=C23, edgecolor=SURFACE, linewidth=2, zorder=3)
# 2019 as an open ring on top, so it stays visible when it sits on the 2023 dot
ax.scatter(plot_df[2019], y, s=150, facecolor="none", edgecolor=C19, linewidth=2.2, zorder=4)

for yi, (_, row) in zip(y, plot_df.iterrows()):
    drop = row[2021] - row[2019]
    ax.text(row[2021] - 0.4, yi, f"{drop:+.1f}", ha="right", va="center", fontsize=9.5, color=INK_2)

ax.set_yticks(y)
labels = ax.set_yticklabels(plot_df.index, fontsize=10.5, color=INK)
labels[0].set_fontweight("bold")
ax.axhline((y[0] + y[1]) / 2, color=GRID, lw=0.8)
ax.set_xlim(np.floor(lo.min()) - 2, np.ceil(hi.max()) + 1)
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
ax.set_xlabel("Life expectancy at birth (years)")

handles = [Line2D([], [], marker="o", ls="", ms=10, mfc="none", mec=C19, mew=2.2, label="2019 (before)")] +           [Line2D([], [], marker="o", ls="", ms=9, color=c, mec=SURFACE, mew=2, label=l)
           for c, l in [(C21, "2021 (low point)"), (C23, "2023 (after)")]]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1.06), ncol=3)

headline(fig, "COVID-19 erased years of progress, then most of it came back",
         f"{n_down} of {n_all} countries and territories had a lower life expectancy in 2021 than in 2019. "
         "Shown: the 12 largest drops.\nNumbers next to the red dots are the change from 2019 to 2021, in years.",
         left=0.17, top=0.78)
fig.subplots_adjust(left=0.17, right=0.97, bottom=0.1)
footer(fig, left=0.17)
save(fig, "05_covid_shock.png")
finish(fig)


# %% [markdown]
# ### Chart 6: Regional summary heatmap
#
# This is Pivot 1 turned into a picture. Darker blue means longer lives. The values are printed in each cell,
# so the chart also works as a table.


# %%
hm = pv_region_decade.copy()
hm.index = hm.index.str.replace("Latin America and the Caribbean", "Latin America & Caribbean")
cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
vmin, vmax = 35, 82

fig, ax = plt.subplots(figsize=(12, 6.2))
ax.grid(False)
mesh = ax.pcolormesh(hm.values, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors=SURFACE, linewidth=3)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        v = hm.values[i, j]
        rgb = cmap((v - vmin) / (vmax - vmin))[:3]
        lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        ax.text(j + 0.5, i + 0.5, f"{v:.0f}", ha="center", va="center", fontsize=11.5,
                color="white" if lum < 0.45 else INK, fontweight="bold" if i == 0 else "normal")

ax.set_xticks(np.arange(hm.shape[1]) + 0.5)
xt = [c if c != "2020s" else "2020s*" for c in hm.columns]
ax.set_xticklabels(xt, fontsize=10.5, color=INK_2)
ax.set_yticks(np.arange(hm.shape[0]) + 0.5)
ax.set_yticklabels(hm.index, fontsize=10.5, color=INK)
ax.get_yticklabels()[0].set_fontweight("bold")
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.spines["bottom"].set_visible(False)

headline(fig, "Africa today has still not reached Europe's level of the 1950s",
         "Average life expectancy at birth by region and decade, in years.", left=0.21, top=0.78)
fig.subplots_adjust(left=0.21, right=0.97, bottom=0.08)
footer(fig, SOURCE + "  *2020s covers 2020 to 2023.", left=0.21)
save(fig, "06_region_decade_heatmap.png")
finish(fig)


# %% [markdown]
# ### Chart 7: Interactive country explorer
#
# One HTML page with everything in it:
# - **Country menu.** Pick a country from the list or click it on the map. The other countries fade out so the chosen one stands out, and the map zooms in on it (you can turn the zoom off).
# - **Time lapse.** Press Play or drag the year slider to move from 1950 to 2023. Everything on the page follows the year.
# - **Side panel.** Three stat tiles (life expectancy, change since 1950, world rank), a line chart against the world average and the middle 80% of countries, and a strip showing where the country stands among all others that year.
# - **Table view.** The same numbers as a table, under "Show the numbers as a table".


# %%
# ---------------------------------------------------------------
# Chart 7: interactive country explorer, saved as ONE self-contained HTML file.
# Pick a country (menu or click on the map): the others fade out and the side
# panel shows stat tiles, a line chart vs the world, and its place among all countries.
# The time lapse (Play button + year slider) lives inside the page itself.
# ---------------------------------------------------------------
import json
import urllib.request

EXPLORER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Life expectancy explorer, 1950 to 2023</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.2.0/plotly.min.js"></script>
<style>
  :root {
    --surface: #fcfcfb; --card: #ffffff; --ink: #0b0b0b; --ink-2: #52514e;
    --muted: #6f6e69; --grid: #e6e5e1; --line: #d3d2cc; --blue: #2a78d6; --wash: #f3f2ee;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--surface); color: var(--ink);
         font-family: Inter, "Helvetica Neue", Helvetica, Arial, sans-serif; }
  .wrap { max-width: 1320px; margin: 0 auto; padding: 28px 28px 20px; }
  h1 { font-size: 25px; font-weight: 700; letter-spacing: -0.01em; margin: 0 0 6px; }
  .sub { color: var(--ink-2); font-size: 14.5px; margin: 0; line-height: 1.5; }

  .controls { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 14px 22px;
              padding: 18px 0 18px; border-bottom: 1px solid var(--grid); margin-bottom: 20px; }
  .ctl { display: flex; flex-direction: column; gap: 5px; }
  .ctl > span { font-size: 11.5px; font-weight: 600; color: var(--muted);
                text-transform: uppercase; letter-spacing: 0.05em; }
  select, button { font: inherit; font-size: 14px; color: var(--ink); border-radius: 8px; }
  select { min-width: 260px; padding: 8px 10px; border: 1px solid var(--line); background: var(--card); }
  button { padding: 8px 14px; border: 1px solid var(--line); background: var(--card); cursor: pointer; }
  button:hover { background: var(--wash); }
  button.primary { background: var(--ink); color: #fff; border-color: var(--ink); min-width: 92px; }
  button.primary:hover { background: #2b2b2a; }
  select:focus-visible, button:focus-visible, input:focus-visible { outline: 2px solid var(--blue); outline-offset: 2px; }
  .check { display: flex; align-items: center; gap: 7px; font-size: 14px; color: var(--ink-2); padding-bottom: 9px; }
  .check input { accent-color: var(--blue); width: 16px; height: 16px; }
  .timeline { display: flex; align-items: center; gap: 14px; flex: 1; min-width: 340px; }
  .timeline input[type=range] { flex: 1; accent-color: var(--blue); }
  .year { font-size: 30px; font-weight: 700; font-variant-numeric: tabular-nums; min-width: 76px; text-align: right; }

  .layout { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(0, 1fr); gap: 22px; }
  @media (max-width: 980px) { .layout { grid-template-columns: 1fr; } }
  .card { background: var(--card); border: 1px solid var(--grid); border-radius: 14px; padding: 16px 18px; }
  .card h2 { font-size: 14px; font-weight: 600; margin: 0; }
  .card h2 span { font-weight: 400; color: var(--muted); }
  #map { width: 100%; aspect-ratio: 1.65 / 1; min-height: 300px; }
  .stack { display: flex; flex-direction: column; gap: 22px; min-width: 0; }
  #rank { height: 330px; }

  .subject { font-size: 23px; font-weight: 700; margin: 0; letter-spacing: -0.01em; }
  .hint { font-size: 13px; color: var(--muted); margin: 4px 0 0; min-height: 18px; }
  .tiles { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin: 14px 0 4px; }
  .tile { background: var(--wash); border-radius: 10px; padding: 10px 12px; }
  .tile .label { font-size: 12px; color: var(--ink-2); }
  .tile .value { font-size: 24px; font-weight: 600; font-variant-numeric: tabular-nums; margin-top: 2px; }
  .tile .value small { font-size: 13px; font-weight: 500; color: var(--ink-2); margin-left: 3px; }
  .tile .note { font-size: 11.5px; color: var(--muted); margin-top: 2px; }
  .section { margin-top: 14px; }
  #line { height: 290px; }
  #strip { height: 190px; }

  details { margin-top: 12px; font-size: 13px; }
  summary { cursor: pointer; color: var(--ink-2); font-weight: 500; }
  table { border-collapse: collapse; width: 100%; margin-top: 8px; font-variant-numeric: tabular-nums; }
  th, td { padding: 5px 8px; text-align: right; border-bottom: 1px solid var(--grid); }
  th:first-child, td:first-child { text-align: left; }
  th { font-weight: 600; color: var(--ink-2); font-size: 12px; }
  footer { font-size: 11.5px; color: var(--muted); margin-top: 16px; line-height: 1.6; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Life expectancy explorer, 1950 to 2023</h1>
  <p class="sub">Pick a country to take a closer look. The other countries fade into the background, and the panel on the right shows its story over time.</p>

  <div class="controls">
    <label class="ctl"><span>Country</span>
      <select id="country" aria-label="Choose a country"></select>
    </label>
    <div class="ctl"><span>&nbsp;</span><button id="clear" type="button">Show world</button></div>
    <label class="check"><input id="zoom" type="checkbox" checked> Zoom to country</label>
    <div class="ctl" style="flex:1;min-width:340px"><span>Year</span>
      <div class="timeline">
        <button id="play" class="primary" type="button" aria-label="Play the time lapse">&#9654; Play</button>
        <input id="year" type="range" min="1950" max="2023" step="1" value="2023" aria-label="Year">
        <div id="yearLabel" class="year" aria-live="polite">2023</div>
      </div>
    </div>
  </div>

  <div class="layout">
    <div class="stack">
      <div class="card">
        <h2>Life expectancy at birth <span id="mapYear">in 2023, years</span></h2>
        <div id="map"></div>
      </div>
      <div class="card">
        <h2 id="rankTitle">Ranking</h2>
        <div id="rank"></div>
      </div>
    </div>

    <div class="card">
      <p id="subject" class="subject">World</p>
      <p id="hint" class="hint"></p>
      <div class="tiles">
        <div class="tile"><div class="label" id="t1l"></div><div class="value" id="t1v"></div><div class="note" id="t1n"></div></div>
        <div class="tile"><div class="label" id="t2l"></div><div class="value" id="t2v"></div><div class="note" id="t2n"></div></div>
        <div class="tile"><div class="label" id="t3l"></div><div class="value" id="t3v"></div><div class="note" id="t3n"></div></div>
      </div>
      <div class="section">
        <h2>Over time <span>compared with the world</span></h2>
        <div id="line"></div>
      </div>
      <div class="section">
        <h2>Where it stands <span id="stripYear">among all countries in 2023</span></h2>
        <div id="strip"></div>
      </div>
      <details>
        <summary>Show the numbers as a table</summary>
        <table id="table"></table>
      </details>
    </div>
  </div>

  <footer>
    Source: Our World in Data, life expectancy at birth. The gray band shows the middle 80% of countries (10th to 90th percentile).
    Some small countries and territories are too small to appear on the map, but you can still pick them from the list.
  </footer>
</div>

<script>
const DATA = __DATA__;
const TOPO = __TOPO__;
if (TOPO) {
  window.PlotlyGeoAssets = window.PlotlyGeoAssets || {};
  window.PlotlyGeoAssets.topojson = window.PlotlyGeoAssets.topojson || {};
  window.PlotlyGeoAssets.topojson["world_110m"] = TOPO;
}

const INK = "#0b0b0b", INK2 = "#52514e", MUTED = "#6f6e69", GRID = "#e6e5e1", SURFACE = "#ffffff", BLUE = "#2a78d6";
const RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
              "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];
const SCALE = RAMP.map((c, i) => [i / (RAMP.length - 1), c]);
const FONT = { family: "Inter, Helvetica Neue, Arial, sans-serif", color: INK2, size: 12 };
const CONFIG = { displayModeBar: false, responsive: true };
const YEARS = DATA.years, FIRST = YEARS[0], LAST = YEARS[YEARS.length - 1];
const BIG = new Set(["RUS", "CAN", "USA", "CHN", "BRA", "AUS", "KAZ", "ARG", "IND", "DZA", "COD", "GRL",
                     "SAU", "MEX", "IDN", "SDN", "LBY", "IRN", "MNG", "PER", "TCD", "NER", "AGO", "MLI"]);

const byCode = {};
DATA.countries.forEach(c => { byCode[c.c] = c; });
const CENTROIDS = {};
if (TOPO && TOPO.objects && TOPO.objects.countries) {
  TOPO.objects.countries.geometries.forEach(g => {
    if (g.id && g.properties && g.properties.ct) CENTROIDS[g.id] = g.properties.ct;
  });
}

// Ranks per year (1 = highest life expectancy)
const RANKS = YEARS.map((_, i) => {
  const rows = DATA.countries.filter(c => c.v[i] !== null).sort((a, b) => b.v[i] - a.v[i]);
  const r = {}; rows.forEach((c, k) => { r[c.c] = k + 1; });
  return { r, n: rows.length };
});

const state = { year: LAST, code: null, zoom: true, playing: false, timer: null };
const $ = id => document.getElementById(id);
const yi = () => state.year - FIRST;
const fmt = v => (v === null || v === undefined) ? "n/a" : v.toFixed(1);
const signed = v => (v >= 0 ? "+" : "") + v.toFixed(1);

// ---------- controls ----------
function buildSelect() {
  const sel = $("country");
  const opt0 = document.createElement("option");
  opt0.value = ""; opt0.textContent = "World (no country selected)";
  sel.appendChild(opt0);
  [...DATA.countries].sort((a, b) => a.n.localeCompare(b.n)).forEach(c => {
    const o = document.createElement("option");
    o.value = c.c; o.textContent = c.n; sel.appendChild(o);
  });
  sel.addEventListener("change", () => select(sel.value || null));
  $("clear").addEventListener("click", () => select(null));
  $("zoom").addEventListener("change", e => { state.zoom = e.target.checked; renderMap(); });
  $("year").addEventListener("input", e => { stop(); setYear(+e.target.value); });
  $("play").addEventListener("click", togglePlay);
}

function select(code) {
  state.code = code && byCode[code] ? code : null;
  $("country").value = state.code || "";
  renderAll();
}

function setYear(y) {
  state.year = Math.max(FIRST, Math.min(LAST, y));
  $("year").value = state.year;
  $("yearLabel").textContent = state.year;
  renderAll();
}

function togglePlay() { state.playing ? stop() : play(); }
function play() {
  if (state.year >= LAST) setYear(FIRST);
  state.playing = true;
  $("play").innerHTML = "&#10074;&#10074; Pause";
  state.timer = setInterval(() => {
    if (state.year >= LAST) { stop(); return; }
    setYear(state.year + 1);
  }, 180);
}
function stop() {
  state.playing = false;
  clearInterval(state.timer);
  $("play").innerHTML = "&#9654; Play";
}

// ---------- map ----------
function renderMap() {
  const i = yi(), sel = state.code, ranks = RANKS[i];
  const part = (rows, opacity, lineW, lineC, hover) => ({
    type: "choropleth", coloraxis: "coloraxis",
    locations: rows.map(c => c.c), z: rows.map(c => c.v[i]), text: rows.map(c => c.n),
    customdata: rows.map(c => ranks.r[c.c] ? ranks.r[c.c] + " of " + ranks.n : "n/a"),
    hovertemplate: hover ? "<b>%{text}</b><br>%{z:.1f} years<br>Rank %{customdata}<extra></extra>" : undefined,
    hoverinfo: hover ? undefined : "skip",
    marker: { opacity, line: { width: lineW, color: lineC } }
  });
  const others = DATA.countries.filter(c => c.c !== sel);
  const traces = [part(others, sel ? 0.2 : 1, 0.4, SURFACE, true)];
  if (sel) traces.push(part([byCode[sel]], 1, 2, INK, true));
  const geo = {
    projection: { type: "natural earth", scale: 1 }, showframe: false, showcoastlines: false,
    showcountries: true, countrycolor: SURFACE, showland: true, landcolor: "#ecebe7",
    bgcolor: "rgba(0,0,0,0)", lataxis: { range: [-58, 86] }, domain: { x: [0, 1], y: [0.11, 1] }
  };
  const ct = sel ? CENTROIDS[sel] : null;
  if (ct && state.zoom) {
    geo.projection = { type: "natural earth", scale: BIG.has(sel) ? 1.9 : 3.2, rotation: { lon: ct[0] } };
    geo.center = { lon: ct[0], lat: ct[1] };
    geo.lataxis = { range: [-90, 90] };
  }
  geo.domain = { x: [0, 1], y: [0.11, 1] };
  const layout = {
    geo, margin: { l: 0, r: 0, t: 6, b: 0 }, paper_bgcolor: "rgba(0,0,0,0)", font: FONT,
    coloraxis: { cmin: 25, cmax: 87, colorscale: SCALE,
      colorbar: { title: { text: "Years", side: "top", font: { size: 11, color: INK2 } }, orientation: "h",
                  thickness: 9, len: 0.34, x: 0.01, xanchor: "left", y: 0, yanchor: "bottom",
                  tickfont: { size: 10, color: MUTED }, outlinewidth: 0 } },
    uirevision: (sel || "world") + (state.zoom ? "-z" : ""),
    hoverlabel: { bgcolor: "#fff", bordercolor: GRID, font: { color: INK, size: 12 } }
  };
  Plotly.react("map", traces, layout, CONFIG);
  $("mapYear").textContent = "in " + state.year + ", years";
}

function renderRank() {
  const i = yi(), rk = RANKS[i], sel = state.code;
  const ordered = DATA.countries.filter(c => c.v[i] !== null).sort((a, b) => b.v[i] - a.v[i]);
  let rows, title;
  if (sel && rk.r[sel]) {
    const p = rk.r[sel] - 1, lo = Math.max(0, Math.min(p - 4, ordered.length - 9));
    rows = ordered.slice(lo, lo + 9);
    title = "Closest neighbors in the ranking, " + state.year;
  } else {
    rows = ordered.slice(0, 5).concat([null], ordered.slice(-5));
    title = "Highest and lowest life expectancy, " + state.year;
  }
  $("rankTitle").textContent = title;
  const label = c => c ? "#" + rk.r[c.c] + "  " + c.n : " ";
  const ys = rows.map((c, k) => c ? label(c) : " ".repeat(k + 1));
  const trace = {
    type: "bar", orientation: "h", y: ys, x: rows.map(c => c ? c.v[i] : null),
    marker: { color: rows.map(c => c && c.c === sel ? BLUE : "#c9c8c2") },
    text: rows.map(c => c ? c.v[i].toFixed(1) : ""), textposition: "outside", cliponaxis: false,
    textfont: { color: INK, size: 11.5 }, customdata: rows.map(c => c ? c.c : ""),
    hovertemplate: "<b>%{y}</b><br>%{x:.1f} years<extra></extra>", width: 0.62
  };
  const layout = {
    margin: { l: 190, r: 40, t: 6, b: 24 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: FONT, showlegend: false,
    xaxis: { range: [0, 92], gridcolor: GRID, zeroline: false, tickfont: { color: MUTED }, dtick: 20, fixedrange: true },
    yaxis: { autorange: "reversed", tickfont: { color: INK, size: 12 }, fixedrange: true },
    hoverlabel: { bgcolor: "#fff", bordercolor: GRID, font: { color: INK, size: 12 } }
  };
  Plotly.react("rank", [trace], layout, CONFIG);
}

// ---------- side panel ----------
function subjectSeries() {
  return state.code ? { name: byCode[state.code].n, v: byCode[state.code].v } : { name: "World", v: DATA.world };
}

function renderPanel() {
  const i = yi(), s = subjectSeries(), v = s.v[i], w = DATA.world[i];
  $("subject").textContent = s.name;
  const set = (k, label, value, unit, note) => {
    $(k + "l").textContent = label;
    const el = $(k + "v"); el.textContent = value;
    if (unit) { const sm = document.createElement("small"); sm.textContent = unit; el.appendChild(sm); }
    $(k + "n").textContent = note || "";
  };
  set("t1", "Life expectancy in " + state.year, fmt(v), "years", "");
  if (state.year === FIRST) set("t2", "Change since 1950", "0.0", "years", "1950 is the first year");
  else set("t2", "Change since 1950", signed(v - s.v[0]), "years", "from " + fmt(s.v[0]) + " in 1950");

  if (state.code) {
    const rk = RANKS[i];
    set("t3", "Rank in " + state.year, rk.r[state.code] ? "#" + rk.r[state.code] : "n/a", "of " + rk.n,
        signed(v - w) + " years vs world average");
    $("hint").textContent = "Click another country on the map, or press Show world to go back.";
  } else {
    const above = DATA.countries.filter(c => c.v[i] !== null && c.v[i] > w).length;
    const total = DATA.countries.filter(c => c.v[i] !== null).length;
    set("t3", "Countries above world average", String(above), "of " + total, "in " + state.year);
    $("hint").textContent = "Click a country on the map or pick one from the list.";
  }
  renderTable(s);
}

function renderLine() {
  const i = yi(), s = subjectSeries();
  const traces = [
    { x: YEARS, y: DATA.p90, mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
    { x: YEARS, y: DATA.p10, mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: "rgba(138,137,132,0.16)",
      name: "Middle 80% of countries", customdata: DATA.p90,
      hovertemplate: "%{y:.0f} to %{customdata:.0f}<extra>Middle 80%</extra>" },
    { x: YEARS, y: DATA.world, mode: "lines", line: { color: state.code ? INK2 : BLUE, width: state.code ? 1.6 : 2.5 },
      name: "World average", hovertemplate: "%{y:.1f}<extra>World</extra>" }
  ];
  if (state.code) {
    traces.push({ x: YEARS, y: s.v, mode: "lines", line: { color: BLUE, width: 2.5 }, name: s.name,
                  hovertemplate: "%{y:.1f}<extra>" + s.name.replace(/[<>&]/g, "") + "</extra>" });
  }
  traces.push({ x: [state.year], y: [s.v[i]], mode: "markers", hoverinfo: "skip", showlegend: false,
                marker: { size: 11, color: BLUE, line: { color: SURFACE, width: 2 } } });
  const lo = Math.min(20, Math.floor(Math.min(...s.v.filter(x => x !== null)) / 10) * 10);
  const layout = {
    margin: { l: 34, r: 8, t: 30, b: 26 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: FONT, hovermode: "x unified", showlegend: true,
    legend: { orientation: "h", x: 0, y: 1.18, font: { size: 11.5, color: INK2 }, traceorder: "reversed" },
    xaxis: { range: [FIRST - 1, LAST + 1], showgrid: false, zeroline: false, tickvals: [1950, 1970, 1990, 2010],
             tickfont: { color: MUTED }, linecolor: GRID, showspikes: false },
    yaxis: { range: [lo, 90], gridcolor: GRID, zeroline: false, tickfont: { color: MUTED },
             tickvals: [20, 40, 60, 80].filter(t => t > lo) },
    shapes: [{ type: "line", x0: state.year, x1: state.year, yref: "paper", y0: 0, y1: 1,
               line: { color: "#b9b8b2", width: 1 } }],
    hoverlabel: { bgcolor: "#fff", bordercolor: GRID, font: { color: INK, size: 12 } }
  };
  Plotly.react("line", traces, layout, CONFIG);
}

function renderStrip() {
  const i = yi();
  const rows = DATA.countries.map((c, k) => ({ c, v: c.v[i], j: (((k * 0.6180339) % 1) - 0.5) * 0.8 }))
                             .filter(r => r.v !== null);
  const traces = [{
    x: rows.map(r => r.v), y: rows.map(r => r.j), text: rows.map(r => r.c.n), customdata: rows.map(r => r.c.c),
    mode: "markers", type: "scatter", hovertemplate: "<b>%{text}</b><br>%{x:.1f} years<extra></extra>",
    marker: { size: 7, color: "#bdbcb6", line: { color: SURFACE, width: 1 } }, showlegend: false
  }];
  const annotations = [], w = DATA.world[i];
  annotations.push({ x: w, y: -0.62, yref: "y", text: "World " + w.toFixed(1), showarrow: false,
                     font: { size: 11, color: INK2 }, yanchor: "top" });
  if (state.code) {
    const r = rows.find(r => r.c.c === state.code);
    if (r) {
      traces.push({ x: [r.v], y: [r.j], mode: "markers", hoverinfo: "skip", showlegend: false,
                    marker: { size: 14, color: BLUE, line: { color: SURFACE, width: 2 } } });
      annotations.push({ x: r.v, y: r.j, text: "<b>" + r.c.n.replace(/[<>&]/g, "") + " " + r.v.toFixed(1) + "</b>",
                         showarrow: true, arrowhead: 0, arrowwidth: 1, arrowcolor: INK2,
                         axref: "x", ax: r.v, ayref: "y", ay: 0.62, yanchor: "bottom",
                         font: { size: 12, color: INK }, bgcolor: "rgba(255,255,255,0.9)" });
    }
  }
  const layout = {
    margin: { l: 8, r: 8, t: 10, b: 26 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: FONT, hovermode: "closest",
    xaxis: { range: [10, 90], gridcolor: GRID, zeroline: false, tickfont: { color: MUTED }, dtick: 10 },
    yaxis: { range: [-0.95, 0.95], visible: false, fixedrange: true },
    shapes: [{ type: "line", x0: w, x1: w, y0: -0.6, y1: 0.6, line: { color: INK2, width: 1.2 } }],
    annotations,
    hoverlabel: { bgcolor: "#fff", bordercolor: GRID, font: { color: INK, size: 12 } }
  };
  Plotly.react("strip", traces, layout, CONFIG);
  $("stripYear").textContent = "among all countries in " + state.year;
}

function renderTable(s) {
  const t = $("table");
  t.replaceChildren();
  const head = document.createElement("tr");
  const cols = state.code ? ["Year", s.name, "World", "Rank"] : ["Year", "World", "Middle 80% of countries"];
  cols.forEach(h => { const th = document.createElement("th"); th.textContent = h; head.appendChild(th); });
  t.appendChild(head);
  const marks = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2019, 2021, 2023];
  if (!marks.includes(state.year)) marks.push(state.year);
  marks.sort((a, b) => a - b).forEach(y => {
    const i = y - FIRST, tr = document.createElement("tr");
    const cells = state.code
      ? [y, fmt(s.v[i]), fmt(DATA.world[i]), RANKS[i].r[state.code] ? "#" + RANKS[i].r[state.code] : "n/a"]
      : [y, fmt(DATA.world[i]), DATA.p10[i].toFixed(0) + " to " + DATA.p90[i].toFixed(0)];
    cells.forEach(v => { const td = document.createElement("td"); td.textContent = v; tr.appendChild(td); });
    if (y === state.year) tr.style.fontWeight = "600";
    t.appendChild(tr);
  });
}

function renderAll() { renderMap(); renderRank(); renderPanel(); renderLine(); renderStrip(); }

buildSelect();
renderAll();
document.getElementById("map").on("plotly_click", ev => {
  const p = ev.points && ev.points[0];
  if (p && p.location) { stop(); select(p.location === state.code ? null : p.location); }
});
document.getElementById("rank").on("plotly_click", ev => {
  const p = ev.points && ev.points[0];
  if (p && p.customdata) { stop(); select(p.customdata); }
});
document.getElementById("strip").on("plotly_click", ev => {
  const p = ev.points && ev.points[0];
  if (p && p.customdata) { stop(); select(p.customdata); }
});
</script>
</body>
</html>
"""

years = list(range(1950, 2024))
wide = (countries_modern.pivot_table(index=["entity", "code"], columns="year", values="life_exp")
                        .reindex(columns=years))
world_series = regions[regions["entity"] == "World"].set_index("year")["life_exp"].reindex(years)


def clean(values):
    return [None if pd.isna(x) else round(float(x), 1) for x in values]


payload = {
    "years": years,
    "countries": [{"n": name, "c": code, "v": clean(row.values)} for (name, code), row in wide.iterrows()],
    "world": clean(world_series.values),
    "p10": clean(wide.quantile(0.10).values),   # 10th percentile of countries, each year
    "p90": clean(wide.quantile(0.90).values),   # 90th percentile of countries, each year
}

# Country shapes for the map. Embedding them makes the file work on its own.
# If the download fails, nothing breaks: the page fetches the shapes itself when it opens.
topo = "null"
try:
    url = "https://cdn.jsdelivr.net/npm/sane-topojson@4.0.0/dist/world_110m.json"
    with urllib.request.urlopen(url, timeout=15) as resp:
        topo = resp.read().decode("utf-8")
except Exception as err:
    print("Map shapes not embedded (the page will load them itself):", err)

explorer = (EXPLORER_HTML.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
                         .replace("__TOPO__", topo))
map_path = os.path.join(OUT_DIR, "07_life_expectancy_explorer.html")
with open(map_path, "w", encoding="utf-8") as f:
    f.write(explorer)
print("Saved:", map_path)

import webbrowser
webbrowser.open("file://" + os.path.abspath(map_path))


# %% [markdown]
# ## Key findings
#
# - **The long view.** For centuries life expectancy stayed low, mostly because many children died young. The big rise started in rich countries in the late 1800s and spread worldwide after 1950.
# - **Every region gained.** Africa and Asia gained the most since 1950. Northern America and Europe started high and grew more slowly.
# - **Convergence.** Countries are much closer to each other today than in 1950. The ridgeline shifts right and gets narrower.
# - **Big countries, big gains.** About half of the 15 most populous countries added 30 years or more. Countries that started low gained the most.
# - **COVID-19 was a real shock.** Most countries lost ground between 2019 and 2021. Latin America was hit especially hard. By 2023 most had recovered.
# - **Africa still lags.** Its average today is still below where Europe was in the 1950s.


# %%
print("\nDone. All figures are in:", OUT_DIR)
import platform
if platform.system() == "Darwin":      # on a Mac, open the figures folder in Finder
    os.system(f'open "{OUT_DIR}"')