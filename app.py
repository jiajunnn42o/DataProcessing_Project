# app.py — dataset-agnostic visualisation; all your functions kept intact  # file header comment explaining purpose
import streamlit as st  # import Streamlit for building the web app UI
import numpy as np  # import NumPy for numerical operations
import pandas as pd  # import pandas for data manipulation and analysis
import seaborn as sns  # import seaborn for statistical plots
import matplotlib.pyplot as plt  # import matplotlib's pyplot interface for plotting
import matplotlib as mpl  # import base matplotlib module to configure global styles
from matplotlib.colors import LinearSegmentedColormap  # import tool to build custom color maps
from io import StringIO  # import StringIO to treat strings like file objects
import re  # import regular expression module for pattern matching
import plotly.express as px  # import Plotly Express for interactive charts
import plotly.io as pio  # import Plotly IO to control global Plotly settings
from streamlit_folium import st_folium  # import helper to display Folium maps in Streamlit
import folium  # import Folium for building Leaflet-based maps
from folium.plugins import MarkerCluster  # import MarkerCluster for clustering many map markers
import json  # import json to save and load simple configuration as JSON

# ----------------------------
# 页面设置
# ----------------------------
st.set_page_config(  # set up basic page configuration for the Streamlit app
    page_title="Dataset Explorer",  # set the browser tab title
    layout="wide",  # use wide layout to use more horizontal space
    initial_sidebar_state="expanded",  # show sidebar expanded when app starts
)

# ----------------------------
# 读取 theme（完全以 config.toml 为准）
# ----------------------------
BG   = st.get_option("theme.backgroundColor") or "#FFFFFF"  # read background color from Streamlit theme, fallback to white
TXT  = st.get_option("theme.textColor") or "#262730"  # read main text color, fallback to dark gray
SEC  = st.get_option("theme.secondaryBackgroundColor") or "#F7F7F7"  # read secondary background color, fallback to light gray
PRIMARY = st.get_option("theme.primaryColor") or "#ff41ec"  # read primary accent color, fallback to pink

def _hex_to_rgb(h):  # helper function to convert hex color string to RGB tuple
    h = (h or "#FFFFFF").lstrip("#")  # remove leading "#" and handle None by using white
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))  # convert each pair of characters to integer (R, G, B)

def _luma(hex_color):  # helper function to estimate brightness (luma) of a color
    r, g, b = _hex_to_rgb(hex_color)  # convert hex to RGB values
    return (0.2126*r + 0.7152*g + 0.0722*b) / 255.0  # compute perceptual brightness in range 0–1

GRID = "#E5E7EB" if _luma(BG) >= 0.5 else "#3A3F47"  # choose grid color based on whether background is light or dark

# Matplotlib / Seaborn 同步 theme
mpl.rcdefaults()  # reset matplotlib global configuration to defaults
sns.reset_orig()  # reset seaborn to its original state (no extra themes)
mpl.rcParams.update({  # update matplotlib global style settings
    "figure.facecolor": BG,  # set figure background to match app background
    "axes.facecolor": BG,  # set axes background to match app background
    "savefig.transparent": True,  # save figures with transparent background
    "text.color": TXT,  # set default text color
    "axes.labelcolor": TXT,  # set axes label color
    "axes.edgecolor": GRID,  # set axes border color
    "xtick.color": TXT,  # set x-axis tick color
    "ytick.color": TXT,  # set y-axis tick color
    "axes.titlecolor": TXT,  # set title color
    "grid.color": GRID,  # set grid line color
    "legend.facecolor": "none",  # make legend background transparent
    "legend.edgecolor": GRID,  # set legend border color
})
sns.set_style("whitegrid", {"axes.facecolor": BG, "grid.color": GRID})  # apply seaborn style that respects our background and grid

def apply_theme(ax):  # helper to apply consistent styling to a matplotlib axis
    ax.title.set_color(TXT)  # set title color
    ax.xaxis.label.set_color(TXT)  # set x-axis label color
    ax.yaxis.label.set_color(TXT)  # set y-axis label color
    ax.tick_params(axis="both", colors=TXT)  # set tick colors for both axes
    ax.grid(True, color=GRID, alpha=0.3)  # show grid with chosen color and transparency
    for sp in ax.spines.values():  # loop over all axis borders (spines)
        sp.set_color(GRID)  # set border color
    leg = ax.get_legend()  # get legend object from axis
    if leg:  # if legend exists
        if _luma(BG) >= 0.5:  # if background is light
            leg.get_frame().set_facecolor((1, 1, 1, 0.85))  # use semi-transparent white legend background
        else:  # if background is dark
            leg.get_frame().set_facecolor((0.1, 0.1, 0.1, 0.85))  # use semi-transparent dark legend background
        leg.get_frame().set_edgecolor(GRID)  # set legend border color
        leg.get_frame().set_linewidth(1.0)  # set legend border line width
        if leg.get_title():  # if legend has a title
            leg.get_title().set_color(TXT)  # set legend title color
        for t in leg.get_texts():  # loop through legend labels
            t.set_color(TXT)  # set legend text color

# ---- Brand helpers (GLOBAL) ----
IS_DARK = (_luma(BG) < 0.5)  # boolean flag: True if background is dark theme, False otherwise

def brand_palette(n=6):  # helper to get brand color palette with n colors
    base = [PRIMARY, "#9b8aff", "#6f6f7a", "#d9b3ff", "#b38dff", "#ff99f3"]  # predefined list of brand-like colors
    return base[:max(1, n)]  # return first n colors, but at least 1 color

def new_fig(w=7.2, h=4.2):  # helper to create a new matplotlib figure and axis with default size
    fig, ax = plt.subplots(figsize=(w, h))  # create figure with given width and height
    return fig, ax  # return both figure and axis

sns.set_palette(brand_palette(6))  # set seaborn default color palette to our brand palette (6 colors)

# Plotly theme + color-blind palette
CB_SAFE = [  # define a color-blind friendly color palette
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AB"
]
pio.templates.default = "plotly_dark" if IS_DARK else "plotly_white"  # set Plotly default template based on dark/light mode

# ----------------------------
# 辅助增强：重复值指标 & 安全自动数值转换 & 读CSV缓存
# ----------------------------
def duplicate_metrics(df, subset=None):  # compute basic duplicate row statistics
    mask_first = df.duplicated(subset=subset, keep="first")  # boolean mask: True for duplicated rows except first occurrence
    mask_last  = df.duplicated(subset=subset, keep="last")  # boolean mask: True for duplicated rows except last occurrence
    mask_all   = df.duplicated(subset=subset, keep=False)  # boolean mask: True for all rows that have duplicates
    return {  # return a dictionary of metrics plus some masks
        "rows_marked_duplicate (keep='first')": int(mask_first.sum()),  # number of rows marked as duplicate (keeping first)
        "rows_marked_duplicate (keep='last')" : int(mask_last.sum()),  # number of rows marked as duplicate (keeping last)
        "rows_in_duplicated_groups (all)"     : int(mask_all.sum())  # total rows involved in any duplicate group
    }, mask_first, mask_all  # also return masks so caller can filter rows

@st.cache_data(show_spinner=False)  # cache this function's result in Streamlit to speed up repeated reads
def _read_csv_fast(text):  # fast CSV reader that guesses separator
    return pd.read_csv(StringIO(text), sep=None, engine="python")  # read CSV from string using python engine and auto-separator

def _safe_autocast_numeric(df):  # try to safely convert text columns to numeric if most values look like numbers
    df = df.copy()  # work on a copy to avoid changing original DataFrame directly
    for col in df.columns:  # loop through each column
        if df[col].dtype == "object":  # only consider object (text-like) columns
            s = df[col].astype(str).str.replace(r"[,\s]", "", regex=True)  # remove commas and spaces from values
            frac_numeric = s.str.match(r"^-?\d+(\.\d+)?$").mean()  # compute fraction of values that match numeric pattern
            if frac_numeric > 0.9:  # if more than 90% look like numbers
                df[col] = pd.to_numeric(df[col], errors="coerce")  # convert column to numeric, non-numbers become NaN
    return df  # return updated DataFrame

# ----------------------------
# 智能读取器：CSV/Excel + 世行宽表 reshape
# ----------------------------
def smart_read(uploaded_file):  # main function to smartly read and standardise uploaded dataset
    """
    Robust reader for CSV/Excel with World Bank wide-to-long reshape.
    """  # docstring explaining this function
    name = uploaded_file.name.lower()  # get file name in lowercase to check extension

    def postprocess_df(df: pd.DataFrame) -> pd.DataFrame:  # inner helper to tidy DataFrame after reading
        year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]  # columns that look like year (4 digits)
        if year_cols and ("Country Name" in df.columns or "country" in [c.lower() for c in df.columns]):  # if looks like World Bank format
            id_candidates = ["Country Name", "Country Code", "Indicator Name", "Indicator Code",
                             "Series Name", "Series Code"]  # possible ID columns to keep
            id_vars = [c for c in id_candidates if c in df.columns]  # keep only those that actually exist

            if "Country Name" not in id_vars:  # if "Country Name" is not yet among id variables
                for c in df.columns:  # go through all columns
                    if str(c).lower().strip() in ["country", "country name", "country_name"]:  # if column looks like a country name
                        df = df.rename(columns={c: "Country Name"})  # rename it to "Country Name"
                        id_vars = ["Country Name"] + [x for x in id_vars if x != "Country Name"]  # make sure it's first in id_vars
                        break  # stop after renaming once

            m = df.melt(id_vars=id_vars, value_vars=year_cols,
                        var_name="Year", value_name="CO2_per_capita")  # reshape wide year columns into long format

            if "Indicator Code" in m.columns:  # if there is an indicator code column
                mask = m["Indicator Code"].astype(str).str.contains("EN.GHG.CO2.PC", case=False, na=False)  # keep only CO2 per capita indicator
                if mask.any():  # if at least one row matches
                    m = m[mask]  # filter to those rows

            if "Country Name" not in m.columns:  # ensure we still have Country Name after melt
                raise RuntimeError("Cannot find 'Country Name' after reshape.")  # raise error if Country Name is missing

            m["Year"] = pd.to_numeric(m["Year"], errors="coerce")  # convert Year column to numeric
            m["CO2_per_capita"] = pd.to_numeric(m["CO2_per_capita"], errors="coerce")  # convert value column to numeric
            m = m.dropna(subset=["Year", "CO2_per_capita"])  # drop rows where year or value is missing
            m["Year"] = m["Year"].astype(int)  # store Year as integer
            return m[["Country Name", "Year", "CO2_per_capita"]]  # return only key columns

        # long form unify
        mapping = {}  # prepare dictionary to store column renaming rules
        for c in df.columns:  # inspect each column name
            lc = str(c).lower().strip()  # lowercase and strip spaces
            if lc in ["country", "country name", "country_name"]:  # possible country name column
                mapping[c] = "Country Name"  # map to "Country Name"
            elif lc == "year":  # if exactly "year"
                mapping[c] = "Year"  # rename to "Year"
            elif lc in ["co2_per_capita", "co2 per capita", "co2_pc", "value"]:  # various possible CO2 column names
                mapping[c] = "CO2_per_capita"  # standardise name to "CO2_per_capita"
        if mapping:  # if there are any columns to rename
            df = df.rename(columns=mapping)  # rename them

        if {"Country Name", "Year", "CO2_per_capita"}.issubset(df.columns):  # if our three key columns are present
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")  # convert Year to numeric
            df["CO2_per_capita"] = pd.to_numeric(df["CO2_per_capita"], errors="coerce")  # convert CO2 values to numeric
            df = df.dropna(subset=["Year", "CO2_per_capita"])  # remove rows with missing Year or CO2 value
            df["Year"] = df["Year"].astype(int)  # ensure Year is integer type
        return df  # return post-processed DataFrame (may be generic, not only CO2)

    if name.endswith((".xlsx", ".xls")):  # if file is an Excel file
        df = pd.read_excel(uploaded_file)  # read Excel into pandas DataFrame
        return postprocess_df(df)  # post-process and return

    raw = uploaded_file.read().decode("utf-8", errors="ignore")  # read raw bytes and decode as UTF-8 text
    header_candidates = ["country", "country name", "year", "co2", "value", "indicator code"]  # keywords to detect header row
    lines = raw.splitlines()  # split file into list of lines
    header_row = 0  # default header row index
    for i, ln in enumerate(lines[:50]):  # scan up to first 50 lines
        lower = ln.lower()  # lowercase current line
        if any(k in lower for k in header_candidates):  # if line contains any header keyword
            header_row = i  # treat this line as header row index
            break  # stop scanning
    trimmed = "\n".join(lines[header_row:])  # cut off lines before header row

    try:  # attempt fast automatic CSV parsing
        df = _read_csv_fast(trimmed)  # use our cached auto-separator reader
        return postprocess_df(df)  # post-process and return
    except Exception:  # if fast reader fails
        pass  # ignore error and try other separators
    for sep in [",", ";", "\t", "|"]:  # try common separators: comma, semicolon, tab, pipe
        try:
            df = pd.read_csv(StringIO(trimmed), sep=sep)  # read with chosen separator
            return postprocess_df(df)  # post-process and return if success
        except Exception:
            continue  # if failed, try next separator
    df = pd.read_csv(StringIO(trimmed), sep=None, engine="python", on_bad_lines="skip")  # last fallback: autodetect separator, skip bad lines
    return postprocess_df(df)  # post-process and return DataFrame

# ----------------------------
# 会话状态
# ----------------------------
if "df_raw" not in st.session_state:    st.session_state.df_raw = None  # store original uploaded DataFrame in session state
if "df_clean" not in st.session_state:  st.session_state.df_clean = None  # store cleaned DataFrame in session state
if "pipeline" not in st.session_state:  st.session_state.pipeline = []  # store list of cleaning steps performed
if "generic_view" not in st.session_state: st.session_state.generic_view = None  # store mapped generic view from Column Mapper
if "last_upload_name" not in st.session_state: st.session_state.last_upload_name = None  # remember last uploaded file name
if "roles" not in st.session_state:     st.session_state.roles = {"time": None, "category": None, "value": None}  # store semantic roles of columns
if "auto_done" not in st.session_state: st.session_state.auto_done = False  # flag to indicate whether auto-clean has been run

# ----------------------------
# 默认数据（你的 CO₂ CSV） — 保留，但仅在无数据时使用
# ----------------------------
DEFAULT_PATH = "data/co2_clean_asean.csv"  # relative path to default CO2 dataset
def load_default():  # function to load the default sample dataset
    df = pd.read_csv(DEFAULT_PATH)  # read CSV into DataFrame
    df["Year"] = df["Year"].astype(int)  # ensure Year column is integer
    df["CO2_per_capita"] = pd.to_numeric(df["CO2_per_capita"], errors="coerce")  # convert CO2 column to numeric
    df = df.dropna(subset=["CO2_per_capita"])  # drop rows with missing CO2 values
    return df  # return cleaned default dataset

# ----------------------------
# CO₂ 专用可视化工具（原样保留）
# ----------------------------
ASEAN = [  # list of ASEAN country names used in custom CO2 charts
    "Malaysia","Singapore","Thailand","Indonesia","Vietnam",
    "Philippines","Cambodia","Lao PDR","Myanmar","Brunei Darussalam"
]

def compute_combined(df_base: pd.DataFrame):  # build a combined dataset with ASEAN average added
    asean_mean = (
        df_base[df_base["Country Name"].isin(ASEAN)]  # filter to only ASEAN countries
        .groupby("Year")["CO2_per_capita"]  # group by Year and select CO2_per_capita
        .mean()  # compute mean CO2 per year for ASEAN
        .reset_index()  # reset index to get a normal DataFrame
    )
    asean_mean["Country Name"] = "ASEAN Average"  # label this aggregated series as "ASEAN Average"
    combined = pd.concat([df_base, asean_mean], ignore_index=True)  # append ASEAN average rows to original data
    return combined  # return combined DataFrame

def line_chart(combined, year_range):  # draw line chart for Malaysia vs ASEAN Average vs World
    plot_df = combined[
        combined["Country Name"].isin(["Malaysia", "ASEAN Average", "World"])  # keep 3 series: Malaysia, ASEAN Average, World
        & (combined["Year"].between(*year_range))  # keep years within selected range
    ]
    palette = {"Malaysia": PRIMARY, "ASEAN Average": "#9b8aff", "World": "#6fc3ff"}  # set custom colors for each series
    fig, ax = plt.subplots(figsize=(9.5, 5.2))  # create figure and axis with fixed size
    sns.lineplot(
        data=plot_df, x="Year", y="CO2_per_capita",
        hue="Country Name", palette=palette, marker="o", linewidth=2.5, ax=ax
    )  # draw line plot with markers
    ax.set_ylim(0, 9)  # fix y-axis range from 0 to 9 metric tons per capita
    ax.set_xlabel("Year"); ax.set_ylabel("CO$_2$ (metric tons per capita)")  # label axes
    ax.set_title("Malaysia vs ASEAN Average vs World")  # set chart title
    apply_theme(ax); plt.tight_layout(pad=1.5)  # apply theme styling and adjust layout
    st.pyplot(fig)  # render the matplotlib figure in Streamlit
    plt.close(fig)  # close figure to free memory

def bar_chart(df_base, year_range):  # draw bar chart of latest year CO2 comparison
    latest_year = min(max(df_base["Year"]), year_range[1])  # choose latest year not beyond selected range
    latest = df_base[
        (df_base["Year"] == latest_year) & (df_base["Country Name"].isin(ASEAN + ["World"]))
    ].sort_values("CO2_per_capita", ascending=False)  # filter for latest year and ASEAN+World, sort by CO2 descending

    fig, ax = plt.subplots(figsize=(8.8, 5))  # create figure and axis
    cmap = LinearSegmentedColormap.from_list(
        "rank_grad", [PRIMARY, "#8B8FA7", "#2B2F36"], N=len(latest)
    )  # create gradient colormap from primary to dark colors with N steps
    denom = max(1, len(latest) - 1)  # denominator to space colors evenly, avoid division by zero
    bar_palette = [cmap(i/denom) for i in range(len(latest))]  # compute color for each bar position

    sns.barplot(
        data=latest, x="CO2_per_capita", y="Country Name",
        palette=bar_palette, ax=ax
    )  # draw horizontal bar chart of CO2 by country
    ax.set_xlabel("CO$_2$ (metric tons per capita)"); ax.set_ylabel("Country")  # label axes
    ax.set_title(f"Latest Year Comparison – {latest_year}")  # title includes year
    for i, v in enumerate(latest["CO2_per_capita"]):  # annotate each bar with value
        ax.text(v + 0.1, i, f"{v:.1f}", va="center", color=TXT)  # place text next to bar
    apply_theme(ax); plt.tight_layout(pad=1.5)  # apply styling and adjust layout
    st.pyplot(fig)  # display figure in Streamlit
    plt.close(fig)  # close figure to free memory

def heatmap_chart(df_base, year_range):  # draw heatmap showing CO2 over time for ASEAN countries and World
    hm = df_base[df_base["Country Name"].isin(ASEAN + ["World"])]  # filter data to ASEAN and World
    hm = hm[hm["Year"].between(*year_range)]  # keep only selected year range
    pivot = hm.pivot_table(values="CO2_per_capita",
                           index="Country Name", columns="Year", aggfunc="mean")  # create pivot: countries vs years

    latest_year = min(max(df_base["Year"]), year_range[1])  # compute latest year to use for ordering
    if latest_year in pivot.columns:  # if latest year exists in pivot
        order = pivot[latest_year].sort_values(ascending=False).index  # order countries by latest year's CO2
        pivot = pivot.loc[order]  # reorder rows in pivot by this order

    vals = pivot.values.astype(float)  # extract numeric values from pivot as float array
    vmin, vmax = np.nanpercentile(vals, [5, 95])  # determine color scale range based on 5th and 95th percentiles

    if _luma(BG) < 0.5:  # choose colormap depending on dark/light background
        cmap = LinearSegmentedColormap.from_list("heat_dark", [SEC, "#3b4252", PRIMARY], N=256)  # dark-mode colormap
    else:
        cmap = LinearSegmentedColormap.from_list("heat_light", ["#fff1fa", PRIMARY, "#4a154b"], N=256)  # light-mode colormap

    fig, ax = plt.subplots(figsize=(12, 6))  # create figure and axis
    h = sns.heatmap(
        pivot, cmap=cmap, vmin=vmin, vmax=vmax,
        cbar_kws={"label": "CO$_2$ per capita"},
        ax=ax, linewidths=0.4, linecolor=GRID, square=False
    )  # draw heatmap with color bar label
    cbar = h.collections[0].colorbar  # get color bar object from heatmap
    cbar.ax.yaxis.label.set_color(TXT); cbar.ax.tick_params(colors=TXT)  # set color bar label and tick colors
    if cbar.outline: cbar.outline.set_edgecolor(GRID)  # set color bar border color if exists

    years = list(pivot.columns)  # list of years (columns) in pivot
    step = 2 if len(years) > 15 else 1  # if many years, show every 2nd label to avoid crowding
    ax.set_xticks(range(0, len(years), step))  # set x-axis ticks at chosen step
    ax.set_xticklabels(years[::step], rotation=0)  # label ticks with subset of years, no rotation
    ax.set_xlabel("Year"); ax.set_ylabel("Country")  # label axes
    ax.set_title("Heatmap of CO$_2$ per Capita (ASEAN & World)")  # set heatmap title

    for target, color, lw in [("Malaysia", PRIMARY, 2.0), ("World", "#1f77b4", 1.8)]:  # highlight Malaysia and World rows
        if target in pivot.index:  # if target country exists in pivot
            i = list(pivot.index).index(target)  # find row index for target
            ax.add_patch(plt.Rectangle((0, i), len(pivot.columns), 1, fill=False, lw=lw, edgecolor=color))  # draw rectangle around that row

    apply_theme(ax)  # apply styling to axis
    st.pyplot(fig)  # display heatmap
    plt.close(fig)  # close figure to free memory

# ----------------------------
# 顶部标题（动态：根据上传文件名）
# ----------------------------
page_name = st.session_state.last_upload_name or "Dataset Explorer"  # choose page title: last file name or default
st.title(f"{page_name}")  # display main title at top of page
st.caption("Upload any CSV/Excel. The app adapts to your schema. Cleaning → Visualise → Report.")  # short subtitle explaining app purpose

# ----------------------------
# 页签
# ----------------------------
# ============ 两栏布局：左 = 上传 & 清洗；右 = 可视化 & 报告 ============
col_left, col_right = st.columns([1, 1.2], gap="large")  # create two main columns: left (smaller), right (slightly wider)

# 左边：Upload & Inspect + Clean & Transform
with col_left:  # start content in left column
    tab_upload, tab_clean = st.tabs(["Upload & Inspect", "Clean & Transform"])  # create two tabs within left column
    # ----------------------------
    # 1) Upload & Inspect
    # ----------------------------
    with tab_upload:  # content for "Upload & Inspect" tab
        st.subheader("Upload a dataset (CSV/Excel) or use the sample")  # subheading for upload section
        up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])  # file uploader widget

        colA, colB = st.columns([1,1])  # create two equal columns for buttons/info
        with colA:  # first column
            if st.button("Use sample dataset (CO₂)"):  # button to load default sample dataset
                st.session_state.df_raw = load_default()  # load default dataset into raw DataFrame
                st.session_state.df_clean = None  # clear any previous cleaned data
                st.session_state.pipeline = [{"step":"load_default", "args": {}}]  # reset pipeline with a single step entry
                st.session_state.last_upload_name = "__SAMPLE__ co2_clean_asean.csv"  # set pseudo file name for sample
                st.success("Loaded sample dataset.")  # show success message
        if up is not None and st.session_state.last_upload_name != up.name:  # if user uploaded new file with different name
            try:
                df_raw = smart_read(up)  # read and post-process uploaded file
                st.session_state.df_raw = df_raw  # store as raw DataFrame
                st.session_state.df_clean = None  # clear any cleaned version
                st.session_state.pipeline = [{"step":"upload", "args":{"filename": up.name}}]  # reset pipeline with upload step
                st.session_state.last_upload_name = up.name  # remember this file name
                if len(df_raw) > 500_000:  # if dataset has more than 500k rows
                    st.warning(f"Large dataset detected: {len(df_raw):,} rows. Some operations may be slow; visualisations may use only subsets.")  # warn about performance
                st.success(f"Uploaded: {up.name}  (rows={len(df_raw)}, cols={len(df_raw.columns)})")  # show upload details
            except Exception as e:  # if anything goes wrong when reading file
                st.error(f"Failed to read file: {e}")  # show error message

        df_show = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw  # choose which DataFrame to display (cleaned or raw)

        # --- Status ribbon (enhanced)
        if df_show is not None:  # if we have any dataset to work with
            st.info(
                f"Active dataset: {'cleaned' if st.session_state.df_clean is not None else 'raw'} | "
                f"Rows: {len(df_show):,} | Columns: {len(df_show.columns):,} | "
                f"Pipeline steps: {len(st.session_state.pipeline)}"
            )  # show a compact summary of current dataset and pipeline length

        if df_show is not None:  # if there is a dataset
            st.markdown("**Data preview (first 10 rows)**")  # section title for preview
            st.dataframe(df_show.head(10), use_container_width=True)  # show first 10 rows in interactive table

            st.markdown("**Missing values by column**")  # section title for missing value summary
            st.write(df_show.isna().sum())  # display count of missing values per column

            # Quick data profile
            with st.expander("Quick data profile"):  # collapsible section for quick profiling
                num = df_show.select_dtypes(include="number")  # select numeric columns
                cat = df_show.select_dtypes(exclude="number")  # select non-numeric (categorical/text) columns
                if not num.empty:  # if there are numeric columns
                    st.markdown("**Numeric summary**")  # title for numeric summary
                    st.dataframe(num.describe().T)  # show descriptive stats (transposed to have columns as rows)
                if not cat.empty:  # if there are categorical columns
                    st.markdown("**Categorical unique counts**")  # title for categorical summary
                    st.dataframe(cat.nunique().rename("n_unique"))  # show count of unique values per categorical column

            # ---------- ENHANCED DUPLICATES SECTION ----------
            st.markdown("### Duplicate Analysis")  # section heading for duplicate analysis
            metrics, dup_first_mask, dup_all_mask = duplicate_metrics(df_show, subset=None)  # compute duplicate metrics for entire dataset

            st.markdown(
                f"""
                <div style="
                    background-color:{SEC};
                    border: 1px solid {GRID};
                    border-radius: 12px;
                    padding: 18px 24px;
                    margin: 6px 0 12px 0;
                    box-shadow: 0px 2px 6px rgba(0,0,0,0.15);
                ">
                    <h4 style="margin-bottom:8px; color:{TXT};">Duplicate Rows Summary</h4>
                    <ul style="margin-top:8px;color:{TXT}">
                    <li>Rows flagged (keep='first'): <b style="color:{PRIMARY}">{metrics["rows_marked_duplicate (keep='first')"]:,}</b></li>
                    <li>Rows flagged (keep='last') : <b style="color:{PRIMARY}">{metrics["rows_marked_duplicate (keep='last')"]:,}</b></li>
                    <li>Rows in duplicated groups : <b style="color:{PRIMARY}">{metrics["rows_in_duplicated_groups (all)"]:,}</b></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )  # display a styled HTML box summarising duplicate statistics

            with st.expander("Find & filter duplicates by columns"):  # collapsible section for detailed duplicate filtering
                cols_with_dups = [
                    c for c in df_show.columns
                    if pd.Series(df_show[c]).duplicated(keep=False).any()
                ]  # list of columns that have at least one duplicate value
                cols_pick = st.multiselect(
                    "Columns to consider for duplicates (leave empty = all columns)",
                    options=df_show.columns.tolist(),
                    default=cols_with_dups,
                )  # let user select which columns define duplicates

                keep_mode = st.selectbox(
                    "Mark which occurrence as duplicate",
                    ["first", "last", "none (mark all)"],
                    help="Matches pandas.duplicated(keep=...) behaviour.",
                )  # choose pandas duplicated() keep behaviour
                keep_arg = {"first": "first", "last": "last", "none (mark all)": False}[keep_mode]  # map human option to real argument

                if st.button("Analyse duplicates", key="btn_analyse_dups"):  # button to perform duplicate analysis
                    subset_arg = cols_pick if len(cols_pick) > 0 else None  # subset of columns if user selected any
                    dup_mask = df_show.duplicated(subset=subset_arg, keep=keep_arg)  # compute duplicate mask with chosen options

                    st.info(f"Rows flagged as duplicates: **{int(dup_mask.sum()):,}**")  # show number of flagged duplicates
                    st.dataframe(df_show[dup_mask].head(200), use_container_width=True)  # preview up to 200 duplicate rows

                    if subset_arg:  # if duplicates were based on specific columns
                        cols = subset_arg if isinstance(subset_arg, list) else [subset_arg]  # ensure cols is a list
                        grp = (
                            df_show.loc[dup_mask, cols]
                            .groupby(cols, dropna=False)
                            .size()
                            .reset_index(name="count")
                            .sort_values("count", ascending=False, kind="mergesort")
                        )  # group duplicate rows by selected columns and count occurrences
                        st.markdown("**Duplicate groups (top 50)**")  # section title for duplicate groups
                        st.dataframe(grp.head(50), use_container_width=True)  # show top 50 duplicate groups

                    csv_dups = df_show[dup_mask].to_csv(index=False).encode("utf-8")  # convert duplicate rows to CSV bytes
                    st.download_button(
                        "Download duplicate rows (CSV)",
                        csv_dups,
                        file_name="duplicates_filtered.csv",
                        mime="text/csv",
                        key="dl_dups",
                    )  # provide button to download duplicate rows
            # ---------- END ENHANCED DUPLICATES SECTION ----------

            # ====== Column Mapper（当没有特定列时使用）======
            required = {"Year", "Country Name", "CO2_per_capita"}  # set of columns needed for original CO2 views
            if not required.issubset(df_show.columns):  # if dataset does not naturally have these columns
                st.markdown("### Column Mapper（Universal Data Adaptation）")  # heading for Column Mapper section
                st.info("Map your own columns to enable generic visualisation with time/category/value semantics.")  # explain purpose

                cols = df_show.columns.tolist()  # list of all column names

                time_candidates = [c for c in cols if np.issubdtype(df_show[c].dtype, np.datetime64)]  # columns already as datetime
                if not time_candidates:  # if no datetime columns found
                    for c in cols:  # try to detect time-like columns in object columns
                        if df_show[c].dtype == "object":  # only text columns
                            try:
                                pd.to_datetime(df_show[c], errors="raise", infer_datetime_format=True)  # try converting to datetime
                                time_candidates.append(c)  # if success, add to time candidates
                            except Exception:
                                pass  # ignore if fails

                col_time = st.selectbox("Time/Year（Optional）", ["(none)"] + time_candidates + [c for c in cols if c not in time_candidates])  # choose time column or none
                col_cat  = st.selectbox("Category/Group（Optional）", ["(none)"] + cols)  # choose category column or none

                numeric_like = []  # list to collect columns that can be numeric
                for c in cols:  # check each column
                    if pd.api.types.is_numeric_dtype(df_show[c]):  # if already numeric
                        numeric_like.append(c)  # add directly
                    else:
                        try:
                            pd.to_numeric(df_show[c], errors="raise")  # try converting text to numeric
                            numeric_like.append(c)  # if success, treat as numeric-like
                        except Exception:
                            pass  # ignore columns that cannot be numeric
                col_val  = st.selectbox("Value（Numeric column(s), required）", numeric_like if numeric_like else ["(none)"])  # choose numeric value column

                if st.button("Create mapped view"):  # button to build generic mapped view
                    mapped = df_show.copy()  # copy DataFrame
                    rename = {}  # dict for renaming chosen columns to standard names
                    if col_time != "(none)": rename[col_time] = "Year"  # rename chosen time column to "Year"
                    if col_cat  != "(none)": rename[col_cat]  = "Category"  # rename chosen category column to "Category"
                    if col_val  != "(none)": rename[col_val]  = "Value"  # rename chosen numeric column to "Value"
                    mapped = mapped.rename(columns=rename)  # apply renaming

                    if "Year" in mapped.columns:  # if Year column exists in mapped view
                        try:
                            mapped["Year"] = pd.to_datetime(mapped["Year"], errors="coerce", infer_datetime_format=True)  # convert Year to datetime if possible
                        except Exception:
                            pass  # ignore if conversion fails
                    if "Value" in mapped.columns:  # if Value column exists
                        mapped["Value"] = pd.to_numeric(mapped["Value"], errors="coerce")  # ensure Value is numeric

                    st.session_state.generic_view = mapped  # store mapped DataFrame in session
                    st.session_state.roles = {  # store roles information based on what user selected
                        "time": None if col_time=="(none)" else "Year",
                        "category": None if col_cat=="(none)" else "Category",
                        "value": None if col_val=="(none)" else "Value",
                    }
                    st.success("A universal mapping view has been generated. Use it in the Visualise tab → Generic Visualise.")  # notify user mapping is ready

            # Export / import roles mapping
            with st.expander("Save or load column-role mapping (roles.json)"):  # collapsible panel for saving/loading roles
                roles_json = json.dumps(st.session_state.roles)  # serialise roles dictionary to JSON string
                st.download_button(
                    "Download roles.json",
                    roles_json.encode("utf-8"),
                    file_name="roles.json",
                    mime="application/json",
                    key="dl_roles",
                )  # allow user to download roles mapping as JSON file
                roles_file = st.file_uploader("Import roles.json", type=["json"], key="roles_in")  # uploader to import previously saved roles
                if roles_file is not None:  # if user uploaded a roles JSON
                    try:
                        st.session_state.roles = json.loads(roles_file.read().decode("utf-8"))  # read and parse JSON into roles
                        st.success("Roles mapping imported. Visualisation defaults will follow these roles.")  # confirm import success
                    except Exception as e:  # if something goes wrong
                        st.error(f"Failed to import roles: {e}")  # show error message

    # ----------------------------
    # 2) Clean & Transform (unchanged logic, but safer auto-cast)
    # ----------------------------
    with tab_clean:  # content for "Clean & Transform" tab
        st.subheader("Step-by-step cleaning with preview and pipeline log")  # heading explaining this section

        clean_mode = st.radio(
            "Cleaning Mode",
            ["Manual", "Auto"],
            captions=["Step-by-step manual cleaning", "Auto-run with default cleaning pipeline"],
            horizontal=True,
            key="clean_mode_radio",
        )  # radio buttons to choose between manual and auto cleaning modes

        base = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw  # choose starting DataFrame

        if clean_mode == "Auto":  # if user selected auto cleaning mode
            if st.session_state.df_raw is None and base is None:  # if there is no data
                st.info("No data yet. Go to **Upload & Inspect** to load a dataset.")  # instruct user to upload data
            elif not st.session_state.auto_done:  # if auto cleaning has not been run yet
                src = st.session_state.df_raw if st.session_state.df_raw is not None else base  # choose source DataFrame
                if src is None:  # double-check if still None
                    st.warning("No data to clean. Please upload or load the default dataset in 'Upload & Inspect'.")  # warn user
                else:
                    df = src.copy()  # make a copy to work on
                    df = df.fillna(df.mean(numeric_only=True))  # fill missing numeric values with their column means
                    df = df.drop_duplicates()  # remove duplicate rows
                    df = _safe_autocast_numeric(df)  # safely convert numeric-like text columns to actual numbers

                    st.session_state.df_clean = df  # store cleaned DataFrame
                    st.session_state.pipeline.append({
                        "step": "auto_clean",
                        "args": {"methods": ["fill mean", "drop duplicates", "convert numeric (safe)"]}
                    })  # log this automatic cleaning step in pipeline
                    st.session_state.auto_done = True  # mark that auto cleaning has been done
                    st.success("Auto cleaning completed.")  # show success message
            else:
                st.info("Auto cleaning already applied. Switch to Manual for step-by-step actions.")  # inform that auto-clean is already done
        else:
            st.session_state.auto_done = False  # reset flag when user switches to manual mode

        base = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw  # re-evaluate base DataFrame
        if base is None:  # if still no data
            st.info("No data yet. Go to **Upload & Inspect** to load a dataset.")  # remind user to upload data
        else:
            df_work = base.copy()  # copy DataFrame for applying transformations
            st.markdown("### Actions")  # section heading for cleaning actions

            act = st.selectbox(
                "Choose an action",
                ["(select)", "Filter rows (keep/remove)", "Handle missing values",
                "Remove duplicates", "Cast to numeric", "Min-Max scale",
                "Clip outliers (IQR)", "Standardize text categories"]
            )  # drop-down to select which cleaning action to perform

            if act == "Filter rows (keep/remove)":  # user chooses to filter rows by value
                try:
                    default_idx = next(
                        i for i, c in enumerate(df_work.columns)
                        if str(c).lower().strip() in ["country", "country name", "country_name", "countryname", "country_code"]
                    )  # try to find a default column that looks like country
                except StopIteration:
                    default_idx = 0  # if no country-like column, default to first column

                col = st.selectbox("Column to filter", df_work.columns.tolist(), index=default_idx)  # choose column to filter on

                uniq_vals = pd.Series(df_work[col].astype(str).unique())  # get unique values in that column as strings
                uniq_vals = uniq_vals.sort_values(kind="mergesort").tolist()  # sort them in stable order and convert to list

                use_asean_preset = st.checkbox("Quick preset: Select ASEAN countries")  # option to quickly select ASEAN countries
                preselected = []  # list to store preselected values
                if use_asean_preset:  # if preset is turned on
                    target = set(ASEAN)  # convert ASEAN list to set for fast membership checks
                    preselected = [v for v in uniq_vals if v in target]  # choose only values that match ASEAN names
                    if not preselected:  # if no matches found
                        st.info("No ASEAN names matched in this column. You can still select manually.")  # inform user

                picked = st.multiselect("Pick values", options=uniq_vals, default=preselected)  # allow user to pick values to keep or remove

                pasted = st.text_area("Or paste values (comma/semicolon/newline separated)",
                                    placeholder="Malaysia, Singapore, Thailand\nVietnam")  # another way: paste values in text box
                if pasted.strip():  # if user entered any values
                    extra = [s.strip() for s in re.split(r"[,;\n]", pasted) if s.strip()]  # split by comma, semicolon, or newline
                    picked = sorted(set(picked) | set(extra))  # merge pasted values with existing selection, remove duplicates

                mode = st.radio("Mode", ["Keep only selected", "Remove selected"], horizontal=True)  # choose whether to keep or remove selected values

                if st.button("Apply", key="apply_filter_values"):  # button to apply filter
                    if not picked:  # if no values chosen
                        st.warning("Please select at least one value to proceed.")  # ask user to select values
                    else:
                        before = len(df_work)  # remember row count before filtering
                        mask = df_work[col].astype(str).isin(set(picked))  # build mask where column value is in picked set
                        if mode == "Keep only selected":  # if user wants to keep those values
                            df_work = df_work[mask];  mode_key = "keep"  # keep only matching rows
                        else:
                            df_work = df_work[~mask]; mode_key = "remove"  # drop rows with those values

                        st.session_state.df_clean = df_work  # store new cleaned DataFrame
                        st.session_state.pipeline.append({
                            "step": "filter_values",
                            "args": {
                                "column": col,
                                "mode": mode_key,
                                "values": (picked[:20] + (["..."] if len(picked) > 20 else [])),  # store first 20 values in pipeline log
                                "rows_before": before,
                                "rows_after": len(df_work)
                            }
                        })  # log filter step
                        st.success(f"Filtered ({mode}). Rows: {before} → {len(df_work)}")  # show row count change

            elif act == "Handle missing values":  # user selects missing value handling
                cols = st.multiselect("Columns to impute", df_work.columns.tolist())  # choose columns to treat
                how = st.radio(
                    "Method",
                    ["drop rows", "fill mean", "fill median", "fill mode", "forward fill", "backward fill"],
                    horizontal=True
                )  # choose imputation strategy
                if st.button("Apply", key="apply_impute"):  # button to apply imputation
                    before = len(df_work)  # remember number of rows before operation
                    if how == "drop rows":
                        df_work = df_work.dropna(subset=cols) if cols else df_work.dropna()  # drop rows with any missing in selected cols or all
                    elif how == "fill mean":
                        for c in cols: df_work[c] = df_work[c].fillna(df_work[c].mean())  # fill NaN with column mean
                    elif how == "fill median":
                        for c in cols: df_work[c] = df_work[c].fillna(df_work[c].median())  # fill NaN with column median
                    elif how == "fill mode":
                        for c in cols:
                            mode_val = df_work[c].mode()  # compute most frequent value
                            if not mode_val.empty: df_work[c] = df_work[c].fillna(mode_val[0])  # use first mode value to fill NaN
                    elif how == "forward fill":
                        df_work[cols] = df_work[cols].ffill()  # propagate previous non-null value forward
                    elif how == "backward fill":
                        df_work[cols] = df_work[cols].bfill()  # propagate next non-null value backward

                    st.session_state.df_clean = df_work  # update cleaned DataFrame in session
                    st.session_state.pipeline.append({"step": "impute", "args": {"cols": cols, "method": how}})  # log imputation step
                    st.success(f"Applied: {how}. Rows before/after: {before} → {len(df_work)}")  # report result

            elif act == "Remove duplicates":  # user chooses to remove duplicate rows
                keep = st.selectbox("Keep", ["first", "last", "False (drop all dup)"])  # choose how to treat duplicates
                keep_arg = {"first": "first", "last": "last", "False (drop all dup)": False}[keep]  # map human option to pandas argument
                if st.button("Apply", key="apply_dropdup"):  # button to apply duplicate removal
                    before = len(df_work)  # number of rows before dropping duplicates
                    df_work = df_work.drop_duplicates(keep=keep_arg)  # remove duplicates with chosen keep option
                    st.session_state.df_clean = df_work  # update cleaned DataFrame
                    st.session_state.pipeline.append({"step": "drop_duplicates", "args": {"keep": keep_arg}})  # log step
                    st.success(f"Duplicates removed. Rows: {before} → {len(df_work)}")  # show how many rows remain

            elif act == "Cast to numeric":  # user wants to convert columns to numeric type
                cols = st.multiselect("Columns to cast (errors→NaN)", df_work.columns.tolist())  # select columns to convert
                if st.button("Apply", key="apply_cast"):  # button to perform conversion
                    for c in cols:
                        df_work[c] = pd.to_numeric(df_work[c], errors="coerce")  # convert each selected column to numeric, invalid to NaN
                    st.session_state.df_clean = df_work  # save updated DataFrame
                    st.session_state.pipeline.append({"step": "to_numeric", "args": {"cols": cols}})  # record cast operation
                    st.success("Cast done.")  # show confirmation

            elif act == "Min-Max scale":  # user selects scaling columns to 0-1 range
                cols = st.multiselect("Columns to scale (0-1)", df_work.select_dtypes(include=np.number).columns.tolist())  # choose numeric columns to scale
                if st.button("Apply", key="apply_scale"):  # button to perform scaling
                    for c in cols:
                        lo, hi = df_work[c].min(), df_work[c].max()  # find min and max of column
                        if pd.notna(lo) and pd.notna(hi) and hi != lo:  # only scale if valid and non-constant
                            df_work[c] = (df_work[c] - lo) / (hi - lo)  # apply min-max scaling formula
                    st.session_state.df_clean = df_work  # update cleaned DataFrame
                    st.session_state.pipeline.append({"step": "minmax", "args": {"cols": cols}})  # record scaling step
                    st.success("Scaled.")  # show success message

            elif act == "Clip outliers (IQR)":  # user chooses to clip outliers using IQR method
                cols = st.multiselect("Numeric columns to clip", df_work.select_dtypes(include=np.number).columns.tolist())  # select numeric columns
                if st.button("Apply", key="apply_iqr"):  # button to apply IQR clipping
                    before_stats = df_work[cols].describe().to_dict() if cols else {}  # capture summary statistics before clipping
                    for c in cols:
                        q1, q3 = df_work[c].quantile([0.25, 0.75])  # compute first and third quartile
                        iqr = q3 - q1  # compute interquartile range
                        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr  # lower and upper bounds for non-outliers
                        df_work[c] = df_work[c].clip(lo, hi)  # clip values outside [lo, hi]
                    st.session_state.df_clean = df_work  # save updated DataFrame
                    st.session_state.pipeline.append({"step": "clip_iqr", "args": {"cols": cols, "before": before_stats}})  # log clipping with before-stats
                    st.success("Outliers clipped using IQR method.")  # show success message

            elif act == "Standardize text categories":  # user chooses to standardise text formatting
                cols = st.multiselect("Text columns to standardize", df_work.select_dtypes(include="object").columns.tolist())  # choose text columns
                if st.button("Apply", key="apply_std_text"):  # button to apply standardisation
                    for c in cols:
                        df_work[c] = (
                            df_work[c].astype(str)
                            .str.strip()  # remove leading and trailing spaces
                            .str.replace(r"\s+", " ", regex=True)  # collapse multiple spaces into one
                            .str.normalize("NFKC")  # normalise Unicode characters (e.g., full-width / half-width)
                        )
                    st.session_state.df_clean = df_work  # update cleaned DataFrame
                    st.session_state.pipeline.append({"step": "standardize_text", "args": {"cols": cols}})  # log this text standardisation step
                    st.success("Text categories standardized (trim + whitespace normalization).")  # show confirmation

        # 导出
        st.markdown("### Export cleaned data")  # section title for export
        export_df = (
            st.session_state.df_clean
            if (st.session_state.df_clean is not None and not st.session_state.df_clean.empty)
            else st.session_state.df_raw
        )  # choose DataFrame to export: cleaned if available, otherwise raw
        if export_df is not None:  # if there is data
            csv = export_df.to_csv(index=False).encode("utf-8")  # convert DataFrame to CSV bytes without row index
            st.download_button("Download cleaned.csv", csv, file_name="cleaned.csv", mime="text/csv")  # provide download button for cleaned data

        # 流水线
        st.markdown("### Pipeline log")  # section title for pipeline log
        st.json(st.session_state.pipeline)  # show pipeline steps as JSON for transparency

# 右边：Visualise + Report
with col_right:  # start content in right column
    tab_viz, tab_report = st.tabs(["Results", "Report"])  # create two tabs: Results (visualisations) and Report
    # ----------------------------
    # 3) Visualise (dataset-agnostic by default)
    # ----------------------------
    with tab_viz:  # content for Results tab
        st.subheader("Results")  # subheading for visualisation section

        # Pick the active dataframe
        if st.session_state.df_clean is not None and not st.session_state.df_clean.empty:  # if cleaned data exists and is non-empty
            df_base = st.session_state.df_clean  # use cleaned data for plots
        elif st.session_state.df_raw is not None and not st.session_state.df_raw.empty:  # otherwise if raw data exists
            df_base = st.session_state.df_raw  # use raw data
        else:
            # 还没有任何 dataset（没上传 & 没按 sample 按钮）
            df_base = None  # no dataset available

        if df_base is None:  # if still no data
            st.info("No dataset yet. Please upload a CSV/Excel or click 'Use sample dataset (CO₂)' on the left.")  # instruct user to load data
        else:
            # Choose data source for plotting (Column Mapper if provided)
            gv = st.session_state.get("generic_view", None)  # get generic mapped view if exists
            use_generic = st.toggle("Use Column Mapper view if available", value=True if gv is not None else False)  # toggle whether to use mapped view
            plot_df = gv if (use_generic and gv is not None) else df_base.copy()  # choose plotting DataFrame based on toggle

            # Detect column types
            num_cols = plot_df.select_dtypes(include="number").columns.tolist()  # identify numeric columns
            dt_cols  = [c for c in plot_df.columns if np.issubdtype(plot_df[c].dtype, np.datetime64)]  # identify datetime columns
            cat_cols = [c for c in plot_df.columns if c not in num_cols + dt_cols]  # treat remaining columns as categorical

            # ---- INTERACTIVE MODE ----
            st.markdown("### Interactive mode (Plotly / Leaflet)")  # title for interactive visualisation section
            inter_mode = st.radio("Pick interactive engine", ["Auto", "Plotly", "Leaflet (map)"], horizontal=True)  # choose interactive engine

            # Color-blind friendly toggle
            cb = st.toggle("Color-blind friendly palette", value=False)  # toggle for using color-blind-friendly palette
            palette_seq = CB_SAFE if cb else None  # choose palette sequence depending on toggle

            # Helper: detect geo possibilities
            def _has_latlon(cols):  # helper to check if we have latitude/longitude columns
                lc = [str(c).lower() for c in cols]  # lowercase all column names
                return ("lat" in lc or "latitude" in lc) and ("lon" in lc or "long" in lc or "lng" in lc or "longitude" in lc)  # check for lat and lon presence

            def _country_col(cols):  # helper to guess a country column
                for c in cols:  # loop through column names
                    lc = str(c).lower()  # lowercase column name
                    if any(k in lc for k in ["country", "nation", "location", "iso", "region"]):  # if name suggests a location/country
                        return c  # return this column name
                return None  # otherwise return None

            # Recommend chart
            recommended = None  # default: no recommendation
            if _has_latlon(plot_df.columns):  # if we detect latitude and longitude
                recommended = "map_latlon"  # recommend map with lat/lon points
            elif _country_col(plot_df.columns):  # else if there is a country-like column
                recommended = "map_country" if num_cols else None  # recommend choropleth if numeric metric exists
            elif dt_cols and num_cols:  # if we have both datetime and numeric
                recommended = "line"  # recommend line chart
            elif len(num_cols) >= 2:  # if at least two numeric columns
                recommended = "scatter"  # recommend scatter plot
            elif num_cols:  # if exactly one numeric column
                recommended = "hist"  # recommend histogram
            elif cat_cols and num_cols:  # if categorical and numeric exist
                recommended = "bar"  # recommend bar chart

            # auto route to map if appropriate
            if inter_mode in ["Auto", "Plotly"]:  # if engine is Auto or Plotly
                if inter_mode == "Auto" and recommended in ["map_latlon", "map_country"]:  # if recommended visual is a map
                    inter_mode = "Leaflet (map)"  # automatically switch engine to Leaflet map

            role = st.session_state.roles  # read column role preferences from session

            # --- Plotly path (with Line index option added) ---
            if inter_mode in ["Auto", "Plotly"]:  # if we are using Plotly-based charts
                st.write("**Preview of current plotting data:**")  # small caption
                st.dataframe(plot_df.head(5), use_container_width=True)  # show first 5 rows to understand columns

                chart_options = ["Line (time)", "Line (index)", "Bar (agg)", "Scatter", "Histogram", "Correlation Heatmap"]  # available Plotly chart types
                default_idx = 0 if recommended == "line" else (2 if recommended == "bar" else (3 if recommended == "scatter" else (4 if recommended == "hist" else 0)))  # choose default chart index based on recommendation
                chart = st.selectbox("Chart Type (interactive)", chart_options, index=default_idx)  # select chart type

                if chart == "Line (time)":  # time-series line chart
                    x_candidates = dt_cols + cat_cols + num_cols  # possible X-axis columns (time, category, or numeric)
                    if not num_cols or not x_candidates:  # need at least one numeric Y and one X
                        st.warning("You need at least one numeric column for Y and one column for X.")  # show warning
                    else:
                        pref_x = role.get("time") or role.get("category")  # preferred X based on roles (time or category)
                        pref_y = role.get("value")  # preferred Y based on value role
                        x_index = x_candidates.index(pref_x) if pref_x in x_candidates else 0  # default X index based on role if available
                        y_index = num_cols.index(pref_y) if pref_y in num_cols else 0  # default Y index based on role if available

                        x = st.selectbox("X Column", x_candidates, index=x_index)  # choose X column
                        y = st.selectbox("Y (Numeric Column)", num_cols, index=y_index)  # choose Y numeric column
                        color = st.selectbox("Grouping/Color (optional)", ["(none)"] + [c for c in plot_df.columns if c not in [x, y]])  # optional grouping/color column
                        hover = st.multiselect("Extra hover fields", [c for c in plot_df.columns if c not in [x, y]])  # columns to show on hover
                        if plot_df[x].dtype == "object":  # if X is stored as object (string dates)
                            try:
                                plot_df[x] = pd.to_datetime(plot_df[x], errors="coerce", infer_datetime_format=True)  # convert to datetime
                            except Exception:
                                pass  # ignore if fails
                        fig = px.line(
                            plot_df.sort_values(x),
                            x=x, y=y,
                            color=None if color == "(none)" else color,
                            markers=True, hover_data=hover,
                            title=f"{y} over {x}",
                            color_discrete_sequence=palette_seq
                        )  # build line chart with optional color and hover columns
                        fig.update_traces(hovertemplate=f"{x}=%{{x}}<br>{y}=%{{y:.3f}}<extra></extra>")  # custom hover text format
                        st.plotly_chart(fig, use_container_width=True)  # display Plotly figure in Streamlit

                elif chart == "Line (index)":  # line chart using row index on X-axis
                    if not num_cols:  # require at least one numeric column
                        st.warning("At least one numeric column is required.")  # show warning message
                    else:
                        pref_y = role.get("value")  # preferred numeric column based on role
                        y_index = num_cols.index(pref_y) if pref_y in num_cols else 0  # default Y index
                        y = st.selectbox("Numeric column (Y)", num_cols, index=y_index)  # choose Y column
                        df_idx = plot_df.reset_index()  # reset index to get index as a column
                        idx = df_idx.columns[0]  # index column name after reset_index
                        fig = px.line(df_idx, x=idx, y=y, markers=True, title=f"{y} over row index",
                                    color_discrete_sequence=palette_seq)  # create line chart using index as X
                        fig.update_traces(hovertemplate=f"Index=%{{x}}<br>{y}=%{{y:.3f}}<extra></extra>")  # custom hover text
                        st.plotly_chart(fig, use_container_width=True)  # show the chart

                elif chart == "Bar (agg)":  # aggregated bar chart
                    if not num_cols or not (cat_cols or dt_cols):  # need one grouping column and one numeric
                        st.warning("A grouping column (category or time) and a numeric column are required.")  # show warning
                    else:
                        pref_x = role.get("category") or role.get("time")  # prefer Category or Time as grouping
                        pref_y = role.get("value")  # preferred numeric column
                        group_candidates = cat_cols + dt_cols  # possible grouping columns
                        x_index = group_candidates.index(pref_x) if pref_x in group_candidates else 0  # default group column index
                        y_index = num_cols.index(pref_y) if pref_y in num_cols else 0  # default numeric column index

                        x = st.selectbox("Grouping Column (X)", group_candidates, index=x_index)  # choose X (group) column
                        y = st.selectbox("Numeric Column (Y)", num_cols, index=y_index)  # choose numeric column
                        agg = st.selectbox("Aggregation", ["mean","sum","median"])  # choose aggregation method
                        df_agg = getattr(plot_df.groupby(x)[y], agg)().reset_index()  # compute aggregated values by group
                        if x in dt_cols:  # if grouping by datetime column
                            fig = px.line(df_agg.sort_values(x), x=x, y=y, markers=True,
                                        title=f"{agg.title()} of {y} over {x}",
                                        color_discrete_sequence=palette_seq)  # plot aggregated series as line over time
                            fig.update_traces(hovertemplate=f"{x}=%{{x}}<br>{y}=%{{y:.3f}}<extra></extra>")  # custom hover
                        else:
                            fig = px.bar(df_agg, x=y, y=x, orientation="h",
                                        title=f"{agg.title()} of {y} by {x}",
                                        color_discrete_sequence=palette_seq)  # plot aggregated values as horizontal bars
                            fig.update_traces(hovertemplate=f"{x}=%{{y}}<br>{y}=%{{x:.3f}}<extra></extra>")  # custom hover
                        st.plotly_chart(fig, use_container_width=True)  # show chart

                elif chart == "Scatter":  # scatter plot
                    if len(num_cols) < 2:  # need at least two numeric columns for X and Y
                        st.warning("At least two numeric columns are required.")  # show warning
                    else:
                        pref_y = role.get("value")  # preferred Y column
                        y_index = num_cols.index(pref_y) if pref_y in num_cols else 1 if len(num_cols) > 1 else 0  # choose Y index
                        x = st.selectbox("X (numeric)", num_cols, key="px_sc_x")  # choose numeric X column
                        y = st.selectbox("Y (numeric)", num_cols, index=y_index, key="px_sc_y")  # choose numeric Y column
                        color = st.selectbox("Color (optional)", ["(none)"] + cat_cols, key="px_sc_c")  # optional color grouping
                        size  = st.selectbox("Bubble size (optional)", ["(none)"] + num_cols, key="px_sc_s")  # optional size mapping
                        hover = st.multiselect("Extra hover fields", [c for c in plot_df.columns if c not in [x, y]])  # extra fields for hover
                        fig = px.scatter(
                            plot_df, x=x, y=y,
                            color=None if color=="(none)" else color,
                            size=None if size=="(none)" else size,
                            hover_data=hover, trendline=None, title=f"{y} vs {x}",
                            color_discrete_sequence=palette_seq
                        )  # build scatter plot with options
                        fig.update_traces(hovertemplate=f"{x}=%{{x:.3f}}<br>{y}=%{{y:.3f}}<extra></extra>")  # custom hover
                        st.plotly_chart(fig, use_container_width=True)  # show plot

                elif chart == "Histogram":  # histogram chart
                    if not num_cols:  # must have at least one numeric column
                        st.warning("No numeric columns detected.")  # show warning
                    else:
                        pref_y = role.get("value")  # preferred numeric column
                        col_index = num_cols.index(pref_y) if pref_y in num_cols else 0  # select default histogram column
                        col = st.selectbox("Numeric column", num_cols, index=col_index)  # choose column to plot histogram
                        bins = st.slider("Bins", 5, 80, 30)  # choose number of bins
                        fig = px.histogram(plot_df, x=col, nbins=bins,
                                        title=f"Histogram — {col}",
                                        color_discrete_sequence=palette_seq)  # create histogram
                        fig.update_traces(hovertemplate=f"{col}=%{{x}}<br>Count=%{{y}}<extra></extra>")  # custom hover
                        st.plotly_chart(fig, use_container_width=True)  # display chart

                else:  # Correlation Heatmap
                    nums = plot_df.select_dtypes(include="number")  # select numeric columns
                    if nums.shape[1] < 2:  # need at least two numeric columns for correlation
                        st.warning("A correlation heatmap requires at least two numeric columns.")  # show warning
                    else:
                        corr = nums.corr(numeric_only=True)  # compute correlation matrix
                        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap",
                                        color_continuous_scale=["#fff1fa", "#9b8aff", PRIMARY])  # build heatmap
                        st.plotly_chart(fig, use_container_width=True)  # display correlation heatmap

            # --- LEAFLET MAP path (auto or forced) ---
            if inter_mode == "Leaflet (map)":  # if user or auto chose Leaflet map mode
                st.write("**Map preview (Leaflet via folium):**")  # caption for map section
                cols = list(plot_df.columns)  # list of column names
                lat_candidates = [c for c in cols if str(c).lower() in ["lat","latitude"]]  # columns likely to be latitude
                lon_candidates = [c for c in cols if str(c).lower() in ["lon","long","lng","longitude"]]  # columns likely to be longitude
                country_col = _country_col(cols)  # detect a country-like column if exists

                if lat_candidates and lon_candidates:  # if we have both lat and lon candidates
                    lat_col = st.selectbox("Latitude column", lat_candidates)  # user selects latitude column
                    lon_col = st.selectbox("Longitude column", lon_candidates)  # user selects longitude column
                    popup_col = st.selectbox("Popup/label column (optional)", ["(none)"] + [c for c in cols if c not in [lat_col, lon_col]])  # optional popup label column
                    center_lat = plot_df[lat_col].dropna().astype(float).mean() if not plot_df[lat_col].dropna().empty else 3.1390  # compute map center latitude or use default (KL)
                    center_lon = plot_df[lon_col].dropna().astype(float).mean() if not plot_df[lon_col].dropna().empty else 101.6869  # compute map center longitude or default
                    m = folium.Map(location=[center_lat, center_lon], tiles="OpenStreetMap", zoom_start=4)  # create base map
                    cluster = MarkerCluster().add_to(m)  # create marker cluster layer to handle many points
                    for _, row in plot_df.dropna(subset=[lat_col, lon_col]).iterrows():  # loop through rows that have valid lat/lon
                        popup_text = None if popup_col=="(none)" else str(row[popup_col])  # get popup text if chosen
                        folium.CircleMarker(
                            location=[float(row[lat_col]), float(row[lon_col])],
                            radius=4, weight=1, fill=True, popup=popup_text
                        ).add_to(cluster)  # add circle marker for each point to cluster
                    st_folium(m, use_container_width=True, returned_objects=[])  # render Folium map inside Streamlit

                elif country_col and num_cols:  # if no lat/lon but we have a country-like column and numeric columns
                    target_metric = st.selectbox("Metric (numeric)", num_cols)  # choose numeric metric for choropleth
                    mode = st.selectbox("Country reference", ["auto (names)", "ISO-2", "ISO-3"])  # choose how countries are encoded
                    locmode = {"auto (names)":"country names", "ISO-2":"ISO-2", "ISO-3":"ISO-3"}[mode]  # map option to Plotly locationmode
                    fig = px.choropleth(
                        plot_df.dropna(subset=[country_col]),
                        locations=country_col, locationmode=locmode,
                        color=target_metric, hover_name=country_col,
                        title=f"{target_metric} by Country",
                        color_continuous_scale="RdPu"
                    )  # create choropleth map for chosen metric
                    fig.update_geos(fitbounds="locations", visible=False)  # fit map bounds to data and hide base geography
                    st.plotly_chart(fig, use_container_width=True)  # show choropleth map
                else:
                    st.info("No lat/lon or country columns detected—switch to Plotly charts or map your columns with the Column Mapper.")  # instruct user if no geo info

            # ====== Optional: keep your original CO₂/ASEAN charts if the schema matches ======
            required_cols = {"Year", "Country Name", "CO2_per_capita"}  # columns required for legacy CO2 charts
            has_co2_view = required_cols.issubset(df_base.columns)  # check if current data has all required columns
            if has_co2_view:  # if yes, show legacy CO2 section
                with st.expander("Legacy CO₂/ASEAN charts (from your original code)"):  # collapsible container for original charts
                    ymin, ymax = int(df_base["Year"].min()), int(df_base["Year"].max())  # compute min and max year in data
                    year_range = st.slider("Year range", min_value=ymin, max_value=ymax, value=(max(1990, ymin), ymax), step=1)  # choose year range for charts

                    labels = {
                        "line": "Line: Country vs ASEAN vs World",
                        "bar":  "Bar: Latest Year (ASEAN + World)",
                        "heat": "Heatmap: ASEAN & World",
                    }  # descriptive labels for radio options
                    view = st.radio("Chart", options=["line","bar","heat"], format_func=lambda k: labels[k], horizontal=True)  # choose which legacy chart to show
                    combined = compute_combined(df_base)  # build combined dataset with ASEAN Average
                    if view == "line":   line_chart(combined, year_range)  # show line chart if chosen
                    elif view == "bar":  bar_chart(df_base, year_range)  # show bar chart if chosen
                    else:                heatmap_chart(df_base, year_range)  # otherwise show heatmap

    # ----------------------------
    # 4) Report（自动方法摘要）
    # ----------------------------
    with tab_report:  # content for Report tab
        st.subheader("Methods summary (auto-generated)")  # heading for methods summary
        if len(st.session_state.pipeline)==0:  # if no pipeline steps were recorded
            st.info("No pipeline steps yet. Perform some cleaning in the **Clean & Transform** tab.")  # tell user to perform cleaning first
        else:
            lines = ["# Data Processing Methods", "", "This dataset was processed inside the system as follows:"]  # start building markdown content
            for i, step in enumerate(st.session_state.pipeline, start=1):  # iterate through pipeline steps with index
                lines.append(f"{i}. **{step['step']}** — params: `{step['args']}`")  # add numbered list item describing each step
            md = "\n".join(lines)  # join lines into single markdown string
            st.markdown(md)  # display methods summary in markdown format

            st.download_button(
                "Download methods.md",
                md.encode("utf-8"),
                file_name="methods.md",
                mime="text/markdown"
            )  # allow user to download the methods description as a markdown file

st.divider()  # draw a horizontal divider at bottom of app
st.caption("Upload any dataset → Clean → Visualise. Generic visualisations adapt to your columns; CO₂ charts appear only if applicable.")  # final caption summarising app behaviour
